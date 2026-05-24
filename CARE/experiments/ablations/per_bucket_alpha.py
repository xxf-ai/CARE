"""
experiments/ablations/per_bucket_alpha.py — Per-bucket α 分段门控

解决全局 α 在 L1 桶性能退化的问题：
  L0 (count=0):  w_txt = 1.0（恒为纯文本，不受 α 影响）
  L1 (1-4):      w_txt = 1/(1 + α₁·count)
  L2+ (≥5):      w_txt = 1/(1 + α₂·count)

在验证集上 grid-search 最优 (α₁, α₂)，与全局 α 对比分层指标。

使用:
  python experiments/ablations/per_bucket_alpha.py --dataset baby
  python experiments/ablations/per_bucket_alpha.py --dataset baby office sports --seeds 42
"""

import sys, json, argparse, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.care         import CARE
from data_utils.dataset  import MARADataset, CacheManager
from evaluate            import build_user_text_pref, get_cold_items

DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "ablations" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]
COLD_THRESHOLD = 5


def load_cache_gpu(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def build_item_counts(data_dir, n_items, device):
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        counts[int(iid)] += 1
    return counts


@torch.no_grad()
def evaluate_per_bucket_alpha(model, cache, test_set, item_counts, device,
                                alpha_l1, alpha_l2plus):
    """使用分段 α 评估: L0→w=1, L1→α₁, L2+→α₂."""
    tau = getattr(model, "tau_score", 1.0)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)
    all_item_repr = model.get_all_item_cf_repr()

    all_hr   = {k: [] for k in K_LIST}
    all_ndcg = {k: [] for k in K_LIST}

    # Per-bucket accumulators
    bucket_hr   = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}
    bucket_ndcg = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [B, 100]
        cand_flat = cand.reshape(-1)

        # CF scores
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # Text scores
        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        # ── Per-bucket α weights ──────────────────────────────────
        cand_counts = item_counts[cand]  # [B, 100]
        # L0 (count=0): w_txt = 1.0 (pure text)
        # L1 (1-4):    w_txt = 1/(1 + α₁·count)
        # L2+ (≥5):    w_txt = 1/(1 + α₂·count)
        w_txt = torch.where(
            cand_counts == 0,
            torch.ones_like(cand_counts, dtype=torch.float32),
            torch.where(
                cand_counts < 5,
                1.0 / (1.0 + alpha_l1 * cand_counts),
                1.0 / (1.0 + alpha_l2plus * cand_counts)
            )
        )

        fused = (1.0 - w_txt) * cf + w_txt * txt

        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()
        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            r = int(ranks[j])
            cnt = int(pos_counts[j])

            if cnt == 0:
                bucket = "L0"
            elif cnt <= 4:
                bucket = "L1"
            elif cnt <= 20:
                bucket = "L2"
            else:
                bucket = "L3"

            for k in K_LIST:
                hit = float(r <= k)
                ndcg_v = (1.0 / np.log2(r + 1)) if r <= k else 0.0
                all_hr[k].append(hit)
                all_ndcg[k].append(ndcg_v)
                bucket_hr[bucket][k].append(hit)
                bucket_ndcg[bucket][k].append(ndcg_v)

    def _metrics(hr_dict, ndcg_dict):
        m = {}
        for k in K_LIST:
            m[f"hr@{k}"]   = float(np.mean(hr_dict[k])) if hr_dict[k] else 0.0
            m[f"ndcg@{k}"] = float(np.mean(ndcg_dict[k])) if ndcg_dict[k] else 0.0
        return m

    result = {"full": _metrics(all_hr, all_ndcg)}
    for b in ["L0", "L1", "L2", "L3"]:
        result[b] = _metrics(bucket_hr[b], bucket_ndcg[b])
    return result


def grid_search(model, cache, data_dir, device, dataset):
    """Grid-search 最优 (α₁, α₂)。用验证集选，返回最优组合。"""
    val_set = MARADataset(data_dir, split="val", seed=42, n_neg=99)

    with open(data_dir / "stats.json") as f:
        n_items = json.load(f)["n_items"]
    item_counts = build_item_counts(data_dir, n_items, device)

    # α₁ candidates: smaller → more text weight → better for L1
    # α₂ candidates: same as CARE global → preserve warm performance
    alpha_l1_cands = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    alpha_l2plus_cands = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

    best_ndcg = -1
    best_combo = (0.1, 0.5)

    print(f"\n  Grid-search 分段 α (验证集 NDCG@10):")
    print(f"  {'α₁ (L1)':<12} {'α₂ (L2+)':<12} {'Full N@10':<12} {'L0 N@10':<12} {'L1 N@10':<12} {'L2 N@10':<12} {'L3 N@10':<12}")
    print(f"  {'─'*84}")

    for a1 in alpha_l1_cands:
        for a2 in alpha_l2plus_cands:
            m = evaluate_per_bucket_alpha(model, cache, val_set, item_counts, device, a1, a2)
            fn = m["full"]["ndcg@10"]
            if fn > best_ndcg:
                best_ndcg = fn
                best_combo = (a1, a2)

            print(f"  {a1:<12.3f} {a2:<12.3f} {m['full']['ndcg@10']:<12.4f} "
                  f"{m['L0']['ndcg@10']:<12.4f} {m['L1']['ndcg@10']:<12.4f} "
                  f"{m['L2']['ndcg@10']:<12.4f} {m['L3']['ndcg@10']:<12.4f}")

    print(f"\n  最优: α₁={best_combo[0]}, α₂={best_combo[1]} (val NDCG@10={best_ndcg:.4f})")
    return best_combo


def compare_global_vs_per_bucket(model, cache, data_dir, device, dataset,
                                   alpha_global, alpha_l1, alpha_l2plus):
    """在测试集上对比全局 α vs 分段 α，输出完整分层指标。"""
    test_set = MARADataset(data_dir, split="test", seed=42, n_neg=99)

    with open(data_dir / "stats.json") as f:
        n_items = json.load(f)["n_items"]
    item_counts = build_item_counts(data_dir, n_items, device)

    # Global α
    m_global = evaluate_per_bucket_alpha(model, cache, test_set, item_counts, device,
                                           alpha_global, alpha_global)
    # Per-bucket α
    m_pb = evaluate_per_bucket_alpha(model, cache, test_set, item_counts, device,
                                       alpha_l1, alpha_l2plus)

    print(f"\n{'='*70}")
    print(f"  全局 α={alpha_global} vs 分段 α=(α₁={alpha_l1}, α₂={alpha_l2plus})")
    print(f"  Dataset: {dataset}")
    print(f"{'='*70}")

    buckets = [("full", "Full"), ("L0", "L0 (0)"), ("L1", "L1 (1-4)"),
               ("L2", "L2 (5-20)"), ("L3", "L3 (>20)")]

    print(f"\n  {'Bucket':<16} {'Global N@10':>14} {'PB N@10':>14} {'Δ':>10}")
    print(f"  {'─'*58}")
    for bkey, bname in buckets:
        g = m_global[bkey]["ndcg@10"]
        p = m_pb[bkey]["ndcg@10"]
        delta = p - g
        marker = " ⬆" if delta > 0.001 else (" ⬇" if delta < -0.001 else "")
        print(f"  {bname:<16} {g:>14.4f} {p:>14.4f} {delta:>+10.4f}{marker}")

    return {"global": m_global, "per_bucket": m_pb, "alpha_l1": alpha_l1, "alpha_l2plus": alpha_l2plus}


def run_one_dataset(dataset, device):
    data_dir  = DATA_DIR / dataset
    cache_dir = CACHE_DIR / dataset

    # Load model (seed 42)
    ckpt_path = CKPT_DIR / dataset / "seed42" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {"d": 64, "tau_score": 1.0})
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)

    model = CARE(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load cache + build user_text_pref
    cache = load_cache_gpu(cache_dir, data_dir, device)
    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], stats["n_users"], device)

    # Get current best α
    try:
        vr = json.load(open(ROOT / "results" / f"{dataset}_care_results.json"))
        alpha_global = vr["per_seed"][0].get("gate_best_alpha", 0.5)
    except Exception:
        alpha_global = 0.5

    print(f"\n{'='*70}")
    print(f"  Per-Bucket α — {dataset}  (global α={alpha_global})")
    print(f"{'='*70}")

    # Grid-search on validation set
    (alpha_l1, alpha_l2plus) = grid_search(model, cache, data_dir, device, dataset)

    # Compare on test set
    result = compare_global_vs_per_bucket(model, cache, data_dir, device, dataset,
                                            alpha_global, alpha_l1, alpha_l2plus)
    result["alpha_global"] = alpha_global
    return result


def main():
    parser = argparse.ArgumentParser(description="Per-bucket α 分段门控")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="用于评估的 seeds（门控权重 grid-search 仅用 seed=42）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else [args.dataset]

    all_results = {}
    for ds in datasets:
        r = run_one_dataset(ds, device)
        if r:
            all_results[ds] = r

    # Save
    out_path = RESULT_DIR / "per_bucket_alpha_results.json"
    with open(out_path, "w") as f:
        # Convert to serializable
        serializable = {}
        for ds, r in all_results.items():
            serializable[ds] = {
                "alpha_global": r["alpha_global"],
                "alpha_l1": r["alpha_l1"],
                "alpha_l2plus": r["alpha_l2plus"],
                "global": r["global"],
                "per_bucket": r["per_bucket"],
            }
        json.dump(serializable, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
