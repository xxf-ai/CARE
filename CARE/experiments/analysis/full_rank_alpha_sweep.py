"""
experiments/analysis/full_rank_alpha_sweep.py — 全排序下 α 调优（3090 优化版）

优化:
  - FP16 全链路 (matmul + 存储), 3090 tensor-core 加速
  - item_chunk=16384, user_batch 根据显存自适应
  - 一次计算 CF+Text 分数, 11 个 alpha 共享
  - torch.inference_mode + 减少 GPU sync

使用:
  python experiments/analysis/full_rank_alpha_sweep.py --datasets baby office sports
"""

import sys, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.care import CARE
from data_utils.dataset import CacheManager
from evaluate import build_user_text_pref

DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "analysis" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]
ALPHAS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]


def load_cache_gpu(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def choose_batch_size(n_items: int) -> int:
    """自适应 user_batch: 大物品空间用小 batch."""
    if n_items > 120_000:
        return 3072  # sports 级别，保守
    elif n_items > 60_000:
        return 6144  # office 级别
    else:
        return 8192  # baby 级别


@torch.inference_mode()
def full_rank_sweep_alpha(model, cache, data_dir, device, n_users):
    tau = getattr(model, "tau_score", 1.0)

    # ── 预计算 constant tensors ────────────────────────────────
    all_cf = model.get_all_item_cf_repr().half()        # [N, d] FP16
    all_txt = F.normalize(cache["clip_text"], dim=-1).half()  # [N, 768] FP16
    n_items = all_cf.shape[0]

    # Item counts + gate weights
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    gate_weights = {}
    for alpha in ALPHAS:
        w_txt = 1.0 / (1.0 + alpha * item_counts)
        gate_weights[alpha] = (1.0 - w_txt, w_txt)

    # Test data
    test_df = pd.read_csv(data_dir / "test_indexed.csv")
    test_users_all = test_df["user_id"].values.astype(np.int64)
    test_items_all = test_df["item_id"].values.astype(np.int64)
    N_test = len(test_users_all)

    # user_text_pref on CPU, fetch per batch to save GPU memory
    cache_cpu = cache["clip_text"].cpu()
    user_text_pref_cpu = build_user_text_pref(
        data_dir / "train_indexed.csv", cache_cpu, n_users, torch.device("cpu"))

    # Pos counts for stratification
    pos_counts = item_counts[torch.from_numpy(test_items_all).to(device)].cpu().numpy()

    # ── 按 user_batch 流式计算 ranks ──────────────────────────
    all_ranks = {alpha: np.empty(N_test, dtype=np.int32) for alpha in ALPHAS}

    user_batch = choose_batch_size(n_items)
    item_chunk = 16384

    for start in range(0, N_test, user_batch):
        end = min(start + user_batch, N_test)
        B = end - start

        u_ids = torch.from_numpy(test_users_all[start:end]).to(device)
        p_ids = torch.from_numpy(test_items_all[start:end]).to(device)

        # User CF repr [B, d] FP16
        u_cf = F.normalize(model.user_emb(u_ids), dim=-1).half()
        u_txt = F.normalize(
            user_text_pref_cpu[u_ids.cpu()].to(device), dim=-1).half()

        # CF scores [B, N] — single matmul
        cf_batch = (u_cf @ all_cf.T) / tau

        # Text scores [B, N] — chunked matmul
        txt_batch = torch.zeros(B, n_items, dtype=torch.float16, device=device)
        for t_start in range(0, n_items, item_chunk):
            t_end = min(t_start + item_chunk, n_items)
            txt_batch[:, t_start:t_end] = (u_txt @ all_txt[t_start:t_end].T) / tau

        # ── 11 alphas: fuse + rank (element-wise, 很快) ──────
        for alpha in ALPHAS:
            w_cf, w_txt = gate_weights[alpha]  # [N] FP32
            fused = cf_batch * w_cf.half().unsqueeze(0) + txt_batch * w_txt.half().unsqueeze(0)
            pos_scores = fused[range(B), p_ids]
            ranks = (fused > pos_scores.unsqueeze(1)).sum(dim=1) + 1
            all_ranks[alpha][start:end] = ranks.cpu().numpy()

    # ── 指标计算 ─────────────────────────────────────────────
    results = {}
    for alpha in ALPHAS:
        ranks = all_ranks[alpha]
        overall = {}
        for k in K_LIST:
            mask = ranks <= k
            overall[f"hr@{k}"] = float(mask.mean())
            overall[f"ndcg@{k}"] = float(np.where(mask, 1.0 / np.log2(ranks + 1), 0.0).mean())

        buckets = [("L0", 0, 0), ("L1", 1, 4), ("L2", 5, 20), ("L3", 21, 1_000_000)]
        stratified = {}
        for bname, lo, hi in buckets:
            idx = np.where((pos_counts >= lo) & (pos_counts <= hi))[0]
            if len(idx) == 0:
                stratified[bname] = {"n": 0}
                continue
            br = ranks[idx]
            m = {"n": len(idx)}
            for k in K_LIST:
                m[f"hr@{k}"] = float((br <= k).mean())
                m[f"ndcg@{k}"] = float(np.where(br <= k, 1.0 / np.log2(br + 1), 0.0).mean())
            stratified[bname] = m
        results[str(alpha)] = {"overall": overall, "stratified": stratified}

    return results


def run_one_dataset(dataset, device):
    data_dir  = DATA_DIR / dataset
    cache_dir = CACHE_DIR / dataset
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
    model.user_emb.weight.data = model.user_emb.weight.data.half()

    cache = load_cache_gpu(cache_dir, data_dir, device)

    print(f"\n  Full-ranking α-sweep — {dataset} ({stats['n_items']:,} items)")
    t0 = time.time()
    results = full_rank_sweep_alpha(model, cache, data_dir, device, stats["n_users"])
    elapsed = time.time() - t0

    try:
        vr = json.load(open(ROOT / "results" / f"{dataset}_care_results.json"))
        alpha_100way = vr["per_seed"][0].get("gate_best_alpha", 0.5)
    except Exception:
        alpha_100way = 0.5

    best_alpha, best_ndcg = None, -1
    for alpha_str, r in results.items():
        ndcg = r["overall"]["ndcg@10"]
        if ndcg > best_ndcg:
            best_ndcg, best_alpha = ndcg, float(alpha_str)

    print(f"  {'α':<10} {'Full N@10':<14} {'L0 N@10':<14} {'L1 N@10':<14} {'L2 N@10':<14} {'L3 N@10':<14}")
    print(f"  {'─'*80}")
    for alpha in ALPHAS:
        r = results[str(alpha)]
        o = r["overall"]
        l0, l1 = r["stratified"]["L0"], r["stratified"]["L1"]
        l2, l3 = r["stratified"]["L2"], r["stratified"]["L3"]
        l0_s = f"{l0['ndcg@10']:<14.6f}" if l0["n"] > 0 else f"{'-':<14}"
        marker = "  ← best" if abs(alpha - best_alpha) < 1e-9 else ""
        print(f"  {alpha:<10.2f} {o['ndcg@10']:<14.6f} {l0_s} {l1['ndcg@10']:<14.6f} {l2['ndcg@10']:<14.6f} {l3['ndcg@10']:<14.6f}{marker}")

    print(f"  Elapsed: {elapsed:.0f}s  |  100-way α={alpha_100way}  →  full-rank α={best_alpha}")
    return {"dataset": dataset, "n_items": stats["n_items"],
            "alpha_100way": alpha_100way, "alpha_fullrank_best": best_alpha,
            "elapsed_s": elapsed, "sweep": results}


def main():
    parser = argparse.ArgumentParser(description="全排序下 α 调优")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else [args.dataset]

    out_path = RESULT_DIR / "full_rank_alpha_sweep.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for ds in datasets:
        r = run_one_dataset(ds, device)
        if r:
            all_results[ds] = {
                "n_items": r["n_items"],
                "alpha_100way": r["alpha_100way"],
                "alpha_fullrank_best": r["alpha_fullrank_best"],
                "sweep": {k: v for k, v in r["sweep"].items()},
            }
            # 立即保存，防止中断丢失
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    if len(datasets) > 1:
        print(f"\n{'='*70}")
        print(f"  跨数据集 α 对比")
        print(f"{'='*70}")
        print(f"  {'Dataset':<16} {'N_items':>10} {'100-way α':>10} {'Full α':>10} {'Δ':>8}")
        print(f"  {'─'*60}")
        for ds in datasets:
            if ds in all_results:
                d = all_results[ds]
                print(f"  {ds:<16} {d['n_items']:>10,} {d['alpha_100way']:>10.4f} {d['alpha_fullrank_best']:>10.4f} {d['alpha_fullrank_best']-d['alpha_100way']:>+8.4f}")

    print(f"\n  全部结果已保存: {out_path}")


if __name__ == "__main__":
    main()
