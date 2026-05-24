"""
experiments/baselines/run_stratified_baselines.py — 基线模型分层评估

对 LightGCN、BM3、FREEDOM (及 BPR-MF、VBPR) 重跑 Table 3 格式的分层评估，
证明 CF 在零交互物品上的崩溃不是 BPR-MF 特有的。

用法:
  python experiments/baselines/run_stratified_baselines.py --dataset baby --seeds 42 123 456
  python experiments/baselines/run_stratified_baselines.py --dataset office --seeds 42
  python experiments/baselines/run_stratified_baselines.py --dataset sports --seeds 42 123 456
  python experiments/baselines/run_stratified_baselines.py --dataset baby --models BPR LIGHTGCN
"""

import sys, json, argparse, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_utils.dataset import MARADataset, CacheManager
from experiments.baselines.models.baselines import BPR_MF, LightGCN_Rec, VBPR, BM3, FREEDOM
from experiments.baselines.models.coldstart_models import CCFCRec, MARec
from experiments.baselines.models.graph_models import LGMRec, PEARL
from experiments.baselines.models.prompt_models import PromptMM
from evaluate import get_cold_items, build_user_text_pref

DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "baselines" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]

# ── 模型构建 ──────────────────────────────────────────────────────────────

def build_model(model_name, n_users, n_items, dataset):
    """构建指定类型的基线模型。"""
    d = 64
    if model_name == "BPR":
        return BPR_MF(n_users, n_items, d=d)
    elif model_name == "LIGHTGCN":
        user_path = CACHE_DIR / dataset / "user_emb.npy"
        item_path = CACHE_DIR / dataset / "item_emb.npy"
        if user_path.exists() and item_path.exists():
            user_emb_np = np.load(user_path)
            item_emb_np = np.load(item_path)
        else:
            print(f"  [WARN] LightGCN 预训练嵌入缺失，使用随机初始化")
            user_emb_np = np.random.randn(n_users, d).astype(np.float32) * 0.01
            item_emb_np = np.random.randn(n_items, d).astype(np.float32) * 0.01
        return LightGCN_Rec(user_emb_np, item_emb_np, d=d)
    elif model_name == "VBPR":
        return VBPR(n_users, n_items, clip_dim=768, d=d)
    elif model_name == "BM3":
        return BM3(n_users, n_items, clip_dim=768, collab_dim=d, d=d)
    elif model_name == "FREEDOM":
        return FREEDOM(n_users, n_items, clip_dim=768, collab_dim=d, d=d)
    elif model_name == "CCFCRec":
        return CCFCRec(n_users, n_items, text_dim=768, d=d)
    elif model_name == "MARec":
        return MARec(n_users, n_items, text_dim=768, d=d)
    elif model_name == "LGMRec":
        return LGMRec(n_users, n_items, text_dim=768, d=d)
    elif model_name == "PEARL":
        return PEARL(n_users, n_items, text_dim=768, d=d)
    elif model_name == "PromptMM":
        return PromptMM(n_users, n_items, text_dim=768, d=d)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── 分层评估核心 ──────────────────────────────────────────────────────────

@torch.no_grad()
def stratified_eval(model, model_name, cache, test_df, item_counts, n_items,
                    device, seed, data_dir, alpha=1.0):
    """对单个模型运行 L0/L1/L2/L3 分层评估（CF only / Text only / CARE gate）。"""
    tau = 1.0
    buckets = [
        ("L0 (zero-shot)", 0, 0),
        ("L1 (1-4)",       1, 4),
        ("L2 (5-20)",      5, 20),
        ("L3 (>20)",      21, 1_000_000),
    ]

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)

    strategies = ["cf_only", "text_only", "gate"]
    bucket_ranks = {bname: {s: [] for s in strategies} for bname, _, _ in buckets}

    base_model = getattr(model, "_orig_mod", model)

    # 确定 CF embedding 来源：尝试调用 get_user_cf_repr / get_all_item_cf_repr
    can_extract_cf = (hasattr(base_model, 'get_user_cf_repr') and
                      hasattr(base_model, 'get_all_item_cf_repr'))

    if not can_extract_cf:
        print(f"  [WARN] {model_name}: 无 get_user_cf_repr/get_all_item_cf_repr, 仅报告 text_only")
        strategies = ["text_only"]

    clip_text = cache["clip_text"]
    user_text_pref = cache["user_text_pref"]

    # 预计算 all_item_cf_repr
    if can_extract_cf:
        all_item_cf = base_model.get_all_item_cf_repr()  # [N_items, d]

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # Text scores (zero-shot CLIP, 模型无关)
        u_txt = F.normalize(user_text_pref[users], dim=-1)
        i_txt = F.normalize(clip_text[cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau
        txt_ranks = ((txt_scores > txt_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        cf_ranks = gate_ranks = None
        if can_extract_cf:
            # 使用模型自己的 get_user_cf_repr / get_all_item_cf_repr
            u_cf = base_model.get_user_cf_repr(users)
            i_cf = all_item_cf[cand_flat].reshape(B, 100, -1)
            cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau
            cf_ranks = ((cf_scores > cf_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

            cand_counts = item_counts[cand]
            w_txt = 1.0 / (1.0 + alpha * cand_counts)
            gate_scores = (1.0 - w_txt) * cf_scores + w_txt * txt_scores
            gate_ranks = ((gate_scores > gate_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        # Stratify
        pos_counts = item_counts[pos_items].cpu().numpy()
        for j in range(B):
            cnt = int(pos_counts[j])
            for bname, lo, hi in buckets:
                if lo <= cnt <= hi:
                    bucket_ranks[bname]["text_only"].append(int(txt_ranks[j]))
                    if can_extract_cf:
                        bucket_ranks[bname]["cf_only"].append(int(cf_ranks[j]))
                        bucket_ranks[bname]["gate"].append(int(gate_ranks[j]))
                    break

    # Compute metrics
    result = {}
    for bname, _, _ in buckets:
        bres = {}
        for strategy in strategies:
            ranks_arr = np.array(bucket_ranks[bname][strategy])
            if len(ranks_arr) == 0:
                bres[strategy] = None
                continue
            m = {"n": len(ranks_arr)}
            for k in K_LIST:
                m[f"hr@{k}"]   = float((ranks_arr <= k).mean())
                m[f"ndcg@{k}"] = float(
                    np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1), 0.0).mean())
            bres[strategy] = m
        result[bname] = bres

    return result


def evaluate_one_baseline(model_name, ckpt_path, data_dir, cache_dir,
                          device, seed, dataset, alpha=1.0):
    """加载单个基线模型 checkpoint 并运行分层评估。"""
    if not ckpt_path.exists():
        print(f"  [跳过] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    # 加载 cache
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in cache_mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)

    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)

    # item interaction counts
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    # Build model + load weights
    model = build_model(model_name, n_users, n_items, dataset).to(device)
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_df = pd.read_csv(data_dir / "test_indexed.csv")
    return stratified_eval(model, model_name, cache, test_df, item_counts,
                           n_items, device, seed, data_dir, alpha=alpha)


# ── 打印工具 ──────────────────────────────────────────────────────────────

def _print_stratified(model_name, seed, result, alpha):
    print(f"\n  Stratified (α={alpha}):")
    print(f"  {'Bucket':<18} {'n':>6}  {'CF N@10':>8}  {'TXT N@10':>8}  {'GATE N@10':>8}")
    print(f"  {'─'*58}")
    for bname in ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]:
        b = result.get(bname, {})
        cf  = b.get("cf_only", {})
        txt = b.get("text_only", {})
        gt  = b.get("gate", {})
        n = cf.get("n", 0) if cf else 0
        cfn = f"{cf.get('ndcg@10', 0):.4f}" if cf else "N/A"
        tn  = f"{txt.get('ndcg@10', 0):.4f}" if txt else "N/A"
        gn  = f"{gt.get('ndcg@10', 0):.4f}" if gt else "N/A"
        print(f"  {bname:<18} {n:>6}  {cfn:>8}  {tn:>8}  {gn:>8}")


def _print_multi_seed_summary(all_results):
    print(f"\n{'='*80}")
    print(f"  多 Seed 汇总 — NDCG@10 mean ± std")
    print(f"{'='*80}")

    buckets_order = ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]
    strategies = ["cf_only", "text_only", "gate"]

    for model_name, seeds_dict in all_results.items():
        print(f"\n  [{model_name}]")
        header = f"  {'Bucket':<18}"
        for s in strategies:
            header += f"  {s:>18}"
        print(header)
        print(f"  {'─'*76}")
        for bname in buckets_order:
            row = f"  {bname:<18}"
            for strategy in strategies:
                vals = []
                for seed, data in seeds_dict.items():
                    b = data["result"].get(bname, {})
                    m = b.get(strategy, {}) if b else {}
                    if m and m.get("ndcg@10") is not None:
                        vals.append(m["ndcg@10"])
                if len(vals) >= 2:
                    row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
                elif len(vals) == 1:
                    row += f"  {vals[0]:.4f}"
                else:
                    row += f"  {'N/A':>18}"
            print(row)
    print(f"{'='*80}")


def _load_care_stratified(dataset, seeds, alpha):
    """BPR 行直接复用 CARE 主实验结果中的分层数据，保证与论文 Table 3 一致。"""
    care_path = ROOT / "results" / f"{dataset}_care_results.json"
    if not care_path.exists():
        print(f"  [WARN] CARE 结果文件不存在: {care_path}, 回退到 checkpoint 评估")
        return None

    with open(care_path) as f:
        care = json.load(f)

    bucket_map = {
        "0 (zero-shot)": "L0 (zero-shot)",
        "1-4 (cold)":    "L1 (1-4)",
        "5-20 (warm)":   "L2 (5-20)",
        ">20 (hot)":     "L3 (>20)",
    }

    all_results = {}
    json_seeds = care.get("seeds", seeds)
    for i, per_seed in enumerate(care["per_seed"]):
        gs = per_seed.get("gate_strat")
        if gs is None:
            continue
        seed = json_seeds[i] if i < len(json_seeds) else seeds[i] if i < len(seeds) else i

        result = {}
        for orig_bname, new_bname in bucket_map.items():
            b = gs.get(orig_bname, {})
            result[new_bname] = {
                "cf_only":   b.get("cf_only"),
                "text_only": b.get("text_only"),
                "gate":      b.get("gate"),
            }
        all_results[seed] = {"result": result, "alpha": alpha, "ckpt": "CARE (from paper)"}
    return all_results


# ── 主入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="基线模型分层评估 (Table 3 格式)")
    parser.add_argument("--dataset", default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--models", nargs="+",
                        default=["BPR", "LIGHTGCN", "VBPR", "BM3", "FREEDOM"])
    parser.add_argument("--ckpt_root", type=str, default=None,
                        help="checkpoint 根目录，默认 checkpoints/{dataset}/")
    parser.add_argument("--alpha", type=float, default=None,
                        help="CARE gate α，不指定则从主实验结果自动读取")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = DATA_DIR / args.dataset
    cache_dir = CACHE_DIR / args.dataset

    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else (CKPT_DIR / args.dataset)

    # 模型名 → checkpoint 子目录名
    ckpt_subdirs = {
        "BPR":      "bpr",
        "LIGHTGCN": "lightgcn",
        "VBPR":     "vbpr",
        "BM3":      "bm3",
        "FREEDOM":  "freedom",
        "CCFCRec":  "ccfcrec",
        "MARec":    "marec",
        "LGMRec":   "lgmrec",
        "PEARL":    "pearl",
        "PromptMM": "promptmm",
    }

    def _find_ckpt(subdir, seed):
        """搜索 checkpoint 文件，覆盖常见的目录结构。"""
        candidates = [
            ckpt_root / subdir / f"seed{seed}" / "best_model.pt",
            ckpt_root / f"seed{seed}" / subdir / "best_model.pt",
            ckpt_root / f"{subdir}_seed{seed}" / "best_model.pt",
            ckpt_root / subdir / f"best_model_seed{seed}.pt",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # 自动读取最佳 alpha
    if args.alpha is None:
        try:
            care_results = json.load(
                open(ROOT / "results" / f"{args.dataset}_care_results.json"))
            best_alpha = care_results["per_seed"][0].get("gate_best_alpha", 1.0)
        except Exception:
            best_alpha = 1.0 if args.dataset in ("baby", "yelp") else 0.5
        alphas_to_try = [best_alpha]
        print(f"  自动选择 α={best_alpha} (来自主实验)")
    else:
        alphas_to_try = [args.alpha]

    print(f"\n{'='*70}")
    print(f"  基线分层评估  dataset={args.dataset}  seeds={args.seeds}")
    print(f"  models={args.models}  alpha(s)={alphas_to_try}")
    print(f"{'='*70}")

    all_results = defaultdict(dict)

    for model_name in args.models:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")

        # BPR 行直接复用 CARE 主实验分层数据，保证与论文 Table 3 一致
        if model_name == "BPR":
            bpr_results = _load_care_stratified(args.dataset, args.seeds, alphas_to_try[0])
            if bpr_results:
                all_results[model_name] = bpr_results
                for seed, data in bpr_results.items():
                    _print_stratified(model_name, seed, data["result"], alphas_to_try[0])
                continue

        subdir = ckpt_subdirs.get(model_name, model_name.lower())
        for seed in args.seeds:
            ckpt_path = _find_ckpt(subdir, seed)

            if ckpt_path is None:
                print(f"  seed={seed}: checkpoint 未找到 (root={ckpt_root})")
                continue

            print(f"  seed={seed}: {ckpt_path}")

            for alpha in alphas_to_try:
                result = evaluate_one_baseline(
                    model_name, ckpt_path, data_dir, cache_dir,
                    device, seed, args.dataset, alpha=alpha)
                if result is not None:
                    all_results[model_name][seed] = {
                        "result": result, "alpha": alpha, "ckpt": str(ckpt_path),
                    }
                    _print_stratified(model_name, seed, result, alpha)

    if not all_results:
        print("\n[错误] 没有任何模型评估成功，请检查 checkpoint 路径")
        return

    _print_multi_seed_summary(all_results)

    # 保存
    out_path = RESULT_DIR / f"{args.dataset}_stratified_baselines.json"
    serializable = {}
    for model_name, seeds_dict in all_results.items():
        serializable[model_name] = {}
        for seed, data in seeds_dict.items():
            serializable[model_name][str(seed)] = {
                "alpha": data["alpha"],
                "ckpt": data["ckpt"],
                "result": data["result"],
            }

    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "seeds": args.seeds,
            "models": list(all_results.keys()),
            "timestamp": datetime.datetime.now().isoformat(),
            "results": serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
