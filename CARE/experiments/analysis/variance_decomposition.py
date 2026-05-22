"""
experiments/analysis/variance_decomposition.py — CF vs Text 分数判别力分析

核心发现（不同于最初的方差假说）:
  CF 分数的方差跨冷度保持恒定，但正样本得分 (pos_score) 从 L0 的负值
  翻转到 L3 的正值——即 CF 对冷物品给出的是系统性错误信号，而不仅仅是噪声。
  Text 的正样本得分始终为正，这解释了为什么门控函数必须大幅抑制冷物品的 CF。

分析指标:
  1. 每冷度分桶的 z-score = (pos - mean) / std — 正样本判别力
  2. CF z-score 跨冷度的符号翻转曲线
  3. Text z-score 跨冷度的稳定性
  4. 基于判别力翻转点的理论 α 推导

使用:
  python experiments/analysis/variance_decomposition.py --dataset baby --seed 42
  python experiments/analysis/variance_decomposition.py --datasets baby office sports --seeds 42
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.vara import VARA
from data_utils.dataset import MARADataset, CacheManager
from evaluate import build_user_text_pref

DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "analysis" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


@torch.no_grad()
def collect_score_statistics(dataset, seed, device):
    """收集每测试样本的 CF 和 Text 分数统计量。

    关键新增: pos_zscore = (pos_score - mean(100 scores)) / std(100 scores)
    这个量衡量模型在 100 个候选中正确识别正样本的能力。
    z > 0 → 正样本得分高于均值（正确方向）
    z < 0 → 正样本得分低于均值（错误方向）
    """
    data_dir  = DATA_DIR / dataset
    cache_dir = CACHE_DIR / dataset
    ckpt_path = CKPT_DIR / dataset / f"seed{seed}" / "best_model.pt"

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {"d": 64, "tau_score": 1.0})
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)

    model = VARA(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tau = getattr(model, "tau_score", 1.0)

    cache = load_cache(cache_dir, data_dir, device)
    clip_text = cache["clip_text"]

    user_text_pref = build_user_text_pref(
        data_dir / "train_indexed.csv", clip_text, stats["n_users"], device)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(stats["n_items"], device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)

    all_item_cf = model.get_all_item_cf_repr()

    records = []

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [B, 100]
        cand_flat = cand.reshape(-1)

        # ── CF scores [B, 100] ──────────────────────────────────
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_cf[cand_flat].reshape(B, 100, -1)
        cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # ── Text scores [B, 100] ─────────────────────────────────
        u_txt = F.normalize(user_text_pref[users], dim=-1)
        i_txt = F.normalize(clip_text[cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        # ── Per-sample statistics ────────────────────────────────
        cf_mean  = cf_scores.mean(dim=1)
        cf_std   = cf_scores.std(dim=1)
        cf_pos   = cf_scores[:, 0]
        # z-score: how many stds is the positive item above/below the mean
        cf_z = ((cf_pos - cf_mean) / cf_std.clamp(min=1e-8)).cpu().numpy()

        txt_mean = txt_scores.mean(dim=1)
        txt_std  = txt_scores.std(dim=1)
        txt_pos  = txt_scores[:, 0]
        txt_z = ((txt_pos - txt_mean) / txt_std.clamp(min=1e-8)).cpu().numpy()

        cf_pos_np  = cf_pos.cpu().numpy()
        cf_mean_np = cf_mean.cpu().numpy()
        cf_std_np  = cf_std.cpu().numpy()
        cf_var_np  = cf_scores.var(dim=1).cpu().numpy()
        txt_pos_np = txt_pos.cpu().numpy()
        txt_mean_np = txt_mean.cpu().numpy()
        txt_std_np = txt_std.cpu().numpy()
        txt_var_np = txt_scores.var(dim=1).cpu().numpy()

        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            cnt = int(pos_counts[j])
            if cnt == 0:
                bucket = "L0"
            elif cnt <= 4:
                bucket = "L1"
            elif cnt <= 20:
                bucket = "L2"
            else:
                bucket = "L3"

            records.append({
                "bucket":        bucket,
                "item_count":    cnt,
                "cf_mean":       float(cf_mean_np[j]),
                "cf_std":        float(cf_std_np[j]),
                "cf_var":        float(cf_var_np[j]),
                "cf_pos_score":  float(cf_pos_np[j]),
                "cf_zscore":     float(cf_z[j]),
                "txt_mean":      float(txt_mean_np[j]),
                "txt_std":       float(txt_std_np[j]),
                "txt_var":       float(txt_var_np[j]),
                "txt_pos_score": float(txt_pos_np[j]),
                "txt_zscore":    float(txt_z[j]),
            })

    return records


def bucket_analysis(records, dataset_name):
    """分桶 z-score 分析 + 判别力翻转点检测。"""
    buckets = ["L0", "L1", "L2", "L3"]
    bucket_names = {"L0": "L0 (zero-shot)", "L1": "L1 (1-4)",
                    "L2": "L2 (5-20)", "L3": "L3 (>20)"}

    print(f"\n{'='*85}")
    print(f"  CF vs Text 分数判别力分析 — {dataset_name}")
    print(f"  Key metric: z-score = (pos - mean_candidates) / std_candidates")
    print(f"  z > 0 → model discriminates positive correctly")
    print(f"  z < 0 → model systematically fails (pos score below average)")
    print(f"{'='*85}")

    summary = {}

    for bucket in buckets:
        subset = [r for r in records if r["bucket"] == bucket]
        if not subset:
            continue

        n = len(subset)
        cf_z  = np.mean([r["cf_zscore"] for r in subset])
        txt_z = np.mean([r["txt_zscore"] for r in subset])
        cf_pos  = np.mean([r["cf_pos_score"] for r in subset])
        txt_pos = np.mean([r["txt_pos_score"] for r in subset])
        cf_mean = np.mean([r["cf_mean"] for r in subset])
        txt_mean = np.mean([r["txt_mean"] for r in subset])
        cf_std  = np.mean([r["cf_std"] for r in subset])
        txt_std = np.mean([r["txt_std"] for r in subset])
        cf_var  = np.mean([r["cf_var"] for r in subset])
        txt_var = np.mean([r["txt_var"] for r in subset])

        # Fraction of samples where pos_score > mean (i.e., model is "correct")
        cf_correct_frac = np.mean([r["cf_zscore"] > 0 for r in subset])
        txt_correct_frac = np.mean([r["txt_zscore"] > 0 for r in subset])

        summary[bucket] = {
            "n": n,
            "cf_zscore": cf_z, "txt_zscore": txt_z,
            "cf_pos": cf_pos, "txt_pos": txt_pos,
            "cf_mean": cf_mean, "txt_mean": txt_mean,
            "cf_std": cf_std, "txt_std": txt_std,
            "cf_var": cf_var, "txt_var": txt_var,
            "cf_correct_frac": cf_correct_frac,
            "txt_correct_frac": txt_correct_frac,
        }

        print(f"\n  {bucket_names[bucket]:<18}  n={n:>6}")
        print(f"  {'─'*72}")
        print(f"  {'':<18}  {'pos_score':>10}  {'z-score':>10}  {'correct%':>10}"
              f"  {'mean':>10}  {'std':>10}")
        print(f"  {'CF':<18}  {cf_pos:>10.4f}  {cf_z:>10.4f}  {cf_correct_frac:>9.1%}"
              f"  {cf_mean:>10.4f}  {cf_std:>10.4f}")
        print(f"  {'Text':<18}  {txt_pos:>10.4f}  {txt_z:>10.4f}  {txt_correct_frac:>9.1%}"
              f"  {txt_mean:>10.4f}  {txt_std:>10.4f}")

    # ── Discriminability crossover analysis ──────────────────────
    print(f"\n{'='*85}")
    print(f"  判别力分析 — {dataset_name}")
    print(f"{'='*85}")

    l0 = summary.get("L0", {})
    l3 = summary.get("L3", {})

    cf_z_l0 = l0.get("cf_zscore", 0)
    cf_z_l3 = l3.get("cf_zscore", 0)
    txt_z_l0 = l0.get("txt_zscore", 0)
    txt_z_l3 = l3.get("txt_zscore", 0)

    print(f"  CF  z-score:   L0={cf_z_l0:.4f}  →  L3={cf_z_l3:.4f}  "
          f"(Δ={cf_z_l3 - cf_z_l0:+.2f}, flips sign: {cf_z_l0 < 0 < cf_z_l3})")
    print(f"  Text z-score:  L0={txt_z_l0:.4f}  →  L3={txt_z_l3:.4f}  "
          f"(Δ={txt_z_l3 - txt_z_l0:+.2f}, consistently {'positive' if txt_z_l0 > 0 else 'negative'})")

    # Key interpretation
    if cf_z_l0 < 0 and cf_z_l3 > 0:
        print(f"\n  >>> CF signal REVERSES from harmful (z={cf_z_l0:.2f}) to helpful (z={cf_z_l3:.2f}).")
        print(f"  >>> Text signal is consistently {'helpful' if txt_z_l0 > 0 else 'weak'} (z={txt_z_l0:.2f}→{txt_z_l3:.2f}).")
        print(f"  >>> This explains why CARE suppresses CF for cold items: CF is actively WRONG, not just noisy.")
    elif cf_z_l0 < 0 and cf_z_l3 < 0:
        print(f"\n  >>> WARNING: CF z-score is negative even for warm items. Check model quality.")

    return summary


def derive_alpha_from_discriminability(summary, dataset_name):
    """从判别力翻转点推导理论 α。

    思路: CF 的 z-score 随 count 增加从负翻转为正。
    令 c0 为 z-score 刚好等于 0 的交互数（翻转点）。
    在翻转点附近，CF 和 Text 的贡献应大致相当。
    若 w_txt = 1/(1+α·c0) = 0.5 (即 CF 和 Text 权重相等) 在翻转点:
      α = 1/c0

    这给出了 α 的直观解释: α 是翻转交互数的倒数。
    小 α → 翻转需要更多交互 → 保守（更依赖 CF）
    大 α → 翻转在更少的交互 → 激进（更依赖 Text）
    """
    if "L0" not in summary or "L1" not in summary or "L3" not in summary:
        print("  [WARN] 分桶数据不完整")
        return None

    l0_cf_z = summary["L0"]["cf_zscore"]
    l1_cf_z = summary["L1"]["cf_zscore"]
    l3_cf_z = summary["L3"]["cf_zscore"]

    # 线性插值翻转点: L1 median count ≈ 2.5, L2 median ≈ 10
    # z(c) ≈ z_l1 + (c - 2.5) * (z_l3 - z_l1) / (c_l3_median - 2.5)
    # 解 z(c0) = 0 → c0
    c_l1_median = 2.5
    c_l3_median = 50.0  # approximate median for >20 items

    if l3_cf_z > l1_cf_z and l1_cf_z < 0:
        # z increases with count, flip happens between L1 and L3
        slope = (l3_cf_z - l1_cf_z) / (c_l3_median - c_l1_median)
        c0 = c_l1_median - l1_cf_z / slope if abs(slope) > 1e-8 else float("inf")
    else:
        c0 = float("inf")  # no clear crossover

    alpha_from_crossover = 1.0 / c0 if c0 > 0 and c0 < 1e6 else None

    # Alternative: use L0/L3 ratio to estimate transition steepness
    # If z_cf(c) ≈ a·log(1+c) + b, then w_txt should be ~1 when z_cf < 0
    # This justifies inverse function form with α controlling transition

    # Load experimental best α
    try:
        vara_path = ROOT / "results" / f"{dataset_name}_vara_results.json"
        with open(vara_path) as f:
            vara = json.load(f)
        alpha_exp = vara["per_seed"][0].get("gate_best_alpha", None)
    except Exception:
        alpha_exp = None

    print(f"\n  CF z-score trajectory: L0={l0_cf_z:.4f} → L1={l1_cf_z:.4f} → L3={l3_cf_z:.4f}")
    print(f"  估计翻转点 c0:           {c0:.1f} interactions" if c0 < 1e6 else "  翻转点: 未检测到")
    if alpha_from_crossover:
        print(f"  从翻转点推导 α ≈ 1/c0:   {alpha_from_crossover:.4f}")
    if alpha_exp is not None:
        print(f"  实验最优 α_exp:           {alpha_exp}")
        if alpha_from_crossover:
            print(f"  比值 α_cross/α_exp:       {alpha_from_crossover/alpha_exp:.2f}")

    return {
        "c0_crossover": c0 if c0 < 1e6 else None,
        "alpha_from_crossover": alpha_from_crossover,
        "alpha_exp": alpha_exp,
        "l0_cf_z": l0_cf_z, "l1_cf_z": l1_cf_z, "l3_cf_z": l3_cf_z,
    }


def main():
    parser = argparse.ArgumentParser(description="CF vs Text 分数判别力分析")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else [args.dataset]

    all_alpha_results = {}
    all_summaries = {}

    for ds in datasets:
        all_records = []
        for seed in args.seeds:
            print(f"\n  [{ds}] seed={seed} — 收集分数统计...")
            records = collect_score_statistics(ds, seed, device)
            if records:
                all_records.extend(records)

        if not all_records:
            print(f"  [SKIP] {ds}: 无数据")
            continue

        summary = bucket_analysis(all_records, ds)
        all_summaries[ds] = summary
        alpha_result = derive_alpha_from_discriminability(summary, ds)
        all_alpha_results[ds] = alpha_result

    # ── Multi-dataset comparison ────────────────────────────────
    if len(datasets) > 1:
        print(f"\n{'='*85}")
        print(f"  跨数据集判别力对比")
        print(f"{'='*85}")
        print(f"  {'Dataset':<16} {'CF z L0':>10} {'CF z L3':>10}"
              f"  {'TXT z L0':>10} {'c0':>8} {'α_cross':>10} {'α_exp':>10}")
        print(f"  {'─'*80}")
        for ds in datasets:
            s = all_summaries.get(ds, {})
            r = all_alpha_results.get(ds, {})
            if s and r:
                l0 = s.get("L0", {})
                l3 = s.get("L3", {})
                c0 = r.get("c0_crossover")
                ac = r.get("alpha_from_crossover")
                ae = r.get("alpha_exp")
                print(f"  {ds:<16} {l0.get('cf_zscore',0):>10.4f} {l3.get('cf_zscore',0):>10.4f}"
                      f"  {l0.get('txt_zscore',0):>10.4f} "
                      f"  {c0 if c0 else 'N/A':>8} "
                      f"  {ac if ac else 'N/A':>10} "
                      f"  {ae if ae else 'N/A':>10}")

    # Save
    out_path = RESULT_DIR / "variance_decomposition_results.json" 
    with open(out_path, "w") as f:
        json.dump({
            "dataset": datasets[0] if len(datasets) == 1 else datasets,
            "seeds": args.seeds,
            "summaries": all_summaries,
            "crossover_analysis": all_alpha_results,
        }, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
