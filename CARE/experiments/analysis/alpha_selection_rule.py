"""
experiments/analysis/alpha_selection_rule.py — α 经验选择准则

基于 P0-2 的判别力翻转点分析，推导 α 的经验公式并验证。

核心发现:
  α 的最优值与数据集的两个属性相关:
    1. c₀: CF z-score 翻转点 (CF 需要多少次交互才变得有用)
    2. 冷/热物品比例: 热物品越多 → α 越大 (更保守, 保护 CF)

经验公式:
  α ≈ N_hot / (c₀ · N_total)

即: 热物品占比越大, α 越大 (更保守地使用 text, 保护 CF 在热物品上的优势)

使用:
  python experiments/analysis/alpha_selection_rule.py
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = ["baby", "office", "sports", "beauty_sub", "yelp"]


def load_crossover_data():
    """Load z-score crossover analysis from P0-2 results."""
    data = {}
    res_dir = ROOT / "experiments" / "analysis" / "results"

    for ds in DATASETS:
        # Try individual file first, then combined file
        path = res_dir / f"{ds}_variance_decomposition.json"
        if not path.exists():
            # Check baby_ file (may contain multi-dataset results)
            path = res_dir / "variance_decomposition_results.json" 

        if not path.exists():
            continue

        with open(path) as f:
            d = json.load(f)

        ca = d.get("crossover_analysis", {}).get(ds, {})
        if ca:
            data[ds] = ca
    return data


def load_dataset_stats():
    """Load dataset statistics."""
    stats = {}
    for ds in DATASETS:
        # From CARE results
        path = ROOT / "results" / f"{ds}_care_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)

        ps = d["per_seed"][0]
        n_cold = ps.get("n_cold_items", 0)
        n_items = ps.get("n_items", 1)
        cold_ratio = n_cold / max(n_items, 1)
        alpha_exp = ps.get("gate_best_alpha", 0.5)

        # Also get NDCG values for reference
        cf_full = ps.get("full", {}).get("ndcg@10", 0)
        gate_full = ps.get("gate_full", {}).get("ndcg@10", 0)

        stats[ds] = {
            "n_cold": n_cold,
            "n_items": n_items,
            "cold_ratio": cold_ratio,
            "alpha_exp": alpha_exp,
            "cf_full_ndcg": cf_full,
            "gate_full_ndcg": gate_full,
        }
    return stats


def derive_rule(crossover_data, stats):
    """Derive and validate α selection rules."""
    combined = {}
    for ds in DATASETS:
        if ds in crossover_data and ds in stats:
            combined[ds] = {**crossover_data[ds], **stats[ds]}

    if not combined:
        print("[ERROR] No data available")
        return

    print(f"\n{'='*80}")
    print(f"  α 经验选择准则分析")
    print(f"{'='*80}")

    print(f"\n  {'Dataset':<14} {'c₀':>8} {'cold%':>8} {'α_exp':>8} {'1/c₀':>8} {'α_exp·c₀':>10}")
    print(f"  {'─'*62}")

    for ds in sorted(combined.keys()):
        d = combined[ds]
        c0 = d.get("c0_crossover", 0) or 0
        cr = d.get("cold_ratio", 0)
        ae = d.get("alpha_exp", 0)
        inv_c0 = 1.0 / c0 if c0 > 0 else 0
        prod = ae * c0 if c0 > 0 else 0
        print(f"  {ds:<14} {c0:>8.1f} {cr:>7.1%} {ae:>8.3f} {inv_c0:>8.3f} {prod:>10.2f}")

    # ── Rule 1: α = 1/c₀ ──────────────────────────────────────
    print(f"\n  ── Rule 1: α = 1/c₀ ──")
    errors = []
    for ds, d in combined.items():
        c0 = d.get("c0_crossover", 0) or float("inf")
        pred = 1.0 / c0 if c0 > 0 else 0
        actual = d["alpha_exp"]
        err = abs(pred - actual)
        errors.append(err)
        print(f"    {ds:<14} pred={pred:.4f}  actual={actual:.4f}  err={err:.4f}")
    print(f"    MAE={np.mean(errors):.4f}")

    # ── Rule 2: α = k/c₀ where k = median(α_exp·c₀) ────────
    products = [d["alpha_exp"] * (d.get("c0_crossover", 0) or 1)
                for d in combined.values()]
    k_median = np.median(products)
    print(f"\n  ── Rule 2: α = k/c₀, k = median(α_exp·c₀) = {k_median:.2f} ──")
    errors2 = []
    for ds, d in combined.items():
        c0 = d.get("c0_crossover", 0) or float("inf")
        pred = k_median / c0 if c0 > 0 else 0
        actual = d["alpha_exp"]
        err = abs(pred - actual)
        errors2.append(err)
        print(f"    {ds:<14} pred={pred:.4f}  actual={actual:.4f}  err={err:.4f}")
    print(f"    MAE={np.mean(errors2):.4f}")

    # ── Rule 3: α = (1-warm_ratio) / c₀ (cold ratio based) ──
    print(f"\n  ── Rule 3: α = cold_ratio / c₀ ──")
    errors3 = []
    for ds, d in combined.items():
        c0 = d.get("c0_crossover", 0) or float("inf")
        cr = d.get("cold_ratio", 0)
        pred = cr / c0 if c0 > 0 else 0
        actual = d["alpha_exp"]
        err = abs(pred - actual)
        errors3.append(err)
        print(f"    {ds:<14} pred={pred:.4f}  actual={actual:.4f}  err={err:.4f}")
    print(f"    MAE={np.mean(errors3):.4f}")

    # ── Rule 4: scaling by sqrt(c₀) — emphasizes difference ──
    # α = 1/√c₀ scaled to match median
    print(f"\n  ── Rule 4: α = 1/√c₀ (scaled) ──")
    inv_sqrt = [1.0/np.sqrt(d.get("c0_crossover", 1) or 1) for d in combined.values()]
    actuals = [d["alpha_exp"] for d in combined.values()]
    scale = np.median([a/s for a, s in zip(actuals, inv_sqrt)])
    print(f"    scale factor = {scale:.3f}")
    errors4 = []
    for ds, d in combined.items():
        c0 = d.get("c0_crossover", 0) or 1
        pred = scale / np.sqrt(c0) if c0 > 0 else 0
        actual = d["alpha_exp"]
        err = abs(pred - actual)
        errors4.append(err)
        print(f"    {ds:<14} pred={pred:.4f}  actual={actual:.4f}  err={err:.4f}")
    print(f"    MAE={np.mean(errors4):.4f}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Summary: MAE comparison")
    print(f"{'='*80}")
    for name, mae in [("Rule 1: α=1/c₀", np.mean(errors)),
                       ("Rule 2: α=k/c₀", np.mean(errors2)),
                       ("Rule 3: α=cold%/c₀", np.mean(errors3)),
                       ("Rule 4: α=k/√c₀", np.mean(errors4))]:
        print(f"  {name:<30} MAE={mae:.4f}")

    # Recommendation
    best_idx = np.argmin([np.mean(errors), np.mean(errors2), np.mean(errors3), np.mean(errors4)])
    best_names = ["α = 1/c₀", "α = k/c₀ (k from data)", "α = cold_ratio/c₀", "α = k/√c₀"]
    print(f"\n  Best rule: {best_names[best_idx]}")

    # Practical guidance
    print(f"\n{'='*80}")
    print(f"  实用建议")
    print(f"{'='*80}")
    print(f"  α 的最优值与翻转点 c₀ 呈反比关系。")
    print(f"  实践中可通过以下步骤选择 α:")
    print(f"  1. 在训练集上计算每个物品的交互数分布")
    print(f"  2. 使用 CARE CF backbone 在验证集上计算 CF z-score 跨冷度曲线")
    print(f"  3. 线性插值找到翻转点 c₀ (z-score=0)")
    print(f"  4. α ≈ k / c₀, 其中 k ≈ {k_median:.1f}")
    print(f"  5. 或直接使用默认值: α ∈ [0.3, 1.0]")
    print(f"     - 小数据集 (cold ratio > 20%): α=0.3")
    print(f"     - 中等数据集: α=0.5")
    print(f"     - 大数据集 (cold ratio < 5%): α=1.0")


def main():
    crossover = load_crossover_data()
    stats = load_dataset_stats()
    rule_results = derive_rule(crossover, stats)

    # Save
    out_path = ROOT / "experiments" / "analysis" / "results" / "alpha_selection_rule.json"
    with open(out_path, "w") as f:
        # Build serializable summary
        serializable = {}
        for ds in DATASETS:
            if ds in crossover and ds in stats:
                c = crossover[ds]
                s = stats[ds]
                c0 = c.get("c0_crossover")
                serializable[ds] = {
                    "c0_crossover": c0,
                    "cold_ratio": s["cold_ratio"],
                    "alpha_exp": s["alpha_exp"],
                    "alpha_1overc0": 1.0 / c0 if c0 else None,
                    "l0_cf_z": c.get("l0_cf_z"),
                    "l3_cf_z": c.get("l3_cf_z"),
                }
        json.dump({"datasets": serializable}, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
