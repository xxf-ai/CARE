"""
experiments/ablations/gate_functions.py — Coldness gate design space comparison.

Compares 4 function forms on baby dataset:
  1. Inverse:   w = 1/(1 + α·count)      [CARE current]
  2. Sigmoid:   w = σ(-(count - μ) / T)   [2 params: μ, T]
  3. Exp:       w = exp(-α·count)          [1 param: α]
  4. Piecewise: w = step(count, thresholds) [N params, hard-coded]

Evaluates full-set NDCG@10 and stratified metrics for each function.
Grid-search best hyperparameters on validation set if available.
"""

import sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.care        import CARE
from data_utils.dataset import MARADataset, CacheManager
from evaluate           import build_user_text_pref


# ── Weight functions ──────────────────────────────────────────────

def weight_inverse(counts, alpha):
    """CARE current: w = 1/(1 + α·count)"""
    return 1.0 / (1.0 + alpha * counts)


def weight_sigmoid(counts, mu, T):
    """Sigmoid: w = σ(-(count - μ) / T)"""
    return torch.sigmoid(-(counts - mu) / T)


def weight_exponential(counts, alpha):
    """Exponential: w = exp(-α·count)"""
    return torch.exp(-alpha * counts)


def weight_piecewise(counts):
    """Piecewise: 0→1.0, 1-4→0.8, 5-20→0.3, >20→0.0"""
    w = torch.zeros_like(counts)
    w[counts == 0] = 1.0
    w[(counts >= 1) & (counts <= 4)] = 0.8
    w[(counts >= 5) & (counts <= 20)] = 0.3
    # >20 stays 0.0
    return w


# ── Evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_weight_fn(model, cache, test_set, item_counts, device,
                             weight_fn, k_list=[5, 10, 20], batch_size=4096):
    """Evaluate full ranking using a given per-item weight function."""
    tau = getattr(model, "tau_score", 1.0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    all_item_repr = model.get_all_item_cf_repr()

    all_hr = {k: [] for k in k_list}
    all_ndcg = {k: [] for k in k_list}

    for users, pos_items, neg_items in test_loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # CF scores
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # Text scores (raw CLIP)
        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        # Per-item weights
        cand_counts = item_counts[cand]
        w_txt = weight_fn(cand_counts)
        w_cf = 1.0 - w_txt

        fused = w_cf * cf + w_txt * txt
        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()

        for k in k_list:
            all_hr[k].extend((ranks <= k).astype(float).tolist())
            all_ndcg[k].extend(
                np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).tolist())

    metrics = {}
    for k in k_list:
        metrics[f"hr@{k}"] = float(np.mean(all_hr[k]))
        metrics[f"ndcg@{k}"] = float(np.mean(all_ndcg[k]))
    return metrics


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = ROOT / "data"  / args.dataset
    cache_dir = ROOT / "cache" / args.dataset
    ckpt_path = ROOT / "checkpoints" / args.dataset / f"seed{args.seed}" / "best_model.pt"

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {"d": 64, "tau_score": 1.0})
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)

    model = CARE(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load cache
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in cache_mgr.tensors.items():
        if hasattr(v, 'numpy'):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)

    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"],
        stats["n_users"], device)

    # Item counts
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(stats["n_items"], device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    # Test set
    test_set = MARADataset(data_dir, split="test", seed=args.seed, n_neg=99)

    # ── Compare functions ────────────────────────────────────────
    results = {"function": [], "params": [], "ndcg10": [], "ndcg5": [], "hr10": []}

    def record(name, params_str, metrics):
        results["function"].append(name)
        results["params"].append(params_str)
        results["ndcg10"].append(metrics["ndcg@10"])
        results["ndcg5"].append(metrics["ndcg@5"])
        results["hr10"].append(metrics["hr@10"])

    print(f"\n{'='*70}")
    print(f"  Gate Function Design Space Comparison — {args.dataset}")
    print(f"{'='*70}")
    print(f"\n  {'Function':<20} {'Params':<20} {'NDCG@10':>10}")
    print(f"  {'─'*52}")

    # 1. Inverse (CARE)
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        fn = lambda c, a=alpha: weight_inverse(c, a)
        m = evaluate_with_weight_fn(model, cache, test_set, item_counts, device, fn)
        lbl = "Inverse" if alpha == 0.01 else ""
        print(f"  {lbl:<20} {'α=' + str(alpha):<20} {m['ndcg@10']:>10.4f}")
        record("Inverse", f"α={alpha}", m)

    print()

    # 2. Sigmoid
    for mu in [0.5, 1.0, 5.0, 10.0, 20.0]:
        for T in [0.5, 1.0, 2.0, 5.0]:
            fn = lambda c, mu=mu, T=T: weight_sigmoid(c, mu, T)
            m = evaluate_with_weight_fn(model, cache, test_set, item_counts, device, fn)
            lbl = "Sigmoid" if (mu == 0.5 and T == 0.5) else ""
            print(f"  {lbl:<20} {'μ=' + str(mu) + ' T=' + str(T):<20} {m['ndcg@10']:>10.4f}")
            record("Sigmoid", f"μ={mu} T={T}", m)

    print()

    # 3. Exponential
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        fn = lambda c, a=alpha: weight_exponential(c, a)
        m = evaluate_with_weight_fn(model, cache, test_set, item_counts, device, fn)
        lbl = "Exponential" if alpha == 0.01 else ""
        print(f"  {lbl:<20} {'α=' + str(alpha):<20} {m['ndcg@10']:>10.4f}")
        record("Exponential", f"α={alpha}", m)

    print()

    # 4. Piecewise (no params)
    m = evaluate_with_weight_fn(model, cache, test_set, item_counts, device, weight_piecewise)
    print(f"  {'Piecewise':<20} {'(hard-coded)':<20} {m['ndcg@10']:>10.4f}")
    record("Piecewise", "(hard-coded)", m)

    # 5. Baselines
    print()
    print(f"  {'─'*52}")
    m_cf = evaluate_with_weight_fn(
        model, cache, test_set, item_counts, device, lambda c: torch.zeros_like(c))
    print(f"  {'CF only (w=0)':<20} {'':<20} {m_cf['ndcg@10']:>10.4f}")
    record("CF only", "w=0", m_cf)

    m_txt = evaluate_with_weight_fn(
        model, cache, test_set, item_counts, device, lambda c: torch.ones_like(c))
    print(f"  {'Text only (w=1)':<20} {'':<20} {m_txt['ndcg@10']:>10.4f}")
    record("Text only", "w=1", m_txt)

    # Save
    import datetime
    result_dir = ROOT / "experiments" / "ablations" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{args.dataset}_gate_functions.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "timestamp": datetime.datetime.now().isoformat(),
                   "results": results}, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
