"""
evaluation/full_rank.py — Full-ranking evaluation (all items, not 100-way).

Addresses reviewer concern about 100-way ranking protocol reliability
(Rendle 2020, Wetzel RecSys 2024). Ranks each test positive against ALL
items, not just 99 random negatives.

CF scores: batch matmul, O(B × N × d) — fast.
Text scores: batch matmul, O(B × N × 768) — slower but batchable.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_utils.dataset import MARADataset


K_LIST = [5, 10, 20]


@torch.no_grad()
def full_rank_evaluate(model, cache, data_dir, seed, device,
                        cf_batch=2048, text_batch=256):
    """Full-ranking: score against ALL items for each test user.

    Returns dict with 4 strategies per coldness level:
      cf_only, text_only, gate (with best α), fixed β=0.5/0.7
    """
    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=0)
    # Extract only (user, pos_item) — no negatives needed
    test_loader = DataLoader(test_set, batch_size=cf_batch,
                             shuffle=False, num_workers=2)

    n_items = cache["clip_text"].shape[0]
    tau = getattr(model, "tau_score", 1.0)

    # ── Precompute item reprs ────────────────────────────────────
    all_cf = model.get_all_item_cf_repr()       # [N, d]
    all_txt = F.normalize(cache["clip_text"], dim=-1)  # [N, 768]

    # Coldness weights (CARE function)
    import json
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    # Build user_text_pref
    with open(data_dir / "stats.json") as f:
        n_users = json.load(f)["n_users"]
    from evaluate import build_user_text_pref
    user_text_pref = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)

    # ── Evaluate ─────────────────────────────────────────────────
    all_ranks = {"cf_only": [], "text_only": [], "gate": []}
    all_pos_counts = []

    for users, pos_items, _ in test_loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        B = users.size(0)

        # --- CF scores: [B, d] @ [d, N] = [B, N] ---
        u_cf = F.normalize(model.user_emb(users), dim=-1)
        cf_all = (u_cf @ all_cf.T) / tau  # [B, N]

        # --- Text scores: batch over items if needed ---
        u_txt = F.normalize(user_text_pref[users], dim=-1)  # [B, 768]
        txt_all = torch.zeros(B, n_items, device=device)
        for t_start in range(0, n_items, text_batch):
            t_end = min(t_start + text_batch, n_items)
            i_txt_batch = all_txt[t_start:t_end]  # [batch, 768]
            txt_all[:, t_start:t_end] = (u_txt @ i_txt_batch.T) / tau

        # --- Gate scores (α = best per dataset) ---
        # Use α=0.5 as default; per-dataset override below
        alpha = 0.5
        w_txt = 1.0 / (1.0 + alpha * item_counts.unsqueeze(0))  # [1, N]
        gate_all = (1.0 - w_txt) * cf_all + w_txt * txt_all

        # --- Ranks ---
        cf_ranks = (cf_all > cf_all[range(B), pos_items].unsqueeze(1)).sum(dim=1) + 1
        txt_ranks = (txt_all > txt_all[range(B), pos_items].unsqueeze(1)).sum(dim=1) + 1
        gate_ranks = (gate_all > gate_all[range(B), pos_items].unsqueeze(1)).sum(dim=1) + 1

        all_ranks["cf_only"].extend(cf_ranks.cpu().tolist())
        all_ranks["text_only"].extend(txt_ranks.cpu().tolist())
        all_ranks["gate"].extend(gate_ranks.cpu().tolist())
        all_pos_counts.extend(item_counts[pos_items].cpu().tolist())

    # ── Compute metrics ──────────────────────────────────────────
    # Per coldness level
    buckets = [
        ("0 (zero-shot)", 0, 0),
        ("1-4 (cold)",    1, 4),
        ("5-20 (warm)",   5, 20),
        (">20 (hot)",    21, 1_000_000),
    ]

    result = {"overall": {}, "stratified": {}}

    for strategy in ["cf_only", "text_only", "gate"]:
        ranks = np.array(all_ranks[strategy])
        # Overall
        m = {}
        for k in K_LIST:
            m[f"ndcg@{k}"] = float(
                np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean())
            m[f"hr@{k}"] = float((ranks <= k).mean())
        result["overall"][strategy] = m

        # Stratified
        for bname, lo, hi in buckets:
            idx = [i for i, c in enumerate(all_pos_counts) if lo <= c <= hi]
            if not idx:
                continue
            br = ranks[idx]
            sm = {"n": len(idx)}
            for k in K_LIST:
                sm[f"ndcg@{k}"] = float(
                    np.where(br <= k, 1.0 / np.log2(br + 1), 0.0).mean())
                sm[f"hr@{k}"] = float((br <= k).mean())
            result["stratified"].setdefault(bname, {})[strategy] = sm

    return result


def print_full_rank_report(result, dataset_name):
    """Pretty-print full-ranking results."""
    print(f"\n{'='*70}")
    print(f"  Full-Ranking Evaluation — {dataset_name}")
    print(f"{'='*70}")

    # Overall
    print(f"\n  Overall (all test items):")
    print(f"  {'Strategy':<12} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'HR@10':>8}")
    for s in ["cf_only", "text_only", "gate"]:
        m = result["overall"][s]
        print(f"  {s:<12} {m['ndcg@5']:>8.4f} {m['ndcg@10']:>8.4f} "
              f"{m['ndcg@20']:>8.4f} {m['hr@10']:>8.4f}")

    # Stratified
    print(f"\n  Stratified by coldness:")
    print(f"  {'Bucket':<18} {'n':>6}  {'CF N@10':>8} {'TXT N@10':>8} {'GATE N@10':>8}")
    for bname in ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]:
        b = result["stratified"].get(bname, {})
        cf  = b.get("cf_only", {})
        txt = b.get("text_only", {})
        gt  = b.get("gate", {})
        n = cf.get("n", 0)
        cfn = f"{cf.get('ndcg@10',0):.4f}" if cf else "N/A"
        tn  = f"{txt.get('ndcg@10',0):.4f}" if txt else "N/A"
        gn  = f"{gt.get('ndcg@10',0):.4f}" if gt else "N/A"
        print(f"  {bname:<18} {n:>6}  {cfn:>8}  {tn:>8}  {gn:>8}")


# ── Standalone runner ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from models.vara import VARA

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = ROOT / "data" / args.dataset
    cache_dir = ROOT / "cache" / args.dataset
    ckpt_path = ROOT / "checkpoints" / args.dataset / f"seed{args.seed}" / "best_model.pt"

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    model = VARA(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load cache
    from data_utils.dataset import CacheManager
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in cache_mgr.tensors.items():
        if hasattr(v, 'numpy'):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)

    result = full_rank_evaluate(model, cache, data_dir, args.seed, device)
    print_full_rank_report(result, args.dataset)

    # Save results
    import datetime
    result_dir = ROOT / "results" / "full_rank"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{args.dataset}_seed{args.seed}_fullrank.json"
    # Convert numpy types for JSON
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "timestamp": datetime.datetime.now().isoformat(),
                   "result": to_native(result)}, f, indent=2)
    print(f"\n  结果已保存: {out_path}")
