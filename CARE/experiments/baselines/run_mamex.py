"""
experiments/baselines/run_mamex.py — Train and evaluate MAMEX-style baseline.

Runs per-item learnable modality gating (MoE-inspired) on baby/office/sports,
then reports full-set, cold-start, and stratified metrics for comparison with CARE.
"""

import sys, os, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from experiments.baselines.mamex_style import MAMEXStyle, mamex_loss
from data_utils.dataset       import MARADataset, CacheManager
from evaluation.evaluator     import Evaluator
from evaluate                 import (eval_cold_rerank, eval_coldness_gated,
                                       eval_coldness_stratified, get_cold_items,
                                       build_user_text_pref)

DATA_DIR    = ROOT / "data"
CACHE_DIR   = ROOT / "cache"
RESULTS_DIR = ROOT / "experiments" / "baselines" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(cache_dir, device):
    cache = {}
    for key in ["clip_text", "clip_cls", "item_emb", "user_emb"]:
        p = cache_dir / f"{key}.npy"
        if p.exists():
            cache[key] = torch.from_numpy(
                np.array(np.load(p), dtype=np.float32)).to(device)
    return cache


def build_item_counts(data_dir, n_items, device):
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    counts = np.zeros(n_items, dtype=np.float32)
    for iid in train_df["item_id"].values:
        counts[int(iid)] += 1
    return torch.from_numpy(counts).to(device)


CFG = {
    "d": 64, "text_dim": 768,
    "lr": 1e-3, "weight_decay": 1e-5,
    "tau_score": 1.0,
    "max_epochs": 50, "patience": 6, "warmup_ratio": 0.10,
    "batch_size": 16384, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "val_every": 3,
    "use_amp": True,
}


def train_mamex(data_dir, cache_dir, device, seed):
    set_seed(seed)
    cfg = dict(CFG)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_users_gpu = torch.from_numpy(
        train_df["user_id"].values.astype(np.int64)).to(device)
    train_items_gpu = torch.from_numpy(
        train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_users_gpu)

    if n_train > 1_000_000:
        cfg["batch_size"] = 65536
    elif n_train > 500_000:
        cfg["batch_size"] = 32768

    val_set    = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"],
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    cache = load_cache(cache_dir, device)
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)

    model = MAMEXStyle(n_users, n_items, cfg).to(device)
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    steps_per_epoch = (n_train + cfg["batch_size"] - 1) // cfg["batch_size"]
    total_steps     = steps_per_epoch * cfg["max_epochs"]
    warmup_steps    = max(1, int(total_steps * cfg["warmup_ratio"]))
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                          eta_min=cfg["lr"] * 0.01),
    ], milestones=[warmup_steps])

    evaluator = Evaluator(model, cache, cfg["k_list"], device,
                          cfg["eval_batch"], cfg["neg_batch"])

    best_ndcg10, best_epoch, patience_cnt = 0.0, 0, 0
    warmup_epoch_end = max(1, int(cfg["max_epochs"] * cfg["warmup_ratio"]))

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        ep_loss = ep_cf = ep_txt = ep_bal = 0.0
        t0 = time.time()
        grad_clip = 0.5 if epoch <= warmup_epoch_end else 1.0

        perm = torch.randperm(n_train, device=device)
        epoch_u = train_users_gpu[perm]
        epoch_i = train_items_gpu[perm]

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos_items = epoch_u[start:end], epoch_i[start:end]
            neg_items = torch.randint(0, n_items, (users.size(0),), device=device)

            with autocast("cuda", enabled=use_amp):
                outputs = model(users, pos_items, neg_items, cache)
                loss, info = mamex_loss(outputs)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += info["loss_total"]
            ep_cf   += info["loss_bpr_cf"]
            ep_txt  += info["loss_bpr_txt"]
            ep_bal  += info["loss_balance"]

        nb = (n_train + cfg["batch_size"] - 1) // cfg["batch_size"]
        elapsed = time.time() - t0

        do_eval = (epoch % cfg["val_every"] == 0) or (epoch <= 5)
        if do_eval:
            val_m  = evaluator.evaluate(val_loader, desc=f"MAMEX ep{epoch}")
            ndcg10 = val_m.get("ndcg@10", 0.0)
            print(f"    ep{epoch:2d}  loss={ep_loss/nb:.4f}"
                  f"(cf={ep_cf/nb:.4f} txt={ep_txt/nb:.4f} bal={ep_bal/nb:.4f})"
                  f"  {elapsed:.0f}s  NDCG@10={ndcg10:.4f}")

            if ndcg10 > best_ndcg10:
                best_ndcg10, best_epoch, patience_cnt = ndcg10, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break
        else:
            print(f"    ep{epoch:2d}  loss={ep_loss/nb:.4f}"
                  f"(cf={ep_cf/nb:.4f} txt={ep_txt/nb:.4f} bal={ep_bal/nb:.4f})"
                  f"  {elapsed:.0f}s")
            patience_cnt += 1

    if best_state:
        model.load_state_dict(best_state)
    print(f"  best val NDCG@10={best_ndcg10:.4f} @ epoch {best_epoch}")
    return model, cache


# ── MAMEX-specific evaluation (uses trained text_adapter, not raw CLIP) ──

@torch.no_grad()
def mamex_eval_cold_rerank(model, cache, test_df, cold_items, n_items, device, cfg,
                            seed, data_dir, cold_beta):
    """Cold-start rerank using MAMEX's trained text adapter."""
    k_list = cfg.get("k_list", [5, 10, 20])
    tau   = model.tau_score
    w_txt = cold_beta
    w_cf  = 1.0 - cold_beta

    all_test_iids = pd.Series(test_df["item_id"].values)
    cold_indices = all_test_iids[all_test_iids.isin(cold_items)].index.tolist()
    if not cold_indices:
        return {}

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=cfg.get("n_neg", 99))
    cold_subset = torch.utils.data.Subset(test_set, cold_indices)
    loader = DataLoader(cold_subset, batch_size=cfg.get("eval_batch", 4096),
                        shuffle=False, num_workers=2)

    all_hr = {k: [] for k in k_list}
    all_ndcg = {k: [] for k in k_list}
    all_item_repr = model.get_all_item_cf_repr()

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # CF scores
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # Trained adapter text scores
        u_txt = model.text_adapter(cache["user_text_pref"][users])
        u_txt = F.normalize(u_txt, dim=-1)
        i_txt = model.text_adapter(cache["clip_text"][cand_flat])
        i_txt = F.normalize(i_txt, dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        fused = w_cf * cf_scores + w_txt * txt_scores
        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()

        for k in k_list:
            all_hr[k].extend((ranks <= k).astype(float).tolist())
            all_ndcg[k].extend(
                np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).tolist())

    metrics = {}
    for k in k_list:
        metrics[f"hr@{k}"]   = float(np.mean(all_hr[k])) if all_hr[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(all_ndcg[k])) if all_ndcg[k] else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = DATA_DIR / args.dataset
    cache_dir = CACHE_DIR / args.dataset

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  MAMEX-style  dataset={args.dataset}  seed={seed}")
        print(f"{'='*60}")

        model, cache = train_mamex(data_dir, cache_dir, device, seed)
        model.eval()

        # Full evaluation
        with open(data_dir / "stats.json") as f:
            stats = json.load(f)
        n_items = stats["n_items"]
        test_df = pd.read_csv(data_dir / "test_indexed.csv")
        item_counts = build_item_counts(data_dir, n_items, device)
        cold_items = get_cold_items(data_dir)

        # CF-only
        test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
        test_loader = DataLoader(test_set, batch_size=16384, shuffle=False, num_workers=4)
        evaluator = Evaluator(model, cache, [5, 10, 20], device, 16384, 64)
        full_cf = evaluator.evaluate(test_loader, desc="Full CF")
        print(f"  Full CF NDCG@10={full_cf.get('ndcg@10', 0):.4f}")

        # Cold-start text fusion (trained adapter)
        cold_b50 = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, CFG, seed, data_dir,
            cold_beta=0.5) if cold_items else {}

        # Coldness-gated (using CARE's function, but with trained adapter scores)
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
            gm = eval_coldness_gated(
                model, cache, test_df, item_counts, n_items, device, CFG, seed, data_dir, alpha)
            print(f"  Gate(α={alpha}) Full NDCG@10={gm.get('ndcg@10', 0):.4f}")

        # Stratified at best α
        strat = eval_coldness_stratified(
            model, cache, test_df, item_counts, n_items, device, CFG, seed, data_dir, 0.5)
        print(f"  Stratified (α=0.5):")
        for bname in ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]:
            b = strat.get(bname, {})
            for s in ["cf_only", "text_only", "gate"]:
                m = b.get(s)
                if m:
                    print(f"    {bname:<18} {s:<10} N@10={m['ndcg@10']:.4f}  n={m['n']}")

        all_results.append({
            "seed": seed,
            "full_cf": full_cf,
            "cold_beta05": cold_b50,
            "stratified": {bn: {s: {k: v for k, v in m.items() if k != 'n'} if m else None
                                for s, m in b.items()}
                          for bn, b in strat.items()},
        })

    # Save
    out_path = RESULTS_DIR / f"{args.dataset}_mamex.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
