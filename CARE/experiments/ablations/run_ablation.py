"""
experiments/ablations/run_ablation.py — VARA ablation experiments

3 variants (share one CF backbone per seed):
  A1: VARA       — CF + raw CLIP text fusion on cold-start items
  A2: w/o Text   — CF only (no text, pure BPR-MF)
  A3: SBERT      — CF + SBERT text fusion (proves CLIP joint space matters)

Training: BPR-MF (GPU-side, same for all variants).
Evaluation: CF-only full/warm/cold + cold-start text fusion at β=0.5, 0.7, 1.0.

Output: experiments/ablations/results/{dataset}_ablations.json
"""

import sys, os, json, argparse, time, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from models.vara          import VARA
from data_utils.dataset   import MARADataset, CacheManager
from evaluation.evaluator import Evaluator
from losses.vara_loss     import vara_total_loss
from evaluate             import (eval_cold_rerank, eval_coldness_gated,
                                   eval_coldness_stratified, get_cold_items,
                                   build_user_text_pref)

DATA_DIR    = ROOT / "data"
CACHE_DIR   = ROOT / "cache"
CKPT_DIR    = ROOT / "experiments" / "ablations" / "checkpoints"
RESULTS_DIR = ROOT / "experiments" / "ablations" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "d": 64,
    "lr": 1e-3, "weight_decay": 1e-5,
    "margin": 0.1, "tau_score": 1.0,
    "max_epochs": 50, "patience": 6, "warmup_ratio": 0.10,
    "batch_size": 16384, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "margin_warmup_start": 5, "margin_warmup_end": 30,
    "val_every": 3,
    "use_amp": True, "compile": True,
}


# ═══════════════════════════════════════════════════════════════════════
# Training (pure BPR-MF, GPU-side data)
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(cache_dir, device):
    """Load cache tensors to GPU (clip_text for evaluation only)."""
    cache = {}
    for key in ["clip_text", "clip_cls", "item_emb", "user_emb"]:
        p = cache_dir / f"{key}.npy"
        if p.exists():
            cache[key] = torch.from_numpy(
                np.array(np.load(p), dtype=np.float32)).to(device)
    return cache


def _get_margin(epoch, cfg):
    s, e, t = cfg["margin_warmup_start"], cfg["margin_warmup_end"], cfg["margin"]
    if epoch <= s: return 0.0
    if epoch >= e: return t
    return t * (epoch - s) / (e - s)


def train_one(data_dir, cache_dir, device, seed):
    """Train BPR-MF and return best model + cache."""
    set_seed(seed)
    cfg = dict(CFG)

    # ── Training data on GPU ─────────────────────────────────────
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

    # ── Validation ───────────────────────────────────────────────
    val_set    = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"],
                            shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    # ── Cache + stats ────────────────────────────────────────────
    cache = load_cache(cache_dir, device)
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    # ── Model ────────────────────────────────────────────────────
    model = VARA(n_users, n_items, cfg).to(device)
    if cfg.get("compile", True) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

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
    ckpt_dir = CKPT_DIR / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    warmup_epoch_end = max(1, int(cfg["max_epochs"] * cfg["warmup_ratio"]))

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        ep_loss = ep_bpr = ep_gap = 0.0
        t0 = time.time()
        grad_clip = 0.5 if epoch <= warmup_epoch_end else 1.0
        current_margin = _get_margin(epoch, cfg)

        perm = torch.randperm(n_train, device=device)
        epoch_u = train_users_gpu[perm]
        epoch_i = train_items_gpu[perm]

        for start in range(0, n_train, cfg["batch_size"]):
            end       = min(start + cfg["batch_size"], n_train)
            users     = epoch_u[start:end]
            pos_items = epoch_i[start:end]
            neg_items = torch.randint(0, n_items, (users.size(0),), device=device)

            with autocast("cuda", enabled=use_amp):
                outputs = model(users, pos_items, neg_items)
                loss, info = vara_total_loss(outputs, margin=current_margin)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += info["loss_total"]
            ep_bpr  += info["loss_bpr"]
            with torch.no_grad():
                ep_gap += (outputs["score_cf_pos"] - outputs["score_cf_neg"]).mean().item()

        nb = (n_train + cfg["batch_size"] - 1) // cfg["batch_size"]
        elapsed = time.time() - t0

        val_every = cfg.get("val_every", 3)
        do_eval = (epoch % val_every == 0) or (epoch <= 5)

        if do_eval:
            val_m  = evaluator.evaluate(val_loader, desc=f"Ep{epoch}")
            ndcg10 = val_m.get("ndcg@10", 0.0)
            print(f"    ep{epoch:2d}  loss={ep_loss/nb:.4f}  bpr={ep_bpr/nb:.4f}"
                  f"  gap={ep_gap/nb:.3f}  {elapsed:.0f}s  NDCG@10={ndcg10:.4f}")

            if ndcg10 > best_ndcg10:
                best_ndcg10, best_epoch, patience_cnt = ndcg10, epoch, 0
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "val_ndcg10": ndcg10, "cfg": cfg},
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_cnt += 1
                if patience_cnt >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break
        else:
            print(f"    ep{epoch:2d}  loss={ep_loss/nb:.4f}  bpr={ep_bpr/nb:.4f}"
                  f"  gap={ep_gap/nb:.3f}  {elapsed:.0f}s")
            patience_cnt += 1

    print(f"  best val NDCG@10={best_ndcg10:.4f} @ epoch {best_epoch}")
    return model, cache, ckpt_dir / "best_model.pt"


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (3 strategies sharing one trained model)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all(model, cache, data_dir, cfg, seed, device, text_source="clip"):
    """Full eval: CF-only full/warm/cold + cold text fusion at β=0.5, 0.7, 1.0.

    text_source: "clip" or "sbert" — which text cache to use.
    """
    base_model = getattr(model, "_orig_mod", model)

    # ── CF-only evaluation ───────────────────────────────────────
    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=cfg.get("n_neg", 99))
    test_loader = DataLoader(test_set, batch_size=cfg.get("eval_batch", 2048),
                             shuffle=False, num_workers=4)

    evaluator = Evaluator(model, cache, cfg.get("k_list", [5, 10, 20]),
                          device, eval_batch=cfg.get("eval_batch", 2048),
                          neg_batch=cfg.get("neg_batch", 20))

    full_cf = evaluator.evaluate(test_loader, desc="Full CF")

    # ── Warm / Cold subsets ──────────────────────────────────────
    cold_items = get_cold_items(data_dir)
    test_df = pd.read_csv(data_dir / "test_indexed.csv")
    all_test_iids = pd.Series(test_df["item_id"].values)

    n_items = len(cache["clip_text"])

    warm_indices = all_test_iids[~all_test_iids.isin(cold_items)].index.tolist()
    cold_indices = all_test_iids[all_test_iids.isin(cold_items)].index.tolist()

    warm_cf = {}
    cold_cf = {}
    if warm_indices:
        warm_cf = evaluator.evaluate(
            DataLoader(torch.utils.data.Subset(test_set, warm_indices),
                       batch_size=cfg.get("eval_batch", 2048),
                       shuffle=False, num_workers=2),
            desc="Warm CF")
    if cold_indices:
        cold_cf = evaluator.evaluate(
            DataLoader(torch.utils.data.Subset(test_set, cold_indices),
                       batch_size=cfg.get("eval_batch", 2048),
                       shuffle=False, num_workers=2),
            desc="Cold CF")

    # ── Build user_text_pref for cold rerank ─────────────────────
    with open(data_dir / "stats.json") as f:
        n_users = json.load(f)["n_users"]

    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)

    # Build user_image_pref if CLIP image features available
    has_image = "clip_cls" in cache
    if has_image:
        from evaluate import build_user_image_pref
        cache["user_image_pref"] = build_user_image_pref(
            data_dir / "train_indexed.csv", cache["clip_cls"], n_users, device)

    # ── Cold-start text fusion ───────────────────────────────────
    cold_b50 = cold_b70 = cold_b100 = {}
    if cold_items:
        cold_b50 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.5)
        cold_b70 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.7)
        cold_b100 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=1.0)

    # ── Cold-start image fusion (if image features available) ────
    cold_img_b50 = cold_img_b70 = cold_img_b100 = {}
    if cold_items and has_image:
        cold_img_b50 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.5, modality="image")
        cold_img_b70 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.7, modality="image")
        cold_img_b100 = eval_cold_rerank(
            base_model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=1.0, modality="image")

    return {
        "full_cf": full_cf,
        "warm_cf": warm_cf if warm_cf else None,
        "cold_cf": cold_cf if cold_cf else None,
        "cold_beta05": cold_b50 if cold_b50 else None,
        "cold_beta07": cold_b70 if cold_b70 else None,
        "cold_beta10": cold_b100 if cold_b100 else None,
        "cold_img_beta05": cold_img_b50 if cold_img_b50 else None,
        "cold_img_beta07": cold_img_b70 if cold_img_b70 else None,
        "cold_img_beta10": cold_img_b100 if cold_img_b100 else None,
        "has_image": has_image,
        "n_cold_items": len(cold_items),
        "cold_ratio": len(cold_items) / max(len(all_test_iids), 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = {
    "A1": {"name": "VARA",          "desc": "CF + raw CLIP text fusion"},
    "A2": {"name": "w/o Text",      "desc": "CF only (pure BPR-MF)"},
    "A3": {"name": "SBERT Text",    "desc": "CF + SBERT text fusion"},
    "A4": {"name": "Coldness Gate", "desc": "Per-item coldness-gated fusion w=1/(1+α·count)"},
}


def build_item_counts(data_dir, n_items, device):
    """Item training interaction counts → GPU tensor [n_items]."""
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    counts = np.zeros(n_items, dtype=np.float32)
    for iid in train_df["item_id"].values:
        counts[int(iid)] += 1
    return torch.from_numpy(counts).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--variants", nargs="+", default=["A1", "A2", "A3"])
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = DATA_DIR / args.dataset
    cache_dir = CACHE_DIR / args.dataset

    all_results = {vk: [] for vk in args.variants}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Training BPR-MF  dataset={args.dataset}  seed={seed}")
        print(f"{'='*60}")

        model, cache, ckpt_path = train_one(data_dir, cache_dir, device, seed)

        # Load best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base_model = getattr(model, "_orig_mod", model)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        base_model.load_state_dict(state_dict, strict=False)
        model.eval()

        # ── Precompute item counts for A4 ──────────────────────
        with open(data_dir / "stats.json") as f:
            n_items = json.load(f)["n_items"]
        item_counts = build_item_counts(data_dir, n_items, device)

        # ── Evaluate with each variant strategy ─────────────────
        for vkey in args.variants:
            vinfo = VARIANTS[vkey]
            print(f"  eval: {vkey} ({vinfo['name']})")

            # Reload cache for each variant (A3 uses SBERT text)
            eval_cache = dict(cache)  # shallow copy, tensors shared
            if vkey == "A3":
                sbert_path = cache_dir / "sbert_text.npy"
                if not sbert_path.exists():
                    print(f"    [WARN] sbert_text.npy not found, skipping A3 for this seed")
                    continue
                sbert = torch.from_numpy(
                    np.array(np.load(sbert_path), dtype=np.float32)).to(device)
                eval_cache["clip_text"] = sbert  # replace with SBERT

            if vkey == "A4":
                # Coldness-gated: evaluate with different α values
                m = evaluate_all(model, eval_cache, data_dir, CFG, seed, device)
                # Add per-α coldness-gated full-test evaluation
                for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
                    gm = eval_coldness_gated(
                        model, eval_cache, pd.read_csv(data_dir / "test_indexed.csv"),
                        item_counts, n_items, device, CFG, seed, data_dir, alpha)
                    m[f"gate_a{str(alpha).replace('.','_')}"] = gm
                all_results[vkey].append(m)

                def _n10(d): return d.get("ndcg@10", 0) if d else 0
                print(f"    Full={_n10(m['full_cf']):.4f}  ColdCF={_n10(m['cold_cf']):.4f}"
                      f"  →β.5={_n10(m['cold_beta05']):.4f}"
                      f"  β.7={_n10(m['cold_beta07']):.4f}"
                      f"  β1={_n10(m['cold_beta10']):.4f}")
                if m.get("has_image"):
                    print(f"    Img β=0.5:{_n10(m.get('cold_img_beta05')):.4f}"
                          f"  β=0.7:{_n10(m.get('cold_img_beta07')):.4f}"
                          f"  β=1.0:{_n10(m.get('cold_img_beta10')):.4f}")
                for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
                    gk = f"gate_a{str(alpha).replace('.','_')}"
                    print(f"    α={alpha:.2f}  Full={_n10(m[gk]):.4f}")

                # ── Stratified coldness analysis ──────────────────────
                best_alpha = 0.1 if args.dataset == "office" else 0.5
                strat = eval_coldness_stratified(
                    model, eval_cache, pd.read_csv(data_dir / "test_indexed.csv"),
                    item_counts, n_items, device, CFG, seed, data_dir, best_alpha)
                has_img = eval_cache.get("clip_cls") is not None
                strat_cols = ["cf_only", "text_only", "gate"]
                if has_img:
                    strat_cols += ["image_only", "gate_image", "multimodal"]
                print(f"\n    Coldness Stratified (α={best_alpha}):")
                header = f"    {'Bucket':<18} {'n':>6}"
                for s in strat_cols:
                    header += f"  {s[:6]:>8}"
                print(header)
                print(f"    {'─'*60}")
                for bname in ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]:
                    b = strat.get(bname, {})
                    n = b.get("cf_only", {}).get("n", 0) if b.get("cf_only") else 0
                    row = f"    {bname:<18} {n:>6}"
                    for s in strat_cols:
                        m = b.get(s)
                        row += f"  {m['ndcg@10']:.4f}" if m else "      N/A"
                    print(row)
            else:
                m = evaluate_all(model, eval_cache, data_dir, CFG, seed, device,
                                text_source="sbert" if vkey == "A3" else "clip")
                all_results[vkey].append(m)

                def _n10(d): return d.get("ndcg@10", 0) if d else 0
                print(f"    Full={_n10(m['full_cf']):.4f}  ColdCF={_n10(m['cold_cf']):.4f}"
                      f"  →β.5={_n10(m['cold_beta05']):.4f}"
                      f"  β.7={_n10(m['cold_beta07']):.4f}"
                      f"  β1={_n10(m['cold_beta10']):.4f}")
                if m.get("has_image"):
                    print(f"    Img β=0.5:{_n10(m.get('cold_img_beta05')):.4f}"
                          f"  β=0.7:{_n10(m.get('cold_img_beta07')):.4f}"
                          f"  β=1.0:{_n10(m.get('cold_img_beta10')):.4f}")

    # ── Aggregate and report ─────────────────────────────────────
    def summarize(results_list, key):
        vals = [r.get(key) for r in results_list if r.get(key)]
        if not vals:
            return None
        nested = {}
        for v in vals:
            for k, val in v.items():
                nested.setdefault(k, []).append(val)
        out = {}
        for k, arr in nested.items():
            out[k] = float(np.mean(arr))
            out[k + "_std"] = float(np.std(arr))
        return out

    out_data = {"_meta": {"dataset": args.dataset, "seeds": args.seeds,
                          "timestamp": datetime.datetime.now().isoformat()},
                "results": {}}

    print(f"\n{'='*100}")
    print(f"  Ablation Results — {args.dataset} (mean over {len(args.seeds)} seeds)")
    print(f"{'='*100}")
    header = (f"  {'Variant':<16} {'Full CF':>8} {'Warm CF':>8} {'Cold CF':>8} "
              f"{'C→β.50':>8} {'C→β.70':>8} {'C→β1.0':>8}")
    print(header)
    print(f"  {'─'*88}")

    for vkey in args.variants:
        res_list = all_results.get(vkey, [])
        if not res_list:
            continue

        agg = {}
        for mk in ["full_cf", "warm_cf", "cold_cf", "cold_beta05", "cold_beta07", "cold_beta10"]:
            s = summarize(res_list, mk)
            if s:
                agg[mk] = s

        def _v(k):
            a = agg.get(k, {})
            return f"{a.get('ndcg@10', 0):.4f}" if a else "   N/A"

        print(f"  {VARIANTS[vkey]['name']:<16} {_v('full_cf'):>8} {_v('warm_cf'):>8} "
              f"{_v('cold_cf'):>8} {_v('cold_beta05'):>8} "
              f"{_v('cold_beta07'):>8} {_v('cold_beta10'):>8}")

        out_data["results"][vkey] = {
            "name": VARIANTS[vkey]["name"],
            "desc": VARIANTS[vkey]["desc"],
            "aggregated": {mk: agg[mk] for mk in agg},
        }

    print(f"{'='*100}")

    # ── Save ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"{args.dataset}_ablations.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
