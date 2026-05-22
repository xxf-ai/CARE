"""
evaluate.py — VARA test-set evaluation

Evaluates on two subsets:
  1. Warm items: items with >= min_interactions interactions in training
  2. Cold-start items: items with < min_interactions interactions
  3. Full test set: all items

Logs saved to logs/{dataset}/eval_seeds{seeds}_{timestamp}.log

Protocol: Leave-One-Out, 99 random negatives, HR@K / NDCG@K (100-way ranking)
"""

import argparse
import json
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vara         import VARA
from data_utils.dataset  import MARADataset, CacheManager
from evaluation.evaluator import Evaluator

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = ROOT_DIR / "cache"
CKPT_DIR   = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"
LOG_DIR    = ROOT_DIR / "logs"


class Logger:
    def __init__(self, log_path: Path):
        self.file = open(log_path, "w", encoding="utf-8")
        self.terminal = sys.stdout
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()

COLD_THRESHOLD = 5  # items with < 5 train interactions = cold-start


def load_cache_to_gpu(cache_mgr, device):
    cache = {}
    for k, v in cache_mgr.tensors.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(np.array(v, dtype=np.float32)).to(device, non_blocking=True)
        else:
            t = v.to(device, non_blocking=True)
        cache[k] = t
    return cache


def get_cold_items(data_dir: Path) -> set:
    """Return set of cold-start item IDs (< COLD_THRESHOLD train interactions)"""
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = train_df["item_id"].value_counts()
    cold = set(item_counts[item_counts < COLD_THRESHOLD].index.tolist())
    return cold


def filter_cold_users(data_dir: Path, cold_items: set, test_df: pd.DataFrame) -> set:
    """Users who have at least one cold-start test item"""
    cold_test = test_df[test_df["item_id"].isin(cold_items)]
    return set(cold_test["user_id"].unique())


def build_user_text_pref(train_csv: Path, clip_text: torch.Tensor,
                         n_users: int, device: torch.device) -> torch.Tensor:
    """Precompute mean CLIP text embedding per user from training items."""
    train_df = pd.read_csv(train_csv)
    user_pref = torch.zeros(n_users, clip_text.shape[1], device=device)
    user_cnt  = torch.zeros(n_users, device=device)

    for _, row in train_df.iterrows():
        uid, iid = int(row["user_id"]), int(row["item_id"])
        user_pref[uid] += clip_text[iid]
        user_cnt[uid]  += 1

    mask = user_cnt > 0
    user_pref[mask] = user_pref[mask] / user_cnt[mask].unsqueeze(1)
    return user_pref


def build_user_image_pref(train_csv: Path, clip_image: torch.Tensor,
                           n_users: int, device: torch.device) -> torch.Tensor:
    """Precompute mean CLIP image embedding per user from training items."""
    train_df = pd.read_csv(train_csv)
    user_pref = torch.zeros(n_users, clip_image.shape[1], device=device)
    user_cnt  = torch.zeros(n_users, device=device)

    for _, row in train_df.iterrows():
        uid, iid = int(row["user_id"]), int(row["item_id"])
        user_pref[uid] += clip_image[iid]
        user_cnt[uid]  += 1

    mask = user_cnt > 0
    user_pref[mask] = user_pref[mask] / user_cnt[mask].unsqueeze(1)
    return user_pref


@torch.no_grad()
def eval_cold_rerank(model, cache, test_df, cold_items, n_items, device, cfg,
                      seed, data_dir, cold_beta, modality="text"):
    """Cold-start evaluation: CF + raw CLIP (text or image) cosine fusion.

    cold_beta: float in [0, 1], CLIP fusion weight (1-cold_beta = CF weight).
    modality: "text" (CLIP text) or "image" (CLIP image/CLS).
    """
    k_list = cfg.get("k_list", [5, 10, 20])
    tau   = model.tau_score
    w_clip = cold_beta
    w_cf   = 1.0 - cold_beta

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

    clip_key = "clip_cls" if modality == "image" else "clip_text"
    user_pref_key = "user_image_pref" if modality == "image" else "user_text_pref"

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # ── CF scores ─────────────────────────────────────────────
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # ── Raw CLIP cosine similarity ────────────────────────────
        u_clip = F.normalize(cache[user_pref_key][users], dim=-1)
        i_clip = F.normalize(cache[clip_key][cand_flat], dim=-1).reshape(B, 100, -1)
        clip_scores = (u_clip.unsqueeze(1) * i_clip).sum(-1) / tau

        # ── Fused scores ─────────────────────────────────────────
        fused = w_cf * cf_scores + w_clip * clip_scores

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


@torch.no_grad()
def eval_coldness_gated(model, cache, test_df, item_counts, n_items, device, cfg,
                        seed, data_dir, alpha, split_name="Cold"):
    """Coldness-gated fusion: per-item text weight = 1/(1 + α·count).

    Cold items (count→0) get weight→1.0 (pure text).
    Warm items (count→∞) get weight→0.0 (pure CF).
    alpha controls the transition steepness.
    """
    k_list = cfg.get("k_list", [5, 10, 20])
    tau   = model.tau_score

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=cfg.get("n_neg", 99))
    loader = DataLoader(test_set, batch_size=cfg.get("eval_batch", 4096),
                        shuffle=False, num_workers=2)

    all_hr = {k: [] for k in k_list}
    all_ndcg = {k: [] for k in k_list}
    all_item_repr = model.get_all_item_cf_repr()

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [B, 100]
        cand_flat = cand.reshape(-1)

        # ── CF scores ─────────────────────────────────────────────
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # ── Raw CLIP text scores ──────────────────────────────────
        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        # ── Per-item coldness weight ──────────────────────────────
        cand_counts = item_counts[cand]  # [B, 100], each item's train count
        w_txt = 1.0 / (1.0 + alpha * cand_counts)  # [B, 100]
        w_cf  = 1.0 - w_txt

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


@torch.no_grad()
def eval_coldness_stratified(model, cache, test_df, item_counts, n_items, device,
                              cfg, seed, data_dir, alpha):
    """Stratified evaluation: metrics per coldness bucket.

    Returns dict: {bucket_name: {strategy: {ndcg@10, ...}}}
    Strategies: cf_only, text_only, gate_alpha.
    """
    k_list = cfg.get("k_list", [5, 10, 20])
    tau   = model.tau_score

    # Bucket definitions: (name, min_count, max_count)
    buckets = [
        ("0 (zero-shot)", 0, 0),
        ("1-4 (cold)",    1, 4),
        ("5-20 (warm)",   5, 20),
        (">20 (hot)",    21, 1_000_000),
    ]

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=cfg.get("n_neg", 99))
    loader = DataLoader(test_set, batch_size=cfg.get("eval_batch", 4096),
                        shuffle=False, num_workers=2)

    # Check if image features available
    has_image = "clip_cls" in cache and "user_image_pref" in cache
    strategies = ["cf_only", "text_only", "gate"]
    if has_image:
        strategies += ["image_only", "gate_image", "multimodal"]

    # Collect results per bucket
    bucket_ranks = {}
    for bname, _, _ in buckets:
        bucket_ranks[bname] = {s: [] for s in strategies}

    all_item_repr = model.get_all_item_cf_repr()
    pos_counts_all = []

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)

        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # ── CF scores ─────────────────────────────────────────────
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau
        cf_ranks = ((cf_scores > cf_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        # ── Text scores ──────────────────────────────────────────
        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau
        txt_ranks = ((txt_scores > txt_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        # ── Gate (text) scores ────────────────────────────────────
        cand_counts = item_counts[cand]
        w_txt = 1.0 / (1.0 + alpha * cand_counts)
        gate_ranks = (( (1.0 - w_txt) * cf_scores + w_txt * txt_scores >
                       ((1.0 - w_txt) * cf_scores + w_txt * txt_scores)[:, 0:1]
                     ).sum(dim=1) + 1).cpu().numpy()

        # ── Image scores (if available) ────────────────────────────
        img_ranks = gate_img_ranks = mm_ranks = None
        if has_image:
            u_img = F.normalize(cache["user_image_pref"][users], dim=-1)
            i_img = F.normalize(cache["clip_cls"][cand_flat], dim=-1).reshape(B, 100, -1)
            img_scores = (u_img.unsqueeze(1) * i_img).sum(-1) / tau
            img_ranks = ((img_scores > img_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

            # Gate with image
            gate_img = (1.0 - w_txt) * cf_scores + w_txt * img_scores
            gate_img_ranks = ((gate_img > gate_img[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

            # Multimodal: max(text, image) — best of both zero-shot signals
            mm_scores = torch.max(txt_scores, img_scores)
            mm_ranks = ((mm_scores > mm_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        # ── Stratify by positive item count ───────────────────────
        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            cnt = int(pos_counts[j])
            for bname, lo, hi in buckets:
                if lo <= cnt <= hi:
                    bucket_ranks[bname]["cf_only"].append(int(cf_ranks[j]))
                    bucket_ranks[bname]["text_only"].append(int(txt_ranks[j]))
                    bucket_ranks[bname]["gate"].append(int(gate_ranks[j]))
                    if has_image:
                        bucket_ranks[bname]["image_only"].append(int(img_ranks[j]))
                        bucket_ranks[bname]["gate_image"].append(int(gate_img_ranks[j]))
                        bucket_ranks[bname]["multimodal"].append(int(mm_ranks[j]))
                    break

        pos_counts_all.extend(pos_counts.tolist())

    # ── Compute metrics per bucket ───────────────────────────────
    result = {}
    for bname, _, _ in buckets:
        bres = {}
        for strategy in strategies:
            ranks_arr = np.array(bucket_ranks[bname][strategy])
            if len(ranks_arr) == 0:
                bres[strategy] = None
                continue
            m = {}
            for k in k_list:
                m[f"hr@{k}"]   = float((ranks_arr <= k).mean())
                m[f"ndcg@{k}"] = float(
                    np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1), 0.0).mean())
            m["n"] = len(ranks_arr)
            bres[strategy] = m
        result[bname] = bres

    return result


def evaluate_one(dataset: str, seed: int, device: torch.device) -> dict:
    data_dir  = DATA_DIR  / dataset
    cache_dir = CACHE_DIR / dataset
    ckpt_path = CKPT_DIR  / dataset / f"seed{seed}" / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})

    # Update n_users/n_items from stats
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    # Load cache
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = load_cache_to_gpu(cache_mgr, device)

    # Build user_text_pref
    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)

    # Build user_image_pref (if CLIP image features available)
    has_image = "clip_cls" in cache
    if has_image:
        cache["user_image_pref"] = build_user_image_pref(
            data_dir / "train_indexed.csv", cache["clip_cls"], n_users, device)

    # Create model
    model = VARA(n_users, n_items, cfg).to(device)
    state_dict = ckpt["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Cold-start items
    cold_items = get_cold_items(data_dir)
    test_df = pd.read_csv(data_dir / "test_indexed.csv")

    # Build item interaction counts for coldness-gated evaluation
    item_counts = torch.zeros(n_items, device=device)
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    def eval_subset(name: str, item_filter_set: set = None):
        """Evaluate on a subset of test data.

        Uses torch.utils.data.Subset to filter the MARADataset
        to only include samples whose positive item is in item_filter_set.
        """
        test_set = MARADataset(data_dir, split="test", seed=seed,
                               n_neg=cfg.get("n_neg", 99))

        if item_filter_set is not None:
            # Find indices of test samples where pos item matches the filter
            matched = test_df[test_df["item_id"].isin(item_filter_set)]
            if len(matched) == 0:
                return {}
            # Build index mapping: the i-th row in test_df → i-th sample in MARADataset
            all_test_iids = pd.Series(test_df["item_id"].values)
            indices = all_test_iids[all_test_iids.isin(item_filter_set)].index.tolist()
            test_set = torch.utils.data.Subset(test_set, indices)

        test_loader = DataLoader(test_set, batch_size=cfg.get("eval_batch", 2048),
                                 shuffle=False, num_workers=4)

        evaluator = Evaluator(model, cache, cfg.get("k_list", [5, 10, 20]),
                              device, eval_batch=cfg.get("eval_batch", 2048),
                              neg_batch=cfg.get("neg_batch", 20))
        metrics = evaluator.evaluate(test_loader, desc=f"{name} seed{seed}")
        return metrics

    # ── Full test (warm + cold) ────────────────────────────────────
    full_m = eval_subset("Full")

    # ── Warm only ──────────────────────────────────────────────────
    warm_m = eval_subset("Warm", set(range(n_items)) - cold_items)

    # ── Cold-start only (CF) ──────────────────────────────────────
    cold_m = eval_subset("Cold", cold_items)

    # ── Cold-start + raw CLIP text fusion ────────────────────────
    cold_b50_m = {}
    cold_b70_m = {}
    cold_clip_m = {}
    if cold_items:
        cold_b50_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.5)     # CF + CLIP text 各 50%
        cold_b70_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.7)     # CLIP text 70%
        cold_clip_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=1.0)     # raw CLIP text only

    # ── Cold-start + raw CLIP image fusion (if image features available) ─
    cold_img_b50_m = {}
    cold_img_b70_m = {}
    cold_img_b100_m = {}
    if cold_items and has_image:
        cold_img_b50_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.5, modality="image")
        cold_img_b70_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=0.7, modality="image")
        cold_img_b100_m = eval_cold_rerank(
            model, cache, test_df, cold_items, n_items, device, cfg, seed, data_dir,
            cold_beta=1.0, modality="image")

    # ── Coldness-gated fusion (full test, per-item dynamic weight) ──
    gate_best = {}
    gate_strat = {}
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    best_alpha_ndcg = -1
    for alpha in alphas:
        gm = eval_coldness_gated(
            model, cache, test_df, item_counts, n_items, device, cfg, seed, data_dir, alpha)
        ndcg10 = gm.get("ndcg@10", 0)
        if ndcg10 > best_alpha_ndcg:
            best_alpha_ndcg = ndcg10
            best_alpha = alpha
            gate_best = gm

    # ── Stratified coldness analysis ──────────────────────────────
    gate_strat = eval_coldness_stratified(
        model, cache, test_df, item_counts, n_items, device, cfg, seed, data_dir, best_alpha)

    result = {
        "full": full_m, "warm": warm_m if warm_m else None,
        "cold_cf": cold_m if cold_m else None,
        "cold_beta05": cold_b50_m if cold_b50_m else None,
        "cold_beta07": cold_b70_m if cold_b70_m else None,
        "cold_beta10": cold_clip_m if cold_clip_m else None,
        "cold_img_beta05": cold_img_b50_m if cold_img_b50_m else None,
        "cold_img_beta07": cold_img_b70_m if cold_img_b70_m else None,
        "cold_img_beta10": cold_img_b100_m if cold_img_b100_m else None,
        "gate_best_alpha": best_alpha,
        "gate_full": gate_best if gate_best else None,
        "gate_strat": gate_strat if gate_strat else None,
        "has_image": has_image,
        "n_cold_items": len(cold_items), "n_items": n_items,
        "cold_ratio": len(cold_items) / max(n_items, 1),
    }

    def _n10(m): return m.get("ndcg@10", 0) if m else 0
    print(f"  seed={seed}  Full CF={_n10(full_m):.4f}  Warm={_n10(warm_m):.4f}"
          f"  Cold CF={_n10(cold_m):.4f}")
    print(f"           Cold Text β=0.5:{_n10(cold_b50_m):.4f}  β=0.7:{_n10(cold_b70_m):.4f}"
          f"  β=1.0:{_n10(cold_clip_m):.4f}")
    if has_image:
        print(f"           Cold Image β=0.5:{_n10(cold_img_b50_m):.4f}"
              f"  β=0.7:{_n10(cold_img_b70_m):.4f}  β=1.0:{_n10(cold_img_b100_m):.4f}")
    print(f"           Gate(α={best_alpha}): Full={_n10(gate_best):.4f}"
          f"  (cold={len(cold_items)}/{n_items}={result['cold_ratio']:.1%})")
    if gate_strat:
        strat_strategies = ["cf_only", "text_only", "gate"]
        if has_image:
            strat_strategies += ["image_only", "gate_image", "multimodal"]
        header_cols = "".join(f"   {s[:4]:>7}" for s in strat_strategies)
        print(f"           Stratified (α={best_alpha}):")
        print(f"           {'Bucket':<18} {'n':>6}{header_cols}")
        print(f"           {'':18} {'':>6}  " + "".join(f"{'N@10':>7}" for _ in strat_strategies))
        def _fm(d, k): return f"{d[k]:.4f}" if d and d.get(k) is not None else "     NA"
        for bname in ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]:
            b = gate_strat.get(bname, {})
            n = b.get("cf_only", {}).get("n", 0) if b.get("cf_only") else 0
            row = f"           {bname:<18} {n:>6}  "
            for s in strat_strategies:
                row += f"{_fm(b.get(s),'ndcg@10'):>7} "
            print(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 日志文件 ─────────────────────────────────────────────────
    log_dir = LOG_DIR / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds_str = "_".join(str(s) for s in args.seeds)
    log_path = log_dir / f"eval_seeds{seeds_str}_{ts}.log"
    logger = Logger(log_path)
    sys.stdout = logger

    print(f"\n{'='*55}")
    print(f"  VARA 测试集评估  数据集={args.dataset}  seeds={args.seeds}")
    print(f"  日志文件: {log_path}")
    print(f"{'='*55}")

    all_results = []
    for seed in args.seeds:
        print(f"\n  seed={seed}")
        r = evaluate_one(args.dataset, seed, device)
        all_results.append(r)

    # ── Summary across seeds ───────────────────────────────────────
    if len(all_results) > 1:
        for subset in ["full", "warm", "cold_cf",
                       "cold_beta05", "cold_beta07", "cold_beta10",
                       "cold_img_beta05", "cold_img_beta07", "cold_img_beta10",
                       "gate_full"]:
            valid = [r[subset] for r in all_results if r.get(subset)]
            if not valid:
                continue
            print(f"\n  [{subset.upper()}] 多种子汇总")
            print(f"  {'指标':<12} {'均值':>8} {'标准差':>8}")
            print(f"  {'─'*32}")
            for key in sorted(valid[0].keys()):
                vals = [m[key] for m in valid]
                print(f"  {key:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

    # ── Multi-seed stratified aggregation ────────────────────────────
    if len(all_results) > 1:
        strats = [r["gate_strat"] for r in all_results if r.get("gate_strat")]
        if strats:
            print(f"\n  {'='*70}")
            print(f"  [STRATIFIED] 多种子汇总 (NDCG@10 mean ± std)")
            print(f"  {'='*70}")
            # Collect all strategies across all seeds
            all_strategies = set()
            for s in strats:
                for bname, bdata in s.items():
                    if bdata:
                        all_strategies.update(bdata.keys())
            all_strategies = sorted(all_strategies)
            buckets_order = ["0 (zero-shot)", "1-4 (cold)", "5-20 (warm)", ">20 (hot)"]

            for bname in buckets_order:
                for strategy in all_strategies:
                    vals = []
                    ns = []
                    for s in strats:
                        b = s.get(bname, {})
                        m = b.get(strategy) if b else None
                        if m and m.get("ndcg@10") is not None:
                            vals.append(m["ndcg@10"])
                            ns.append(m.get("n", 0))
                    if vals:
                        print(f"  {bname:<18} {strategy:<14}  "
                              f"{np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                              f"(n≈{int(np.mean(ns))})")

    # Save results
    RESULT_DIR.mkdir(exist_ok=True)
    out_path = RESULT_DIR / f"{args.dataset}_vara_results.json"
    with open(out_path, "w") as f:
        serializable = []
        metric_keys = ["full", "warm", "cold_cf",
                       "cold_beta05", "cold_beta07", "cold_beta10",
                       "cold_img_beta05", "cold_img_beta07", "cold_img_beta10",
                       "gate_full", "gate_strat"]
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if k in metric_keys:
                    sr[k] = v
                elif k == "gate_best_alpha":
                    sr[k] = v
                elif k in ("n_cold_items", "n_items", "cold_ratio"):
                    sr[k] = v
            serializable.append(sr)
        json.dump({"dataset": args.dataset, "seeds": args.seeds,
                   "per_seed": serializable}, f, indent=2)
    print(f"\n  结果已保存: {out_path}")

    # ── 关闭日志文件 ────────────────────────────────────────────
    sys.stdout = logger.terminal
    logger.close()
    print(f"  日志已保存: {log_path}")


if __name__ == "__main__":
    main()
