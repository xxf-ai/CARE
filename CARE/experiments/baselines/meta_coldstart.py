"""
experiments/baselines/meta_coldstart.py — Meta-learning cold-start baselines

实现两个元学习风格的冷启动基线:

1. MetaEmb (simplified):
   用 warm items 的 (clip_text → CF_embedding) 映射训练一个投影器，
   冷物品通过投影器从 CLIP text 生成伪 CF embedding。
   训练: MSE(project(clip_text_warm), cf_emb_warm)
   推理: cold_item_cf = project(clip_text_cold)

2. ProtoNet (simplified):
   为每个冷度级别学习原型 embedding。冷物品的 embedding 是其所属
   冷度原型的加权组合: e = α·proto_coldness + (1-α)·content_proj(text)

训练后在冻结 backbone 上评估 CF-only + CARE-gate 冷启动性能。

使用:
  python experiments/baselines/meta_coldstart.py --dataset baby --seeds 42 123 456
  python experiments/baselines/meta_coldstart.py --dataset yelp --seeds 42
"""

import sys, json, argparse, time, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.vara import VARA
from data_utils.dataset import MARADataset, CacheManager
from evaluate import build_user_text_pref, get_cold_items

DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "baselines" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]
COLD_THRESHOLD = 5


# ══════════════════════════════════════════════════════════════════════
# MetaEmb: Content → CF projector
# ══════════════════════════════════════════════════════════════════════

class MetaEmbProjector(nn.Module):
    """Project CLIP text features into CF embedding space.

    Trained with MSE loss on warm items: ||project(text) - cf_emb||²
    """

    def __init__(self, text_dim=768, cf_dim=64, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, cf_dim, bias=False),
        )

    def forward(self, text_features):
        return self.proj(text_features)


# ══════════════════════════════════════════════════════════════════════
# ProtoNet: Coldness prototype + content projector
# ══════════════════════════════════════════════════════════════════════

class ProtoNetProjector(nn.Module):
    """Learn coldness-level prototypes + content projector.

    Item embedding = gate(coldness) * proto[coldness_bucket] + (1-gate) * proj(text)
    """

    def __init__(self, text_dim=768, cf_dim=64, n_protos=5, hidden=128):
        super().__init__()
        self.prototypes = nn.Embedding(n_protos, cf_dim)  # 5 coldness levels
        self.content_proj = nn.Sequential(
            nn.Linear(text_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, cf_dim, bias=False),
        )
        # Gate: coldness_bucket → interpolation weight
        self.gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_features, coldness_bucket):
        """coldness_bucket: int tensor [B], 0=zero-shot, 1=1-4, 2=5-20, 3=>20, 4=unused"""
        proto = self.prototypes(coldness_bucket)
        content = self.content_proj(text_features)
        g = self.gate(coldness_bucket.float().unsqueeze(-1))
        return g * proto + (1 - g) * content


# ══════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════

def get_coldness_bucket(item_counts, n_items):
    """Map item counts to coldness bucket indices."""
    buckets = torch.zeros(n_items, dtype=torch.long, device=item_counts.device)
    buckets[item_counts == 0] = 0
    buckets[(item_counts >= 1) & (item_counts < COLD_THRESHOLD)] = 1
    buckets[(item_counts >= COLD_THRESHOLD) & (item_counts <= 20)] = 2
    buckets[item_counts > 20] = 3
    return buckets


def load_cache_gpu(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_item_counts(data_dir, n_items, device):
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        counts[int(iid)] += 1
    return counts


# ══════════════════════════════════════════════════════════════════════
# MetaEmb training
# ══════════════════════════════════════════════════════════════════════

def train_metaemb(model, cache, item_counts, data_dir, device, seed,
                   lr=1e-3, epochs=50, patience=8):
    """Train MetaEmb projector on warm items."""
    set_seed(seed)
    projector = MetaEmbProjector(text_dim=768, cf_dim=model.d).to(device)

    # Get CF embeddings for all items
    with torch.no_grad():
        all_cf = model.get_all_item_cf_repr()  # [N, d]

    clip_text = cache["clip_text"]
    n_items = clip_text.shape[0]

    # Split: warm items for training, cold for validation
    warm_mask = item_counts >= COLD_THRESHOLD
    cold_mask = item_counts < COLD_THRESHOLD

    warm_indices = warm_mask.nonzero(as_tuple=True)[0]
    cold_indices = cold_mask.nonzero(as_tuple=True)[0]

    if len(warm_indices) == 0:
        print("  [WARN] No warm items, skipping MetaEmb")
        return None

    # Validation: hold out 10% of warm items
    n_val = max(100, len(warm_indices) // 10)
    perm = torch.randperm(len(warm_indices), device=device)
    val_idx = warm_indices[perm[:n_val]]
    train_idx = warm_indices[perm[n_val:]]

    optimizer = optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        projector.train()
        # Batch training
        batch_size = 4096
        ep_loss, nb = 0.0, 0
        shuf = train_idx[torch.randperm(len(train_idx), device=device)]

        for start in range(0, len(shuf), batch_size):
            end = min(start + batch_size, len(shuf))
            idx = shuf[start:end]

            target = all_cf[idx]
            pred = projector(clip_text[idx])
            loss = F.mse_loss(pred, target)
            # Also add cosine similarity loss for direction
            cos_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            total_loss = loss + 0.1 * cos_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            ep_loss += total_loss.item()
            nb += 1

        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            projector.eval()
            with torch.no_grad():
                pred_val = projector(clip_text[val_idx])
                target_val = all_cf[val_idx]
                val_loss = F.mse_loss(pred_val, target_val).item()
            projector.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in projector.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

    if best_state:
        projector.load_state_dict(best_state)
    projector.eval()
    return projector


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_metaemb(model, projector, cache, test_set, item_counts, device, alpha=0.5):
    """Evaluate MetaEmb-style cold-start: projected text as CF for cold items.

    For cold items: use projected_text_emb as CF embedding.
    For warm items: use original CF embedding.
    Gating: CARE-style w_txt = 1/(1+α·count).
    """
    tau = getattr(model, "tau_score", 1.0)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)

    # Precompute hybrid item representations
    n_items = cache["clip_text"].shape[0]
    cold_mask = item_counts < COLD_THRESHOLD
    warm_mask = ~cold_mask

    all_cf = model.get_all_item_cf_repr()  # [N, d]
    all_proj = projector(cache["clip_text"])  # projected from text
    all_proj = F.normalize(all_proj, dim=-1)

    all_item_hybrid = all_cf.clone()
    all_item_hybrid[cold_mask] = all_proj[cold_mask]

    # For warm items, optionally blend: hybrid = (1-β)*cf + β*proj
    # (warm items benefit from CF, cold from projection)
    hybrid_repr = torch.where(
        warm_mask.unsqueeze(1),
        all_cf,  # warm: pure CF
        all_proj,  # cold: pure projection
    )

    all_hr = {k: [] for k in K_LIST}
    all_ndcg = {k: [] for k in K_LIST}
    b_hr = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}
    b_ndcg = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}

    for users, pos_items, neg_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # CF scores (using hybrid repr for cold items)
        u_cf = model.get_user_cf_repr(users)
        i_cf = hybrid_repr[cand_flat].reshape(B, 100, -1)
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        # Text scores (raw CLIP, same as CARE)
        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        # CARE gate
        cand_counts = item_counts[cand]
        w_txt = 1.0 / (1.0 + alpha * cand_counts)
        fused = (1.0 - w_txt) * cf + w_txt * txt

        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()
        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            r = int(ranks[j])
            cnt = int(pos_counts[j])
            if cnt == 0: bn = "L0"
            elif cnt <= 4: bn = "L1"
            elif cnt <= 20: bn = "L2"
            else: bn = "L3"
            for k in K_LIST:
                hit = float(r <= k)
                ndcg_v = (1.0 / np.log2(r + 1)) if r <= k else 0.0
                all_hr[k].append(hit); all_ndcg[k].append(ndcg_v)
                b_hr[bn][k].append(hit); b_ndcg[bn][k].append(ndcg_v)

    def _m(hr_d, ndcg_d):
        m = {}
        for k in K_LIST:
            m[f"hr@{k}"] = float(np.mean(hr_d[k])) if hr_d[k] else 0.0
            m[f"ndcg@{k}"] = float(np.mean(ndcg_d[k])) if ndcg_d[k] else 0.0
        return m

    result = {"full": _m(all_hr, all_ndcg)}
    for b in ["L0", "L1", "L2", "L3"]:
        result[b] = _m(b_hr[b], b_ndcg[b])
    return result


@torch.no_grad()
def evaluate_cf_only(model, test_set, device):
    """Baseline CF-only evaluation."""
    tau = getattr(model, "tau_score", 1.0)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)
    all_item = model.get_all_item_cf_repr()
    all_hr = {k: [] for k in K_LIST}
    all_ndcg = {k: [] for k in K_LIST}

    for users, pos_items, neg_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)
        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item[cand_flat].reshape(B, 100, -1)
        scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau
        ranks = (scores > scores[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()
        for j in range(B):
            r = int(ranks[j])
            for k in K_LIST:
                all_hr[k].append(float(r <= k))
                all_ndcg[k].append((1.0 / np.log2(r + 1)) if r <= k else 0.0)

    m = {}
    for k in K_LIST:
        m[f"hr@{k}"] = float(np.mean(all_hr[k]))
        m[f"ndcg@{k}"] = float(np.mean(all_ndcg[k]))
    return m


# ══════════════════════════════════════════════════════════════════════
# Main per dataset
# ══════════════════════════════════════════════════════════════════════

def run_one_dataset(dataset, device, seeds):
    data_dir  = DATA_DIR / dataset
    cache_dir = CACHE_DIR / dataset

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    # Get CARE best α
    try:
        vr = json.load(open(ROOT / "results" / f"{dataset}_vara_results.json"))
        alpha_best = vr["per_seed"][0].get("gate_best_alpha", 0.5)
    except Exception:
        alpha_best = 0.5

    cache = load_cache_gpu(cache_dir, data_dir, device)
    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)
    item_counts = build_item_counts(data_dir, n_items, device)
    test_set = MARADataset(data_dir, split="test", seed=42, n_neg=99)

    print(f"\n{'='*70}")
    print(f"  MetaEmb Cold-Start Baseline — {dataset}")
    print(f"{'='*70}")

    metaemb_results = []

    for seed in seeds:
        ckpt_path = CKPT_DIR / dataset / f"seed{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] seed={seed}: checkpoint 不存在")
            continue

        print(f"\n  Seed={seed}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("cfg", {"d": 64, "tau_score": 1.0})

        model = VARA(n_users, n_items, cfg).to(device)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Train MetaEmb projector
        print(f"    Training MetaEmb projector...")
        t0 = time.time()
        projector = train_metaemb(model, cache, item_counts, data_dir, device, seed)
        if projector is None:
            continue
        print(f"    {time.time()-t0:.0f}s")

        # Evaluate
        # CF-only (original)
        m_cf = evaluate_cf_only(model, test_set, device)
        # MetaEmb (projected CF for cold + CARE gate)
        m_meta = evaluate_metaemb(model, projector, cache, test_set, item_counts, device, alpha_best)

        print(f"    CF-only  Full N@10={m_cf['ndcg@10']:.4f}")
        print(f"    MetaEmb Full N@10={m_meta['full']['ndcg@10']:.4f}")
        print(f"    MetaEmb L0 N@10={m_meta['L0']['ndcg@10']:.4f}  "
              f"L1 N@10={m_meta['L1']['ndcg@10']:.4f}")

        metaemb_results.append({
            "seed": seed,
            "cf_only": m_cf,
            "metaemb": m_meta,
            "alpha": alpha_best,
        })

    return metaemb_results


def main():
    parser = argparse.ArgumentParser(description="MetaEmb 冷启动基线")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else [args.dataset]

    all_results = {}
    for ds in datasets:
        r = run_one_dataset(ds, device, args.seeds)
        if r:
            all_results[ds] = r

    # Save — merge with existing file if present
    out_path = RESULT_DIR / "meta_coldstart_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(all_results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
