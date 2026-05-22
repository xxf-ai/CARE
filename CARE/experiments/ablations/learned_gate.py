"""
experiments/ablations/learned_gate.py — 轻量学习门控 (Tiny MLP Gate)

在冻结的 VARA CF backbone 上，训练极少参数的门控网络：
  Variant A (2 param):  w_txt = σ(a·log(1+c) + b)
  Variant B (4 param):  w_txt = σ(w₂·relu(w₁·log(1+c) + b₁) + b₂)

其中 σ 是 sigmoid, c 是物品交互数。CARE 的 inverse 函数可被视为
Variant A 在 a=-1, b=log(α) 时的特例。

训练: BPR loss on fused scores, 只更新 gate 参数, CF backbone 冻结。
      在验证集上早停选最优 gate。

使用:
  python experiments/ablations/learned_gate.py --dataset baby
  python experiments/ablations/learned_gate.py --dataset baby office sports --seeds 42
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

from models.vara         import VARA
from data_utils.dataset  import MARADataset, CacheManager
from evaluate            import build_user_text_pref

DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "ablations" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]


# ══════════════════════════════════════════════════════════════════════
# Gate variants
# ══════════════════════════════════════════════════════════════════════

class LearnedGate2Param(nn.Module):
    """2-parameter gate: w = σ(a·log(1+c) + b)"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(-1.0))   # slope in log space
        self.b = nn.Parameter(torch.tensor(0.0))     # bias (log-α equivalent)

    def forward(self, item_counts):
        """item_counts: Tensor of any shape, 返回同形 w_txt ∈ (0,1)"""
        # log(1+c) to avoid log(0) and smooth near zero
        x = torch.log1p(item_counts.float())
        return torch.sigmoid(self.a * x + self.b)

    def extra_repr(self):
        return f"a={self.a.item():.4f}, b={self.b.item():.4f}"


class LearnedGate4Param(nn.Module):
    """4-parameter gate: w = σ(w2·relu(w1·log(1+c) + b1) + b2)"""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0, -1.0]))  # [2]
        self.b1 = nn.Parameter(torch.zeros(2))             # [2]
        self.w2 = nn.Parameter(torch.tensor([1.0, -1.0]))   # [2]
        self.b2 = nn.Parameter(torch.tensor(0.0))           # scalar

    def forward(self, item_counts):
        x = torch.log1p(item_counts.float()).unsqueeze(-1)  # [..., 1]
        h = F.relu(x * self.w1 + self.b1)                    # [..., 2]
        logit = (h * self.w2).sum(dim=-1) + self.b2          # [...]
        return torch.sigmoid(logit)


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def load_cache_gpu(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def build_item_counts(data_dir, n_items, device):
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        counts[int(iid)] += 1
    return counts


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_gate(model, gate, cache, test_set, item_counts, device):
    """Full evaluation with learned gate."""
    tau = getattr(model, "tau_score", 1.0)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)
    all_item_repr = model.get_all_item_cf_repr()

    all_hr = {k: [] for k in K_LIST}
    all_ndcg = {k: [] for k in K_LIST}
    # Per-bucket
    b_hr = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}
    b_ndcg = {b: {k: [] for k in K_LIST} for b in ["L0", "L1", "L2", "L3"]}

    for users, pos_items, neg_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        u_cf = model.get_user_cf_repr(users)
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        cand_counts = item_counts[cand]
        w_txt = gate(cand_counts)

        fused = (1.0 - w_txt) * cf + w_txt * txt
        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
        ranks = ranks.cpu().numpy()
        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            r = int(ranks[j])
            cnt = int(pos_counts[j])
            if cnt == 0:
                bn = "L0"
            elif cnt <= 4:
                bn = "L1"
            elif cnt <= 20:
                bn = "L2"
            else:
                bn = "L3"
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
def evaluate_care_gate(model, cache, test_set, item_counts, device, alpha):
    """Evaluate with CARE inverse gate w=1/(1+α·c) for comparison."""
    tau = getattr(model, "tau_score", 1.0)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)
    all_item_repr = model.get_all_item_cf_repr()
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
        i_cf = all_item_repr[cand_flat].reshape(B, 100, -1)
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau

        u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
        i_txt = F.normalize(cache["clip_text"][cand_flat], dim=-1).reshape(B, 100, -1)
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau

        w_txt = 1.0 / (1.0 + alpha * item_counts[cand])
        fused = (1.0 - w_txt) * cf + w_txt * txt
        ranks = (fused > fused[:, 0:1]).sum(dim=1) + 1
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
# Training
# ══════════════════════════════════════════════════════════════════════

def train_gate(model, gate, cache, data_dir, device, n_items, seed,
               lr=1e-2, epochs=30, patience=5, batch_size=16384):
    """Train gate on frozen CF backbone with BPR loss."""
    set_seed(seed)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_users = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_items = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_users)

    item_counts = build_item_counts(data_dir, n_items, device)
    tau = getattr(model, "tau_score", 1.0)

    # Precompute all_item_cf once
    all_item_cf = model.get_all_item_cf_repr()

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=99)

    optimizer = optim.AdamW(gate.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ndcg, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        gate.train()
        perm = torch.randperm(n_train, device=device)
        eu = train_users[perm]
        ei = train_items[perm]
        ep_loss, nb = 0.0, 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)
            B = users.size(0)

            with torch.no_grad():
                # CF scores
                u_cf = model.get_user_cf_repr(users)
                pos_cf = (u_cf * all_item_cf[pos]).sum(-1) / tau
                neg_cf = (u_cf * all_item_cf[neg]).sum(-1) / tau

                # Text scores
                u_txt = F.normalize(cache["user_text_pref"][users], dim=-1)
                pos_txt = (u_txt * F.normalize(cache["clip_text"][pos], dim=-1)).sum(-1) / tau
                neg_txt = (u_txt * F.normalize(cache["clip_text"][neg], dim=-1)).sum(-1) / tau

            # Gate weights
            w_pos = gate(item_counts[pos])
            w_neg = gate(item_counts[neg])

            # Fused scores
            fused_pos = (1.0 - w_pos) * pos_cf + w_pos * pos_txt
            fused_neg = (1.0 - w_neg) * neg_cf + w_neg * neg_txt

            loss = -F.logsigmoid(fused_pos - fused_neg - 0.1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            nb += 1

        scheduler.step()

        # Validate every 3 epochs
        if epoch % 3 == 0 or epoch == epochs:
            gate.eval()
            vm = evaluate_gate(model, gate, cache, val_set, item_counts, device)
            ndcg = vm["full"]["ndcg@10"]
            gate.train()

            if ndcg > best_val_ndcg:
                best_val_ndcg = ndcg
                best_state = {k: v.clone() for k, v in gate.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

    if best_state:
        gate.load_state_dict(best_state)
    gate.eval()
    return gate, best_val_ndcg


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def run_one_dataset(dataset, device, gate_variant="2param"):
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

    model = VARA(stats["n_users"], stats["n_items"], cfg).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    cache = load_cache_gpu(cache_dir, data_dir, device)
    cache["user_text_pref"] = build_user_text_pref(
        data_dir / "train_indexed.csv", cache["clip_text"], stats["n_users"], device)

    # Get CARE best α for comparison
    try:
        vr = json.load(open(ROOT / "results" / f"{dataset}_vara_results.json"))
        alpha_best = vr["per_seed"][0].get("gate_best_alpha", 0.5)
    except Exception:
        alpha_best = 0.5

    print(f"\n{'='*70}")
    print(f"  Learned Gate ({gate_variant}) — {dataset}")
    print(f"{'='*70}")

    gate = LearnedGate2Param() if gate_variant == "2param" else LearnedGate4Param()
    gate = gate.to(device)
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"  Gate params: {n_params}")

    # Train gate
    t0 = time.time()
    gate, val_ndcg = train_gate(model, gate, cache, data_dir, device,
                                 stats["n_items"], seed=42, lr=1e-2, epochs=30)
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.0f}s  val NDCG@10={val_ndcg:.4f}")

    # Print learned parameters
    if gate_variant == "2param":
        print(f"  Learned: a={gate.a.item():.4f}, b={gate.b.item():.4f}")
    else:
        print(f"  Learned w1={gate.w1.data}, b1={gate.b1.data}")
        print(f"  Learned w2={gate.w2.data}, b2={gate.b2.item():.4f}")

    # Test evaluation
    item_counts = build_item_counts(data_dir, stats["n_items"], device)
    test_set = MARADataset(data_dir, split="test", seed=42, n_neg=99)

    m_learned = evaluate_gate(model, gate, cache, test_set, item_counts, device)
    m_care = evaluate_care_gate(model, cache, test_set, item_counts, device, alpha_best)

    # Per-bucket comparison
    print(f"\n  Test comparison (CARE α={alpha_best} vs Learned {gate_variant}):")
    print(f"  {'Bucket':<16} {'CARE N@10':>12} {'Learned N@10':>12} {'Δ':>10}")
    print(f"  {'─'*54}")
    for bkey, bname in [("full", "Full"), ("L0", "L0 (0)"), ("L1", "L1 (1-4)"),
                         ("L2", "L2 (5-20)"), ("L3", "L3 (>20)")]:
        c = m_care["ndcg@10"] if bkey == "full" else m_care.get("ndcg@10", 0)
        l = m_learned[bkey]["ndcg@10"]
        # For CARE per-bucket, we can't easily get it from evaluate_care_gate
        # (it only returns full-set). Use m_learned comparison only for full.
        if bkey == "full":
            print(f"  {bname:<16} {c:>12.4f} {l:>12.4f} {l-c:>+10.4f}")

    print(f"\n  Per-bucket (Learned gate only):")
    for b in ["L0", "L1", "L2", "L3"]:
        print(f"  {b:<16} NDCG@10={m_learned[b]['ndcg@10']:.4f}")

    return {
        "dataset": dataset,
        "gate_variant": gate_variant,
        "n_params": n_params,
        "alpha_care": alpha_best,
        "learned_params": {
            "a": gate.a.item(), "b": gate.b.item()
        } if gate_variant == "2param" else {},
        "val_ndcg": val_ndcg,
        "test_care_full": m_care,
        "test_learned": m_learned,
    }


def main():
    parser = argparse.ArgumentParser(description="轻量学习门控")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--variant", default="2param", choices=["2param", "4param"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else [args.dataset]

    all_results = []
    for ds in datasets:
        r = run_one_dataset(ds, device, args.variant)
        if r:
            all_results.append(r)

    out_path = RESULT_DIR / f"learned_gate_{args.variant}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
