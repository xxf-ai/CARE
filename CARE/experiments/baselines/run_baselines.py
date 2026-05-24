"""
experiments/baselines/run_baselines.py — 基线模型训练 + 分层评估

训练 BPR-MF, LightGCN, VBPR, BM3, FREEDOM 基线，保存 checkpoint，
然后自动运行 Table 3 格式的分层评估。

用法:
  python experiments/baselines/run_baselines.py --dataset baby --seeds 42 123 456
  python experiments/baselines/run_baselines.py --dataset baby --models BPR LIGHTGCN --seeds 42
  python experiments/baselines/run_baselines.py --dataset baby --skip_train   # 仅评估已有 ckpt
"""

import sys, json, argparse, time, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_utils.dataset import MARADataset, CacheManager
from evaluation.evaluator import Evaluator
from experiments.baselines.models.baselines import BPR_MF, LightGCN_Rec, VBPR, BM3, FREEDOM
from evaluate import get_cold_items, build_user_text_pref
from experiments.baselines.run_stratified_baselines import (
    stratified_eval, build_model, _print_stratified, _print_multi_seed_summary,
)

DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "baselines" / "results"
LOG_DIR    = ROOT / "logs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [5, 10, 20]

BASE_CFG = {
    "d": 64, "lr": 1e-3, "weight_decay": 1e-5,
    "max_epochs": 30, "patience": 6,
    "batch_size": 16384, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "val_every": 3, "use_amp": True,
}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_to_device(batch, device):
    """将 DataLoader batch 移到 GPU。"""
    users, pos_items, neg_items = batch
    return (
        users.to(device),
        pos_items.to(device),
        neg_items.to(device),
    )


def _save_ckpt(model, optimizer, epoch, best_metric, ckpt_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
    }, ckpt_path)


def _load_ckpt(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device, weights_only=False)


# ═══════════════════════════════════════════════════════════════════════════
# 各模型训练入口
# ═══════════════════════════════════════════════════════════════════════════

def train_bpr(data_dir, cache, n_users, n_items, device, seed, cfg):
    """训练 BPR-MF。"""
    model = BPR_MF(n_users, n_items, d=cfg["d"]).to(device)
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "bpr" / f"seed{seed}"
    best_ndcg, patience = 0.0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0
        nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            loss = model(users, pos, neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"BPR ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


def train_lightgcn(data_dir, cache, n_users, n_items, device, seed, cfg):
    """训练 LightGCN 推荐模型（冻结预训练嵌入 + 可学习投影）。"""
    user_emb_np = np.load(CACHE_DIR / args_dataset / "user_emb.npy")
    item_emb_np = np.load(CACHE_DIR / args_dataset / "item_emb.npy")

    model = LightGCN_Rec(user_emb_np, item_emb_np, d=cfg["d"]).to(device)
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "lightgcn" / f"seed{seed}"
    best_ndcg, patience = 0.0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0; nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            loss = model(users, pos, neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"LIGHTGCN ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


def train_vbpr(data_dir, cache, n_users, n_items, device, seed, cfg):
    """训练 VBPR（CF + CLIP 视觉特征）。"""
    clip_cls = cache.get("clip_cls")
    if clip_cls is None:
        print("  [SKIP] VBPR 需要 clip_cls.npy，未找到")
        return None, 0.0
    clip_cls = clip_cls.to(device)

    # 构建 user visual preference
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    user_vis_pref = torch.zeros(n_users, 768, device=device)
    user_cnt = torch.zeros(n_users, device=device)
    for uid, iid in zip(train_df["user_id"].values, train_df["item_id"].values):
        user_vis_pref[int(uid)] += clip_cls[int(iid)]
        user_cnt[int(uid)] += 1
    mask = user_cnt > 0
    user_vis_pref[mask] = user_vis_pref[mask] / user_cnt[mask].unsqueeze(1)

    model = VBPR(n_users, n_items, clip_dim=768, d=cfg["d"]).to(device)

    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "vbpr" / f"seed{seed}"
    best_ndcg, patience = 0.0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0; nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            loss = model(users, pos, neg,
                        user_vis_pref[users], clip_cls[pos], clip_cls[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"VBPR ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


def train_bm3(data_dir, cache, n_users, n_items, device, seed, cfg):
    """训练 BM3（CLIP 视觉 + LightGCN 协同 + 对比学习）。"""
    clip_cls = cache.get("clip_cls")
    if clip_cls is None:
        print("  [SKIP] BM3 需要 clip_cls.npy，未找到")
        return None, 0.0

    # Load LightGCN embeddings for collaborative features
    col_path = CACHE_DIR / args_dataset / "item_emb.npy"
    if col_path.exists():
        col_emb = torch.from_numpy(np.load(col_path).astype(np.float32)).to(device)
    else:
        col_emb = torch.randn(n_items, cfg["d"], device=device) * 0.01

    clip_cls = clip_cls.to(device)

    model = BM3(n_users, n_items, clip_dim=768, collab_dim=col_emb.shape[1], d=cfg["d"]).to(device)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "bm3" / f"seed{seed}"
    best_ndcg, patience = 0.0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0; nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            loss = model(users, pos, neg,
                        clip_cls[pos], clip_cls[neg],
                        col_emb[pos], col_emb[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"BM3 ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


def train_freedom(data_dir, cache, n_users, n_items, device, seed, cfg):
    """训练 FREEDOM（视觉 + 文本 + 协同融合）。

    FREEDOM 无 item_emb，无法用标准 Evaluator。验证用 BPR loss 近似。
    """
    clip_cls = cache.get("clip_cls")
    clip_text = cache.get("clip_text")
    if clip_cls is None or clip_text is None:
        print("  [SKIP] FREEDOM 需要 clip_cls.npy 和 clip_text.npy")
        return None, 0.0

    col_path = CACHE_DIR / args_dataset / "item_emb.npy"
    if col_path.exists():
        col_emb = torch.from_numpy(np.load(col_path).astype(np.float32)).to(device)
    else:
        col_emb = torch.randn(n_items, cfg["d"], device=device) * 0.01

    clip_cls = clip_cls.to(device)
    clip_text = clip_text.to(device)

    model = FREEDOM(n_users, n_items, clip_dim=768, collab_dim=col_emb.shape[1], d=cfg["d"]).to(device)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    # 验证集 BPR loss 评估（FREEDOM 无独立 item_emb，无法计算标准 NDCG@10）
    val_df = pd.read_csv(data_dir / "val_indexed.csv")
    val_users = torch.from_numpy(val_df["user_id"].values.astype(np.int64)).to(device)
    val_items = torch.from_numpy(val_df["item_id"].values.astype(np.int64)).to(device)
    val_neg = torch.randint(0, n_items, (len(val_users),), device=device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "freedom" / f"seed{seed}"
    best_loss, patience = float("inf"), 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        eu, ei = train_u[perm], train_i[perm]
        ep_loss = 0.0; nb = 0

        for start in range(0, n_train, cfg["batch_size"]):
            end = min(start + cfg["batch_size"], n_train)
            users, pos = eu[start:end], ei[start:end]
            neg = torch.randint(0, n_items, (users.size(0),), device=device)

            loss = model(users,
                        clip_cls[pos], clip_cls[neg],
                        clip_text[pos], clip_text[neg],
                        col_emb[pos], col_emb[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            model.eval()
            with torch.no_grad():
                val_loss = model(val_users,
                               clip_cls[val_items], clip_cls[val_neg],
                               clip_text[val_items], clip_text[val_neg],
                               col_emb[val_items], col_emb[val_neg])
                vl = val_loss.item()
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  val_loss={vl:.4f}")
            if vl < best_loss:
                best_loss, patience = vl, 0
                _save_ckpt(model, optimizer, epoch, best_loss, ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_loss, ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_loss


# ── 调度 ──────────────────────────────────────────────────────────────────

TRAINERS = {
    "BPR":      train_bpr,
    "LIGHTGCN": train_lightgcn,
    "VBPR":     train_vbpr,
    "BM3":      train_bm3,
    "FREEDOM":  train_freedom,
}


def load_cache_gpu(cache_dir, data_dir, device):
    """加载所有可用的 cache 特征到 GPU。"""
    cache_mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in cache_mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def run_stratified_for_model(model_name, data_dir, cache_dir, device, seeds, dataset, alpha):
    """加载已训练的 checkpoint 并跑分层评估。BPR 复用 CARE 原始数据保证一致性。"""
    from experiments.baselines.run_stratified_baselines import evaluate_one_baseline, _load_care_stratified

    # BPR 行直接复用 CARE 主实验分层数据
    if model_name == "BPR":
        bpr = _load_care_stratified(dataset, seeds, alpha)
        if bpr:
            for seed, data in bpr.items():
                _print_stratified(model_name, seed, data["result"], alpha)
            return bpr

    results = {}
    subdir = {"BPR": "bpr", "LIGHTGCN": "lightgcn", "VBPR": "vbpr",
              "BM3": "bm3", "FREEDOM": "freedom"}[model_name]

    for seed in seeds:
        ckpt_path = CKPT_DIR / dataset / subdir / f"seed{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  seed={seed}: checkpoint 不存在 {ckpt_path}")
            continue

        print(f"  seed={seed}: 分层评估...")
        result = evaluate_one_baseline(
            model_name, ckpt_path, data_dir, cache_dir, device, seed, dataset, alpha=alpha)
        if result is not None:
            results[seed] = {"result": result, "alpha": alpha, "ckpt": str(ckpt_path)}
            _print_stratified(model_name, seed, result, alpha)

    return results


# ── 主入口 ────────────────────────────────────────────────────────────────

args_dataset = None  # module-level, set by main()

def main():
    global args_dataset

    parser = argparse.ArgumentParser(description="基线模型训练 + 分层评估")
    parser.add_argument("--dataset", default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--models", nargs="+",
                        default=["BPR", "LIGHTGCN", "VBPR", "BM3", "FREEDOM"])
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练，仅运行分层评估")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过分层评估，仅训练")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    args_dataset = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = DATA_DIR / args.dataset
    cache_dir = CACHE_DIR / args.dataset

    cfg = dict(BASE_CFG)
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.epochs:     cfg["max_epochs"] = args.epochs
    if args.lr:         cfg["lr"] = args.lr

    # Auto-scale batch size
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]
    n_train_approx = stats.get("n_train", n_users * 10)
    if n_train_approx > 1_000_000:
        cfg["batch_size"] = 65536
    elif n_train_approx > 500_000:
        cfg["batch_size"] = 32768

    # α
    if args.alpha is not None:
        alpha = args.alpha
    else:
        try:
            vr = json.load(open(ROOT / "results" / f"{args.dataset}_care_results.json"))
            alpha = vr["per_seed"][0].get("gate_best_alpha", 1.0)
        except Exception:
            alpha = 1.0 if args.dataset in ("baby", "yelp") else 0.5

    print(f"\n{'='*70}")
    print(f"  基线训练 + 分层评估  dataset={args.dataset}  seeds={args.seeds}")
    print(f"  models={args.models}  α={alpha}  batch={cfg['batch_size']}")
    print(f"{'='*70}")

    # ── 加载 cache ──────────────────────────────────────────────────
    if not args.skip_train:
        print("\n  加载特征缓存...")
        cache = load_cache_gpu(cache_dir, data_dir, device)
    else:
        cache = {}

    # ── 训练 ─────────────────────────────────────────────────────────
    train_results = {}
    if not args.skip_train:
        for model_name in args.models:
            trainer = TRAINERS.get(model_name)
            if trainer is None:
                print(f"\n  [SKIP] 未知模型: {model_name}")
                continue

            if model_name == "BPR":
                print(f"\n  [SKIP] BPR 不需训练，分层评估直接复用 CARE 数据")
                continue

            print(f"\n{'─'*50}")
            print(f"  Train: {model_name}")
            print(f"{'─'*50}")

            train_results[model_name] = {}
            for seed in args.seeds:
                set_seed(seed)
                t0 = time.time()
                print(f"  seed={seed} ...")
                ckpt_path, best_ndcg = trainer(data_dir, cache, n_users, n_items, device, seed, cfg)
                elapsed = time.time() - t0
                rm = "N/A" if ckpt_path is None else f"NDCG@10={best_ndcg:.4f}"
                print(f"  seed={seed}: {rm}  ({elapsed:.0f}s)")
                train_results[model_name][seed] = {
                    "ckpt": str(ckpt_path) if ckpt_path else None,
                    "best_val_ndcg10": best_ndcg,
                }

        # 保存训练摘要
        summary_path = RESULT_DIR / f"{args.dataset}_baselines.json"
        existing = {}
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)

        new_results = {}
        for mname, seeds_dict in train_results.items():
            test_m = {}
            for seed, info in seeds_dict.items():
                test_m[f"seed{seed}"] = info
            new_results[mname] = test_m

        existing["_meta"] = {
            "dataset": args.dataset, "seeds": args.seeds,
            "epochs": cfg["max_epochs"], "lr": cfg["lr"],
            "total_time_sec": 0,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        existing["results"] = new_results
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  训练摘要已保存: {summary_path}")

    # ── 分层评估 ─────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print(f"  分层评估")
        print(f"{'='*70}")

        all_stratified = {}
        for model_name in args.models:
            print(f"\n  [{model_name}]")
            strat_results = run_stratified_for_model(
                model_name, data_dir, cache_dir, device, args.seeds, args.dataset, alpha)
            if strat_results:
                all_stratified[model_name] = strat_results

        if all_stratified:
            _print_multi_seed_summary(all_stratified)

            out_path = RESULT_DIR / f"{args.dataset}_stratified_baselines.json"
            serializable = {}
            for mname, seeds_dict in all_stratified.items():
                serializable[mname] = {}
                for seed, data in seeds_dict.items():
                    serializable[mname][str(seed)] = {
                        "alpha": data["alpha"],
                        "ckpt": data["ckpt"],
                        "result": data["result"],
                    }
            with open(out_path, "w") as f:
                json.dump({
                    "dataset": args.dataset, "seeds": args.seeds,
                    "models": list(all_stratified.keys()),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": serializable,
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  分层结果已保存: {out_path}")

    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
