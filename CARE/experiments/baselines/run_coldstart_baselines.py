"""
experiments/baselines/run_coldstart_baselines.py — 专用冷启动基线训练 + 分层评估

实现 4 个专门针对冷启动的推荐方法:
  - DropoutNet (NeurIPS 2017): 训练时随机丢弃 item CF embedding，模拟冷启动条件
  - CLCRec    (WWW 2021): 对比学习将冷启动物品的 CF embedding 拉向内容表示
  - CCFCRec   (WWW 2023): 两路 CF + 对比学习对齐 content 与 co-occurrence 表示
  - MARec     (RecSys 2024): Metadata alignment，warm items 上训练 CLIP→CF projector

用法:
  python experiments/baselines/run_coldstart_baselines.py --dataset baby --seeds 42 123 456
  python experiments/baselines/run_coldstart_baselines.py --dataset baby --models DropoutNet --seeds 42
  python experiments/baselines/run_coldstart_baselines.py --dataset baby --models CCFCRec MARec --seeds 42
"""

import sys, json, argparse, time, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_utils.dataset import MARADataset, CacheManager
from evaluation.evaluator import Evaluator
from evaluate import build_user_text_pref
from experiments.baselines.models.coldstart_models import CCFCRec, MARec

DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "experiments" / "baselines" / "results"
LOG_DIR    = ROOT / "logs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

COLD_THRESHOLD = 5
K_LIST = [5, 10, 20]
BASE_CFG = {
    "d": 64, "text_dim": 768,
    "lr": 1e-3, "weight_decay": 1e-5,
    "max_epochs": 30, "patience": 6,
    "batch_size": 16384, "eval_batch": 16384, "neg_batch": 64,
    "k_list": [5, 10, 20], "n_neg": 99,
    "val_every": 3,
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# DropoutNet
# ═══════════════════════════════════════════════════════════════════════════

class DropoutNet(nn.Module):
    """
    DropoutNet (NeurIPS 2017) 简化版。
    核心思想: 训练时随机丢弃 item CF embedding，强迫模型依赖内容特征，
    从而使冷启动物品（无 CF 信号）在推理时也能通过内容通道获得合理表示。
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64, dropout_rate=0.5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.content_proj = nn.Sequential(
            nn.Linear(text_dim, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, d, bias=False),
            nn.LayerNorm(d),
        )
        self.dropout_rate = dropout_rate
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def _item_combined(self, item_ids, item_text):
        cf = self.item_emb(item_ids)
        ct = self.content_proj(item_text)
        if self.training:
            mask = (torch.rand(item_ids.size(0), 1, device=cf.device)
                    >= self.dropout_rate).float()
            cf = cf * mask
        return cf + ct

    def get_score(self, users, items, item_text):
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self._item_combined(items, item_text), dim=-1)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_items, neg_items, pos_text, neg_text):
        pos = self.get_score(users, pos_items, pos_text)
        neg = self.get_score(users, neg_items, neg_text)
        diff = pos - neg
        l_bpr = -F.logsigmoid(diff).mean()
        reg = (self.user_emb(users).norm(2).pow(2) +
               self.item_emb(pos_items).norm(2).pow(2) +
               self.item_emb(neg_items).norm(2).pow(2)) / users.size(0)
        return l_bpr + 1e-5 * reg

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    @torch.no_grad()
    def get_full_item_repr(self, item_ids, item_text):
        return F.normalize(self._item_combined(item_ids, item_text), dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# CLCRec
# ═══════════════════════════════════════════════════════════════════════════

class CLCRec(nn.Module):
    """
    CLCRec (WWW 2021) 简化版。
    核心思想: 用对比损失将冷启动物品的 CF embedding 拉向其内容表示，
    使得零交互物品在 CF 空间中也有语义上有意义的表示。
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64, lambda_cl=0.1, temp=0.2):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.content_proj = nn.Sequential(
            nn.Linear(text_dim, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, d, bias=False),
            nn.LayerNorm(d),
        )
        self.lambda_cl = lambda_cl
        self.temp = temp
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_score(self, users, items):
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.item_emb(items), dim=-1)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_items, neg_items,
                is_cold_pos, is_cold_neg, pos_text, neg_text):
        # BPR on CF scores
        pos = self.get_score(users, pos_items)
        neg = self.get_score(users, neg_items)
        l_bpr = -F.logsigmoid(pos - neg).mean()

        # Contrastive: 拉冷启动物品的 CF embedding 向内容投影
        l_cl = torch.tensor(0.0, device=users.device)
        n_cold = 0

        # 正样本中的冷启动物品
        if is_cold_pos.any():
            cold_mask = is_cold_pos
            item_cf = F.normalize(self.item_emb(pos_items[cold_mask]), dim=-1)
            item_txt = F.normalize(self.content_proj(pos_text[cold_mask]), dim=-1)
            # InfoNCE: 每个冷物品的 CF 应与自己的 content 匹配
            sim = (item_cf * item_txt).sum(dim=-1) / self.temp
            l_cl = l_cl - sim.mean()
            n_cold += cold_mask.sum().item()

        # 负样本中的冷启动物品
        if is_cold_neg.any():
            cold_mask = is_cold_neg
            item_cf = F.normalize(self.item_emb(neg_items[cold_mask]), dim=-1)
            item_txt = F.normalize(self.content_proj(neg_text[cold_mask]), dim=-1)
            sim = (item_cf * item_txt).sum(dim=-1) / self.temp
            l_cl = l_cl - sim.mean()
            n_cold += cold_mask.sum().item()

        if n_cold > 0:
            l_cl = l_cl / n_cold

        reg = (self.user_emb(users).norm(2).pow(2) +
               self.item_emb(pos_items).norm(2).pow(2) +
               self.item_emb(neg_items).norm(2).pow(2)) / users.size(0)

        return l_bpr + self.lambda_cl * l_cl + 1e-5 * reg

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# 分层评估（复用 run_stratified_baselines 的逻辑）
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def stratified_eval_coldstart(model, model_name, cache, item_counts, n_items,
                               device, seed, data_dir, alpha=1.0):
    """对冷启动基线模型运行 L0/L1/L2/L3 分层评估。"""
    tau = 1.0
    buckets = [
        ("L0 (zero-shot)", 0, 0),
        ("L1 (1-4)",       1, 4),
        ("L2 (5-20)",      5, 20),
        ("L3 (>20)",      21, 1_000_000),
    ]

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)
    strategies = ["cf_only", "text_only", "gate"]
    bucket_ranks = {bname: {s: [] for s in strategies} for bname, _, _ in buckets}

    base_model = getattr(model, "_orig_mod", model)
    can_extract_cf = (hasattr(base_model, 'get_user_cf_repr') and
                      hasattr(base_model, 'get_all_item_cf_repr'))

    clip_text = cache["clip_text"]
    user_text_pref = cache["user_text_pref"]
    if can_extract_cf:
        all_item_cf = base_model.get_all_item_cf_repr()

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        cand_flat = cand.reshape(-1)

        # Text scores
        u_txt = F.normalize(user_text_pref[users], dim=-1)
        i_txt = F.normalize(clip_text[cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(-1) / tau
        txt_ranks = ((txt_scores > txt_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        cf_ranks = gate_ranks = None
        if can_extract_cf:
            u_cf = base_model.get_user_cf_repr(users)
            i_cf = all_item_cf[cand_flat].reshape(B, 100, -1)
            cf_scores = (u_cf.unsqueeze(1) * i_cf).sum(-1) / tau
            cf_ranks = ((cf_scores > cf_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

            cand_counts = item_counts[cand]
            w_txt = 1.0 / (1.0 + alpha * cand_counts)
            gate_scores = (1.0 - w_txt) * cf_scores + w_txt * txt_scores
            gate_ranks = ((gate_scores > gate_scores[:, 0:1]).sum(dim=1) + 1).cpu().numpy()

        pos_counts = item_counts[pos_items].cpu().numpy()
        for j in range(B):
            cnt = int(pos_counts[j])
            for bname, lo, hi in buckets:
                if lo <= cnt <= hi:
                    bucket_ranks[bname]["text_only"].append(int(txt_ranks[j]))
                    if can_extract_cf:
                        bucket_ranks[bname]["cf_only"].append(int(cf_ranks[j]))
                        bucket_ranks[bname]["gate"].append(int(gate_ranks[j]))
                    break

    result = {}
    for bname, _, _ in buckets:
        bres = {}
        for strategy in strategies:
            ranks_arr = np.array(bucket_ranks[bname][strategy])
            if len(ranks_arr) == 0:
                bres[strategy] = None
                continue
            m = {"n": len(ranks_arr)}
            for k in K_LIST:
                m[f"hr@{k}"]   = float((ranks_arr <= k).mean())
                m[f"ndcg@{k}"] = float(
                    np.where(ranks_arr <= k, 1.0 / np.log2(ranks_arr + 1), 0.0).mean())
            bres[strategy] = m
        result[bname] = bres
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════════

def load_cache_gpu(cache_dir, device):
    mgr = CacheManager(cache_dir, DATA_DIR / args_dataset, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


def _save_ckpt(model, optimizer, epoch, best_metric, ckpt_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_metric": best_metric}, ckpt_path)


def train_dropoutnet(data_dir, cache, n_users, n_items, device, seed, cfg):
    model = DropoutNet(n_users, n_items, text_dim=768, d=cfg["d"],
                        dropout_rate=cfg.get("dropout_rate", 0.5)).to(device)

    clip_text = cache["clip_text"].to(device)
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "dropoutnet" / f"seed{seed}"
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
                        clip_text[pos], clip_text[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"DN ep{epoch}")
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


def train_clcrec(data_dir, cache, n_users, n_items, device, seed, cfg):
    model = CLCRec(n_users, n_items, text_dim=768, d=cfg["d"],
                    lambda_cl=cfg.get("lambda_cl", 0.1),
                    temp=cfg.get("temp", 0.2)).to(device)

    clip_text = cache["clip_text"].to(device)
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.from_numpy(train_df["user_id"].values.astype(np.int64)).to(device)
    train_i = torch.from_numpy(train_df["item_id"].values.astype(np.int64)).to(device)
    n_train = len(train_u)

    # 计算训练集中每个物品的交互次数（用于判断冷启动）
    item_counts = torch.zeros(n_items, dtype=torch.long, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"], shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device, cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "clcrec" / f"seed{seed}"
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

            is_cold_pos = item_counts[pos] < COLD_THRESHOLD
            is_cold_neg = item_counts[neg] < COLD_THRESHOLD

            loss = model(users, pos, neg,
                        is_cold_pos, is_cold_neg,
                        clip_text[pos], clip_text[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"CLCRec ep{epoch}")
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


# ═══════════════════════════════════════════════════════════════════════════
# CCFCRec
# ═══════════════════════════════════════════════════════════════════════════

def train_ccfcrec(data_dir, cache, n_users, n_items, device, seed, cfg):
    model = CCFCRec(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
    clip_text = cache["clip_text"].to(device)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    train_i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    n_train = len(train_u)

    item_counts = torch.zeros(n_items, dtype=torch.long, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"],
                            shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device,
                          cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "ccfcrec" / f"seed{seed}"
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
            is_cold_pos = item_counts[pos] < COLD_THRESHOLD
            loss = model(users, pos, neg, is_cold_pos,
                         clip_text[pos], clip_text[neg])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"CCFCRec ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg,
                           ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg,
                   ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


# ═══════════════════════════════════════════════════════════════════════════
# MARec
# ═══════════════════════════════════════════════════════════════════════════

def train_marec(data_dir, cache, n_users, n_items, device, seed, cfg):
    model = MARec(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
    clip_text = cache["clip_text"].to(device)

    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    train_u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    train_i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    n_train = len(train_u)

    item_counts = torch.zeros(n_items, dtype=torch.long, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1
    is_warm = item_counts >= COLD_THRESHOLD

    val_set = MARADataset(data_dir, split="val", seed=seed, n_neg=cfg["n_neg"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch"],
                            shuffle=False, num_workers=4)
    evaluator = Evaluator(model, {}, cfg["k_list"], device,
                          cfg["eval_batch"], cfg["neg_batch"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ckpt_dir = CKPT_DIR / args_dataset / "marec" / f"seed{seed}"
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
            is_warm_pos = is_warm[pos]
            loss = model(users, pos, neg, is_warm_pos, clip_text[pos])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        if epoch % cfg["val_every"] == 0 or epoch <= 5:
            val_m = evaluator.evaluate(val_loader, desc=f"MARec ep{epoch}")
            ndcg = val_m.get("ndcg@10", 0)
            print(f"    ep{epoch:2d}  loss={ep_loss/max(nb,1):.4f}  NDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg, patience = ndcg, 0
                _save_ckpt(model, optimizer, epoch, best_ndcg,
                           ckpt_dir / "best_model.pt")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"    早停 @ epoch {epoch}")
                    break

    if not (ckpt_dir / "best_model.pt").exists():
        _save_ckpt(model, optimizer, epoch, best_ndcg,
                   ckpt_dir / "best_model.pt")
    return ckpt_dir / "best_model.pt", best_ndcg


TRAINERS = {"DropoutNet": train_dropoutnet, "CLCRec": train_clcrec,
            "CCFCRec": train_ccfcrec, "MARec": train_marec}


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

args_dataset = None

def main():
    global args_dataset
    parser = argparse.ArgumentParser(description="冷启动基线训练 + 分层评估")
    parser.add_argument("--dataset", default="baby",
                        choices=["baby", "beauty_sub", "office", "sports", "yelp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--models", nargs="+", default=["DropoutNet", "CLCRec"])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
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

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    if args.alpha is not None:
        alpha = args.alpha
    else:
        try:
            vr = json.load(open(ROOT / "results" / f"{args.dataset}_vara_results.json"))
            alpha = vr["per_seed"][0].get("gate_best_alpha", 1.0)
        except Exception:
            alpha = 1.0 if args.dataset in ("baby", "yelp") else 0.5

    print(f"\n{'='*70}")
    print(f"  冷启动基线训练 + 分层评估  dataset={args.dataset}  seeds={args.seeds}")
    print(f"  models={args.models}  α={alpha}")
    print(f"{'='*70}")

    # ── 加载 cache ──────────────────────────────────────────────────
    cache = {}
    if not args.skip_train or not args.skip_eval:
        print("\n  加载特征缓存...")
        cache = load_cache_gpu(cache_dir, device)

    # ── 训练 ─────────────────────────────────────────────────────────
    train_results = {}
    if not args.skip_train:
        for model_name in args.models:
            trainer = TRAINERS.get(model_name)
            if trainer is None:
                print(f"\n  [SKIP] 未知模型: {model_name}")
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

    # ── 分层评估 ─────────────────────────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print(f"  分层评估")
        print(f"{'='*70}")

        # 构建 user_text_pref + item_counts
        cache["user_text_pref"] = build_user_text_pref(
            data_dir / "train_indexed.csv", cache["clip_text"], n_users, device)
        train_df = pd.read_csv(data_dir / "train_indexed.csv")
        item_counts = torch.zeros(n_items, device=device)
        for iid in train_df["item_id"].values:
            item_counts[int(iid)] += 1

        all_stratified = {}
        ckpt_subdirs = {"DropoutNet": "dropoutnet", "CLCRec": "clcrec",
                        "CCFCRec": "ccfcrec", "MARec": "marec"}

        for model_name in args.models:
            print(f"\n  [{model_name}]")
            subdir = ckpt_subdirs.get(model_name)
            if subdir is None:
                continue

            strat_results = {}
            for seed in args.seeds:
                ckpt_path = CKPT_DIR / args.dataset / subdir / f"seed{seed}" / "best_model.pt"
                if not ckpt_path.exists():
                    print(f"  seed={seed}: checkpoint 不存在 {ckpt_path}")
                    continue

                print(f"  seed={seed}: {ckpt_path}")

                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                if model_name == "DropoutNet":
                    model = DropoutNet(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
                elif model_name == "CLCRec":
                    model = CLCRec(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
                elif model_name == "CCFCRec":
                    model = CCFCRec(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
                elif model_name == "MARec":
                    model = MARec(n_users, n_items, text_dim=768, d=cfg["d"]).to(device)
                else:
                    print(f"  [SKIP] Unknown model: {model_name}")
                    continue

                state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                result = stratified_eval_coldstart(
                    model, model_name, cache, item_counts, n_items,
                    device, seed, data_dir, alpha=alpha)
                if result is not None:
                    strat_results[seed] = {"result": result, "alpha": alpha, "ckpt": str(ckpt_path)}
                    _print_stratified(model_name, seed, result, alpha)

            if strat_results:
                all_stratified[model_name] = strat_results

        if all_stratified:
            _print_multi_seed_summary(all_stratified)

            out_path = RESULT_DIR / f"{args.dataset}_coldstart_baselines.json"

            # 读取已有结果并在 seed 级别合并,避免覆盖
            existing = {}
            if out_path.exists():
                try:
                    with open(out_path) as f:
                        prev = json.load(f)
                    existing = prev.get("results", {})
                except Exception:
                    pass

            merged_results = dict(existing)
            for mname, seeds_dict in all_stratified.items():
                if mname not in merged_results:
                    merged_results[mname] = {}
                # seed 级别合并: 只更新新 seed,不覆盖已有 seed
                for seed_str, data in seeds_dict.items():
                    merged_results[mname][seed_str] = data

            all_models = sorted(merged_results.keys())

            with open(out_path, "w") as f:
                json.dump({
                    "dataset": args.dataset,
                    "models": all_models,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": merged_results,
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  结果已保存: {out_path}")

    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"{'='*70}")


def _print_stratified(model_name, seed, result, alpha):
    print(f"\n  Stratified (α={alpha}):")
    print(f"  {'Bucket':<18} {'n':>6}  {'CF N@10':>8}  {'TXT N@10':>8}  {'GATE N@10':>8}")
    print(f"  {'─'*58}")
    for bname in ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]:
        b = result.get(bname, {})
        cf  = b.get("cf_only", {})
        txt = b.get("text_only", {})
        gt  = b.get("gate", {})
        n = cf.get("n", 0) if cf else 0
        cfn = f"{cf.get('ndcg@10', 0):.4f}" if cf else "N/A"
        tn  = f"{txt.get('ndcg@10', 0):.4f}" if txt else "N/A"
        gn  = f"{gt.get('ndcg@10', 0):.4f}" if gt else "N/A"
        print(f"  {bname:<18} {n:>6}  {cfn:>8}  {tn:>8}  {gn:>8}")


def _print_multi_seed_summary(all_results):
    print(f"\n{'='*80}")
    print(f"  多 Seed 汇总 — NDCG@10 mean ± std")
    print(f"{'='*80}")

    buckets_order = ["L0 (zero-shot)", "L1 (1-4)", "L2 (5-20)", "L3 (>20)"]
    strategies = ["cf_only", "text_only", "gate"]

    for model_name, seeds_dict in all_results.items():
        print(f"\n  [{model_name}]")
        header = f"  {'Bucket':<18}"
        for s in strategies:
            header += f"  {s:>18}"
        print(header)
        print(f"  {'─'*76}")
        for bname in buckets_order:
            row = f"  {bname:<18}"
            for strategy in strategies:
                vals = []
                for seed, data in seeds_dict.items():
                    b = data["result"].get(bname, {})
                    m = b.get(strategy, {}) if b else {}
                    if m and m.get("ndcg@10") is not None:
                        vals.append(m["ndcg@10"])
                if len(vals) >= 2:
                    row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
                elif len(vals) == 1:
                    row += f"  {vals[0]:.4f}"
                else:
                    row += f"  {'N/A':>18}"
            print(row)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
