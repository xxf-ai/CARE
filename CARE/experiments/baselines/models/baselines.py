"""
experiments/baselines/models/baselines.py — 所有基线模型实现

包含：
  BPR_MF     : 矩阵分解 + BPR
  LightGCN   : 直接复用预训练 LightGCN 嵌入（cache/item_emb + user_emb）
  VBPR       : 视觉 BPR（CLIP CLS + 用户嵌入）
  BM3        : 多模态对比学习推荐（简化版）
  FREEDOM    : 去噪多模态图推荐（简化版，用于上界对比）

所有模型统一接口：
  model.get_score(user_ids, item_ids) -> Tensor [B]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# BPR-MF
# ══════════════════════════════════════════════════════════════════════
class BPR_MF(nn.Module):
    """纯矩阵分解，BPR 损失，作为最基础的协同过滤基线。"""

    def __init__(self, n_users: int, n_items: int, d: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_score(self, user_ids, item_ids):
        u = F.normalize(self.user_emb(user_ids), dim=-1)
        i = F.normalize(self.item_emb(item_ids), dim=-1)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    def forward(self, users, pos_items, neg_items):
        pos = self.get_score(users, pos_items)
        neg = self.get_score(users, neg_items)
        loss = -F.logsigmoid(pos - neg).mean()
        reg  = (self.user_emb(users).norm(2).pow(2) +
                self.item_emb(pos_items).norm(2).pow(2) +
                self.item_emb(neg_items).norm(2).pow(2)) / users.size(0)
        return loss + 1e-5 * reg


# ══════════════════════════════════════════════════════════════════════
# LightGCN（复用预训练嵌入）
# ══════════════════════════════════════════════════════════════════════
class LightGCN_Rec(nn.Module):
    """
    直接使用 cache/ 中预训练好的 LightGCN 嵌入做推荐。
    不需要重新训练图卷积，仅在嵌入上加一层线性投影微调。
    """

    def __init__(self, user_emb_np: np.ndarray, item_emb_np: np.ndarray, d: int = 64):
        super().__init__()
        n_u, d_in = user_emb_np.shape
        n_i, _    = item_emb_np.shape

        # 冻结预训练嵌入，加可学习投影
        self.user_base = nn.Embedding.from_pretrained(
            torch.from_numpy(user_emb_np.astype(np.float32)), freeze=True)
        self.item_base = nn.Embedding.from_pretrained(
            torch.from_numpy(item_emb_np.astype(np.float32)), freeze=True)

        self.user_proj = nn.Linear(d_in, d, bias=False)
        self.item_proj = nn.Linear(d_in, d, bias=False)
        nn.init.eye_(self.user_proj.weight) if d == d_in else nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.eye_(self.item_proj.weight) if d == d_in else nn.init.xavier_uniform_(self.item_proj.weight)

    def get_score(self, user_ids, item_ids):
        u = F.normalize(self.user_proj(self.user_base(user_ids)), dim=-1)
        i = F.normalize(self.item_proj(self.item_base(item_ids)), dim=-1)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_proj(self.user_base(users)), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_proj(self.item_base.weight), dim=-1)

    def forward(self, users, pos_items, neg_items):
        pos = self.get_score(users, pos_items)
        neg = self.get_score(users, neg_items)
        return -F.logsigmoid(pos - neg).mean()


# ══════════════════════════════════════════════════════════════════════
# VBPR
# ══════════════════════════════════════════════════════════════════════
class VBPR(nn.Module):
    """
    Visual BPR: 用户嵌入 + CLIP CLS 视觉特征投影，BPR 损失。
    用户侧：可学习嵌入 u_id + 视觉偏好投影 u_vis（从 user_pref）
    物品侧：可学习嵌入 i_id + 视觉投影 i_vis（从 clip_cls）
    """

    def __init__(self, n_users: int, n_items: int, clip_dim: int = 768, d: int = 64):
        super().__init__()
        self.user_id_emb  = nn.Embedding(n_users, d)
        self.item_id_emb  = nn.Embedding(n_items, d)
        self.user_vis_proj = nn.Linear(clip_dim, d, bias=False)
        self.item_vis_proj = nn.Linear(clip_dim, d, bias=False)
        nn.init.normal_(self.user_id_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.user_vis_proj.weight)
        nn.init.xavier_uniform_(self.item_vis_proj.weight)

    def get_score(self, user_ids, item_ids, user_pref, item_vis):
        u = F.normalize(self.user_id_emb(user_ids) +
                        self.user_vis_proj(user_pref), dim=-1)
        i = F.normalize(self.item_id_emb(item_ids) +
                        self.item_vis_proj(item_vis), dim=-1)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_id_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_id_emb.weight, dim=-1)

    def forward(self, users, pos_items, neg_items, user_pref, pos_vis, neg_vis):
        pos = self.get_score(users, pos_items, user_pref, pos_vis)
        neg = self.get_score(users, neg_items, user_pref, neg_vis)
        return -F.logsigmoid(pos - neg).mean()


# ══════════════════════════════════════════════════════════════════════
# BM3（简化版）
# ══════════════════════════════════════════════════════════════════════
class BM3(nn.Module):
    """
    BM3: Bootstrap Multimodal Recommendation（简化版）
    核心：用 CLIP CLS + LightGCN emb 做多模态表示，in-batch 对比学习。
    参考：SIGIR 2023 "Bootstrap Latent Representations for Multi-modal
          Recommendation"
    """

    def __init__(self, n_users: int, n_items: int,
                 clip_dim: int = 768, collab_dim: int = 64, d: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, d)
        self.item_emb  = nn.Embedding(n_items, d)

        # 视觉模态投影
        self.vis_proj  = nn.Sequential(
            nn.Linear(clip_dim, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout))
        # 协同模态投影
        self.col_proj  = nn.Sequential(
            nn.Linear(collab_dim, d), nn.BatchNorm1d(d), nn.ReLU())

        # 融合门
        self.gate = nn.Sequential(nn.Linear(d * 3, d), nn.Sigmoid())

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def _item_repr(self, item_ids, vis_feat, col_feat):
        e_id  = self.item_emb(item_ids)
        e_vis = self.vis_proj(vis_feat)
        e_col = self.col_proj(col_feat)
        g = self.gate(torch.cat([e_id, e_vis, e_col], dim=-1))
        return F.normalize(g * e_vis + (1 - g) * e_col + e_id, dim=-1)

    def get_score(self, user_ids, item_ids, vis_feat, col_feat):
        u = F.normalize(self.user_emb(user_ids), dim=-1)
        i = self._item_repr(item_ids, vis_feat, col_feat)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    def forward(self, users, pos_items, neg_items, pos_vis, neg_vis,
                pos_col, neg_col, tau: float = 0.2):
        u     = F.normalize(self.user_emb(users), dim=-1)
        e_pos = self._item_repr(pos_items, pos_vis, pos_col)
        e_neg = self._item_repr(neg_items, neg_vis, neg_col)

        # BPR
        pos_score = (u * e_pos).sum(-1)
        neg_score = (u * e_neg).sum(-1)
        l_bpr = -F.logsigmoid(pos_score - neg_score).mean()

        # In-batch contrastive: cap at MAX_IB to avoid OOM on large batches
        MAX_IB = 4096
        B = u.size(0)
        if B > MAX_IB:
            idx = torch.randperm(B, device=u.device)[:MAX_IB]
            u_ib, e_ib = u[idx], e_pos[idx]
        else:
            u_ib, e_ib = u, e_pos

        logits = torch.matmul(u_ib, e_ib.T) / tau
        labels = torch.arange(u_ib.size(0), device=u_ib.device)
        l_cl   = (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels)) / 2.0

        return l_bpr + 0.1 * l_cl


# ══════════════════════════════════════════════════════════════════════
# FREEDOM（简化版）
# ══════════════════════════════════════════════════════════════════════
class FREEDOM(nn.Module):
    """
    FREEDOM 简化版：去噪图 + 多模态融合。
    原版用 item-item 图去噪，这里用固定 LightGCN emb + 视觉特征。
    参考：MM 2023 "FREEDOM: Free Large Language Model Encounters
          Multimodal Recommendation"
    """

    def __init__(self, n_users: int, n_items: int,
                 clip_dim: int = 768, collab_dim: int = 64, d: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.vis_proj = nn.Linear(clip_dim, d, bias=False)
        self.txt_proj = nn.Linear(clip_dim, d, bias=False)
        self.col_proj = nn.Linear(collab_dim, d, bias=False)
        self.fusion   = nn.Linear(d * 3, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def _item_repr(self, vis, txt, col):
        e = torch.cat([self.vis_proj(vis),
                       self.txt_proj(txt),
                       self.col_proj(col)], dim=-1)
        return F.normalize(self.fusion(e), dim=-1)

    def get_score(self, user_ids, vis, txt, col):
        u = F.normalize(self.user_emb(user_ids), dim=-1)
        i = self._item_repr(vis, txt, col)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_vis, neg_vis, pos_txt, neg_txt,
                pos_col, neg_col):
        u     = F.normalize(self.user_emb(users), dim=-1)
        e_pos = self._item_repr(pos_vis, pos_txt, pos_col)
        e_neg = self._item_repr(neg_vis, neg_txt, neg_col)
        pos_s = (u * e_pos).sum(-1)
        neg_s = (u * e_neg).sum(-1)
        return -F.logsigmoid(pos_s - neg_s).mean()
