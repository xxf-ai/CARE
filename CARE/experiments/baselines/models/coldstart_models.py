"""
experiments/baselines/models/coldstart_models.py — 冷启动专用基线模型

  CCFCRec  : Contrastive Collaborative Filtering for Cold-Start Item Recommendation (WWW 2023)
            两路 CF: Content CF (text→embedding) + Co-occurrence CF (BPR-MF)
            对比损失将 content repr 拉向 co-occurrence repr (仅 cold items)
  MARec    : Metadata Alignment for cold-start Recommendation (RecSys 2024)
            在 warm items 上训练 CLIP text → CF embedding projector (MSE)
            冷启动物品用 projector 输出替代 CF embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# CCFCRec — WWW 2023
# ══════════════════════════════════════════════════════════════════════════════
class CCFCRec(nn.Module):
    """
    CCFCRec 简化版。
    两路 CF:
      - Co-occurrence CF Module: 标准 BPR-MF (user_emb + item_emb)
      - Content CF Module:       从 CLIP text 生成 content-based CF embedding
    对比损失: 将冷启动物品的 content repr 拉向其 co-occurrence repr

    get_all_item_cf_repr() 返回 item_emb (co-occurrence 部分可提取)
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64,
                 lambda_cl=0.1, temp=0.2):
        super().__init__()
        # Co-occurrence CF
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        # Content CF
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
        """Co-occurrence CF score (BPR-MF style)."""
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.item_emb(items), dim=-1)
        return (u * i).sum(dim=-1)

    def get_content_score(self, users, items, item_text):
        """Content CF score."""
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.content_proj(item_text), dim=-1)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_items, neg_items,
                is_cold_pos, pos_text, neg_text):
        # BPR on co-occurrence CF scores
        pos = self.get_score(users, pos_items)
        neg = self.get_score(users, neg_items)
        l_bpr = -F.logsigmoid(pos - neg).mean()

        # Contrastive: 将冷物品的 content repr 拉向 co-occurrence repr
        l_cl = torch.tensor(0.0, device=users.device)
        n_cl = 0
        if is_cold_pos.any():
            cold_mask = is_cold_pos
            cf_i = F.normalize(self.item_emb(pos_items[cold_mask]), dim=-1)
            ct_i = F.normalize(self.content_proj(pos_text[cold_mask]), dim=-1)
            # 每个冷物品的 CF embedding 应与 content embedding 相似
            sim = (cf_i * ct_i).sum(dim=-1) / self.temp
            l_cl = l_cl - sim.mean()
            n_cl += cold_mask.sum().item()
        if n_cl > 0:
            l_cl = l_cl / n_cl

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


# ══════════════════════════════════════════════════════════════════════════════
# MARec — RecSys 2024
# ══════════════════════════════════════════════════════════════════════════════
class MARec(nn.Module):
    """
    MARec 简化版: Metadata Alignment for cold-start Recommendation.

    核心: 在 warm items 上训练 CLIP text → CF embedding 的 projector (MSE loss),
    冷启动物品用 projected embedding 替代。

    训练: BPR loss + λ_align * MSE(projector(text_warm), item_emb(warm))
    推理: cold items → projector(text), warm items → item_emb

    get_all_item_cf_repr() 返回 item_emb
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64,
                 lambda_align=0.1):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.projector = nn.Sequential(
            nn.Linear(text_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d, bias=False),
        )
        self.lambda_align = lambda_align
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def get_score(self, users, items):
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.item_emb(items), dim=-1)
        return (u * i).sum(dim=-1)

    def get_aligned_score(self, users, items, item_text):
        """使用 projected text embedding 代替 item_emb (用于冷启动物品)."""
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.projector(item_text), dim=-1)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_items, neg_items, is_warm_pos, pos_text):
        # BPR on CF scores
        pos = self.get_score(users, pos_items)
        neg = self.get_score(users, neg_items)
        l_bpr = -F.logsigmoid(pos - neg).mean()

        # Alignment loss: MSE(projector(text), item_emb) on warm items only
        l_align = torch.tensor(0.0, device=users.device)
        if is_warm_pos.any():
            warm_mask = is_warm_pos
            projected = self.projector(pos_text[warm_mask])
            target = self.item_emb(pos_items[warm_mask]).detach()
            l_align = F.mse_loss(projected, target)

        reg = (self.user_emb(users).norm(2).pow(2) +
               self.item_emb(pos_items).norm(2).pow(2) +
               self.item_emb(neg_items).norm(2).pow(2)) / users.size(0)

        return l_bpr + self.lambda_align * l_align + 1e-5 * reg

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)
