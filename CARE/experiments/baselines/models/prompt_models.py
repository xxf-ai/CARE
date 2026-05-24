"""
experiments/baselines/models/prompt_models.py — Prompt-based 基线模型

  PromptMM : Multi-Modal Knowledge Distillation for Recommendation with
             Prompt-Tuning (WWW 2024) 简化版
             learnable soft prompts as key/value bank
             item CLIP text queries prompts → enhanced repr
             i = i_cf + i_enhanced, score = dot(u, i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptMM(nn.Module):
    """
    PromptMM 简化版: learnable prompt bank 增强 item 文本表示.

    核心:
      - 每个 item 的 CLIP text vector 作为 query, 查询 learnable prompt bank
      - 返回 prompt 加权组合作为增强文本表示
      - item = i_cf + i_enhanced_text, score = dot(u, i)

    get_all_item_cf_repr() 返回 item_emb
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64,
                 n_prompts=8, n_heads=4):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.text_proj = nn.Linear(text_dim, d, bias=False)

        # Learnable prompt bank
        self.prompts = nn.Parameter(torch.randn(n_prompts, d) * 0.02)

        # Cross-attention: item text (Q) attends to prompt bank (K, V)
        self.cross_attn = nn.MultiheadAttention(
            d, n_heads, batch_first=True, dropout=0.1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.text_proj.weight)

    def _enhance_item_text(self, item_text):
        """
        item_text: [B, text_dim]
        returns: [B, d] — prompt-enhanced text representation
        Item text queries the prompt bank, returns weighted prompt combination.
        """
        i_txt = self.text_proj(item_text)  # [B, d]

        B = item_text.size(0)
        q = i_txt.unsqueeze(1)                        # [B, 1, d] — query
        kv = self.prompts.unsqueeze(0).expand(B, -1, -1)  # [B, P, d] — key/value
        attn_out, _ = self.cross_attn(q, kv, kv)
        return attn_out.squeeze(1)  # [B, d]

    def get_score(self, users, items, item_text):
        u = F.normalize(self.user_emb(users), dim=-1)
        i_cf = self.item_emb(items)
        i_txt_enhanced = self._enhance_item_text(item_text)
        i = F.normalize(i_cf + i_txt_enhanced, dim=-1)
        return (u * i).sum(dim=-1)

    def forward(self, users, pos_items, neg_items, pos_text, neg_text):
        pos_score = self.get_score(users, pos_items, pos_text)
        neg_score = self.get_score(users, neg_items, neg_text)
        l_bpr = -F.logsigmoid(pos_score - neg_score).mean()

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
