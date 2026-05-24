"""
models/care.py — CARE (Coldness-Adaptive Reranking Engine)

Simplified: CF backbone (BPR-MF) trains on collaborative signals.
At inference, raw CLIP text cosine similarity is fused for cold-start items.
No trained text adapter — CLIP text works zero-shot as visual-semantic proxy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARE(nn.Module):

    def __init__(self, n_users: int, n_items: int, cfg: dict):
        super().__init__()

        d = cfg.get("d", 64)

        # ── CF Backbone (BPR-MF) ──────────────────────────────────
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.tau_score = cfg.get("tau_score", 1.0)
        self.d = d

    # ── Scoring ──────────────────────────────────────────────────

    def cf_score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """CF score: normalized dot product / τ"""
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.item_emb(items), dim=-1)
        return (u * i).sum(dim=-1) / self.tau_score

    # ── Forward (training) ──────────────────────────────────────

    def forward(self, users, pos_items, neg_items, cache=None) -> dict:
        score_cf_pos = self.cf_score(users, pos_items)
        score_cf_neg = self.cf_score(users, neg_items)
        return {"score_cf_pos": score_cf_pos, "score_cf_neg": score_cf_neg}

    # ── Inference helpers (used by Evaluator) ───────────────────

    @torch.no_grad()
    def get_user_cf_repr(self, users: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self) -> torch.Tensor:
        return F.normalize(self.item_emb.weight, dim=-1)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
