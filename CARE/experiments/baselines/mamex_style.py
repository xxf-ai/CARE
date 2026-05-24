"""
MAMEX-style baseline: per-item learnable modality gating via MoE.

Key differences from CARE:
  - Text adapter: trained Linear(768→64) projection (not zero-shot)
  - Gating: small MLP takes [user_cf, item_cf, user_text, item_text] → [w_cf, w_txt]
  - Load-balancing loss: prevents gating from collapsing to single modality

This serves as the "fusion paradigm" upper bound for ablation comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMEXStyle(nn.Module):
    """MAMEX-inspired per-item modality gating with trained text adapter."""

    def __init__(self, n_users, n_items, cfg):
        super().__init__()
        d = cfg.get("d", 64)
        text_dim = cfg.get("text_dim", 768)

        # ── CF Backbone ──────────────────────────────────────────
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # ── Trained text adapter (MAMEX uses trained projection) ─
        self.text_adapter = nn.Sequential(
            nn.Linear(text_dim, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, d, bias=False),
            nn.LayerNorm(d),
        )

        # ── Per-item gating network ──────────────────────────────
        # Input: [user_cf_score, item_cf_score, user_text_score, item_text_score]
        # Output: [w_cf, w_txt] via softmax
        self.gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        self.tau_score = cfg.get("tau_score", 1.0)
        self.d = d

    # ── Scoring ──────────────────────────────────────────────────

    def cf_score(self, users, items):
        u = F.normalize(self.user_emb(users), dim=-1)
        i = F.normalize(self.item_emb(items), dim=-1)
        return (u * i).sum(dim=-1) / self.tau_score

    def text_score(self, user_text_pref, item_text):
        u = F.normalize(self.text_adapter(user_text_pref), dim=-1)
        i = F.normalize(self.text_adapter(item_text), dim=-1)
        return (u * i).sum(dim=-1) / self.tau_score

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, users, pos_items, neg_items, cache=None):
        # CF scores
        score_cf_pos = self.cf_score(users, pos_items)
        score_cf_neg = self.cf_score(users, neg_items)

        # Text scores (trained adapter)
        user_txt = cache["user_text_pref"][users]
        pos_txt  = cache["clip_text"][pos_items]
        neg_txt  = cache["clip_text"][neg_items]

        score_txt_pos = self.text_score(user_txt, pos_txt)
        score_txt_neg = self.text_score(user_txt, neg_txt)

        return {
            "score_cf_pos": score_cf_pos, "score_cf_neg": score_cf_neg,
            "score_txt_pos": score_txt_pos, "score_txt_neg": score_txt_neg,
        }

    # ── Inference helpers ────────────────────────────────────────

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_emb.weight, dim=-1)

    @torch.no_grad()
    def compute_gated_scores(self, user_ids, cand_ids, cache):
        """Per-item gated fusion for evaluation."""
        B, K = cand_ids.shape

        # CF scores
        u_cf = F.normalize(self.user_emb(user_ids), dim=-1)       # [B, d]
        i_cf = F.normalize(self.item_emb(cand_ids), dim=-1)        # [B, K, d]
        cf = (u_cf.unsqueeze(1) * i_cf).sum(-1) / self.tau_score   # [B, K]

        # Text scores
        u_txt = self.text_adapter(cache["user_text_pref"][user_ids])  # [B, d]
        i_txt = self.text_adapter(cache["clip_text"][cand_ids])       # [B, K, d]
        txt = (u_txt.unsqueeze(1) * i_txt).sum(-1) / self.tau_score   # [B, K]

        # ── Per-item gating ────────────────────────────────────
        # Gate input: [cf_per_user, cf_per_item, txt_per_user, txt_per_item]
        # Use mean over items as "user-level" signal
        cf_user = cf.mean(dim=1, keepdim=True).expand(-1, K)       # [B, K]
        txt_user = txt.mean(dim=1, keepdim=True).expand(-1, K)

        gate_in = torch.stack([cf_user, cf, txt_user, txt], dim=-1)  # [B, K, 4]
        gate_out = self.gate(gate_in)                                 # [B, K, 2]
        w = F.softmax(gate_out, dim=-1)                               # [B, K, 2]
        w_cf, w_txt = w[..., 0], w[..., 1]

        return w_cf * cf + w_txt * txt

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Training loss for MAMEX ────────────────────────────────────

def mamex_loss(outputs, lambda_balance=0.01):
    """BPR on both CF and text + load-balancing regularization."""
    # BPR on CF
    diff_cf = outputs["score_cf_pos"] - outputs["score_cf_neg"]
    l_cf = -F.logsigmoid(diff_cf).mean()

    # BPR on text
    diff_txt = outputs["score_txt_pos"] - outputs["score_txt_neg"]
    l_txt = -F.logsigmoid(diff_txt).mean()

    # Load balancing: encourage CF and text to have similar average scores
    # (prevents gating network from ignoring one modality entirely)
    cf_mean = outputs["score_cf_pos"].mean()
    txt_mean = outputs["score_txt_pos"].mean()
    l_balance = (cf_mean - txt_mean) ** 2

    loss = l_cf + l_txt + lambda_balance * l_balance

    info = {
        "loss_total": loss.item(),
        "loss_bpr_cf": l_cf.item(),
        "loss_bpr_txt": l_txt.item(),
        "loss_balance": l_balance.item(),
    }
    return loss, info
