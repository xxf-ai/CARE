"""
losses/care_loss.py — CARE training loss

BPR loss on CF scores. Raw CLIP text is applied zero-shot at inference only.
"""

import torch
import torch.nn.functional as F


def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss"""
    diff = pos_score - neg_score - margin
    return -F.logsigmoid(diff).mean()


def care_total_loss(
    outputs: dict,
    margin: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """BPR loss on CF backbone scores."""
    l_cf = bpr_loss(
        outputs["score_cf_pos"],
        outputs["score_cf_neg"],
        margin=margin,
    )
    info = {"loss_total": l_cf.item(), "loss_bpr": l_cf.item()}
    return l_cf, info
