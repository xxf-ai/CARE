"""losses/bpr.py — In-batch BPR 损失"""
import torch
import torch.nn.functional as F


def bpr_loss(
    pos_score: torch.Tensor,   # [B]
    neg_score: torch.Tensor,   # [B]（随机负样本，保留兼容）
    margin: float = 0.0,
) -> torch.Tensor:
    diff = pos_score - neg_score - margin
    return -F.logsigmoid(diff).mean()


def inbatch_bpr_loss(
    u_repr:   torch.Tensor,   # [B, d] 用户表示
    e_i_pos:  torch.Tensor,   # [B, d] 正样本物品表示
    e_i_neg:  torch.Tensor,   # [B, d] 随机负样本物品表示
    margin:   float = 0.5,
    inbatch_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    混合 BPR：随机负样本 BPR + batch 内负样本 BPR

    Batch 内负样本：把同一 batch 里其他用户的正样本当作当前用户的负样本
    B=1024 时相当于每个用户有 1023 个负样本，远难于随机负样本
    """
    B = u_repr.size(0)

    # ── 随机负样本 BPR ──────────────────────────────────────────
    pos_score = (u_repr * e_i_pos).sum(-1)   # [B]
    neg_score = (u_repr * e_i_neg).sum(-1)   # [B]
    l_random  = bpr_loss(pos_score, neg_score, margin=margin)

    # ── Batch 内负样本 BPR ──────────────────────────────────────
    # 打分矩阵 [B, B]：第 i 行第 j 列 = user_i 对 item_j 的分数
    score_mat = u_repr @ e_i_pos.t()           # [B, B]
    # 对角线是正样本分数，其余是 batch 内负样本分数
    pos_diag  = score_mat.diag()               # [B]
    # 每个用户取最难的 batch 内负样本（最大非对角值）
    mask = torch.eye(B, device=u_repr.device).bool()
    score_mat_masked = score_mat.masked_fill(mask, -1e9)
    hardest_neg = score_mat_masked.max(dim=1).values  # [B]
    l_inbatch = bpr_loss(pos_diag, hardest_neg, margin=margin)

    loss = (1 - inbatch_weight) * l_random + inbatch_weight * l_inbatch
    return loss, {"bpr_random": l_random.item(), "bpr_inbatch": l_inbatch.item()}
