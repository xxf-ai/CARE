"""
evaluation/evaluator.py — VARA Evaluator (optimized for 40GB GPU)

Uses CF backbone scoring (cos/τ) for efficient batch evaluation.
All-item repr matrix precomputed once and reused across batches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Evaluator:

    def __init__(self, model, cache, k_list, device,
                 eval_batch=4096, neg_batch=64):
        self.model      = model
        self.cache      = cache
        self.k_list     = k_list
        self.device     = device
        self.eval_batch = eval_batch
        self.neg_batch  = neg_batch

    @torch.no_grad()
    def evaluate(self, loader, desc=""):
        self.model.eval()

        base_model = getattr(self.model, "_orig_mod", self.model)
        tau_score  = getattr(base_model, "tau_score", 1.0)

        # ── 预计算全量物品矩阵（一次，所有 batch 复用）─────────
        all_item_repr = base_model.get_all_item_cf_repr()  # [N_items, d]

        all_hr   = {k: [] for k in self.k_list}
        all_ndcg = {k: [] for k in self.k_list}

        for batch in loader:
            users, pos_items, neg_items = batch
            users     = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            B = users.size(0)
            if neg_items.dim() == 1:
                neg_items = neg_items.unsqueeze(1)

            # ── User repr + pos repr ────────────────────────────
            u_repr = base_model.get_user_cf_repr(users)       # [B, d]
            i_pos  = all_item_repr[pos_items]                  # [B, d]
            pos_scores = (u_repr * i_pos).sum(-1, keepdim=True) / tau_score

            # ── 负样本 scores：单次 bmm（49 items in one shot）─
            neg_ids = neg_items                            # [B, 99]
            i_neg   = all_item_repr[neg_ids]               # [B, 99, d]
            neg_scores = torch.bmm(
                u_repr.unsqueeze(1), i_neg.transpose(1, 2)
            ).squeeze(1) / tau_score                        # [B, 99]

            # ── Rank ────────────────────────────────────────────
            all_scores = torch.cat([pos_scores, neg_scores], dim=1)
            ranks = (all_scores > pos_scores).sum(dim=1) + 1
            ranks = ranks.cpu().numpy()

            for k in self.k_list:
                all_hr[k].extend((ranks <= k).astype(float).tolist())
                all_ndcg[k].extend(
                    np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).tolist())

        metrics = {}
        for k in self.k_list:
            metrics[f"hr@{k}"]   = float(np.mean(all_hr[k]))
            metrics[f"ndcg@{k}"] = float(np.mean(all_ndcg[k]))

        return metrics
