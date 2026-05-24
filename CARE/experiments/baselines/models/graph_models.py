"""
experiments/baselines/models/graph_models.py — 基于图的基线模型

  LGMRec : Local and Global Graph Learning for Multimodal Recommendation (AAAI 2024)
          解耦协同与文本嵌入,两路独立 LightGCN 传播,最终融合
  PEARL  : dual-layer graph learning for multimodal recommendation (2025)
          双图学习: 交互图 GCN + item-item 文本亲和图 GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bipartite_norm(edge_user, edge_item, n_users, n_items):
    """对称归一化: norm[u,i] = deg(u)^{-1/2} * deg(i)^{-1/2}"""
    user_deg = torch.zeros(n_users, device=edge_user.device)
    item_deg = torch.zeros(n_items, device=edge_user.device)
    user_deg.scatter_add_(0, edge_user, torch.ones_like(edge_user, dtype=torch.float))
    item_deg.scatter_add_(0, edge_item, torch.ones_like(edge_item, dtype=torch.float))
    norm = user_deg[edge_user].pow(-0.5) * item_deg[edge_item].pow(-0.5)
    return norm


# ══════════════════════════════════════════════════════════════════════════════
# LGMRec — AAAI 2024
# ══════════════════════════════════════════════════════════════════════════════
class LGMRec(nn.Module):
    """
    LGMRec 简化版: 解耦 CF 与 Text 嵌入, 两路独立 bipartite GCN 传播, 最终融合。
    get_all_item_cf_repr() 返回 item_cf_emb (协同部分可提取)。
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64, n_layers=1):
        super().__init__()
        self.user_cf_emb = nn.Embedding(n_users, d)
        self.item_cf_emb = nn.Embedding(n_items, d)
        self.user_txt_emb = nn.Embedding(n_users, d)
        self.item_txt_proj = nn.Linear(text_dim, d, bias=False)
        self.n_layers = n_layers
        nn.init.normal_(self.user_cf_emb.weight, std=0.01)
        nn.init.normal_(self.item_cf_emb.weight, std=0.01)
        nn.init.normal_(self.user_txt_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.item_txt_proj.weight)
        self.graph_built = False

    def set_graph(self, edge_index, n_users, n_items):
        """edge_index: [2, E], row=user[0..n_u), col=item[0..n_i)"""
        self.edge_u = edge_index[0]
        self.edge_i = edge_index[1]
        self.n_users = n_users
        self.n_items = n_items
        self.norm = _bipartite_norm(self.edge_u, self.edge_i, n_users, n_items)
        self.graph_built = True

    def _propagate(self, user_emb, item_emb):
        """单层 bipartite LightGCN 传播。"""
        norm = self.norm.to(dtype=user_emb.dtype).unsqueeze(-1)

        u_msg = torch.zeros_like(user_emb)
        u_msg.index_add_(0, self.edge_u, item_emb[self.edge_i] * norm)

        i_msg = torch.zeros_like(item_emb)
        i_msg.index_add_(0, self.edge_i, user_emb[self.edge_u] * norm)

        return u_msg, i_msg

    def _get_representations(self, clip_text):
        with torch.cuda.amp.autocast(enabled=False):
            return self._get_representations_impl(clip_text)

    def _get_representations_impl(self, clip_text):
        clip_text = clip_text.float()
        u_cf = self.user_cf_emb.weight.float()
        i_cf = self.item_cf_emb.weight.float()
        u_txt = self.user_txt_emb.weight.float()
        i_txt = self.item_txt_proj(clip_text)

        if self.graph_built:
            u_cf_stack, i_cf_stack = [u_cf], [i_cf]
            u_txt_stack, i_txt_stack = [u_txt], [i_txt]
            for _ in range(self.n_layers):
                u_cf_msg, i_cf_msg = self._propagate(u_cf, i_cf)
                u_txt_msg, i_txt_msg = self._propagate(u_txt, i_txt)
                u_cf = u_cf + u_cf_msg; i_cf = i_cf + i_cf_msg
                u_txt = u_txt + u_txt_msg; i_txt = i_txt + i_txt_msg
                u_cf_stack.append(u_cf); i_cf_stack.append(i_cf)
                u_txt_stack.append(u_txt); i_txt_stack.append(i_txt)
            u_cf = torch.stack(u_cf_stack).mean(0)
            i_cf = torch.stack(i_cf_stack).mean(0)
            u_txt = torch.stack(u_txt_stack).mean(0)
            i_txt = torch.stack(i_txt_stack).mean(0)

        u = F.normalize(u_cf + u_txt, dim=-1)
        i = F.normalize(i_cf + i_txt, dim=-1)
        return u, i, u_cf, i_cf

    def get_score(self, users, items, clip_text):
        u, i, _, _ = self._get_representations(clip_text)
        return (u[users] * i[items]).sum(dim=-1)

    def forward(self, users, pos_items, neg_items, pos_text, neg_text):
        return self.forward_full(users, pos_items, neg_items,
                                 pos_text.new_zeros(self.n_items, pos_text.shape[1]))

    def forward_full(self, users, pos_items, neg_items, clip_text_all):
        u, i, _, _ = self._get_representations(clip_text_all)
        pos = (u[users] * i[pos_items]).sum(dim=-1)
        neg = (u[users] * i[neg_items]).sum(dim=-1)
        l_bpr = -F.logsigmoid(pos - neg).mean()
        reg = (self.user_cf_emb(users).norm(2).pow(2) +
               self.item_cf_emb(pos_items).norm(2).pow(2) +
               self.item_cf_emb(neg_items).norm(2).pow(2)) / users.size(0)
        return l_bpr + 1e-5 * reg

    @torch.no_grad()
    def get_user_cf_repr(self, users):
        return F.normalize(self.user_cf_emb(users), dim=-1)

    @torch.no_grad()
    def get_all_item_cf_repr(self):
        return F.normalize(self.item_cf_emb.weight, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# PEARL — 2025
# ══════════════════════════════════════════════════════════════════════════════
class PEARL(nn.Module):
    """
    PEARL 简化版: 双图学习
      - Interaction Graph: user-item bipartite GCN (cf_repr)
      - Affinity Graph: item-item 文本余弦 top-K → GCN (affinity_repr)
    融合: α * i_int + (1-α) * i_aff
    """

    def __init__(self, n_users, n_items, text_dim=768, d=64,
                 n_layers=1, alpha_fuse=0.5, top_k_aff=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.n_layers = n_layers
        self.alpha_fuse = alpha_fuse
        self.top_k_aff = top_k_aff
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.graph_built = False

    def set_graphs(self, edge_index, clip_text, n_users, n_items):
        """构建 user-item 交互图 + item-item 文本亲和图。"""
        self.edge_u = edge_index[0]
        self.edge_i = edge_index[1]
        self.n_users = n_users
        self.n_items = n_items
        # 交互图归一化
        self.int_norm = _bipartite_norm(self.edge_u, self.edge_i, n_users, n_items)

        # item-item 文本亲和图: 分块计算 top-K cosine similarity (避免 OOM)
        with torch.no_grad():
            txt_norm = F.normalize(clip_text.float(), dim=-1)
            K = min(self.top_k_aff, n_items - 1)
            chunk_size = 2048
            all_topk = []
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                chunk_sim = torch.mm(txt_norm[start:end], txt_norm.T)  # [chunk, N]
                # 屏蔽自相似度
                chunk_sim[torch.arange(end - start, device=clip_text.device),
                          torch.arange(start, end, device=clip_text.device)] = -1e9
                _, topk = chunk_sim.topk(K, dim=-1)
                all_topk.append(topk)
            topk_idx = torch.cat(all_topk, dim=0)  # [n_items, K]
            # src→dst 边: each item aggregates from its top-K neighbors
            self.ii_src = topk_idx.reshape(-1)  # [n_items * K]
            self.ii_dst = torch.arange(n_items, device=clip_text.device).repeat_interleave(
                self.top_k_aff)

            # 度归一化
            dst_deg = torch.zeros(n_items, device=clip_text.device)
            dst_deg.scatter_add_(0, self.ii_dst, torch.ones_like(self.ii_dst, dtype=torch.float))
            src_deg = torch.zeros(n_items, device=clip_text.device)
            src_deg.scatter_add_(0, self.ii_src, torch.ones_like(self.ii_src, dtype=torch.float))
            self.ii_norm = dst_deg[self.ii_dst].pow(-0.5) * src_deg[self.ii_src].pow(-0.5)

        self.graph_built = True

    def _int_propagate(self, user_emb, item_emb):
        """交互图: user↔item 双向传播。"""
        norm = self.int_norm.to(dtype=user_emb.dtype).unsqueeze(-1)
        u_msg = torch.zeros_like(user_emb)
        u_msg.index_add_(0, self.edge_u, item_emb[self.edge_i] * norm)
        i_msg = torch.zeros_like(item_emb)
        i_msg.index_add_(0, self.edge_i, user_emb[self.edge_u] * norm)
        return u_msg, i_msg

    def _aff_propagate(self, item_emb):
        """文本亲和图: item←item 单向传播。"""
        norm = self.ii_norm.to(dtype=item_emb.dtype).unsqueeze(-1)
        i_msg = torch.zeros_like(item_emb)
        i_msg.index_add_(0, self.ii_dst, item_emb[self.ii_src] * norm)
        return i_msg

    def _get_representations(self):
        with torch.cuda.amp.autocast(enabled=False):
            return self._get_representations_impl()

    def _get_representations_impl(self):
        u = self.user_emb.weight.float()
        i = self.item_emb.weight.float()

        if not self.graph_built:
            return F.normalize(u, dim=-1), F.normalize(i, dim=-1)

        # Interaction GCN
        u_int, i_int = u, i
        for _ in range(self.n_layers):
            u_msg, i_msg = self._int_propagate(u_int, i_int)
            u_int = u_int + u_msg; i_int = i_int + i_msg

        # Affinity GCN
        i_aff = i
        for _ in range(self.n_layers):
            i_aff = i_aff + self._aff_propagate(i_aff)

        u_final = F.normalize(u_int, dim=-1)
        i_final = F.normalize(self.alpha_fuse * i_int + (1 - self.alpha_fuse) * i_aff, dim=-1)
        return u_final, i_final

    def get_score(self, users, items):
        u, i = self._get_representations()
        return (u[users] * i[items]).sum(dim=-1)

    def forward(self, users, pos_items, neg_items, pos_text=None, neg_text=None):
        u, i = self._get_representations()
        pos = (u[users] * i[pos_items]).sum(dim=-1)
        neg = (u[users] * i[neg_items]).sum(dim=-1)
        l_bpr = -F.logsigmoid(pos - neg).mean()
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
