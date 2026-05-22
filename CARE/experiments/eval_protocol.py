"""
experiments/eval_protocol.py — 高速评估协议

不用 DataLoader，直接读 CSV + 全量矩阵运算：
  1. pandas 读 test_indexed.csv，直接拿到 user/pos/neg 的整数 ID
  2. 一次性取出所有用户表示 [N_test, d] 和候选物品表示 [N_test, 100, d]
  3. bmm 一次完成全部打分，CPU 几乎不参与
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

K_LIST = [5, 10, 20]


def _load_test_data(data_dir: Path, split: str, seed: int, n_neg: int):
    """
    直接读预处理好的 CSV，返回 numpy 数组。
    neg_items 列存储格式：用空格分隔的整数字符串，或预计算好的固定负样本文件。
    """
    # 优先找预计算的负样本文件
    neg_file = data_dir / f"{split}_neg_seed{seed}.npy"
    idx_file = data_dir / f"{split}_indexed.csv"

    df = pd.read_csv(idx_file)
    user_ids = df["user_id"].values.astype(np.int64)
    pos_ids  = df["item_id"].values.astype(np.int64)

    if neg_file.exists():
        neg_ids = np.load(neg_file)   # [N, n_neg]
    else:
        # 随机采样固定负样本（用 seed 保证可复现）
        rng      = np.random.default_rng(seed)
        n_items  = int(pd.read_json(data_dir / "stats.json")["n_items"])
        neg_ids  = rng.integers(0, n_items, size=(len(user_ids), n_neg))
        # 确保负样本不等于正样本
        for i in range(len(user_ids)):
            mask = neg_ids[i] == pos_ids[i]
            neg_ids[i][mask] = (neg_ids[i][mask] + 1) % n_items
        np.save(neg_file, neg_ids)
        print(f"    已预计算并缓存负样本: {neg_file}")

    return user_ids, pos_ids, neg_ids   # [N], [N], [N, n_neg]


def evaluate_model(
    score_fn=None,
    data_dir: Path = None,
    split: str = "test",
    seed: int = 42,
    n_neg: int = 99,
    batch_size: int = 4096,
    device: torch.device = None,
    k_list: list = K_LIST,
    get_user_repr_fn=None,
    get_all_item_repr_fn=None,
    n_items: int = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_ids, pos_ids, neg_ids = _load_test_data(data_dir, split, seed, n_neg)
    N = len(user_ids)

    # ── 走矩阵表示路径（最快）────────────────────────────────────────
    if get_user_repr_fn is not None and get_all_item_repr_fn is not None:
        return _eval_repr(get_user_repr_fn, get_all_item_repr_fn,
                          user_ids, pos_ids, neg_ids,
                          batch_size, device, k_list)

    # ── fallback：score_fn 路径 ───────────────────────────────────────
    return _eval_score_fn(score_fn, user_ids, pos_ids, neg_ids,
                          batch_size, device, k_list)


def _eval_repr(get_user_repr_fn, get_all_item_repr_fn,
               user_ids, pos_ids, neg_ids,
               batch_size, device, k_list):
    """
    最快路径：
      - 预计算全量物品矩阵 [N_items, d]，放 GPU
      - 按 batch 取用户表示，索引候选物品，bmm 打分
    """
    N   = len(user_ids)
    n_neg = neg_ids.shape[1]

    with torch.no_grad():
        item_matrix = get_all_item_repr_fn().to(device)   # [N_items, d]

    all_ranks = np.empty(N, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            u_ids = torch.from_numpy(user_ids[start:end]).to(device)
            p_ids = torch.from_numpy(pos_ids[start:end]).to(device)
            n_ids = torch.from_numpy(neg_ids[start:end]).to(device)

            B = u_ids.size(0)
            u_repr = get_user_repr_fn(u_ids)              # [B, d]

            # 候选物品：pos + neg → [B, 1+n_neg, d]
            cand = torch.cat([p_ids.unsqueeze(1), n_ids], dim=1)  # [B, 1+n_neg]
            cand_repr = item_matrix[cand.reshape(-1)].reshape(B, 1 + n_neg, -1)

            # [B, 1, d] × [B, d, 1+n_neg] → [B, 1+n_neg]
            scores = torch.bmm(u_repr.unsqueeze(1),
                               cand_repr.transpose(1, 2)).squeeze(1)

            pos_score = scores[:, 0:1]
            ranks = (scores > pos_score).sum(dim=1) + 1   # [B]
            all_ranks[start:end] = ranks.cpu().numpy()

    return _ranks_to_metrics(all_ranks, k_list)


def _eval_score_fn(score_fn, user_ids, pos_ids, neg_ids,
                   batch_size, device, k_list):
    """fallback：用 score_fn 逐批打分"""
    N     = len(user_ids)
    n_neg = neg_ids.shape[1]
    all_ranks = np.empty(N, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            u_ids = torch.from_numpy(user_ids[start:end]).to(device)
            p_ids = torch.from_numpy(pos_ids[start:end]).to(device)
            n_ids = torch.from_numpy(neg_ids[start:end]).to(device)

            B = u_ids.size(0)
            cand = torch.cat([p_ids.unsqueeze(1), n_ids], dim=1).reshape(-1)
            exp_u = u_ids.unsqueeze(1).expand(-1, 1 + n_neg).reshape(-1)

            scores = score_fn(exp_u, cand).reshape(B, 1 + n_neg)
            pos_score = scores[:, 0:1]
            ranks = (scores > pos_score).sum(dim=1) + 1
            all_ranks[start:end] = ranks.cpu().numpy()

    return _ranks_to_metrics(all_ranks, k_list)


def _ranks_to_metrics(ranks: np.ndarray, k_list: list) -> dict:
    metrics = {}
    for k in k_list:
        metrics[f"hr@{k}"]   = float((ranks <= k).mean())
        metrics[f"ndcg@{k}"] = float(
            np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return metrics


def run_multi_seed(
    model_factory,
    data_dir,
    seeds=(42, 123, 456),
    split="test",
    n_neg=99,
    batch_size=4096,
    device=None,
    k_list=K_LIST,
):
    import numpy as np
    all_results = []
    for seed in seeds:
        print(f"  seed={seed} ...", end=" ", flush=True)
        score_fn, _ = model_factory(seed)
        m = evaluate_model(score_fn, data_dir, split=split, seed=seed,
                           n_neg=n_neg, batch_size=batch_size,
                           device=device, k_list=k_list)
        all_results.append(m)
        print({k: f"{v:.4f}" for k, v in m.items()})

    summary = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        summary[key]          = float(np.mean(vals))
        summary[key + "_std"] = float(np.std(vals))
    return summary
