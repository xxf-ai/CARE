"""
data_utils/dataset.py
---------------------
MARADataset — 训练/验证/测试 Dataset + CacheManager (VARA edition)

CacheManager：
  加载 VARA 所需的 npy 特征文件：
    clip_text.npy (必需) — CLIP 文本语义特征 [N_items, 768]
    item_emb.npy  (可选) — LightGCN 物品嵌入
    user_emb.npy  (可选) — LightGCN 用户嵌入
  user_text_pref 由 train.py 在加载后计算（clip_text 均值）。

MARADataset：
  训练集：在线混合负采样（50% 热门 + 50% 随机）
  验证/测试集：固定 99 个负样本（确保可复现的评估）
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────── Cache Manager ────────────────────────

class CacheManager:
    """
    统一管理所有 npy 特征缓存，预加载并可选择 pin_memory
    """

    # VARA required files: clip_text is essential; item_emb/user_emb optional
    REQUIRED_FILES = [
        "clip_text",     # [N_items, 768] — CLIP text semantics
    ]
    OPTIONAL_FILES = [
        "clip_cls",      # [N_items, 768] — CLIP visual (if images available)
        "item_emb",      # [N_items, d]   — LightGCN embeddings
        "user_emb",      # [N_users, d]   — LightGCN user embeddings
    ]

    def __init__(
        self,
        cache_dir:  Path,
        data_dir:   Path,
        device:     torch.device,
        pin_memory: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.data_dir  = Path(data_dir)
        self.device    = device
        self.tensors   = {}

        print("  预加载特征缓存 (VARA)...")

        # Required: clip_text
        for key in self.REQUIRED_FILES:
            path = self.cache_dir / f"{key}.npy"
            if not path.exists():
                raise FileNotFoundError(f"缓存文件缺失: {path}")
            arr = np.load(path)
            self.tensors[key] = torch.from_numpy(arr.astype(np.float32))
            print(f"    {key}: {tuple(self.tensors[key].shape)} [必需]")

        # Optional: clip_cls, item_emb, user_emb
        for key in self.OPTIONAL_FILES:
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                arr = np.load(path)
                self.tensors[key] = torch.from_numpy(arr.astype(np.float32))
                print(f"    {key}: {tuple(self.tensors[key].shape)} [可选]")
            else:
                print(f"    {key}: 不存在，跳过 [可选]")

    def __getitem__(self, key: str) -> torch.Tensor:
        """将张量移到目标设备（DataLoader worker 中调用）"""
        return self.tensors[key].to(self.device, non_blocking=True)

    def to_device(self, device: torch.device):
        """一次性将所有张量移到 GPU（小数据集时可用）"""
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}
        self.device  = device


# ─────────────────────────── Dataset ─────────────────────────────

class MARADataset(Dataset):
    """
    训练集 Dataset：在线负采样
    """

    def __init__(
        self,
        data_dir: Path,
        split:    str = "train",   # "train" | "val" | "test"
        seed:     int = 42,
        n_neg:    int = 99,         # val/test 负样本数
    ):
        self.data_dir = Path(data_dir)
        self.split    = split
        self.rng      = np.random.default_rng(seed)
        self.n_neg    = n_neg

        # 读取统计信息
        with open(data_dir / "stats.json") as f:
            stats = json.load(f)
        self.n_users = stats["n_users"]
        self.n_items = stats["n_items"]

        # 加载交互数据（整数 ID）
        df = pd.read_csv(data_dir / f"{split}_indexed.csv")
        self.users = df["user_id"].values.astype(np.int64)
        self.items = df["item_id"].values.astype(np.int64)

        # 训练集：构建用户-物品集合用于负采样过滤
        if split == "train":
            self._build_user_pos_sets(data_dir)
        else:
            self._precompute_neg_samples(seed)

    def _build_user_pos_sets(self, data_dir: Path):
        """构建每个用户的正样本集合（用于负采样时过滤）"""
        all_df = pd.concat([
            pd.read_csv(data_dir / "train_indexed.csv"),
            pd.read_csv(data_dir / "val_indexed.csv"),
            pd.read_csv(data_dir / "test_indexed.csv"),
        ])
        self.user_pos: dict[int, set] = {}
        item_cnt = {}
        for uid, iid in zip(all_df["user_id"].values, all_df["item_id"].values):
            self.user_pos.setdefault(int(uid), set()).add(int(iid))
            item_cnt[int(iid)] = item_cnt.get(int(iid), 0) + 1

        # 取交互数 top 20% 的物品作为热门负样本池
        sorted_items = sorted(item_cnt.keys(), key=lambda x: -item_cnt[x])
        top_k = max(100, len(sorted_items) // 5)
        self._popular_items = sorted_items[:top_k]

    def _precompute_neg_samples(self, seed: int):
        """为 val/test 预计算固定 99 个负样本（保证评估可复现）"""
        self.neg_items = np.zeros((len(self.users), self.n_neg), dtype=np.int64)
        rng = np.random.default_rng(seed)
        for i, (uid, pos_iid) in enumerate(zip(self.users, self.items)):
            negs = []
            while len(negs) < self.n_neg:
                cands = rng.integers(0, self.n_items, size=self.n_neg * 2)
                cands = cands[cands != pos_iid][:self.n_neg - len(negs)]
                negs.extend(cands.tolist())
            self.neg_items[i] = negs[:self.n_neg]

    def _sample_neg(self, uid: int, pos_iid: int) -> int:
        """
        混合负采样：50% 热门负样本 + 50% 随机负样本
        热门物品作为负样本更难区分，提供更有效的梯度信号
        """
        pos_set = self.user_pos.get(uid, {pos_iid})
        use_popular = (self.rng.random() < 0.5) and hasattr(self, '_popular_items')
        for _ in range(50):  # 最多尝试50次
            if use_popular:
                neg = int(self.rng.choice(self._popular_items))
            else:
                neg = int(self.rng.integers(0, self.n_items))
            if neg not in pos_set:
                return neg
        # fallback
        return int(self.rng.integers(0, self.n_items))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        uid     = int(self.users[idx])
        pos_iid = int(self.items[idx])

        if self.split == "train":
            neg_iid = self._sample_neg(uid, pos_iid)
            return (
                torch.tensor(uid,     dtype=torch.long),
                torch.tensor(pos_iid, dtype=torch.long),
                torch.tensor(neg_iid, dtype=torch.long),
            )
        else:
            # val/test：返回正样本 + 固定 99 负样本
            neg_iids = torch.from_numpy(self.neg_items[idx])  # [99]
            return (
                torch.tensor(uid,     dtype=torch.long),
                torch.tensor(pos_iid, dtype=torch.long),
                neg_iids,
            )
