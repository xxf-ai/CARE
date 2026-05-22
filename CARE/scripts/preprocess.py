"""
scripts/preprocess.py
---------------------
Step 1: 交互数据预处理

功能：
  - 加载原始 .jsonl / .jsonl.gz 交互文件（自动识别格式）
  - 同时保留 review_text（标题+正文），供后续聚合为 item_reviews.json
  - K-core 迭代过滤（默认 k=5）
  - Beauty 子集采样（目标 3000 用户）
  - 用户级时序划分（Leave-One-Out）
  - 构建 user2id / item2id 整数映射（供模型嵌入层使用）
  - 聚合训练集评论 → item_reviews.json（仅用训练集，防止测试集泄漏）
  - 输出统计摘要

使用方法：
  python scripts/preprocess.py --dataset baby
  python scripts/preprocess.py --dataset beauty_sub
  python scripts/preprocess.py --dataset all   # 依次处理两个数据集
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────── 路径常量 ──────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # MARA/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"

RAW_FILE_MAP = {
    "baby": {
        "interactions": RAW_DIR / "Baby_Products.jsonl",
        "meta":         RAW_DIR / "meta_Baby_Products.jsonl",
    },
    "beauty": {
        "interactions": RAW_DIR / "All_Beauty.jsonl",
        "meta":         RAW_DIR / "meta_All_Beauty.jsonl",
    },
    "office": {
        "interactions": RAW_DIR / "Office_Products.jsonl",
        "meta":         RAW_DIR / "meta_Office_Products.jsonl",
    },
    "sports": {
        "interactions": RAW_DIR / "Sports_and_Outdoors.jsonl",
        "meta":         RAW_DIR / "meta_Sports_and_Outdoors.jsonl",
    },
}

# 每个物品最多保留的评论条数（按评分高低取 top-k，控制文本长度）
MAX_REVIEWS_PER_ITEM = 5

# ─────────────────────────── I/O 工具 ──────────────────────────────

def open_jsonl(path: Path):
    """自动识别 .jsonl 或 .jsonl.gz，返回逐行迭代器"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_interactions(path: Path) -> pd.DataFrame:
    """
    加载交互记录（Amazon Reviews 2023 格式）
    字段：user_id, item_id, timestamp, rating, review_text
    review_text = review_title + " " + review_body
    """
    records = []
    with open_jsonl(path) as f:
        for line in tqdm(f, desc=f"加载交互 {path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = d.get("user_id", "")
            item_id = d.get("parent_asin") or d.get("asin", "")
            ts      = d.get("timestamp", 0)
            rating  = d.get("rating", 0.0)

            if not user_id or not item_id:
                continue

            # Amazon 2023 时间戳单位为毫秒，统一转为秒
            if ts > 1e12:
                ts = ts // 1000

            # 拼接评论标题 + 正文
            review_title = (d.get("title") or "").strip()
            review_body  = (d.get("text")  or "").strip()
            review_text  = f"{review_title} {review_body}".strip()

            records.append({
                "user_id":     user_id,
                "item_id":     item_id,
                "timestamp":   int(ts),
                "rating":      float(rating),
                "review_text": review_text,
            })

    df = pd.DataFrame(records)
    print(f"  原始记录数: {len(df):,}  |  用户: {df['user_id'].nunique():,}  |  物品: {df['item_id'].nunique():,}")
    n_with_review = (df["review_text"].str.len() > 0).sum()
    print(f"  含评论文本: {n_with_review:,}  ({n_with_review / len(df) * 100:.1f}%)")
    return df


# ─────────────────────────── 过滤与采样 ───────────────────────────

def kcore_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    迭代 k-core 过滤：直到所有用户和物品的交互数均 >= k
    """
    iteration = 0
    while True:
        prev_len = len(df)
        user_cnt = df.groupby("user_id").size()
        item_cnt = df.groupby("item_id").size()
        df = df[
            df["user_id"].isin(user_cnt[user_cnt >= k].index) &
            df["item_id"].isin(item_cnt[item_cnt >= k].index)
        ]
        iteration += 1
        if len(df) == prev_len:
            break

    print(f"  {k}-core 过滤完成（{iteration} 轮）"
          f"  →  记录: {len(df):,}  用户: {df['user_id'].nunique():,}  物品: {df['item_id'].nunique():,}")
    return df.reset_index(drop=True)


def sample_beauty_subset(
    df: pd.DataFrame,
    target_users: int = 3000,
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    采样 Beauty 子集，保持 k-core 性质
    """
    rng = np.random.RandomState(seed)

    user_cnt = df.groupby("user_id").size().sort_values(ascending=False)
    candidate_users = user_cnt.index[: target_users * 2].tolist()
    subset = df[df["user_id"].isin(candidate_users)].copy()
    subset = kcore_filter(subset, k=k)

    all_users   = subset["user_id"].unique()
    n_sample    = min(target_users, len(all_users))
    final_users = rng.choice(all_users, n_sample, replace=False)
    result      = subset[subset["user_id"].isin(final_users)].copy()
    result      = kcore_filter(result, k=k)

    print(f"  Beauty 子集采样完成：目标 {target_users} 用户 → 实际 {result['user_id'].nunique():,} 用户")
    return result.reset_index(drop=True)


# ─────────────────────────── 时序划分 ─────────────────────────────

def temporal_split(df: pd.DataFrame):
    """
    用户级 Leave-One-Out 时序划分：
      test:  每用户最后 1 条
      val:   每用户倒数第 2 条
      train: 其余全部（含 review_text，供评论聚合使用）
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_rows, val_rows, test_rows = [], [], []

    for uid, grp in tqdm(df.groupby("user_id"), desc="时序划分"):
        rows = grp.to_dict("records")
        if len(rows) < 3:
            train_rows.extend(rows)
            continue
        test_rows.append(rows[-1])
        val_rows.append(rows[-2])
        train_rows.extend(rows[:-2])

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    test_df  = pd.DataFrame(test_rows)

    print(f"  数据划分 → 训练: {len(train_df):,}  验证: {len(val_df):,}  测试: {len(test_df):,}")
    return train_df, val_df, test_df


# ─────────────────────────── 评论聚合 ─────────────────────────────

def build_item_reviews(train_df: pd.DataFrame, max_per_item: int = MAX_REVIEWS_PER_ITEM) -> dict:
    """
    从训练集聚合每个物品的评论文本（仅用训练集，严格防止测试集泄漏）

    策略：
      - 按 rating 降序排列，优先保留高评分评论（信息量更高）
      - 每个物品最多保留 max_per_item 条
      - 过滤空评论

    返回：{item_id: [review_str, ...]}
    """
    item_reviews: dict[str, list[str]] = defaultdict(list)

    # 按评分降序，优先取高分评论
    sorted_df = train_df.sort_values("rating", ascending=False)

    for _, row in tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="聚合评论"):
        iid  = row["item_id"]
        text = (row.get("review_text") or "").strip()
        if not text:
            continue
        if len(item_reviews[iid]) < max_per_item:
            item_reviews[iid].append(text)

    # 统计
    n_items_with_review = sum(1 for v in item_reviews.values() if v)
    total_reviews       = sum(len(v) for v in item_reviews.values())
    print(f"  评论聚合完成：{n_items_with_review:,} 个物品有评论，共 {total_reviews:,} 条")

    return dict(item_reviews)


# ─────────────────────────── ID 映射 ─────────────────────────────

def build_id_maps(df: pd.DataFrame, train_df: pd.DataFrame):
    user_freq = train_df.groupby("user_id").size().sort_values(ascending=False)
    item_freq = train_df.groupby("item_id").size().sort_values(ascending=False)

    all_users = list(user_freq.index) + [u for u in df["user_id"].unique() if u not in user_freq.index]
    all_items = list(item_freq.index) + [i for i in df["item_id"].unique() if i not in item_freq.index]

    user2id = {u: idx for idx, u in enumerate(all_users)}
    item2id = {i: idx for idx, i in enumerate(all_items)}
    return user2id, item2id


def apply_id_maps(df: pd.DataFrame, user2id: dict, item2id: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].map(user2id)
    df["item_id"] = df["item_id"].map(item2id)
    return df


# ─────────────────────────── 统计摘要 ─────────────────────────────

def print_stats(name, df, train_df, val_df, test_df, user2id, item2id):
    n_users = len(user2id)
    n_items = len(item2id)
    n_inter = len(df)
    density = n_inter / (n_users * n_items) * 100

    print(f"\n{'='*50}")
    print(f"  数据集统计：{name}")
    print(f"{'='*50}")
    print(f"  用户数         : {n_users:>10,}")
    print(f"  物品数         : {n_items:>10,}")
    print(f"  总交互数       : {n_inter:>10,}")
    print(f"  交互密度       : {density:>10.4f}%")
    print(f"  均值交互/用户  : {n_inter/n_users:>10.2f}")
    print(f"  训练/验证/测试 : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(f"{'='*50}")


# ─────────────────────────── 主流程 ───────────────────────────────

def process_dataset(
    dataset_name: str,
    raw_inter_path: Path,
    output_dir: Path,
    k: int = 5,
    beauty_target_users: int = 3000,
    seed: int = 42,
    is_beauty_sub: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*55}")
    print(f"  处理数据集: {dataset_name}")
    print(f"{'#'*55}")

    # 1. 加载原始交互（含 review_text）
    df = load_interactions(raw_inter_path)

    # 2. K-core 过滤
    df = kcore_filter(df, k=k)

    # 3. Beauty 子集采样
    if is_beauty_sub:
        df = sample_beauty_subset(df, target_users=beauty_target_users, k=k, seed=seed)

    # 4. 时序划分
    train_df, val_df, test_df = temporal_split(df)

    # 5. 构建 ID 映射
    user2id, item2id = build_id_maps(df, train_df)

    # 6. 统计
    print_stats(dataset_name, df, train_df, val_df, test_df, user2id, item2id)

    # 7. 保存 CSV（review_text 不存入 CSV，节省磁盘；已单独聚合到 item_reviews.json）
    cols_no_review = ["user_id", "item_id", "timestamp", "rating"]
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df[cols_no_review].to_csv(output_dir / f"{split_name}.csv", index=False)

    # 8. 整数 ID 版 CSV
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        int_df = apply_id_maps(split_df[cols_no_review], user2id, item2id)
        int_df.to_csv(output_dir / f"{split_name}_indexed.csv", index=False)

    # 9. 保存 ID 映射
    with open(output_dir / "user2id.json", "w") as f:
        json.dump(user2id, f, ensure_ascii=False)
    with open(output_dir / "item2id.json", "w") as f:
        json.dump(item2id, f, ensure_ascii=False)

    # 10. 有序 ID 列表（供 cache 脚本索引对应）
    item_ids_ordered = [item for item, _ in sorted(item2id.items(), key=lambda x: x[1])]
    user_ids_ordered = [user for user, _ in sorted(user2id.items(), key=lambda x: x[1])]
    cache_dir = ROOT_DIR / "cache" / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "item_ids.json", "w") as f:
        json.dump(item_ids_ordered, f)
    with open(cache_dir / "user_ids.json", "w") as f:
        json.dump(user_ids_ordered, f)

    # 11. ★ 聚合评论（仅训练集，防止泄漏）→ item_reviews.json
    print("\n  聚合训练集评论...")
    item_reviews = build_item_reviews(train_df, max_per_item=MAX_REVIEWS_PER_ITEM)
    with open(output_dir / "item_reviews.json", "w", encoding="utf-8") as f:
        json.dump(item_reviews, f, ensure_ascii=False)
    print(f"  ✅ 评论已保存：{output_dir / 'item_reviews.json'}")

    # 12. 统计信息
    stats = {
        "n_users":  len(user2id),
        "n_items":  len(item2id),
        "n_train":  len(train_df),
        "n_val":    len(val_df),
        "n_test":   len(test_df),
        "density":  len(df) / (len(user2id) * len(item2id)),
        "k_core":   k,
        "seed":     seed,
        "n_items_with_reviews": sum(1 for v in item_reviews.values() if v),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ 已保存至: {output_dir}")
    return stats


# ─────────────────────────── CLI ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MARA 数据预处理")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["baby", "beauty_sub", "office", "sports", "all"],
        help="要处理的数据集（默认：all，依次处理全部）"
    )
    parser.add_argument("--k",           type=int, default=5,    help="k-core 过滤阈值（默认 5）")
    parser.add_argument("--seed",        type=int, default=42,   help="随机种子（默认 42）")
    parser.add_argument("--beauty_users",type=int, default=3000, help="Beauty 子集目标用户数（默认 3000）")
    args = parser.parse_args()

    datasets_to_process = []
    if args.dataset in ("baby", "all"):
        datasets_to_process.append(("baby", False))
    if args.dataset in ("beauty_sub", "all"):
        datasets_to_process.append(("beauty_sub", True))
    if args.dataset in ("office", "all"):
        datasets_to_process.append(("office", False))
    if args.dataset in ("sports", "all"):
        datasets_to_process.append(("sports", False))

    for ds_name, is_beauty_sub in datasets_to_process:
        raw_key   = "beauty" if is_beauty_sub else ds_name
        raw_inter = RAW_FILE_MAP[raw_key]["interactions"]
        process_dataset(
            dataset_name        = ds_name,
            raw_inter_path      = raw_inter,
            output_dir          = DATA_DIR / ds_name,
            k                   = args.k,
            beauty_target_users = args.beauty_users,
            seed                = args.seed,
            is_beauty_sub       = is_beauty_sub,
        )

    print("\n\n✅ 所有数据集预处理完成！")
    print("  下一步：python scripts/extract_meta.py --dataset all")


if __name__ == "__main__":
    main()
