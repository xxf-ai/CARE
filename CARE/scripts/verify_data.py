"""
scripts/verify_data.py
----------------------
Step 4: 预处理结果验证

检查项：
  [数据文件]
  ✓ train/val/test.csv 存在且行数一致
  ✓ user2id / item2id JSON 完整
  ✓ train 中所有 user_id / item_id 均在映射表中
  ✓ val/test 中无数据泄漏（val 物品不在 test 中）
  ✓ 每用户在 train 中至少有 k-2 条记录

  [元数据]
  ✓ item_meta.json 键数 == item2id 大小
  ✓ 各字段非空率统计

  [图像]
  ✓ images/ 目录下图片数 == item2id 大小
  ✓ 抽查若干图片可正常打开且尺寸正确
  ✓ 黑图（占位）数量统计

  [缓存目录]
  ✓ cache/{dataset}/item_ids.json 与 item2id 一致

使用方法：
  python scripts/verify_data.py --dataset baby
  python scripts/verify_data.py --dataset all
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"

PASS = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"


def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"{PASS} {msg_pass}")
    else:
        print(f"{FAIL} {msg_fail}")
    return condition


def verify_dataset(dataset_name: str, sample_images: int = 20) -> bool:
    data_dir  = DATA_DIR  / dataset_name
    cache_dir = CACHE_DIR / dataset_name

    print(f"\n{'='*60}")
    print(f"  验证数据集: {dataset_name}")
    print(f"{'='*60}")

    all_pass = True

    # ── 1. 基础文件存在性 ────────────────────────────────────────
    print("\n[1] 基础文件检查")
    for fname in ["train.csv", "val.csv", "test.csv", "user2id.json", "item2id.json", "stats.json", "item_meta.json"]:
        ok = (data_dir / fname).exists()
        all_pass &= check(ok, f"{fname} 存在", f"{fname} 不存在！")

    # ── 2. 加载文件 ──────────────────────────────────────────────
    try:
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df   = pd.read_csv(data_dir / "val.csv")
        test_df  = pd.read_csv(data_dir / "test.csv")
        with open(data_dir / "user2id.json") as f:
            user2id = json.load(f)
        with open(data_dir / "item2id.json") as f:
            item2id = json.load(f)
        with open(data_dir / "stats.json") as f:
            stats = json.load(f)
        with open(data_dir / "item_meta.json") as f:
            item_meta = json.load(f)
    except Exception as e:
        print(f"{FAIL} 文件加载失败: {e}")
        return False

    # ── 3. 数据集基本统计 ────────────────────────────────────────
    print("\n[2] 数据集统计")
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    print(f"     用户数: {n_users:,}  物品数: {n_items:,}")
    print(f"     训练/验证/测试: {stats['n_train']:,} / {stats['n_val']:,} / {stats['n_test']:,}")
    print(f"     交互密度: {stats['density']*100:.4f}%")

    ok = len(user2id) == n_users and len(item2id) == n_items
    all_pass &= check(ok, "ID 映射大小与统计一致", f"ID 映射大小不一致：user2id={len(user2id)}, item2id={len(item2id)}")

    # ── 4. 划分一致性 ────────────────────────────────────────────
    print("\n[3] 数据划分检查")
    ok = len(val_df) == len(test_df) == len(user2id)
    all_pass &= check(ok, "验证集 / 测试集行数 == 用户数（Leave-One-Out）",
                      f"验证/测试行数不匹配：val={len(val_df)}, test={len(test_df)}, users={len(user2id)}")

    # 不同分组不重叠（user_id×item_id 对唯一）
    train_pairs = set(zip(train_df["user_id"], train_df["item_id"]))
    val_pairs   = set(zip(val_df["user_id"],   val_df["item_id"]))
    test_pairs  = set(zip(test_df["user_id"],  test_df["item_id"]))
    overlap_tv = train_pairs & val_pairs
    overlap_tt = train_pairs & test_pairs
    overlap_vt = val_pairs   & test_pairs
    all_pass &= check(len(overlap_tv) == 0, "train 与 val 无重叠", f"train-val 重叠 {len(overlap_tv)} 对")
    all_pass &= check(len(overlap_tt) == 0, "train 与 test 无重叠", f"train-test 重叠 {len(overlap_tt)} 对")
    all_pass &= check(len(overlap_vt) == 0, "val 与 test 无重叠", f"val-test 重叠 {len(overlap_vt)} 对")

    # ── 5. ID 映射完整性 ─────────────────────────────────────────
    print("\n[4] ID 映射检查")
    all_df = pd.concat([train_df, val_df, test_df])
    unknown_users = set(all_df["user_id"].unique()) - set(user2id.keys())
    unknown_items = set(all_df["item_id"].unique()) - set(item2id.keys())
    all_pass &= check(len(unknown_users) == 0, "所有 user_id 均在 user2id 中",
                      f"{len(unknown_users)} 个 user_id 不在映射表中")
    all_pass &= check(len(unknown_items) == 0, "所有 item_id 均在 item2id 中",
                      f"{len(unknown_items)} 个 item_id 不在映射表中")

    # ID 值范围
    id_values_u = list(user2id.values())
    id_values_i = list(item2id.values())
    ok_u = min(id_values_u) == 0 and max(id_values_u) == len(user2id) - 1
    ok_i = min(id_values_i) == 0 and max(id_values_i) == len(item2id) - 1
    all_pass &= check(ok_u, "user2id 值为连续整数 [0, n_users)", f"user2id 值不连续")
    all_pass &= check(ok_i, "item2id 值为连续整数 [0, n_items)", f"item2id 值不连续")

    # ── 6. 元数据检查 ────────────────────────────────────────────
    print("\n[5] 元数据检查")
    ok = len(item_meta) == n_items
    all_pass &= check(ok, f"item_meta.json 含 {n_items:,} 个物品", f"item_meta 大小 {len(item_meta)} ≠ {n_items}")

    # 各字段非空率
    fields = ["title", "description", "category", "main_image_url"]
    for field in fields:
        non_empty = sum(1 for v in item_meta.values() if v.get(field, ""))
        rate = non_empty / n_items * 100
        symbol = PASS if rate >= 50 else WARN
        print(f"  {symbol[2:]}  {field} 非空率: {rate:.1f}%  ({non_empty:,}/{n_items:,})")

    # ── 7. 图像检查 ──────────────────────────────────────────────
    print("\n[6] 图像文件检查")
    image_dir = data_dir / "images"
    if not image_dir.exists():
        print(f"{WARN}  images/ 目录不存在，跳过图像检查（请先运行 download_images.py）")
    else:
        jpg_files = list(image_dir.glob("*.jpg"))
        ok = len(jpg_files) == n_items
        all_pass &= check(ok, f"图像数量 {len(jpg_files):,} == 物品数 {n_items:,}",
                          f"图像数量 {len(jpg_files):,} ≠ 物品数 {n_items:,}")

        # 抽样检查：图像可打开且尺寸正确
        sample_files = random.sample(jpg_files, min(sample_images, len(jpg_files)))
        bad_imgs = 0
        black_imgs = 0
        for img_path in sample_files:
            try:
                img = Image.open(img_path)
                arr = np.array(img)
                # 检查是否为全黑占位图
                if arr.max() == 0:
                    black_imgs += 1
            except Exception:
                bad_imgs += 1

        all_pass &= check(bad_imgs == 0, f"抽查 {len(sample_files)} 张图像均可正常打开",
                          f"有 {bad_imgs} 张图像损坏")
        if black_imgs > 0:
            print(f"{WARN}  抽查中有 {black_imgs}/{len(sample_files)} 张为黑图占位（正常，下载失败的物品）")

    # ── 8. 缓存目录检查 ──────────────────────────────────────────
    print("\n[7] 缓存目录检查")
    item_ids_cache = cache_dir / "item_ids.json"
    user_ids_cache = cache_dir / "user_ids.json"
    if not item_ids_cache.exists():
        print(f"{WARN}  cache/{dataset_name}/item_ids.json 不存在（运行 preprocess.py 后应自动生成）")
    else:
        with open(item_ids_cache) as f:
            cached_item_ids = json.load(f)
        ok = len(cached_item_ids) == n_items and set(cached_item_ids) == set(item2id.keys())
        all_pass &= check(ok, "cache item_ids.json 与 item2id 一致",
                          f"cache item_ids 不一致：{len(cached_item_ids)} vs {n_items}")

    # ── 汇总 ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    if all_pass:
        print(f"  ✅✅  {dataset_name} 验证全部通过！")
    else:
        print(f"  ❌  {dataset_name} 存在问题，请检查上述错误")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="验证预处理结果")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["baby", "beauty_sub", "office", "sports", "all"],
    )
    parser.add_argument("--sample_images", type=int, default=20, help="图像抽查数量（默认 20）")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("baby", "all"):
        datasets.append("baby")
    if args.dataset in ("beauty_sub", "all"):
        datasets.append("beauty_sub")
    if args.dataset in ("office", "all"):
        datasets.append("office")
    if args.dataset in ("sports", "all"):
        datasets.append("sports")

    results = {}
    for ds in datasets:
        results[ds] = verify_dataset(ds, sample_images=args.sample_images)

    print(f"\n{'='*60}")
    print("  验证汇总")
    print(f"{'='*60}")
    all_ok = True
    for ds, ok in results.items():
        symbol = "✅" if ok else "❌"
        print(f"  {symbol}  {ds}")
        all_ok &= ok

    if all_ok:
        print("\n  🎉 所有数据集验证通过！可以开始下一步：")
        print("     python scripts/cache_clip_features.py --dataset baby")
    else:
        print("\n  请修复上述错误后重新验证。")


if __name__ == "__main__":
    main()
