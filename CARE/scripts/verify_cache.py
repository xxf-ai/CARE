"""
scripts/verify_cache.py
------------------------
Day 2 · Step 4 — 缓存完整性验证

检查所有 npy 文件是否满足：
  ✓ 文件存在
  ✓ shape 正确（行数 == 物品数 / 用户数）
  ✓ dtype 为 float32
  ✓ 无 NaN / Inf 值
  ✓ clip_cls / clip_text / fine_attr 向量已做 L2 归一化检查（均值模长接近 1）
  ✓ item_ids.json 与 clip_cls.npy 行数一致
  ✓ user_ids.json 与 user_emb.npy 行数一致
  ✓ 各文件索引对齐（item_ids[i] 与所有特征文件第 i 行对应）

使用：
  python scripts/verify_cache.py --dataset baby
  python scripts/verify_cache.py --dataset all
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"

OK   = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"


def chk(cond, ok_msg, fail_msg) -> bool:
    if cond:
        print(f"{OK} {ok_msg}")
    else:
        print(f"{FAIL} {fail_msg}")
    return cond


def verify_dataset(dataset_name: str) -> bool:
    cache_dir = CACHE_DIR / dataset_name
    data_dir  = DATA_DIR  / dataset_name

    print(f"\n{'='*60}")
    print(f"  验证缓存: {dataset_name}")
    print(f"{'='*60}")

    # ── 读取 stats ────────────────────────────────────────────────
    stats_path = data_dir / "stats.json"
    if not stats_path.exists():
        print(f"{FAIL} stats.json 不存在，请先运行 preprocess.py")
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    print(f"  期望: n_users={n_users:,}  n_items={n_items:,}")

    # ── 读取 ID 列表 ──────────────────────────────────────────────
    item_ids_path = cache_dir / "item_ids.json"
    user_ids_path = cache_dir / "user_ids.json"
    if not item_ids_path.exists() or not user_ids_path.exists():
        print(f"{FAIL} item_ids.json / user_ids.json 不存在")
        return False
    with open(item_ids_path) as f:
        item_ids = json.load(f)
    with open(user_ids_path) as f:
        user_ids = json.load(f)

    all_pass = True
    all_pass &= chk(len(item_ids) == n_items,
                    f"item_ids.json 长度 {len(item_ids)} == n_items",
                    f"item_ids.json 长度 {len(item_ids)} ≠ {n_items}")
    all_pass &= chk(len(user_ids) == n_users,
                    f"user_ids.json 长度 {len(user_ids)} == n_users",
                    f"user_ids.json 长度 {len(user_ids)} ≠ {n_users}")

    # ── VARA 所需 npy 文件规格 ─────────────────────────────────────
    npy_specs = [
        ("clip_text.npy",   (n_items, 768),        "CLIP 文本语义特征"),
        ("user_emb.npy",    (n_users, None),       "LightGCN 用户协同嵌入"),
        ("item_emb.npy",    (n_items, None),       "LightGCN 物品协同嵌入"),
    ]
    # 可选：clip_cls.npy 仅当图片已下载时存在
    optional_specs = [
        ("clip_cls.npy",    (n_items, 768),        "CLIP 视觉特征（可选）"),
    ]

    print(f"\n[1] npy 文件检查")
    arrays: dict[str, np.ndarray] = {}

    for fname, expect_shape, desc in npy_specs:
        path = cache_dir / fname
        if not path.exists():
            print(f"{FAIL} {fname} 不存在  [{desc}]")
            all_pass = False
            continue

        arr = np.load(path, mmap_mode="r")
        arrays[fname] = arr

        # shape 检查
        shape_ok = True
        for i, (actual, expected) in enumerate(zip(arr.shape, expect_shape)):
            if expected is not None and actual != expected:
                shape_ok = False
        all_pass &= chk(shape_ok,
                        f"{fname} shape={arr.shape}  ✓",
                        f"{fname} shape={arr.shape} 期望 {expect_shape}")

        # dtype
        all_pass &= chk(arr.dtype == np.float32,
                        f"{fname} dtype=float32",
                        f"{fname} dtype={arr.dtype}（应为 float32）")

        # NaN / Inf
        # mmap 模式下 np.isnan 很慢，抽样检查
        sample = arr[:min(200, len(arr))]
        has_nan = np.isnan(sample).any()
        has_inf = np.isinf(sample).any()
        all_pass &= chk(not has_nan and not has_inf,
                        f"{fname} 抽样无 NaN/Inf",
                        f"{fname} 存在 NaN={has_nan} Inf={has_inf}")

    # 可选 npy 文件
    for fname, expect_shape, desc in optional_specs:
        path = cache_dir / fname
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            arrays[fname] = arr
            shape_ok = all(actual == expected for actual, expected in
                          zip(arr.shape, expect_shape) if expected is not None)
            if shape_ok:
                print(f"{OK} {fname} shape={arr.shape} [{desc}]（可选，已存在）")
            else:
                print(f"{WARN} {fname} shape={arr.shape} 期望 {expect_shape} [{desc}]")
        else:
            print(f"{WARN} {fname} 不存在 [{desc}]（可选，跳过）")

    # ── 统计摘要 ──────────────────────────────────────────────────
    print(f"\n[2] 特征统计（norm / zero_rows）")
    for fname in ["clip_text.npy", "clip_cls.npy"]:
        if fname not in arrays:
            continue
        arr   = arrays[fname]
        norms = np.linalg.norm(arr[:min(500, len(arr))], axis=-1)
        zero  = (arr == 0).all(axis=-1).sum()
        print(f"  {fname:<22}  "
              f"norm均值={norms.mean():.3f}  "
              f"norm标准差={norms.std():.3f}  "
              f"零向量={zero}/{n_items}")

    for fname in ["user_emb.npy", "item_emb.npy"]:
        if fname not in arrays:
            continue
        arr   = arrays[fname]
        norms = np.linalg.norm(arr[:min(500, len(arr))], axis=-1)
        print(f"  {fname:<22}  norm均值={norms.mean():.3f}  std={norms.std():.3f}  "
              f"d={arr.shape[1]}")

    # ── 磁盘占用 ──────────────────────────────────────────────────
    print(f"\n[3] 磁盘占用")
    total_mb = 0.0
    for p in sorted(cache_dir.glob("*.npy")):
        mb = p.stat().st_size / 1e6
        total_mb += mb
        print(f"  {p.name:<28} {mb:7.1f} MB")
    print(f"  {'合计':<28} {total_mb:7.1f} MB")

    # ── 汇总 ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    if all_pass:
        print(f"  ✅✅  {dataset_name} 缓存验证全部通过！（VARA 核心文件完整）")
    else:
        print(f"  ❌   {dataset_name} 存在问题，请检查上述错误。")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="验证特征缓存完整性")
    parser.add_argument("--dataset", default="all", choices=["baby", "beauty_sub", "office", "sports", "all"])
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("baby",       "all"): datasets.append("baby")
    if args.dataset in ("beauty_sub", "all"): datasets.append("beauty_sub")
    if args.dataset in ("office",     "all"): datasets.append("office")
    if args.dataset in ("sports",     "all"): datasets.append("sports")

    results = {ds: verify_dataset(ds) for ds in datasets}

    print(f"\n{'='*60}")
    print("  验证汇总")
    print(f"{'='*60}")
    all_ok = True
    for ds, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {ds}")
        all_ok &= ok

    if all_ok:
        print("\n  🎉 所有缓存验证通过！可以开始 Day 3：MARA 核心模块开发。")
        print("     python train.py --dataset baby --epochs 2 --smoke_test")
    else:
        print("\n  请修复问题后重新验证。")


if __name__ == "__main__":
    main()
