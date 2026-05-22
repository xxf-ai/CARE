"""
experiments/analysis/score_distribution.py — CLIP text vs image 分数分布方差分析

验证论文 §5.2 机制假说：image 分数分布比 text 更平坦（方差更小），
使得 image gating 给 CF 留出更大的负样本压制空间，解释了为何
"image 独立更弱但 gating 更强"。

用法:
  python experiments/analysis/score_distribution.py --dataset baby --seed 42
  python experiments/analysis/score_distribution.py --dataset baby --seeds 42 123 456
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data_utils.dataset import MARADataset, CacheManager
from evaluate import build_user_text_pref, build_user_image_pref

DATA_DIR  = ROOT / "data"
CACHE_DIR = ROOT / "cache"

COLD_THRESHOLD = 5


def load_cache(cache_dir, data_dir, device):
    mgr = CacheManager(cache_dir, data_dir, device, pin_memory=False)
    cache = {}
    for k, v in mgr.tensors.items():
        if isinstance(v, np.ndarray):
            cache[k] = torch.from_numpy(np.array(v, dtype=np.float32)).to(device)
        else:
            cache[k] = v.to(device)
    return cache


@torch.no_grad()
def analyze_distribution(dataset, seed, device):
    data_dir  = DATA_DIR / dataset
    cache_dir = CACHE_DIR / dataset

    cache = load_cache(cache_dir, data_dir, device)

    if "clip_cls" not in cache:
        print(f"  [SKIP] {dataset}: 无 clip_cls.npy (image features)")
        return None

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    clip_text = cache["clip_text"]
    clip_cls  = cache["clip_cls"]

    # Build user preferences
    user_text_pref = build_user_text_pref(
        data_dir / "train_indexed.csv", clip_text, n_users, device)
    user_img_pref = build_user_image_pref(
        data_dir / "train_indexed.csv", clip_cls, n_users, device)

    # Item interaction counts for coldness stratification
    train_df = pd.read_csv(data_dir / "train_indexed.csv")
    item_counts = torch.zeros(n_items, device=device)
    for iid in train_df["item_id"].values:
        item_counts[int(iid)] += 1

    test_set = MARADataset(data_dir, split="test", seed=seed, n_neg=99)
    loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=2)

    # Collect per-sample statistics
    stats_list = []  # each entry: {bucket, txt_mean, txt_std, img_mean, img_std, txt_range, img_range, n_candidates}

    for users, pos_items, neg_items in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B = users.size(0)
        cand = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [B, 100]
        cand_flat = cand.reshape(-1)

        # Text scores [B, 100]
        u_txt = F.normalize(user_text_pref[users], dim=-1)
        i_txt = F.normalize(clip_text[cand_flat], dim=-1).reshape(B, 100, -1)
        txt_scores = (u_txt.unsqueeze(1) * i_txt).sum(dim=-1)  # cos sim

        # Image scores [B, 100]
        u_img = F.normalize(user_img_pref[users], dim=-1)
        i_img = F.normalize(clip_cls[cand_flat], dim=-1).reshape(B, 100, -1)
        img_scores = (u_img.unsqueeze(1) * i_img).sum(dim=-1)

        # Per-sample statistics
        txt_mean = txt_scores.mean(dim=1).cpu().numpy()
        txt_std  = txt_scores.std(dim=1).cpu().numpy()
        img_mean = img_scores.mean(dim=1).cpu().numpy()
        img_std  = img_scores.std(dim=1).cpu().numpy()
        txt_range = (txt_scores.max(dim=1).values - txt_scores.min(dim=1).values).cpu().numpy()
        img_range = (img_scores.max(dim=1).values - img_scores.min(dim=1).values).cpu().numpy()

        # Positive item scores
        txt_pos = txt_scores[:, 0].cpu().numpy()
        img_pos = img_scores[:, 0].cpu().numpy()

        # Coldness bucket
        pos_counts = item_counts[pos_items].cpu().numpy()

        for j in range(B):
            cnt = int(pos_counts[j])
            if cnt == 0:
                bucket = "L0"
            elif cnt <= 4:
                bucket = "L1"
            elif cnt <= 20:
                bucket = "L2"
            else:
                bucket = "L3"

            stats_list.append({
                "bucket": bucket,
                "txt_mean": float(txt_mean[j]),
                "txt_std": float(txt_std[j]),
                "img_mean": float(img_mean[j]),
                "img_std": float(img_std[j]),
                "txt_range": float(txt_range[j]),
                "img_range": float(img_range[j]),
                "txt_pos": float(txt_pos[j]),
                "img_pos": float(img_pos[j]),
            })

    return stats_list


def report(stats_list):
    """打印 text vs image 分布对比报告。"""
    buckets = ["L0", "L1", "L2", "L3"]
    bucket_names = {"L0": "L0 (zero-shot)", "L1": "L1 (1-4)", "L2": "L2 (5-20)", "L3": "L3 (>20)"}

    print(f"\n{'='*85}")
    print(f"  CLIP Text vs Image 分数分布方差分析")
    print(f"{'='*85}")

    # ── Overall ──
    print(f"\n  {'─'*60}")
    print(f"  全量测试集 ({len(stats_list)} samples)")
    print(f"  {'─'*60}")
    _print_comparison(stats_list, "Overall")

    # ── Per bucket ──
    for bucket in buckets:
        subset = [s for s in stats_list if s["bucket"] == bucket]
        if not subset:
            continue
        print(f"\n  {'─'*60}")
        print(f"  {bucket_names[bucket]} ({len(subset)} samples)")
        print(f"  {'─'*60}")
        _print_comparison(subset, bucket_names[bucket])

    # ── Key finding: L0 bucket comparison ──
    l0 = [s for s in stats_list if s["bucket"] == "L0"]
    if l0:
        txt_std_mean = np.mean([s["txt_std"] for s in l0])
        img_std_mean = np.mean([s["img_std"] for s in l0])
        txt_range_mean = np.mean([s["txt_range"] for s in l0])
        img_range_mean = np.mean([s["img_range"] for s in l0])
        ratio_std = img_std_mean / max(txt_std_mean, 1e-8)
        ratio_range = img_range_mean / max(txt_range_mean, 1e-8)

        print(f"\n{'='*85}")
        print(f"  核心发现 (L0 bucket)")
        print(f"{'='*85}")
        print(f"  Text score std:      {txt_std_mean:.4f}")
        print(f"  Image score std:     {img_std_mean:.4f}  (ratio: {ratio_std:.2f}x)")
        print(f"  Text score range:    {txt_range_mean:.4f}")
        print(f"  Image score range:   {img_range_mean:.4f}  (ratio: {ratio_range:.2f}x)")
        if ratio_std < 1.0:
            print(f"\n  => Image 分数分布比 Text 更平坦 ({ratio_std:.2f}x std)，")
            print(f"     验证了 §5.2 机制假说：更平坦的 image 分布给 CF 留出更大的")
            print(f"     负样本压制空间，解释了为何 image gating 优于 text gating。")
        else:
            print(f"\n  => 未观察到预期的 image 分布更平坦效应，需要进一步检查。")


def _print_comparison(stats_list, label):
    txt_std_mean = np.mean([s["txt_std"] for s in stats_list])
    img_std_mean = np.mean([s["img_std"] for s in stats_list])
    txt_range_mean = np.mean([s["txt_range"] for s in stats_list])
    img_range_mean = np.mean([s["img_range"] for s in stats_list])
    txt_pos_mean = np.mean([s["txt_pos"] for s in stats_list])
    img_pos_mean = np.mean([s["img_pos"] for s in stats_list])
    txt_mean_mean = np.mean([s["txt_mean"] for s in stats_list])
    img_mean_mean = np.mean([s["img_mean"] for s in stats_list])

    print(f"  {'指标':<20} {'Text':>10} {'Image':>10} {'Ratio (I/T)':>12}")
    print(f"  {'─'*52}")
    print(f"  {'Score 均值':<20} {txt_mean_mean:>10.4f} {img_mean_mean:>10.4f} {img_mean_mean/max(txt_mean_mean,1e-8):>12.2f}")
    print(f"  {'Score 标准差':<20} {txt_std_mean:>10.4f} {img_std_mean:>10.4f} {img_std_mean/max(txt_std_mean,1e-8):>12.2f}")
    print(f"  {'Score 极差':<20} {txt_range_mean:>10.4f} {img_range_mean:>10.4f} {img_range_mean/max(txt_range_mean,1e-8):>12.2f}")
    print(f"  {'正样本得分':<20} {txt_pos_mean:>10.4f} {img_pos_mean:>10.4f} {img_pos_mean/max(txt_pos_mean,1e-8):>12.2f}")


def main():
    parser = argparse.ArgumentParser(description="CLIP score 分布方差分析")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_stats = []
    for seed in args.seeds:
        print(f"  Analyzing seed={seed} ...")
        s = analyze_distribution(args.dataset, seed, device)
        if s:
            all_stats.extend(s)

    if not all_stats:
        print("[ERROR] No data collected")
        return

    report(all_stats)

    # Save raw stats for paper figures
    out_path = ROOT / "experiments" / "ablations" / "results" / f"{args.dataset}_score_distribution.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "seeds": args.seeds,
            "n_samples": len(all_stats),
            "stats": all_stats[:1000],  # save first 1000 samples to keep file small
        }, f, indent=2)
    print(f"\n  原始数据(前1000条)已保存: {out_path}")


if __name__ == "__main__":
    main()
