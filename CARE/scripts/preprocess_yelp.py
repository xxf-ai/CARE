"""
scripts/preprocess_yelp.py
---------------------------
P2: Yelp dataset preprocessing for CARE generalization experiment.

Converts Yelp Open Dataset into the same format as Amazon data,
then reuses existing pipeline (preprocess.py + cache_clip_features.py).

Usage:
  # 1. Download Yelp JSON from https://www.yelp.com/dataset
  # 2. Place review.json and business.json in data/raw/yelp/
  # 3. Run:
  python scripts/preprocess_yelp.py
  # 4. Then normal pipeline:
  python scripts/cache_clip_features.py --dataset yelp
  python train.py --dataset yelp --seeds 42
  python evaluate.py --dataset yelp --seeds 42 123 456

Output format matches Amazon pipeline exactly:
  data/yelp/train_indexed.csv, val_indexed.csv, test_indexed.csv
  data/yelp/item_meta.json (with clip_text field for CLIP encoding)
  data/yelp/stats.json
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw" / "yelp"

# ── Config ────────────────────────────────────────────────────────
K_CORE   = 5    # Min interactions per user and per item
MAX_REVIEWS_PER_ITEM = 5
MIN_REVIEW_LENGTH = 20  # Filter too-short reviews


def load_yelp(raw_dir: Path):
    """Load Yelp review.json and business.json, return DataFrames."""
    print("  Loading Yelp reviews...")
    reviews = []
    with open(raw_dir / "review.json", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  reviews"):
            r = json.loads(line)
            if len(r.get("text", "")) >= MIN_REVIEW_LENGTH:
                reviews.append({
                    "user_id": r["user_id"],
                    "item_id": r["business_id"],
                    "rating": float(r["stars"]),
                    "timestamp": r["date"],
                    "text": r["text"][:512],  # Truncate for memory
                })
    reviews_df = pd.DataFrame(reviews)

    print("  Loading Yelp businesses...")
    businesses = {}
    with open(raw_dir / "business.json", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  businesses"):
            b = json.loads(line)
            # Build descriptive text from name + categories + attributes
            desc_parts = [b.get("name", "")]
            cats = b.get("categories", "")
            if cats:
                cats_str = ", ".join(cats) if isinstance(cats, list) else str(cats)
                desc_parts.append(cats_str)
            # Add attributes if available
            attrs = b.get("attributes", {})
            if attrs:
                for k, v in attrs.items():
                    if isinstance(v, str) and v.strip():
                        # Clean up nested keys
                        clean_k = k.replace("'", "").replace('"', '')
                        desc_parts.append(f"{clean_k}: {v}")
            clip_text = " ".join(desc_parts)[:256]
            businesses[b["business_id"]] = {
                "title": b.get("name", ""),
                "clip_text": clip_text,
                "categories": cats,
            }

    return reviews_df, businesses


def k_core_filter(df: pd.DataFrame, k: int):
    """Iterative K-core filtering."""
    while True:
        prev = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = set(user_counts[user_counts >= k].index)
        valid_items = set(item_counts[item_counts >= k].index)
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(df) == prev:
            break
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_core", type=int, default=K_CORE)
    parser.add_argument("--sample_users", type=int, default=0,
                       help="Subsample N users for quick testing (0=full)")
    args = parser.parse_args()

    raw_dir = RAW_DIR
    out_dir = DATA_DIR / "yelp"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (raw_dir / "review.json").exists():
        print(f"\n  ⚠ Yelp data not found at {raw_dir}")
        print(f"  Please download from https://www.yelp.com/dataset")
        print(f"  and place review.json + business.json in {raw_dir}")
        return

    # ── Load ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Yelp Preprocessing  (K-core={args.k_core})")
    print(f"{'='*55}")

    reviews_df, businesses = load_yelp(raw_dir)
    print(f"  Raw reviews: {len(reviews_df):,}")
    print(f"  Businesses with metadata: {len(businesses):,}")

    # ── K-core filter ─────────────────────────────────────────────
    reviews_df = k_core_filter(reviews_df, args.k_core)
    print(f"  After K-core (k={args.k_core}): {len(reviews_df):,} reviews")

    # ── Subsample if requested ────────────────────────────────────
    if args.sample_users > 0:
        users = reviews_df["user_id"].unique()
        sample = np.random.choice(users, min(args.sample_users, len(users)), replace=False)
        reviews_df = reviews_df[reviews_df["user_id"].isin(sample)]
        reviews_df = k_core_filter(reviews_df, args.k_core)
        print(f"  After subsampling {args.sample_users} users: {len(reviews_df):,} reviews")

    # ── Leave-One-Out split ──────────────────────────────────────
    reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"])
    reviews_df = reviews_df.sort_values(["user_id", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []
    for uid, grp in reviews_df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        if len(grp) >= 3:
            train_rows.append(grp.iloc[:-2])
            val_rows.append(grp.iloc[-2:-1])
            test_rows.append(grp.iloc[-1:])
        elif len(grp) == 2:
            train_rows.append(grp.iloc[:-1])
            val_rows.append(grp.iloc[-1:])
        else:
            train_rows.append(grp)

    train_df = pd.concat(train_rows) if train_rows else pd.DataFrame()
    val_df   = pd.concat(val_rows)   if val_rows   else pd.DataFrame()
    test_df  = pd.concat(test_rows)  if test_rows  else pd.DataFrame()

    # ── Build ID mappings ────────────────────────────────────────
    all_users = set(train_df["user_id"]) | set(val_df["user_id"]) | set(test_df["user_id"])
    all_items = set(train_df["item_id"]) | set(val_df["item_id"]) | set(test_df["item_id"])
    user2id = {u: i for i, u in enumerate(sorted(all_users))}
    item2id = {i: j for j, i in enumerate(sorted(all_items))}

    def map_ids(df):
        df = df.copy()
        df["user_id"] = df["user_id"].map(user2id)
        df["item_id"] = df["item_id"].map(item2id)
        return df[["user_id", "item_id", "rating", "timestamp"]]

    # ── Save CSVs ─────────────────────────────────────────────────
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        mapped = map_ids(df)
        mapped.to_csv(out_dir / f"{name}_indexed.csv", index=False)
        # Also save original-ID version
        df[["user_id", "item_id", "rating", "timestamp"]].to_csv(
            out_dir / f"{name}.csv", index=False)

    # ── Save item metadata (for CLIP text extraction) ─────────────
    item_meta = {}
    for item_id, idx in item2id.items():
        if item_id in businesses:
            item_meta[str(idx)] = businesses[item_id]
        else:
            item_meta[str(idx)] = {"title": str(item_id)[:50], "clip_text": str(item_id)[:50]}
    with open(out_dir / "item_meta.json", "w", encoding="utf-8") as f:
        json.dump(item_meta, f, ensure_ascii=False)

    # ── Save stats ────────────────────────────────────────────────
    stats = {
        "n_users": len(user2id),
        "n_items": len(item2id),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Save ID mappings ──────────────────────────────────────────
    with open(out_dir / "user2id.json", "w") as f:
        json.dump(user2id, f)
    with open(out_dir / "item2id.json", "w") as f:
        json.dump(item2id, f)

    # ── Cache directory (required by clip feature extraction) ─────
    cache_dir = ROOT_DIR / "cache" / "yelp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    item_ids_ordered = [item for item, _ in sorted(item2id.items(), key=lambda x: x[1])]
    user_ids_ordered = [user for user, _ in sorted(user2id.items(), key=lambda x: x[1])]
    with open(cache_dir / "item_ids.json", "w") as f:
        json.dump(item_ids_ordered, f)
    with open(cache_dir / "user_ids.json", "w") as f:
        json.dump(user_ids_ordered, f)

    # ── Report ────────────────────────────────────────────────────
    print(f"\n  Dataset statistics:")
    print(f"    Users:  {stats['n_users']:,}")
    print(f"    Items:  {stats['n_items']:,}")
    print(f"    Train:  {stats['n_train']:,}")
    print(f"    Val:    {stats['n_val']:,}")
    print(f"    Test:   {stats['n_test']:,}")
    print(f"    Output: {out_dir}")
    print(f"\n  Next steps:")
    print(f"    python scripts/cache_clip_features.py --dataset yelp")
    print(f"    python train.py --dataset yelp --seeds 42")
    print(f"    python evaluate.py --dataset yelp --seeds 42 123 456")


if __name__ == "__main__":
    main()
