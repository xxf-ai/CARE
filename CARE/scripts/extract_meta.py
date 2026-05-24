"""
scripts/extract_meta.py
-----------------------
Step 2: 物品元数据提取

功能：
  - 从 meta_*.jsonl 中提取 item_id 对应的结构化元信息
  - 仅保留已通过 k-core 过滤的物品（按 item2id.json 过滤）
  - 读取 preprocess.py 生成的 item_reviews.json，将评论融入文本
  - 输出 item_meta.json：{item_id: {title, description, category,
                                    clip_text, image_urls, ...}}
    clip_text 字段 = title + description/features + top评论，
    供 cache_clip_features.py 直接使用，无需再拼接
  - 同时输出 image_url_list.json：待下载的 (item_id, url) 列表

clip_text 拼接策略（按优先级填充，截断到 256 字符）：
  title（必填）+ description/features（补充属性）+ 评论（补充感官信息）
  评论最多取 MAX_REVIEW_CHARS 字符，保证 title 信息不被截断

使用方法：
  python scripts/extract_meta.py --dataset baby
  python scripts/extract_meta.py --dataset beauty_sub
  python scripts/extract_meta.py --dataset all
"""

import argparse
import gzip
import json
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────── 路径常量 ──────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"

RAW_META_MAP = {
    "baby":       RAW_DIR / "meta_Baby_Products.jsonl",
    "beauty_sub": RAW_DIR / "meta_All_Beauty.jsonl",
    "office":     RAW_DIR / "meta_Office_Products.jsonl",
    "sports":     RAW_DIR / "meta_Sports_and_Outdoors.jsonl",
}

# clip_text 各部分的字符预算
CLIP_TEXT_TOTAL     = 256   # CLIP tokenizer 截断前的总字符数
TITLE_CHARS         = 80    # title 预留字符（优先保证）
META_CHARS          = 80    # description/features 预留字符
MAX_REVIEW_CHARS    = 96    # 评论最多占用字符（剩余全给评论）
MAX_REVIEWS_IN_TEXT = 3     # 最多拼接几条评论


# ─────────────────────────── I/O 工具 ──────────────────────────────

def open_jsonl(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ─────────────────────────── 字段解析 ─────────────────────────────

def parse_description(raw) -> str:
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw if x).strip()
    if isinstance(raw, str):
        return raw.strip()
    return ""


def parse_features(raw: list) -> str:
    """features 为卖点列表，取前 3 条拼接"""
    if not raw:
        return ""
    return " ".join(str(x).strip() for x in raw[:3] if x)


def parse_categories(raw) -> str:
    if not raw:
        return ""
    flat = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, str):
            flat.append(item)
    return flat[-1] if flat else ""


def parse_image_urls(raw: list) -> list:
    if not raw:
        return []
    main_url, extra_urls = None, []
    for img in raw:
        if not isinstance(img, dict):
            continue
        url = img.get("hi_res") or img.get("large") or img.get("thumb") or ""
        if not url:
            continue
        if img.get("variant") == "MAIN" and main_url is None:
            main_url = url
        else:
            extra_urls.append(url)
    return ([main_url] if main_url else []) + extra_urls


# ─────────────────────────── clip_text 拼接 ───────────────────────

def build_clip_text(
    title: str,
    description: str,
    features: str,
    reviews: list[str],
) -> str:
    """
    按优先级拼接 clip_text，总长不超过 CLIP_TEXT_TOTAL 字符：
      1. title（截断到 TITLE_CHARS）
      2. description 或 features（截断到 META_CHARS）
      3. 最多 MAX_REVIEWS_IN_TEXT 条评论（总计截断到 MAX_REVIEW_CHARS）

    这样即使 description 为空（Amazon Baby/Beauty 常见），
    评论也能补充足够的语义信号。
    """
    parts = []

    # 1. title
    title_part = title[:TITLE_CHARS].strip()
    if title_part:
        parts.append(title_part)

    # 2. meta 文字：description 优先，否则用 features
    meta_src = description if len(description) >= len(features) else features
    meta_part = meta_src[:META_CHARS].strip()
    if meta_part:
        parts.append(meta_part)

    # 3. 评论：取前 MAX_REVIEWS_IN_TEXT 条，合并后截断
    if reviews:
        review_concat = " ".join(reviews[:MAX_REVIEWS_IN_TEXT])
        review_part   = review_concat[:MAX_REVIEW_CHARS].strip()
        if review_part:
            parts.append(review_part)

    text = " ".join(parts)
    return text[:CLIP_TEXT_TOTAL]


# ─────────────────────────── 主提取逻辑 ───────────────────────────

def extract_meta(
    meta_path: Path,
    item_ids_set: set,
    item_reviews: dict,       # {item_id: [review_str, ...]}（来自训练集）
) -> dict:
    """
    从 meta jsonl 中提取元数据，融合评论，生成 clip_text
    返回 {item_id: meta_dict}
    """
    meta_dict = {}
    found = total_in_file = 0

    with open_jsonl(meta_path) as f:
        for line in tqdm(f, desc=f"提取元数据 {meta_path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_in_file += 1
            item_id = item.get("parent_asin", "")
            if item_id not in item_ids_set:
                continue

            found += 1
            title       = item.get("title", "").strip()
            description = parse_description(item.get("description", ""))
            features    = parse_features(item.get("features", []))
            category    = parse_categories(item.get("categories", []))
            image_urls  = parse_image_urls(item.get("images", []))
            reviews     = item_reviews.get(item_id, [])

            meta_dict[item_id] = {
                "title":          title,
                "description":    description,
                "features":       item.get("features", []),
                "category":       category,
                "main_category":  item.get("main_category", ""),
                "price":          item.get("price", ""),
                "avg_rating":     item.get("average_rating", None),
                "rating_count":   item.get("rating_number", None),
                "image_urls":     image_urls,
                "main_image_url": image_urls[0] if image_urls else "",
                # ★ 供 cache_clip_features.py 直接使用的融合文本
                "clip_text":      build_clip_text(title, description, features, reviews),
                # 评论条数（方便调试时检查覆盖率）
                "n_reviews":      len(reviews),
            }

    print(f"  文件内总物品: {total_in_file:,}  |  命中: {found:,}  |  目标: {len(item_ids_set):,}")

    # 补充缺失物品的空记录
    missing = item_ids_set - set(meta_dict.keys())
    if missing:
        print(f"  ⚠️  缺少元数据的物品: {len(missing):,} 个（将使用空记录）")
        for item_id in missing:
            reviews = item_reviews.get(item_id, [])
            meta_dict[item_id] = {
                "title": "", "description": "", "features": [],
                "category": "", "main_category": "", "price": "",
                "avg_rating": None, "rating_count": None,
                "image_urls": [], "main_image_url": "",
                "clip_text": build_clip_text("", "", "", reviews),
                "n_reviews": len(reviews),
            }

    # 统计 clip_text 覆盖质量
    n_with_review_in_text = sum(
        1 for v in meta_dict.values() if v["n_reviews"] > 0
    )
    avg_clip_len = sum(len(v["clip_text"]) for v in meta_dict.values()) / max(len(meta_dict), 1)
    print(f"  clip_text 含评论物品: {n_with_review_in_text:,} / {len(meta_dict):,}"
          f"  |  平均长度: {avg_clip_len:.0f} 字符")

    return meta_dict


def build_image_url_list(meta_dict: dict) -> list:
    return [
        {"item_id": item_id, "url": meta["main_image_url"]}
        for item_id, meta in meta_dict.items()
        if meta.get("main_image_url")
    ]


# ─────────────────────────── 主流程 ───────────────────────────────

def process_dataset(dataset_name: str):
    data_dir = DATA_DIR / dataset_name

    # 读取 item2id（由 preprocess.py 生成）
    item2id_path = data_dir / "item2id.json"
    if not item2id_path.exists():
        raise FileNotFoundError(
            f"未找到 {item2id_path}，请先运行：python scripts/preprocess.py --dataset {dataset_name}"
        )
    with open(item2id_path) as f:
        item2id = json.load(f)
    item_ids_set = set(item2id.keys())

    # ★ 读取评论（由 preprocess.py 生成，仅训练集，无泄漏）
    reviews_path = data_dir / "item_reviews.json"
    if reviews_path.exists():
        with open(reviews_path, encoding="utf-8") as f:
            item_reviews = json.load(f)
        print(f"  已加载评论：{sum(len(v) for v in item_reviews.values()):,} 条"
              f"  覆盖 {len(item_reviews):,} 个物品")
    else:
        item_reviews = {}
        print(f"  ⚠️  未找到 {reviews_path}，将不使用评论文本")
        print(f"       建议先运行：python scripts/preprocess.py --dataset {dataset_name}")

    print(f"\n{'#'*55}")
    print(f"  提取元数据: {dataset_name}  ({len(item_ids_set):,} 个物品)")
    print(f"{'#'*55}")

    # 提取元数据（融合评论）
    meta_path = RAW_META_MAP[dataset_name]
    meta_dict = extract_meta(meta_path, item_ids_set, item_reviews)

    # 保存 item_meta.json
    meta_out = data_dir / "item_meta.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False)
    print(f"  ✅ 元数据已保存：{meta_out}")

    # 保存图片 URL 列表
    url_list     = build_image_url_list(meta_dict)
    url_list_out = data_dir / "image_url_list.json"
    with open(url_list_out, "w") as f:
        json.dump(url_list, f, ensure_ascii=False)

    n_with_img    = len(url_list)
    n_without_img = len(item_ids_set) - n_with_img
    print(f"  有主图 URL: {n_with_img:,}  |  无图（黑图占位）: {n_without_img:,}")
    print(f"  ✅ 图片 URL 列表已保存：{url_list_out}")
    print(f"  下一步：python scripts/download_images.py --dataset {dataset_name}")

    return meta_dict


def main():
    parser = argparse.ArgumentParser(description="提取物品元数据（含评论融合）")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["baby", "beauty_sub", "office", "sports", "all"],
    )
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

    for ds in datasets:
        process_dataset(ds)

    print("\n✅ 所有元数据提取完成！")


if __name__ == "__main__":
    main()
