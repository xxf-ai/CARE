"""
scripts/cache_clip_features.py
-------------------------------
Day 2 · Step 1 — CLIP 特征离线缓存

输出（每个数据集）：
  cache/{dataset}/clip_cls.npy       [N, 768]        CLS token → τ_coarse
  cache/{dataset}/clip_patches.npy   [N, 256, 1024]  Patch tokens → τ_region
  cache/{dataset}/clip_text.npy      [N, 768]        文本编码 → τ_text

维度说明（CLIP ViT-L/14）：
  视觉 transformer 宽度 = 1024，输出 CLS 经 visual.proj 投影到 768
  Patch token 无投影矩阵，保留原始 1024 维供 τ_region 注意力使用
  文本编码器经 text_projection 输出 768 维

效率优化：
  - torch DataLoader 多进程预读图像 (num_workers=4, pin_memory)
  - FP16 推理，显存减半
  - visual 和 text 分开批次，互不阻塞
  - 断点续传：若中途中断，重跑会跳过已写入的文件

预计耗时（RTX 3090, Baby 7050 张）：
  视觉特征 (cls + patches)：~12 min
  文本特征：~2 min
  合计：~15 min

使用：
  python scripts/cache_clip_features.py --dataset baby
  python scripts/cache_clip_features.py --dataset beauty_sub
  python scripts/cache_clip_features.py --dataset all
  python scripts/cache_clip_features.py --dataset baby --batch_size 32  # 显存不足时
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"

# Check GPU decode availability
_HAS_GPU_DECODE = hasattr(torchvision.io, 'decode_jpeg') and torch.cuda.is_available()


# ─────────────────────────── Dataset ──────────────────────────────

class RawImageDataset(Dataset):
    """Load raw JPEG bytes from disk — fast, minimal CPU, zero PIL overhead.

    GPU decode happens in collate_fn via torchvision.io.decode_jpeg.
    """

    def __init__(self, item_ids: list, image_dir: Path):
        self.item_ids  = item_ids
        self.image_dir = image_dir
        # Precompute all valid paths + a fallback black image
        self.paths = [image_dir / f"{iid}.jpg" for iid in item_ids]
        # A valid JPEG black placeholder for missing images
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (0, 0, 0)).save(buf, "JPEG")
        self._black_jpeg = buf.getvalue()

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            return path.read_bytes()
        except Exception:
            return self._black_jpeg


# CLIP normalization constants (ViT-L/14)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _collate_decode_gpu(raw_list, device):
    """GPU-decode a batch of JPEG bytes → [B, 3, 224, 224] float16, CLIP-normalized."""
    tensors = []
    mean = _CLIP_MEAN.to(device)
    std  = _CLIP_STD.to(device)

    for raw in raw_list:
        try:
            data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            if _HAS_GPU_DECODE:
                img = torchvision.io.decode_jpeg(data, device=device)
            else:
                img = torchvision.io.decode_jpeg(data)
                img = img.to(device, non_blocking=True)
        except Exception:
            img = torch.zeros(3, 224, 224, device=device, dtype=torch.uint8)

        img = img.float() / 255.0    # [0, 1]
        # Resize if not exact 224
        if img.shape[-2:] != (224, 224):
            img = F.interpolate(img.unsqueeze(0), (224, 224),
                               mode='bilinear', antialias=True).squeeze(0)
        # CLIP normalize
        img = (img - mean[:, None, None]) / std[:, None, None]
        tensors.append(img.half())

    return torch.stack(tensors)


# ─────────────────────────── 视觉特征提取 ─────────────────────────

def _extract_visual(
    model,
    loader: DataLoader,
    device: torch.device,
    n_items: int,
    visual_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    提取 CLS token（经 ln_post + proj → 768）和
    Patch tokens（经 ln_post，保持 visual_width 维）

    返回 (cls_array [N,768], patches_array [N,256,visual_width])
    """
    proj = model.visual.proj  # [visual_width, 768]

    cls_list    = []
    patch_list  = []

    for imgs in tqdm(loader, desc="  视觉特征"):
        imgs = imgs.to(device, non_blocking=True).half()

        with torch.no_grad():
            # ── patch embedding ──────────────────────────────────
            # conv1: [B, visual_width, grid, grid]
            x = model.visual.conv1(imgs)                   # [B, W, G, G]
            x = x.reshape(x.shape[0], x.shape[1], -1)     # [B, W, G*G]
            x = x.permute(0, 2, 1)                         # [B, N_patch, W]

            # prepend CLS token
            cls_tok = model.visual.class_embedding.to(x.dtype) \
                           .unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tok, x], dim=1)             # [B, N_patch+1, W]

            # positional embedding
            x = x + model.visual.positional_embedding.to(x.dtype)
            x = model.visual.ln_pre(x)

            # transformer (seq_first)
            x = x.permute(1, 0, 2)                         # [N, B, W]
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)                         # [B, N, W]

            # post layer norm
            x = model.visual.ln_post(x)                    # [B, N, W]

            cls_feat   = x[:, 0, :]                        # [B, W]
            patch_feat = x[:, 1:, :]                       # [B, 256, W]

            # project CLS to output dim (768)
            if proj is not None:
                cls_feat = cls_feat @ proj.to(cls_feat.dtype)  # [B, 768]

        cls_list.append(cls_feat.float().cpu())
        # patch_feat NOT collected — saves ~2GB RAM per batch

    cls_arr = torch.cat(cls_list, dim=0).numpy()   # [N, 768]
    return cls_arr, None  # patches no longer needed


# ─────────────────────────── 文本特征提取 ─────────────────────────

def _extract_text(
    model,
    item_ids: list,
    item_meta: dict,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    使用 item_meta[iid]['clip_text'] 字段（已融合评论，由 extract_meta.py 生成）
    批量编码文本特征，输出 [N, 768]
    """
    import clip as clip_lib

    text_list = []
    for i in tqdm(range(0, len(item_ids), batch_size), desc="  文本特征"):
        batch_ids = item_ids[i : i + batch_size]
        texts = []
        for iid in batch_ids:
            meta = item_meta.get(iid, {})
            # 优先用已融合评论的 clip_text 字段；降级用 title
            text = meta.get("clip_text") or meta.get("title") or ""
            texts.append(text[:256])

        tokens = clip_lib.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)  # [B, 768]
        text_list.append(feat.float().cpu())

    return torch.cat(text_list, dim=0).numpy()  # [N, 768]


# ─────────────────────────── 主流程 ───────────────────────────────

def cache_clip_features(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    device_str: str = "cuda",
):
    import clip as clip_lib

    cache_dir = CACHE_DIR / dataset_name
    data_dir  = DATA_DIR  / dataset_name
    image_dir = data_dir / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # CARE 只需要 clip_text.npy，clip_cls.npy 仅在有图片时提取
    need_text   = not (cache_dir / "clip_text.npy").exists()
    need_visual = (not (cache_dir / "clip_cls.npy").exists()) and image_dir.exists()
    if not need_text and not need_visual:
        print(f"  ✅ {dataset_name}: CARE 特征已存在，跳过")
        return

    print(f"\n{'#'*58}")
    print(f"  CLIP 特征缓存: {dataset_name}")
    print(f"{'#'*58}")

    # 加载资源（仅当需要提取时加载 CLIP 模型）
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}  |  batch_size: {batch_size}  |  workers: {num_workers}")

    print("  加载 CLIP ViT-L/14 ...")
    model, preprocess = clip_lib.load("ViT-L/14", device=device)
    model.eval()

    visual_width = model.visual.conv1.out_channels  # ViT-L/14 = 1024

    # 加载 item_ids（由 preprocess.py 生成）
    with open(cache_dir / "item_ids.json") as f:
        item_ids = json.load(f)
    n_items = len(item_ids)
    print(f"  物品数: {n_items:,}")

    # ── 视觉特征（GPU 解码，零 PIL / 零 fork）─────────────────
    if need_visual:
        dataset = RawImageDataset(item_ids, image_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: _collate_decode_gpu(batch, device),
        )
        print(f"    GPU decode: {'✅ nvJPEG' if _HAS_GPU_DECODE else '⚠ CPU fallback'}")
        cls_arr, patches_arr = _extract_visual(model, loader, device, n_items, visual_width)
        np.save(cache_dir / "clip_cls.npy", cls_arr)
        print(f"  clip_cls.npy     {cls_arr.shape}  →  {cache_dir/'clip_cls.npy'}")
        del cls_arr, patches_arr
    else:
        print(f"  ⚠️  跳过视觉特征（图片目录不存在或已缓存）")

    # ── 文本特征（始终提取，不依赖图片）────────────────────────
    if need_text:
        with open(data_dir / "item_meta.json", encoding="utf-8") as f:
            item_meta = json.load(f)

        text_arr = _extract_text(model, item_ids, item_meta, device, batch_size=512)
        np.save(cache_dir / "clip_text.npy", text_arr)
        print(f"  clip_text.npy    {text_arr.shape}  →  {cache_dir/'clip_text.npy'}")
        del text_arr

    print(f"  ✅ {dataset_name} CLIP 缓存完成")


def main():
    parser = argparse.ArgumentParser(description="缓存 CLIP 视觉 + 文本特征")
    parser.add_argument("--dataset",     default="all", choices=["baby", "beauty_sub", "office", "sports", "yelp", "all"])
    parser.add_argument("--batch_size",  type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("baby",       "all"): datasets.append("baby")
    if args.dataset in ("beauty_sub", "all"): datasets.append("beauty_sub")
    if args.dataset in ("office",     "all"): datasets.append("office")
    if args.dataset in ("sports",     "all"): datasets.append("sports")
    if args.dataset in ("yelp",       "all"): datasets.append("yelp")

    for ds in datasets:
        cache_clip_features(ds, args.batch_size, args.num_workers, args.device)

    print("\n✅ CLIP 特征缓存全部完成")
    print("   下一步：python scripts/pretrain_lightgcn.py --dataset all")


if __name__ == "__main__":
    main()
