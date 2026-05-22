#!/bin/bash
# scripts/run_preprocess.sh
# --------------------------
# 一键运行全部预处理步骤（Step 1 ~ 4）
#
# 使用方法：
#   bash scripts/run_preprocess.sh               # 处理全部数据集
#   bash scripts/run_preprocess.sh baby          # 仅处理 Baby
#   bash scripts/run_preprocess.sh beauty_sub    # 仅处理 Beauty 子集
#   SKIP_DOWNLOAD=1 bash scripts/run_preprocess.sh  # 跳过图片下载（网络受限时）
#
# 环境变量：
#   DATASET=all|baby|beauty_sub  (默认 all)
#   SKIP_DOWNLOAD=0|1            (默认 0)
#   IMAGE_WORKERS=8              (图片下载并发数，默认 8)

set -e  # 遇错即停

DATASET=${1:-${DATASET:-all}}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
IMAGE_WORKERS=${IMAGE_WORKERS:-8}

echo "============================================================"
echo "  MARA 数据预处理流水线"
echo "  数据集: $DATASET  |  跳过下载: $SKIP_DOWNLOAD  |  并发数: $IMAGE_WORKERS"
echo "============================================================"

# 确认原始文件存在
echo ""
echo ">>> 检查原始数据文件..."
if [ "$DATASET" = "all" ] || [ "$DATASET" = "baby" ]; then
    [ -f "data/raw/Baby_Products.jsonl" ]      || { echo "❌ 缺少 data/raw/Baby_Products.jsonl";       exit 1; }
    [ -f "data/raw/meta_Baby_Products.jsonl" ] || { echo "❌ 缺少 data/raw/meta_Baby_Products.jsonl";  exit 1; }
fi
if [ "$DATASET" = "all" ] || [ "$DATASET" = "beauty_sub" ]; then
    [ -f "data/raw/All_Beauty.jsonl" ]         || { echo "❌ 缺少 data/raw/All_Beauty.jsonl";          exit 1; }
    [ -f "data/raw/meta_All_Beauty.jsonl" ]    || { echo "❌ 缺少 data/raw/meta_All_Beauty.jsonl";     exit 1; }
fi
echo "  ✅ 原始文件检查通过"

# Step 1: 交互预处理
echo ""
echo ">>> Step 1/4: 交互数据预处理（过滤 + 划分 + ID 映射）..."
python scripts/preprocess.py --dataset "$DATASET"

# Step 2: 元数据提取
echo ""
echo ">>> Step 2/4: 提取物品元数据..."
python scripts/extract_meta.py --dataset "$DATASET"

# Step 3: 图片下载（可跳过）
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo ""
    echo ">>> Step 3/4: 跳过图片下载（SKIP_DOWNLOAD=1）"
    echo "  稍后可运行：python scripts/download_images.py --dataset $DATASET"
else
    echo ""
    echo ">>> Step 3/4: 下载商品图片（预计 1~2 小时）..."
    python scripts/download_images.py --dataset "$DATASET" --workers "$IMAGE_WORKERS"
fi

# Step 4: 验证
echo ""
echo ">>> Step 4/4: 验证预处理结果..."
python scripts/verify_data.py --dataset "$DATASET"

echo ""
echo "============================================================"
echo "  🎉 预处理流水线完成！"
echo ""
echo "  后续步骤（Day 2）："
echo "    python scripts/cache_clip_features.py --dataset baby"
echo "    python scripts/cache_fine_features.py --dataset baby"
echo "    python scripts/pretrain_lightgcn.py   --dataset baby"
echo "    python scripts/verify_cache.py        --dataset baby"
echo "============================================================"
