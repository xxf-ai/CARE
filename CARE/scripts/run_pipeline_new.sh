#!/bin/bash
# scripts/run_pipeline_new.sh
# 一键处理 Office_Products 和 Sports_and_Outdoors 数据集
# VARA 架构：仅需 clip_text.npy + item_emb.npy + user_emb.npy
#
# 用法：bash scripts/run_pipeline_new.sh
#       bash scripts/run_pipeline_new.sh office  # 只处理单个

set -e

DATASET="${1:-all}"
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "============================================"
echo "  VARA 数据预处理流水线"
echo "  数据集: $DATASET"
echo "  输出: clip_text.npy + LightGCN embeddings"
echo "============================================"

# Step 1: 交互数据预处理 (K-core + Leave-One-Out)
echo ""
echo "▶ Step 1/4: 交互数据预处理"
python scripts/preprocess.py --dataset "$DATASET"

# Step 2: 元数据提取 + clip_text 拼接
echo ""
echo "▶ Step 2/4: 元数据提取"
python scripts/extract_meta.py --dataset "$DATASET"

# Step 3: CLIP 文本特征缓存（无需图片，GPU加速）
echo ""
echo "▶ Step 3/4: CLIP 文本特征缓存"
python scripts/cache_clip_features.py --dataset "$DATASET"

# Step 4: LightGCN 协同嵌入预训练
echo ""
echo "▶ Step 4/4: LightGCN 预训练"
python scripts/pretrain_lightgcn.py --dataset "$DATASET"

echo ""
echo "✅ 流水线完成！"
echo ""
echo "缓存输出文件（每个数据集）："
echo "  cache/{dataset}/clip_text.npy   — CLIP 文本语义特征"
echo "  cache/{dataset}/item_emb.npy    — LightGCN 协同物品嵌入"
echo "  cache/{dataset}/user_emb.npy    — LightGCN 协同用户嵌入"
echo "  cache/{dataset}/item_ids.json   — 物品 ID 列表"
echo "  cache/{dataset}/user_ids.json   — 用户 ID 列表"
