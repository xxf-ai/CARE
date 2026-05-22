#!/bin/bash
# scripts/run_cache.sh
# ---------------------
# Day 2 一键运行全部特征预计算（可后台运行，全程无需人工干预）
#
# 使用：
#   bash scripts/run_cache.sh                        # 全部数据集，全部步骤
#   bash scripts/run_cache.sh baby                   # 仅 baby
#   SKIP_FINE=1 bash scripts/run_cache.sh            # 跳过 Qwen-VL（模型未下载时）
#   VLM=internvl2 bash scripts/run_cache.sh          # 用 InternVL2-2B 替代 Qwen-VL
#
# 后台运行并保存日志：
#   nohup bash scripts/run_cache.sh > logs/cache.log 2>&1 &
#   tail -f logs/cache.log

set -e

DATASET=${1:-${DATASET:-all}}
SKIP_FINE=${SKIP_FINE:-0}
VLM=${VLM:-qwen}

mkdir -p logs

echo "============================================================"
echo "  MARA 特征预计算流水线 (Day 2)"
echo "  数据集: $DATASET  |  VLM: $VLM  |  跳过细粒度: $SKIP_FINE"
echo "  开始时间: $(date)"
echo "============================================================"

# ── Step 1：CLIP 特征（~15 min，可双卡并行）─────────────────────
echo ""
echo ">>> Step 1/4: CLIP 视觉 + 文本特征 ..."
if [ "$DATASET" = "all" ]; then
    # 两个数据集并行（同一 GPU，显存足够时）
    python scripts/cache_clip_features.py --dataset baby       2>&1 | tee -a logs/cache.log
    python scripts/cache_clip_features.py --dataset beauty_sub 2>&1 | tee -a logs/cache.log
else
    python scripts/cache_clip_features.py --dataset "$DATASET" 2>&1 | tee -a logs/cache.log
fi
echo "  ✅ Step 1 完成 ($(date))"

# ── Step 2：LightGCN 预训练（~20-30 min）────────────────────────
echo ""
echo ">>> Step 2/4: LightGCN 协同嵌入预训练 ..."
if [ "$DATASET" = "all" ]; then
    python scripts/pretrain_lightgcn.py --dataset baby       2>&1 | tee -a logs/cache.log
    python scripts/pretrain_lightgcn.py --dataset beauty_sub 2>&1 | tee -a logs/cache.log
else
    python scripts/pretrain_lightgcn.py --dataset "$DATASET" 2>&1 | tee -a logs/cache.log
fi
echo "  ✅ Step 2 完成 ($(date))"

# ── Step 3：细粒度属性缓存（~3-5h，带断点续传）──────────────────
if [ "$SKIP_FINE" = "1" ]; then
    echo ""
    echo ">>> Step 3/4: 跳过 Qwen-VL 细粒度特征 (SKIP_FINE=1)"
    echo "  稍后可运行：python scripts/cache_fine_features.py --dataset $DATASET"
else
    echo ""
    echo ">>> Step 3/4: Qwen-VL 细粒度属性缓存（耗时最长，请耐心等待）..."
    if [ "$DATASET" = "all" ]; then
        python scripts/cache_fine_features.py --dataset baby       --model "$VLM" 2>&1 | tee -a logs/cache.log
        python scripts/cache_fine_features.py --dataset beauty_sub --model "$VLM" 2>&1 | tee -a logs/cache.log
    else
        python scripts/cache_fine_features.py --dataset "$DATASET" --model "$VLM" 2>&1 | tee -a logs/cache.log
    fi
    echo "  ✅ Step 3 完成 ($(date))"
fi

# ── Step 4：验证 ─────────────────────────────────────────────────
echo ""
echo ">>> Step 4/4: 验证缓存完整性 ..."
python scripts/verify_cache.py --dataset "$DATASET" 2>&1 | tee -a logs/cache.log
echo "  ✅ Step 4 完成 ($(date))"

echo ""
echo "============================================================"
echo "  🎉 Day 2 特征预计算全部完成！"
echo "  完成时间: $(date)"
echo "  日志保存在: logs/cache.log"
echo ""
echo "  缓存文件位置："
echo "    cache/baby/       (7 个 npy + item_ids.json + user_ids.json)"
echo "    cache/beauty_sub/ (结构同上)"
echo ""
echo "  下一步 (Day 3)："
echo "    python models/mgvt/coarse_tool.py   # 单元测试"
echo "    python models/mgvt/region_tool.py"
echo "============================================================"
