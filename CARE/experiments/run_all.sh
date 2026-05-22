#!/bin/bash
# experiments/run_all.sh — 一键运行全部实验（VARA edition）
#
# 用法（在项目根目录执行）：
#   bash experiments/run_all.sh baby            # 全部步骤
#   bash experiments/run_all.sh office          # Office_Products
#   bash experiments/run_all.sh sports          # Sports_and_Outdoors
#   bash experiments/run_all.sh baby baselines  # 只跑基线
#   bash experiments/run_all.sh baby ablations  # 只跑消融
#   bash experiments/run_all.sh baby tables     # 只生成表格
#
# 日志目录：experiments/logs/
#   baselines_baby_YYYYMMDD_HHMMSS.log
#   ablations_baby_YYYYMMDD_HHMMSS.log
#   alpha_baby_YYYYMMDD_HHMMSS.log
#   tables_baby_YYYYMMDD_HHMMSS.log
#   run_all_summary.log  ← 每次运行的总结追加到此文件

set -e   # 任意步骤失败则停止

DATASET="${1:-baby}"
STEP="${2:-all}"
SEEDS="42 123 456"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJ_ROOT/experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/run_all_summary.log"

# ── 颜色输出 ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log_step() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ▶ $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶ $1" >> "$SUMMARY_LOG"
}

log_done() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1  (耗时: $2)${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1  (耗时: $2)" >> "$SUMMARY_LOG"
}

log_fail() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1 失败！详见日志: $2${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1 失败！详见日志: $2" >> "$SUMMARY_LOG"
}

# ── 写入本次运行头部信息 ─────────────────────────────────────────────
echo "" >> "$SUMMARY_LOG"
echo "================================================================" >> "$SUMMARY_LOG"
echo "  新一轮实验  数据集=$DATASET  步骤=$STEP  时间=$TIMESTAMP" >> "$SUMMARY_LOG"
echo "================================================================" >> "$SUMMARY_LOG"

cd "$PROJ_ROOT"

# ── Step 0: 先运行主模型 test 集评估 ──────────────────────────────────
run_evaluate() {
    local LOG="$LOG_DIR/evaluate_${DATASET}_${TIMESTAMP}.log"
    log_step "Step 0: 主模型 Test 集评估"
    local T0=$SECONDS

    OMP_NUM_THREADS=1 python evaluate.py \
        --dataset "$DATASET" --seeds $SEEDS \
        2>&1 | tee "$LOG"

    local EXIT=${PIPESTATUS[0]}
    local ELAPSED=$((SECONDS - T0))
    if [ $EXIT -eq 0 ]; then
        log_done "主模型评估" "${ELAPSED}s"
        echo "  日志: $LOG" >> "$SUMMARY_LOG"
    else
        log_fail "主模型评估" "$LOG"; return 1
    fi
}

# ── Step 1: 基线实验 ──────────────────────────────────────────────────
run_baselines() {
    local LOG="$LOG_DIR/baselines_${DATASET}_${TIMESTAMP}.log"
    log_step "Step 1: 基线对比实验"
    local T0=$SECONDS

    OMP_NUM_THREADS=1 python experiments/baselines/run_baselines.py \
        --dataset "$DATASET" --seeds $SEEDS \
        2>&1 | tee "$LOG"

    local EXIT=${PIPESTATUS[0]}
    local ELAPSED=$((SECONDS - T0))
    if [ $EXIT -eq 0 ]; then
        log_done "基线实验" "${ELAPSED}s"
        echo "  日志: $LOG" >> "$SUMMARY_LOG"
        echo "  结果: experiments/baselines/results/${DATASET}_baselines.json" >> "$SUMMARY_LOG"
    else
        log_fail "基线实验" "$LOG"; return 1
    fi
}

# ── Step 2: 消融实验 ──────────────────────────────────────────────────
run_ablations() {
    local LOG="$LOG_DIR/ablations_${DATASET}_${TIMESTAMP}.log"
    log_step "Step 2: 消融实验（A1~A6）"
    local T0=$SECONDS

    OMP_NUM_THREADS=1 python experiments/ablations/run_ablation.py \
        --dataset "$DATASET" --seeds $SEEDS \
        2>&1 | tee "$LOG"

    local EXIT=${PIPESTATUS[0]}
    local ELAPSED=$((SECONDS - T0))
    if [ $EXIT -eq 0 ]; then
        log_done "消融实验" "${ELAPSED}s"
        echo "  日志: $LOG" >> "$SUMMARY_LOG"
        echo "  结果: experiments/ablations/results/${DATASET}_ablations.json" >> "$SUMMARY_LOG"
    else
        log_fail "消融实验" "$LOG"; return 1
    fi
}

# ── Step 3: 粒度权重可视化 ────────────────────────────────────────────
run_alpha() {
    local LOG="$LOG_DIR/alpha_${DATASET}_${TIMESTAMP}.log"
    log_step "Step 3: Alpha 权重可视化"
    local T0=$SECONDS

    for SEED in 42 123 456; do
        CKPT="checkpoints/${DATASET}/seed${SEED}/best_model.pt"
        if [ -f "$CKPT" ]; then
            OMP_NUM_THREADS=1 python experiments/analysis/alpha_visualize.py \
                --dataset "$DATASET" --seed $SEED --ckpt "$CKPT" \
                2>&1 | tee -a "$LOG"
        else
            echo "  [警告] 检查点不存在: $CKPT" | tee -a "$LOG"
        fi
    done

    local ELAPSED=$((SECONDS - T0))
    log_done "Alpha 可视化" "${ELAPSED}s"
    echo "  日志: $LOG" >> "$SUMMARY_LOG"
    echo "  图片: experiments/analysis/figures/" >> "$SUMMARY_LOG"
}

# ── Step 4: 生成论文表格 ──────────────────────────────────────────────
run_tables() {
    local LOG="$LOG_DIR/tables_${DATASET}_${TIMESTAMP}.log"
    log_step "Step 4: 生成论文表格"
    local T0=$SECONDS

    python experiments/analysis/make_tables.py \
        --dataset "$DATASET" \
        2>&1 | tee "$LOG"

    local EXIT=${PIPESTATUS[0]}
    local ELAPSED=$((SECONDS - T0))
    if [ $EXIT -eq 0 ]; then
        log_done "论文表格" "${ELAPSED}s"
        echo "  日志: $LOG" >> "$SUMMARY_LOG"
    else
        log_fail "论文表格" "$LOG"; return 1
    fi
}

# ── 根据参数决定运行哪些步骤 ──────────────────────────────────────────
case "$STEP" in
    all)
        run_evaluate
        run_baselines
        run_ablations
        run_alpha
        run_tables
        ;;
    evaluate)   run_evaluate ;;
    baselines)  run_baselines ;;
    ablations)  run_ablations ;;
    alpha)      run_alpha ;;
    tables)     run_tables ;;
    *)
        echo "未知步骤: $STEP"
        echo "可选: all | evaluate | baselines | ablations | alpha | tables"
        exit 1
        ;;
esac

# ── 最终汇总 ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  全部完成！汇总日志: $SUMMARY_LOG${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "  查看基线结果:"
echo "    cat experiments/baselines/results/${DATASET}_baselines.json"
echo "  查看消融结果:"
echo "    cat experiments/ablations/results/${DATASET}_ablations.json"
echo "  查看汇总日志:"
echo "    cat $SUMMARY_LOG"
