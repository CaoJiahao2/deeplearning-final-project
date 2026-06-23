#!/bin/bash
# Task 2: 使用 v7 caption 训练 baseline 和 attention 模型
#
# 用法:
#   bash scripts/train_v7.sh baseline         # 从头训练 baseline
#   bash scripts/train_v7.sh attention        # 从头训练 attention
#   bash scripts/train_v7.sh both             # 训练两者并对比
#   bash scripts/train_v7.sh resume-baseline  # 续训 baseline
#   bash scripts/train_v7.sh resume-attention # 续训 attention
#   bash scripts/train_v7.sh resume-both      # 续训两者
#   bash scripts/train_v7.sh eval             # 评估已有 checkpoint
#
# 输出目录: checkpoint/v7/<arch>/
#   ├── <arch>_best.pth          # 最优 loss 模型
#   ├── <arch>_best_metric.pth   # 最优 BLEU-4 模型
#   ├── <arch>_history.json      # 训练历史
#   ├── <arch>_scores.json       # 评估指标
#   ├── <arch>.log               # 训练日志
#   ├── tb/                      # Tensorboard 日志
#   └── vocab_<arch>.json        # 词表

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/turing1/anaconda3/envs/n_sam3/bin/python"
SCRIPT="train.py"

# ── 默认参数 ──
ANNOTATION="annotations/train_captions_v7.jsonl"
TRAIN_DIR="data/train"
VAL_DIR="data/val"
GPU="${GPU:-0}"
EPOCHS=60          # 总 epoch 数（续训时从上次结束继续到此数）
BATCH_SIZE=64
LR=1e-3
ARCH="${1:-both}"

# ── 训练函数 ──
train_model() {
    local arch="$1"
    local resume="${2:-false}"
    local output_dir="v7/$arch"

    local extra_args=""
    if [ "$resume" = "true" ]; then
        extra_args="--resume"
        echo "============================================================"
        echo "  Resume Training: $arch (-> epoch $EPOCHS)"
    else
        echo "============================================================"
        echo "  Training: $arch ($EPOCHS epochs)"
    fi
    echo "  Annotation: $ANNOTATION"
    echo "  Output: checkpoint/$output_dir/"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
        --arch "$arch" \
        --annotation "$ANNOTATION" \
        --train_dir "$TRAIN_DIR" \
        --val_annotation "$ANNOTATION" \
        --checkpoint_dir "checkpoint" \
        --output_dir "$output_dir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --gpu 0 \
        --fine_tune layer4 \
        --max_caption_len 64 \
        --decode_strategy greedy \
        --fp16 \
        $extra_args

    echo ""
    echo "  [Done] $arch -> checkpoint/$output_dir/"
    echo ""
}

# ── 评估函数 ──
eval_model() {
    local arch="$1"
    local output_dir="v7/$arch"
    local ckpt="checkpoint/$output_dir/${arch}_best.pth"

    echo "============================================================"
    echo "  Evaluating: $arch"
    echo "  Checkpoint: $ckpt"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
        --eval_only \
        --arch "$arch" \
        --checkpoint "$ckpt" \
        --annotation "$ANNOTATION" \
        --train_dir "$TRAIN_DIR" \
        --val_annotation "$ANNOTATION" \
        --checkpoint_dir "checkpoint" \
        --output_dir "$output_dir" \
        --gpu 0 \
        --decode_strategy greedy \
        --beam_size 5

    echo ""
}

# ── 对比函数 ──
compare_models() {
    echo "============================================================"
    echo "  Comparison: baseline vs attention"
    echo "============================================================"
    for arch in baseline attention; do
        echo ""
        echo "--- $arch ---"
        if [ -f "checkpoint/v7/$arch/${arch}_scores.json" ]; then
            cat "checkpoint/v7/$arch/${arch}_scores.json"
        else
            echo "  (no scores file found)"
        fi
    done
    echo ""
    echo "============================================================"
    echo "  Tensorboard:"
    echo "    tensorboard --logdir checkpoint/v7/*/tb"
    echo "============================================================"
}

# ── 主逻辑 ──
case "$ARCH" in
    baseline)
        train_model baseline
        ;;
    attention)
        train_model attention
        ;;
    both)
        train_model baseline
        train_model attention
        compare_models
        ;;
    resume-baseline)
        train_model baseline true
        ;;
    resume-attention)
        train_model attention true
        ;;
    resume-both)
        train_model baseline true
        train_model attention true
        compare_models
        ;;
    eval)
        eval_model baseline
        eval_model attention
        ;;
    *)
        echo "Usage: bash scripts/train_v7.sh {baseline|attention|both|resume-baseline|resume-attention|resume-both|eval}"
        exit 1
        ;;
esac
