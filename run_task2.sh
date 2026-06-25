#!/bin/bash
# Task 2: Image Captioning — Training and Evaluation
#
# Usage:
#   bash run_task2.sh baseline         # Train baseline
#   bash run_task2.sh attention        # Train attention
#   bash run_task2.sh both             # Train both, then compare
#   bash run_task2.sh resume-baseline  # Resume baseline
#   bash run_task2.sh resume-attention # Resume attention
#   bash run_task2.sh resume-both      # Resume both
#   bash run_task2.sh eval             # Evaluate existing checkpoints
#
# Environment: conda activate n_sam3

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="python"

# Default paths
ANNOTATION="annotations/train_captions_v7.jsonl"
TRAIN_DIR="data/train"
VAL_ANNOTATION="annotations/train_captions_v7.jsonl"
CHECKPOINT_DIR="checkpoint"
OUTPUT_DIR="v7"
GPU="${GPU:-0}"
EPOCHS=60          # 总 epoch 数
BATCH_SIZE=64
LR=1e-3
VOCAB="bert-base-uncased"  # 使用 BERT 子词分词
DECODE_STRATEGY="greedy"

MODE="${1:-both}"

# -------------------------------------------------------------------
# Helper: train a single model
# -------------------------------------------------------------------
train_model() {
    local arch="$1"
    local resume="${2:-false}"
    local out="$OUTPUT_DIR/$arch"
    local extra_args=""

    if [ "$resume" = "true" ]; then
        extra_args="--resume"
        echo "============================================================"
        echo "  Resume Training: $arch (-> epoch $EPOCHS)"
    else
        echo "============================================================"
        echo "  Training: $arch ($EPOCHS epochs)"
    fi
    echo "  Output:   $CHECKPOINT_DIR/$out/"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
        --arch "$arch" \
        --annotation "$ANNOTATION" \
        --train_dir "$TRAIN_DIR" \
        --val_annotation "$VAL_ANNOTATION" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$out" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --gpu 0 \
        --fine_tune layer4 \
        --max_caption_len 64 \
        --decode_strategy "$DECODE_STRATEGY" \
        --pretrained_vocab "$VOCAB" \
        --fp16 \
        $extra_args

    echo ""
    echo "  [Done] $arch -> $CHECKPOINT_DIR/$out/"
    echo ""
}

# -------------------------------------------------------------------
# Helper: evaluate a model
# -------------------------------------------------------------------
eval_model() {
    local arch="$1"
    local out="$OUTPUT_DIR/$arch"
    local ckpt="$CHECKPOINT_DIR/$out/${arch}_best.pth"
    echo "============================================================"
    echo "  Evaluating: $arch"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
        --eval_only \
        --arch "$arch" \
        --checkpoint "$ckpt" \
        --annotation "$ANNOTATION" \
        --train_dir "$TRAIN_DIR" \
        --val_annotation "$VAL_ANNOTATION" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$out" \
        --gpu 0 \
        --decode_strategy "$DECODE_STRATEGY" \
        --pretrained_vocab "$VOCAB" \
        --beam_size 5

    echo ""
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
case "$MODE" in
    baseline)
        train_model baseline
        ;;
    attention)
        train_model attention
        ;;
    both)
        train_model baseline
        train_model attention

        echo "============================================================"
        echo "  Comparison: baseline vs attention"
        echo "============================================================"
        for arch in baseline attention; do
            echo ""
            echo "--- $arch ---"
            if [ -f "$CHECKPOINT_DIR/$OUTPUT_DIR/$arch/${arch}_scores.json" ]; then
                cat "$CHECKPOINT_DIR/$OUTPUT_DIR/$arch/${arch}_scores.json"
            else
                echo "  (no scores file found)"
            fi
        done
        echo ""
        echo "============================================================"
        echo "  Tensorboard:"
        echo "    tensorboard --logdir $CHECKPOINT_DIR/$OUTPUT_DIR/*/tb"
        echo "============================================================"
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

        echo "============================================================"
        echo "  Comparison: baseline vs attention"
        echo "============================================================"
        for arch in baseline attention; do
            echo ""
            echo "--- $arch ---"
            if [ -f "$CHECKPOINT_DIR/$OUTPUT_DIR/$arch/${arch}_scores.json" ]; then
                cat "$CHECKPOINT_DIR/$OUTPUT_DIR/$arch/${arch}_scores.json"
            else
                echo "  (no scores file found)"
            fi
        done
        echo ""
        ;;
    eval)
        eval_model baseline
        eval_model attention
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_task2.sh {baseline|attention|both|resume-baseline|resume-attention|resume-both|eval}"
        exit 1
        ;;
esac
