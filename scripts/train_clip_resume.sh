#!/bin/bash
# Resume CLIP ViT-L/14 training from epoch 60 to 120
# Learning rate reduced to 5e-4

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/turing1/anaconda3/envs/n_sam3/bin/python"
CLIP_PATH="/home/turing1/jhcao/final-project/checkpoint/clip-vit-B-14"
GPU=0

echo "============================================================"
echo "  Resume Training: CLIP ViT-L/14 + Attention LSTM"
echo "  From epoch 60 -> 120, LR reduced to 5e-4"
echo "  GPU: $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
    --arch attention \
    --backbone clip \
    --clip_path "$CLIP_PATH" \
    --norm_type clip \
    --annotation annotations/train_captions_v7.jsonl \
    --train_dir data/train \
    --val_annotation annotations/train_captions_v7.jsonl \
    --checkpoint_dir checkpoint \
    --output_dir v7_clip \
    --epochs 120 \
    --batch_size 32 \
    --lr 5e-4 \
    --gpu 0 \
    --fine_tune none \
    --max_caption_len 64 \
    --decode_strategy greedy \
    --pretrained_vocab bert-base-uncased \
    --fp16 \
    --early_stopping_patience 20 \
    --resume

echo ""
echo "  [Done] CLIP attention model -> checkpoint/v7_clip/"
echo ""
