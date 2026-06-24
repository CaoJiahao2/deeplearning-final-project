#!/bin/bash
# Train image captioning model with CLIP ViT-L/14 encoder (frozen)
# + Attention LSTM decoder
#
# Usage: bash scripts/train_clip.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/turing1/anaconda3/envs/n_sam3/bin/python"
CLIP_PATH="/home/turing1/jhcao/final-project/checkpoint/clip-vit-B-14"
GPU=1

echo "============================================================"
echo "  Training: CLIP ViT-L/14 + Attention LSTM"
echo "  Encoder: frozen CLIP ViT-L/14 (304M params)"
echo "  Decoder: Attention LSTM (trainable)"
echo "  GPU: $GPU"
echo "  Epochs: 60"
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
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-3 \
    --gpu 0 \
    --fine_tune none \
    --max_caption_len 64 \
    --decode_strategy greedy \
    --pretrained_vocab bert-base-uncased \
    --fp16 \
    --early_stopping_patience 15

echo ""
echo "  [Done] CLIP attention model -> checkpoint/v7_clip/"
echo ""
