#!/bin/bash
# 单张/批量图片推理脚本
#
# 用法:
#   # 单张推理
#   bash scripts/infer.sh data/val/2002.jpg
#
#   # 指定模型和显卡
#   bash scripts/infer.sh data/val/2002.jpg attention 0
#
#   # 批量推理（传入目录）
#   bash scripts/infer.sh data/val baseline 0
#
#   # 指定 checkpoint 文件
#   bash scripts/infer.sh data/val/2002.jpg baseline 0 checkpoint/v7/baseline/baseline_best_metric.pth
#   bash scripts/infer.sh data/val/2002.jpg attention 3 checkpoint/v7/attention/attention_best_metric.pth

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python"
SCRIPT="scripts/inference.py"

INPUT="${1:-data/val/2002.jpg}"
ARCH="${2:-attention}"
GPU="${3:-0}"
CKPT="${4:-}"

# 判断是单张图片还是目录
if [ -d "$INPUT" ]; then
    MODE="--image_dir $INPUT"
    echo "批量推理: $INPUT"
else
    MODE="--image $INPUT"
    echo "单张推理: $INPUT"
fi
echo "模型: $ARCH | GPU: $GPU"

# 构建推理命令
CMD="$PYTHON $SCRIPT $MODE --arch $ARCH --exp v7/$ARCH --gpu $GPU --strategy greedy"

# 如果指定了 checkpoint
if [ -n "$CKPT" ]; then
    CMD="$CMD --checkpoint $CKPT"
fi

# 执行推理
eval $CMD
