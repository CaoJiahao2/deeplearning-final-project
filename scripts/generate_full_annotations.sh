#!/bin/bash
# Generate annotations for ALL training images using the best prompt version (v3).
# This script needs to be run before training Task 2 models.
#
# Prerequisite: Qwen3.5-9B model must be available at the checkpoint path.
#
# Usage:
#   bash generate_full_annotations.sh [num_images] [gpu] [prompt_version]

set -euo pipefail
cd "$(dirname "$0")"

NUM_IMAGES="${1:-2000}"       # Number of images to process
GPU="${2:-1}"                 # GPU to use
PROMPT_VERSION="${3:-v3}"    # Prompt version (v3 recommended based on comparison)

MODEL_PATH="checkpoint/Qwen3.5-9B"
DATA_DIR="data/train"
OUTPUT_DIR="annotations"

CONDA_ENV="n_sam3"
PYTHON="conda run -n $CONDA_ENV python3"

echo "============================================================"
echo "  Generating Full Annotations"
echo "============================================================"
echo "  Images:        $NUM_IMAGES"
echo "  Prompt:        $PROMPT_VERSION"
echo "  GPU:           cuda:$GPU"
echo "  Output:        $OUTPUT_DIR/train_captions_${PROMPT_VERSION}_full.jsonl"
echo "============================================================"

$PYTHON scripts/generate_annotations.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --split train \
    --num_images "$NUM_IMAGES" \
    --prompt_version "$PROMPT_VERSION" \
    --output_dir "$OUTPUT_DIR" \
    --device_map "cuda:$GPU" \
    --resume

echo ""
echo "Done. Annotations saved to $OUTPUT_DIR/train_captions_${PROMPT_VERSION}_full.jsonl"
echo "Use this file for training with:"
echo "  python train.py --annotation $OUTPUT_DIR/train_captions_${PROMPT_VERSION}_full.jsonl"
