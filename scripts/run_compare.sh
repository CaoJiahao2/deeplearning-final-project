#!/bin/bash
# ============================================================
# 5 组 Prompt 批量对比测试脚本
# 测试前 10 张图片，生成对比报告
# ============================================================

set -e

ENV_NAME="n_sam3"
MODEL_PATH="/home/turing1/jhcao/final-project/checkpoint/Qwen3.5-9B"
DATA_DIR="/mnt/jhcao/final-project/data"
OUTPUT_DIR="/mnt/jhcao/final-project/annotations"
NUM_IMAGES=10
DEVICE="cuda:1"

# 清理旧的标注文件
echo "Cleaning old annotations..."
rm -f ${OUTPUT_DIR}/*.jsonl ${OUTPUT_DIR}/*.md

# 激活环境并运行
echo "Starting 5-prompt comparison test..."
echo "  Images: ${NUM_IMAGES}"
echo "  Device: ${DEVICE}"
echo ""

conda run -n ${ENV_NAME} python scripts/generate_annotations.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --split train \
    --num_images ${NUM_IMAGES} \
    --output_dir ${OUTPUT_DIR} \
    --device_map ${DEVICE} \
    --compare

echo ""
echo "Done! Check results in: ${OUTPUT_DIR}/"
echo "  - train_captions_v1.jsonl ~ v5.jsonl  (各版本标注)"
echo "  - prompt_comparison_report.md          (对比报告)"
