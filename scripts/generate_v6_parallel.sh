#!/bin/bash
# 并行生成 v6 标注：3 张 GPU 同时运行，最后合并结果
# 用法: bash scripts/generate_v6_parallel.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/annotations"
SCRIPT="$SCRIPT_DIR/generate_annotations.py"

mkdir -p "$OUTPUT_DIR"

# 需要处理的 split（train + val）
SPLITS=("train" "val")

for SPLIT in "${SPLITS[@]}"; do
    echo "========================================"
    echo "Processing split: $SPLIT"
    echo "========================================"

    # 统计总图片数
    TOTAL=$(ls "$DATA_DIR/$SPLIT"/*.jpg 2>/dev/null | wc -l)
    echo "Total images in $SPLIT: $TOTAL"

    # 分成 3 片
    CHUNK_SIZE=$(( (TOTAL + 2) / 3 ))

    # 临时目录存放分片结果
    TMP_DIR="$OUTPUT_DIR/tmp_${SPLIT}_v6"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    # 启动 3 个并行进程，分别使用 cuda:0, cuda:1, cuda:2
    PIDS=()
    for SHARD in 0 1 2; do
        START=$((SHARD * CHUNK_SIZE))
        COUNT=$CHUNK_SIZE
        # 最后一个分片处理剩余所有
        if [ $SHARD -eq 2 ]; then
            COUNT=-1
        fi

        CUDA_DEV="cuda:$SHARD"
        OUTPUT_FILE="$TMP_DIR/${SPLIT}_captions_v6_shard${SHARD}.jsonl"

        echo "  [GPU $SHARD] images offset=$START, count=$COUNT -> $OUTPUT_FILE"

        CUDA_VISIBLE_DEVICES=$SHARD python "$SCRIPT" \
            --split "$SPLIT" \
            --prompt_version v6 \
            --num_images "$COUNT" \
            --output_dir "$TMP_DIR" \
            --device_map "cuda:0" \
            --shard_offset "$START" \
            --shard_id "$SHARD" &
        PIDS+=($!)
    done

    # 等待所有分片完成
    echo "  Waiting for all 3 shards to finish..."
    FAILED=0
    for i in 0 1 2; do
        if wait ${PIDS[$i]}; then
            echo "  [GPU $i] Done."
        else
            echo "  [GPU $i] FAILED!"
            FAILED=1
        fi
    done

    if [ $FAILED -eq 1 ]; then
        echo "  ERROR: Some shards failed. Check logs."
        exit 1
    fi

    # 合并分片
    MERGED="$OUTPUT_DIR/${SPLIT}_captions_v6.jsonl"
    > "$MERGED"
    for SHARD in 0 1 2; do
        SHARD_FILE="$TMP_DIR/${SPLIT}_captions_v6_shard${SHARD}.jsonl"
        if [ -f "$SHARD_FILE" ]; then
            cat "$SHARD_FILE" >> "$MERGED"
        fi
    done

    # 按 image_id 排序（按数字排序）
    python3 -c "
import json, sys
records = []
with open('$MERGED') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
records.sort(key=lambda r: int(r['image_id'].split('.')[0]))
with open('$MERGED', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Merged {len(records)} records -> $MERGED')
"

    # 清理临时目录
    rm -rf "$TMP_DIR"
    echo ""
done

echo "All done!"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/train_captions_v6.jsonl "$OUTPUT_DIR"/val_captions_v6.jsonl 2>/dev/null
