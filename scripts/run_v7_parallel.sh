#!/bin/bash
# 3 卡并行生成 v7 caption，完成后合并
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/annotations"
PYTHON="python"
SCRIPT="$SCRIPT_DIR/generate_v6_caption_only.py"   # 已更新为 v7 prompt

mkdir -p "$OUTPUT_DIR"

# GPU 编号列表（用户指定 0, 1, 3 空闲）
GPUS=(0 1 3)

for SPLIT in train val; do
    echo "============================================"
    echo "  Split: $SPLIT"
    echo "============================================"

    TOTAL=$(ls "$DATA_DIR/$SPLIT"/*.jpg 2>/dev/null | wc -l)
    N=${#GPUS[@]}
    CHUNK=$(( (TOTAL + N - 1) / N ))
    echo "  Total images: $TOTAL, chunk size: $CHUNK, GPUs: ${GPUS[*]}"

    # 清理旧分片
    for g in "${GPUS[@]}"; do
        rm -f "$OUTPUT_DIR/${SPLIT}_captions_v7_shard${g}.jsonl"
    done

    PIDS=()
    for i in "${!GPUS[@]}"; do
        GPU_ID=${GPUS[$i]}
        OFFSET=$((i * CHUNK))
        COUNT=$CHUNK
        # 最后一片取剩余全部
        [ $i -eq $((N - 1)) ] && COUNT=-1

        echo "  [GPU $GPU_ID] offset=$OFFSET count=$COUNT"
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON "$SCRIPT" \
            --split "$SPLIT" \
            --gpu 0 \
            --shard_offset "$OFFSET" \
            --shard_count "$COUNT" \
            --shard_id "$GPU_ID" \
            --version v7 \
            --output_dir "$OUTPUT_DIR" &
        PIDS+=($!)
    done

    echo "  Waiting for shards ..."
    FAIL=0
    for i in "${!GPUS[@]}"; do
        wait ${PIDS[$i]} && echo "  [GPU ${GPUS[$i]}] OK" || { echo "  [GPU ${GPUS[$i]}] FAILED"; FAIL=1; }
    done
    [ $FAIL -eq 1 ] && { echo "Aborting."; exit 1; }

    # 合并 + 排序
    MERGED="$OUTPUT_DIR/${SPLIT}_captions_v7.jsonl"
    > "$MERGED"
    for g in "${GPUS[@]}"; do
        SHARD_FILE="$OUTPUT_DIR/${SPLIT}_captions_v7_shard${g}.jsonl"
        [ -f "$SHARD_FILE" ] && cat "$SHARD_FILE" >> "$MERGED"
    done

    $PYTHON -c "
import json
records = []
with open('$MERGED') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
records.sort(key=lambda r: int(r['image_id'].split('.')[0]))
with open('$MERGED', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Merged {len(records)} records -> $MERGED')
"

    # 清理分片
    for g in "${GPUS[@]}"; do
        rm -f "$OUTPUT_DIR/${SPLIT}_captions_v7_shard${g}.jsonl"
    done
    echo ""
done

echo "All done."
ls -lh "$OUTPUT_DIR"/{train,val}_captions_v7.jsonl
