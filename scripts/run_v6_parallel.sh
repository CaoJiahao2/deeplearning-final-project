#!/bin/bash
# 3 卡并行生成 v6 caption（仅 caption），完成后合并
# 用法: conda activate n_sam3 && bash scripts/run_v6_parallel.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/annotations"
PYTHON="/home/turing1/anaconda3/envs/n_sam3/bin/python"
SCRIPT="$SCRIPT_DIR/generate_v6_caption_only.py"

mkdir -p "$OUTPUT_DIR"

for SPLIT in train val; do
    echo "============================================"
    echo "  Split: $SPLIT"
    echo "============================================"

    TOTAL=$(ls "$DATA_DIR/$SPLIT"/*.jpg 2>/dev/null | wc -l)
    CHUNK=$(( (TOTAL + 2) / 3 ))
    echo "  Total images: $TOTAL, chunk size: $CHUNK"

    # 清理旧分片
    rm -f "$OUTPUT_DIR/${SPLIT}_captions_v6_shard"*.jsonl

    PIDS=()
    for SHARD in 0 1 2; do
        OFFSET=$((SHARD * CHUNK))
        COUNT=$CHUNK
        [ $SHARD -eq 2 ] && COUNT=-1   # 最后一片取剩余全部

        echo "  [GPU $SHARD] offset=$OFFSET count=$COUNT"
        CUDA_VISIBLE_DEVICES=$SHARD $PYTHON "$SCRIPT" \
            --split "$SPLIT" \
            --gpu 0 \
            --shard_offset "$OFFSET" \
            --shard_count "$COUNT" \
            --shard_id "$SHARD" \
            --output_dir "$OUTPUT_DIR" \
            --resume &
        PIDS+=($!)
    done

    echo "  Waiting for shards ..."
    FAIL=0
    for i in 0 1 2; do
        wait ${PIDS[$i]} && echo "  [GPU $i] OK" || { echo "  [GPU $i] FAILED"; FAIL=1; }
    done
    [ $FAIL -eq 1 ] && { echo "Aborting."; exit 1; }

    # 合并 + 按 image_id 数字排序
    MERGED="$OUTPUT_DIR/${SPLIT}_captions_v6.jsonl"
    cat "$OUTPUT_DIR/${SPLIT}_captions_v6_shard"{0,1,2}.jsonl > "$MERGED.tmp"
    $PYTHON -c "
import json
records = []
with open('$MERGED.tmp') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
records.sort(key=lambda r: int(r['image_id'].split('.')[0]))
with open('$MERGED', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Merged {len(records)} records -> $MERGED')
"
    rm -f "$MERGED.tmp" "$OUTPUT_DIR/${SPLIT}_captions_v6_shard"*.jsonl
    echo ""
done

echo "All done."
ls -lh "$OUTPUT_DIR"/{train,val}_captions_v6.jsonl
