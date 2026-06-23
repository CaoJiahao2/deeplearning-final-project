#!/bin/bash
# Full pipeline: Generate annotations → Train both models → Compare
#
# Usage:
#   bash run_full_pipeline.sh [gpu]
#
# Steps:
#   1. Generate annotations for all 2000 training images (Task 1)
#   2. Train baseline model (CNN + LSTM)
#   3. Train attention model (CNN + Attention + LSTM)
#   4. Compare results

set -euo pipefail
cd "$(dirname "$0")"

GPU="${1:-0}"
ANNOTATION="annotations/train_captions_v3_full.jsonl"

echo "============================================================"
echo "  Full Pipeline: Annotation → Train → Evaluate"
echo "============================================================"
echo "  GPU: $GPU"
echo ""

# Step 1: Generate annotations (if not already done)
if [ ! -f "$ANNOTATION" ]; then
    echo "[Step 1] Generating annotations for all 2000 images..."
    bash generate_full_annotations.sh 2000 "$GPU" v3
else
    echo "[Step 1] Annotations already exist at $ANNOTATION, skipping."
fi

echo ""

# Step 2: Train baseline
echo "[Step 2] Training baseline (CNN + LSTM)..."
bash run_task2.sh baseline

# Step 3: Train attention
echo "[Step 3] Training attention (CNN + Attention + LSTM)..."
bash run_task2.sh attention

# Step 4: Compare
echo ""
echo "============================================================"
echo "  Final Comparison"
echo "============================================================"
for arch in baseline attention; do
    echo ""
    echo "--- $arch ---"
    if [ -f "checkpoint/${arch}_scores.json" ]; then
        cat "checkpoint/${arch}_scores.json"
    else
        echo "  (no scores file)"
    fi
done
echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
