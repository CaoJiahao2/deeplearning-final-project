# Task 2: Image Captioning Model Training

## Architecture Overview

### 1. Baseline: CNN + LSTM

```
Image (3×224×224)
    │
    ▼
┌──────────────┐
│  ResNet-50   │  Pretrained on ImageNet, fine-tune layer4
│  (Encoder)   │  Output: 2048-dim global feature
└──────┬───────┘
       │  Linear(2048 → 512) + ReLU
       ▼
  image_feat [B, 512]
       │
       ├──→ init_h, init_c  (LSTM initial hidden/cell state)
       │
       ▼
┌──────────────┐
│    LSTM      │  Input: word embedding [B, 512]
│  (Decoder)   │  Hidden: 512-dim, 1 layer
└──────┬───────┘
       │
       ▼
  Linear(512 → vocab_size) → next word prediction
```

- **Encoder**: ResNet-50 pretrained on ImageNet, extracts a 2048-dim global average-pooled feature vector
- **Decoder**: Single-layer LSTM (hidden_dim=512). Image features project to initial hidden state. Each timestep takes word embedding as input.
- **Total params**: ~28.2M (19.7M trainable with layer4 fine-tuning)

### 2. Attention: CNN + Attention + LSTM

```
Image (3×224×224)
    │
    ▼
┌──────────────┐
│  ResNet-50   │  Pretrained on ImageNet, fine-tune layer4
│  (Encoder)   │  Output: spatial features [B, 49, 2048]
└──────┬───────┘
       │  Linear(2048 → 512) + ReLU
       ▼
  spatial_feat [B, 49, 512]
       │
       ├──→ mean → init_h, init_c
       │
       ▼
┌──────────────┐
│  Bahdanau    │  score = V^T tanh(W_h·h_t + W_s·s_i)
│  Attention   │  context = softmax(scores) · spatial_feat
└──────┬───────┘
       │  context [B, 512]
       ▼
┌──────────────┐
│    LSTM      │  Input: concat(word_embed, context) [B, 1024]
│  (Decoder)   │  Hidden: 512-dim, 1 layer
└──────┬───────┘
       │
       ▼
  Linear(1024 → vocab_size) → next word prediction
```

- **Encoder**: Same ResNet-50, but outputs spatial feature maps (7×7 = 49 spatial positions, each 2048-dim → projected to 512-dim)
- **Attention**: Bahdanau (additive) attention. At each timestep, the decoder's hidden state attends to all 49 spatial positions to compute a context vector.
- **Decoder**: LSTM takes concatenation of word embedding and attention context as input. Output combines LSTM hidden state with context for prediction.
- **Total params**: ~28.2M (21.0M trainable with layer4 fine-tuning)

## File Structure

```
models/
  __init__.py         # Package exports
  vocab.py            # Vocabulary class (build, encode, decode, save/load)
  dataset.py          # CaptionDataset + collate_fn for DataLoader
  encoder.py          # CNNEncoder (ResNet-based, global + spatial features)
  decoder.py          # LSTMDecoder (baseline) + AttentionLSTMDecoder (with Bahdanau attention)
  caption_model.py    # CaptionModel (combines encoder + decoder, training + inference)
  metrics.py          # BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr

train.py              # Main training + evaluation script
run_task2.sh          # Train/evaluate single model or both
generate_full_annotations.sh  # Generate Task 1 annotations for all 2000 images
run_full_pipeline.sh  # End-to-end: annotations → train → compare
```

## Usage

### Prerequisites

1. **Generate annotations** (Task 1 must be complete):
   ```bash
   # If annotations not yet generated for all 2000 images:
   bash generate_full_annotations.sh 2000 0 v3
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
   ```

### Training

```bash
# Train baseline (CNN + LSTM)
python train.py --arch baseline --epochs 30 --gpu 0 --fp16

# Train attention model (CNN + Attention + LSTM)
python train.py --arch attention --epochs 30 --gpu 0 --fp16

# Or use the run script:
bash run_task2.sh baseline    # Train baseline
bash run_task2.sh attention   # Train attention
bash run_task2.sh both        # Train both and compare
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `baseline` | `baseline` or `attention` |
| `--backbone` | `resnet50` | `resnet50` or `resnet101` |
| `--embed_dim` | `512` | Embedding dimension |
| `--hidden_dim` | `512` | LSTM hidden dimension |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--fine_tune` | `layer4` | CNN fine-tune depth (`none`, `layer4`, `all`) |
| `--fp16` | off | Mixed precision training |
| `--gpu` | `0` | GPU device index |

### Evaluation

```bash
# Evaluate a trained model
python train.py --eval_only --arch baseline --checkpoint checkpoint/baseline_best.pth --gpu 0

# Beam search decoding
python train.py --eval_only --arch attention --checkpoint checkpoint/attention_best.pth \
    --decode_strategy beam --beam_size 5 --gpu 0
```

### Full Pipeline

```bash
# Run everything end-to-end
bash run_full_pipeline.sh 0
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU-1** | Unigram precision with brevity penalty |
| **BLEU-2** | Bigram precision with brevity penalty |
| **BLEU-3** | Trigram precision with brevity penalty |
| **BLEU-4** | 4-gram precision with brevity penalty |
| **METEOR** | Unigram matching with stemming and synonymy |
| **ROUGE-L** | Longest Common Subsequence based F1 |
| **CIDEr** | TF-IDF weighted n-gram cosine similarity |

## Output Files

After training, the following files are saved in `checkpoint/`:

- `{arch}_best.pth` — Best model by validation loss
- `{arch}_best_metric.pth` — Best model by BLEU-4 score
- `{arch}_epoch{n}.pth` — Periodic checkpoints
- `{arch}_history.json` — Training history (loss, metrics per epoch)
- `{arch}_scores.json` — Final evaluation scores
- `vocab_{arch}.json` — Vocabulary file

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 |
| Max caption length | 64 tokens |
| Embedding dim | 512 |
| LSTM hidden dim | 512 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Grad clipping | 5.0 |
| Teacher forcing | Yes (during training) |
| Decode strategy | Greedy (default), Beam search (optional) |
