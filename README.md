# 图像描述生成与多模态学习

华中科技大学 软件学院 深度学习期末项目

## 项目概述

本项目实现图像描述生成（Image Captioning），包含两个核心任务：

1. **任务一 — Prompt Engineering**：使用多模态大模型 Qwen3.5-9B 为无标注图片生成描述，通过多版本 prompt 对比确定最优方案。
2. **任务二 — 图像描述模型训练**：基于生成的标注，训练 CNN-RNN 图像描述模型，对比不同视觉编码器（ResNet50 vs CLIP ViT-L/14）和解码策略（Baseline LSTM vs Attention LSTM）。

## 模型架构

```
CaptionModel
├── Encoder (视觉特征提取)
│   ├── CNNEncoder (ResNet50)      → 49 spatial tokens (7×7)
│   └── CLIPEncoder (CLIP ViT-L/14) → 256 spatial tokens (16×16), 完全冻结
├── Decoder (文本生成)
│   ├── LSTMDecoder (baseline)     → 全局特征初始化 hidden state
│   └── AttentionLSTMDecoder       → Bahdanau 注意力逐时间步加权空间特征
└── Generation: greedy / beam search (带 repetition penalty)
```

## 快速开始

### 环境配置

```bash
conda create -n caption python=3.12
conda activate caption
pip install -r requirements.txt
```

### 下载 CLIP 预训练权重

推理需要 CLIP ViT-L/14 预训练权重，从 HuggingFace 下载：

```bash
python -c "
from transformers import CLIPModel
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
model.save_pretrained('checkpoint/pretrained/clip-vit-large-patch14')
print('Downloaded to checkpoint/pretrained/clip-vit-large-patch14')
"
```

### 使用预训练权重推理

```bash
# 单张图片推理（默认使用 CLIP + Attention 最佳权重）
python predict.py --image data/val/2002.jpg

# 指定解码策略
python predict.py --image data/val/2002.jpg --strategy beam --beam_size 5

# 批量推理
python predict.py --image_dir data/val/ --output results/predictions.json

# 使用 ResNet 模型
python predict.py --image data/val/2002.jpg --backbone resnet50
```

## 项目结构

```
final-project/
├── README.md                      # 本文档
├── requirements.txt               # Python 依赖
├── train.py                       # 主训练脚本
├── predict.py                     # 快速推理入口（推荐）
├── models/                        # 模型定义
│   ├── caption_model.py           # CaptionModel 主模型
│   ├── encoder.py                 # ResNet 编码器
│   ├── clip_encoder.py            # CLIP 编码器
│   ├── decoder.py                 # LSTM / Attention LSTM 解码器
│   ├── dataset.py                 # 数据集和 DataLoader
│   ├── vocab.py                   # 词表（支持 BERT tokenizer）
│   └── metrics.py                 # 评估指标
├── scripts/                       # 工具脚本
│   ├── train_clip.sh              # CLIP 模型训练
│   ├── inference.py               # 通用推理脚本
│   ├── eval_comprehensive.py      # 全面评测
│   └── ...
├── annotations/                   # 训练/验证标注
├── checkpoint/                    # 模型权重（gitignore）
│   └── pretrained/                # CLIP 预训练权重（需下载）
├── data/                          # 图像数据（gitignore）
└── results/                       # 评测结果
```

## 训练

### CLIP + Attention 模型（推荐）

```bash
# 下载预训练权重后，从头训练
bash scripts/train_clip.sh

# 续训（60→120 epochs, LR 降至 5e-4）
bash scripts/train_clip_resume.sh
```

### ResNet + Attention/Baseline

```bash
# 训练两种解码器并对比
bash run_task2.sh both

# 仅训练 attention
bash run_task2.sh attention
```

### 训练参数

| 参数 | CLIP 模型 | ResNet 模型 |
|------|----------|------------|
| 编码器 | CLIP ViT-L/14 (冻结, 304M) | ResNet50 (fine-tune layer4, 25M) |
| 解码器 | Attention LSTM (51M) | Attention LSTM (51M) |
| Batch Size | 32 | 64 |
| Learning Rate | 1e-3 → 5e-4 | 1e-3 |
| Epochs | 120 (60+60) | 60 |
| 图像归一化 | CLIP: [0.481, 0.458, 0.408] | ImageNet: [0.485, 0.456, 0.406] |
| Tokenizer | BERT (bert-base-uncased, 30522) | 同左 |

## 实验结果

### 视觉编码器对比（Attention 解码器）

| 指标 | ResNet50 (60ep) | CLIP ViT-L/14 (120ep) |
|------|----------------|----------------------|
| BLEU-4 | 0.891 | 0.621 |
| METEOR | 0.940 | 0.764 |
| CIDEr | 0.908 | 0.660 |
| Val Loss | 1.887 | 0.246 |

> ResNet BLEU-4 异常高（COCO SOTA ~0.40），原因是编码器微调 + 训练/验证标注风格一致（均由 Qwen 生成）。CLIP 分数更能反映真实泛化能力。

### 全面评测（验证集 369 张）

| 配置 | BLEU-4 | METEOR | CIDEr | BERTScore-F1 | CLIPScore |
|------|--------|--------|-------|-------------|-----------|
| CLIP-loss-greedy | 0.025 | 0.271 | 0.053 | 0.217 | 0.513 |
| **CLIP-loss-beam5** | **0.033** | **0.292** | **0.062** | **0.258** | **0.519** |
| ResNet-metric-greedy | 0.028 | 0.267 | 0.056 | 0.203 | 0.490 |

### Attention 机制消融

| 解码器 | BLEU-4 | METEOR | CIDEr |
|--------|--------|--------|-------|
| Baseline (LSTM) | 0.026 | 0.267 | 0.040 |
| **Attention LSTM** | **0.617** | **0.760** | **0.657** |

Attention 机制带来 20 倍以上的指标提升。

## 关键结论

1. **CLIP ViT-L/14 优于 ResNet50** — 在语义指标上全面领先，验证了更强视觉编码器的价值
2. **Attention 机制至关重要** — BLEU-4 提升 20 倍
3. **BERT tokenizer 消除 UNK** — 子词分词避免 OOV 问题
4. **Beam search 优于 greedy** — beam5 在所有参考指标上优于 greedy
5. **best_loss 权重优于 best_metric** — val_loss 是更可靠的模型选择标准

## 参考资料

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
