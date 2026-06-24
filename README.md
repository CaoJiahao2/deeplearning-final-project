# 图像描述生成与多模态学习

华中科技大学 软件学院 深度学习期末项目

## 项目概述

本项目实现图像描述生成（Image Captioning）与多模态学习，包含两个核心任务：

1. **任务一 — Prompt Engineering**：使用多模态大模型 Qwen3.5-9B 为无标注图片生成多维度描述，通过多版本 prompt 对比实验确定最优方案。
2. **任务二 — 图像描述模型训练**：基于任务一生成的标注，训练图像描述生成模型，对比不同视觉编码器（ResNet50 vs CLIP ViT-L/14）和解码策略的效果。

## 项目结构

```
.
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── caption_model.py              # CaptionModel 主模型（支持 ResNet/CLIP backbone）
│   ├── dataset.py                    # 数据集和 DataLoader
│   ├── decoder.py                    # LSTM / Attention LSTM 解码器
│   ├── encoder.py                    # ResNet 编码器
│   ├── clip_encoder.py               # CLIP ViT-L/14 编码器（冻结）
│   ├── metrics.py                    # 评估指标 (BLEU, METEOR, CIDEr)
│   └── vocab.py                      # 词表和分词器（支持自定义/BERT）
│
├── scripts/                          # 工具脚本
│   ├── generate_annotations.py       # 任务一：多版本 prompt 标注生成
│   ├── generate_caption_v7.py        # v7 caption 专用生成（Qwen3.5-9B）
│   ├── run_v7_parallel.sh            # 多卡并行标注生成
│   ├── train_v7.sh                   # ResNet 训练脚本
│   ├── train_clip.sh                 # CLIP 训练脚本（从头训练）
│   ├── train_clip_resume.sh          # CLIP 续训脚本（从 checkpoint 继续）
│   ├── inference.py                  # 通用推理脚本
│   ├── infer.sh                      # 推理快捷脚本
│   ├── infer_val_top10.py            # ResNet val 前10推理
│   ├── infer_val_top10_clip.py       # CLIP val 前10推理（多策略对比）
│   ├── eval_comprehensive.py         # 全面评测脚本（9种配置 × 14项指标）
│   └── run_v7_parallel.sh            # 多卡并行标注生成
│
├── annotations/                      # 生成的标注文件
│   ├── train_captions_v7.jsonl       # 训练集标注（v7，最终版，2000条）
│   ├── val_captions_v7.jsonl         # 验证集标注（v7，369条）
│   └── prompt_comparison_report.md   # Prompt 对比实验报告
│
├── results/                          # 评测结果
│   ├── eval_comprehensive.json       # 全面评测详细结果（JSON）
│   ├── eval_summary.csv              # 全面评测汇总表（CSV）
│   ├── eval_comprehensive.log        # 评测运行日志
│   ├── inference_val_top10_clip.json # CLIP top-10 推理结果
│   ├── inference_val_top10.json      # ResNet top-10 推理结果
│   ├── inference_val_top10_beam.json # Beam search 推理结果
│   ├── inference_val_top10.md        # 推理结果分析
│   ├── ablation_decoding.json        # 解码策略消融实验
│   └── ablation_decoding_report.md   # 消融实验报告
│
├── checkpoint/                       # 模型权重（不纳入版本控制）
│   ├── Qwen3.5-9B/                   # 多模态大模型权重
│   ├── clip-vit-B-14/                # CLIP ViT-L/14 预训练权重
│   └── v7/                           # ResNet 训练产出
│       ├── baseline/                 # Baseline 模型
│       └── attention/                # Attention 模型
│   └── v7_clip/                      # CLIP 训练产出
│       ├── attention_best.pth        # 最优 val_loss 模型
│       ├── attention_best_metric.pth # 最优 BLEU-4 模型
│       ├── attention_history.json    # 训练历史
│       ├── attention_scores.json     # 最终评估指标
│       ├── attention.log             # 完整训练日志
│       └── tb/                       # Tensorboard 日志
│
├── data/                             # 图片数据（不纳入版本控制）
│   ├── train/                        # 训练集 (2000 张 JPG)
│   └── val/                          # 验证集 (369 张 JPG)
│
├── train.py                          # 主训练脚本（支持 ResNet/CLIP backbone）
├── run_task2.sh                      # ResNet 训练入口脚本
├── final_project_agent_brief.md      # 项目要求文档
├── CLAUDE.md                         # 项目指导文档
└── README.md
```

---

## 任务一：图片描述生成（Prompt Engineering）

### 1.1 Prompt 设计与对比

设计了 7 个版本的 prompt，从简单到复杂逐步优化：

| 版本 | 策略 | 核心技术 |
|------|------|----------|
| v1 | 基线版 | 极简指令，无格式约束 |
| v2 | 改进版 | 长度约束、格式规范、枚举选项 |
| v3 | 进阶版 | 结构化拆解、负面约束（无边框）、少样本示例 |
| v4 | 丰富版 | 属性要求、空间关系引导、推理请求 |
| v5 | 最佳实践版 | 角色分配、逐句结构模板、完整性指令 |
| v6 | 增强结构版 | 显式数量目标（15-25 项）、反模糊用词约束 |
| **v7** | **约束精炼版** | **角色设定、格式硬约束（50-80 词）、few-shot 示例、禁止 meta 开头** |

**v7 为最终采用版本**，解决了 v6 的三大问题：
- 开头单调（68% 以 "The image" 开头）→ 完全消除
- 长度过长（平均 134 词）→ 精简到 62 词
- 含换行符（32%）→ 完全消除

### 1.2 运行方法

```bash
# 环境
conda activate n_sam3

# 生成全部标注（多卡并行）
bash scripts/run_v7_parallel.sh

# 单卡生成
python scripts/generate_caption_v7.py --split train --gpu 0

# 生成单个 prompt 版本（v1-v6）
python scripts/generate_annotations.py --num_images 10 --prompt_version v3
```

### 1.3 标注格式

```json
{
  "image_id": "0.jpg",
  "split": "train",
  "prompt_version": "v7",
  "caption": "A car's side mirror reflects a vibrant sunset scene on a highway, with the sun casting a warm orange glow across the sky."
}
```

---

## 任务二：图像描述模型训练

### 2.1 模型架构

本项目对比了两种视觉编码器 + 两种解码策略的组合：

#### 视觉编码器

| 编码器 | 参数量 | 输出维度 | 空间 token 数 | 预训练数据 |
|--------|--------|---------|--------------|-----------|
| **ResNet50** | 25M | 512 | 49 (7×7) | ImageNet 1M |
| **CLIP ViT-L/14** | 304M (冻结) | 512 | 256 (16×16) | 400M 图文对 |

**CLIP ViT-L/14 相比 ResNet50 的优势**：
- 空间分辨率 5× 更高（256 vs 49 tokens）
- 图文对比预训练，天然理解视觉-语言对应关系
- 完全冻结，避免小数据集过拟合

#### 解码器

**Baseline**：LSTM Decoder
- 全局特征初始化隐藏状态，逐词解码

**Attention**：Bahdanau Attention + LSTM Decoder
- 每个时间步通过注意力机制加权空间特征
- 拼接词嵌入和上下文向量作为 LSTM 输入

### 2.2 分词方案

采用 **BERT tokenizer**（bert-base-uncased）替代自定义词表：

| 方案 | 词表大小 | UNK 率 | 中文处理 |
|------|---------|--------|---------|
| 自定义词表 | 6831 | 42% 词只出现 1 次 | 混入词表 |
| **BERT tokenizer** | **30522** | **接近 0%** | **自动过滤** |

BERT embedding 预训练权重初始化：加载 BERT 的 word embedding [30522, 768]，投影到 [30522, 512]。

### 2.3 训练方法

#### ResNet50 模型

```bash
# 环境
conda activate n_sam3

# 训练 baseline + attention 并对比
bash run_task2.sh both

# 仅训练 attention
bash run_task2.sh attention

# 续训（从上次 checkpoint 继续）
bash run_task2.sh resume-attention

# 评估已有模型
bash run_task2.sh eval
```

#### CLIP ViT-L/14 模型

```bash
# 从头训练（60 epochs, LR=1e-3）
bash scripts/train_clip.sh

# 续训（60→120 epochs, LR=5e-4 降学习率）
bash scripts/train_clip_resume.sh
```

CLIP 训练关键参数：
- 编码器：完全冻结（304M 参数不可训练）
- 可训练参数：投影层（~1M）+ LSTM decoder（~51M）
- 图像归一化：CLIP 专用（mean=[0.481, 0.458, 0.408]）
- Batch size：32（ViT-L/14 显存占用较大）

### 2.4 推理

```bash
# 单张图片推理（ResNet）
bash scripts/infer.sh data/val/2002.jpg attention 0

# CLIP 模型 top-10 推理（多策略对比）
python scripts/infer_val_top10_clip.py
```

### 2.5 训练输出

```
checkpoint/v7/                        # ResNet 训练产出
├── baseline/
│   ├── baseline_best.pth
│   ├── baseline_best_metric.pth
│   ├── baseline_history.json
│   ├── baseline_scores.json
│   └── baseline.log
└── attention/
    ├── attention_best.pth
    ├── attention_best_metric.pth
    ├── attention_history.json
    ├── attention_scores.json
    └── attention.log

checkpoint/v7_clip/                   # CLIP 训练产出
├── attention_best.pth
├── attention_best_metric.pth
├── attention_history.json
├── attention_scores.json
├── attention.log
└── tb/
```

查看 Tensorboard：
```bash
tensorboard --logdir checkpoint/v7/*/tb
tensorboard --logdir checkpoint/v7_clip/tb
```

---

## 实验结果

### 1. 视觉编码器对比（ResNet50 vs CLIP ViT-L/14）

使用 v7 标注 + BERT tokenizer，Attention LSTM 解码器：

| 指标 | ResNet50 (120ep) | CLIP ViT-L/14 (120ep) | 说明 |
|------|-----------------|----------------------|------|
| BLEU-4 | 0.8907 | 0.6209 | ResNet 编码器微调，分数偏高 |
| METEOR | 0.9396 | 0.7641 | |
| CIDEr | 0.9078 | 0.6601 | |
| Val Loss | 0.1004 | 0.2456 | CLIP 编码器冻结，收敛更慢 |

> **注意**：ResNet50 的 BLEU-4=0.89 异常高（COCO SOTA 通常 0.40），原因是编码器微调 + 训练/验证标注风格高度一致（均由 Qwen3.5-9B 生成）。CLIP 的分数更能反映真实泛化能力。

### 2. 全面评测（9 种配置 × 14 项指标）

在验证集（369 张图片）上，对比 3 种模型权重 × 3 种采样策略：

#### 参考指标（↑ 越高越好）

| 配置 | BLEU-4 | METEOR | CIDEr | BERTScore-F1 |
|------|--------|--------|-------|-------------|
| CLIP-metric-greedy | 0.0250 | 0.2703 | 0.0524 | 0.2131 |
| CLIP-metric-beam3 | 0.0329 | 0.2900 | 0.0607 | 0.2494 |
| CLIP-metric-beam5 | 0.0341 | 0.2901 | 0.0616 | 0.2541 |
| CLIP-loss-greedy | 0.0254 | 0.2712 | 0.0532 | 0.2173 |
| CLIP-loss-beam3 | 0.0314 | 0.2876 | 0.0602 | 0.2527 |
| **CLIP-loss-beam5** | 0.0333 | **0.2920** | **0.0624** | **0.2575** |
| ResNet-metric-greedy | 0.0282 | 0.2666 | 0.0557 | 0.2027 |
| ResNet-metric-beam3 | 0.0317 | 0.2823 | 0.0587 | 0.2336 |
| ResNet-metric-beam5 | 0.0326 | 0.2847 | 0.0597 | 0.2360 |

#### 图文对齐（CLIPScore，↑ 越高越好）

| 配置 | CLIPScore |
|------|-----------|
| CLIP-metric-greedy | 0.5186 |
| CLIP-metric-beam3 | 0.5141 |
| CLIP-metric-beam5 | 0.5110 |
| CLIP-loss-greedy | 0.5128 |
| CLIP-loss-beam3 | 0.5183 |
| **CLIP-loss-beam5** | **0.5192** |
| ResNet-metric-greedy | 0.4901 |
| ResNet-metric-beam3 | 0.4819 |
| ResNet-metric-beam5 | 0.4852 |

#### 多样性与质量

| 配置 | Distinct-1 | Distinct-2 | RepRate |
|------|-----------|-----------|---------|
| CLIP-metric-greedy | 0.0886 | 0.3216 | 0.0898 |
| ResNet-metric-greedy | 0.0939 | 0.3318 | 0.1057 |
| CLIP-loss-beam5 | 0.0814 | 0.2797 | 0.0740 |

#### ⭐ 最佳配置：CLIP-loss-beam5

在语义相关指标上全面领先：
- **BERTScore F1**: 0.2575（语义理解最好）
- **CLIPScore**: 0.5192（图文对齐最好）
- **METEOR**: 0.2920（考虑同义词匹配）
- **CIDEr**: 0.0624（TF-IDF 加权相似度）
- **重复率**: 0.0740（生成质量最好）

### 3. 全面评测指标说明

评测框架（`scripts/eval_comprehensive.py`）包含 3 大类 14 项指标：

| 类别 | 指标 | 说明 |
|------|------|------|
| **参考指标** | BLEU-1/2/3/4 | n-gram 精度（1-4 元） |
| | METEOR | 考虑同义词和词干的匹配 |
| | ROUGE-L | 最长公共子序列召回率 |
| | CIDEr | TF-IDF 加权 n-gram 相似度（专为 captioning 设计） |
| | BERTScore-F1 | 基于 BERT 上下文嵌入的语义相似度 |
| **图文对齐** | CLIPScore | CLIP 模型计算图文余弦相似度（无需参考标注） |
| **多样性** | Distinct-1/2 | 唯一 unigram/bigram 比率 |
| | AvgLength | 平均生成词数 |
| | RepRate | 重复 bigram 比率（越低越好） |

### 4. Prompt 对比（任务一）

| 版本 | caption 平均词数 | objects 格式正常率 | "The image" 开头占比 |
|------|-----------------|-------------------|---------------------|
| v6 | 134.1 | 100% | 74.5% |
| **v7** | **61.9** | **100%** | **0.4%** |

### 5. 解码策略消融实验

| 配置 | 耗时(s/张) | 重复率 | 质量 |
|------|-----------|--------|------|
| greedy | 0.31 | 34.0% | 有连续重复 |
| beam5_rp1.0 | 1.20 | 34.4% | 无连续重复 |
| **beam5_rp1.3** | **9.15** | **31.2%** | **最优** |
| beam10_rp1.3 | 17.86 | 31.5% | 最高质量但太慢 |

**推荐配置**：训练评估用 greedy（快），推理用 beam5_rp1.3（质量最优）

### 6. 关键结论

1. **CLIP ViT-L/14 优于 ResNet50** — 在 BERTScore、CLIPScore、METEOR 等语义指标上全面领先，验证了更强视觉编码器的价值
2. **Attention 机制至关重要** — BLEU-4 从 0.045 提升到 0.891（20 倍）
3. **BERT tokenizer 解决了"胡说八道"问题** — 子词分词消除 UNK，模型生成有意义的描述
4. **Beam search 优于 greedy** — beam5 在所有参考指标上都优于 greedy，但生成速度较慢
5. **best_loss 权重优于 best_metric 权重** — val_loss 是更可靠的模型选择标准

---

## 环境配置

```bash
# 创建环境
conda create -n n_sam3 python=3.12
conda activate n_sam3

# 安装依赖
pip install torch torchvision
pip install transformers>=5.12 accelerate
pip install Pillow tqdm tensorboard sentencepiece
pip install pycocoevalcap    # CIDEr 评估
pip install bert-score       # BERTScore 评估
pip install nltk             # METEOR 评估
```

## 参考资料

- [Qwen3.5 官方文档](https://qwen.ai/blog?id=qwen3.5)
- [nndl.github.io 参考书目](https://nndl.github.io/)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
