# 图像描述生成与多模态学习

华中科技大学 软件学院 深度学习期末项目

## 项目概述

本项目实现图像描述生成（Image Captioning）与多模态学习，包含两个核心任务：

1. **任务一 — Prompt Engineering**：使用多模态大模型 Qwen3.5-9B 为无标注图片生成多维度描述，通过多版本 prompt 对比实验确定最优方案。
2. **任务二 — 图像描述模型训练**：基于任务一生成的标注，训练 CNN + LSTM（with/without Attention）图像描述生成模型，对比分析 attention 机制的效果。

## 项目结构

```
.
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── caption_model.py              # CaptionModel 主模型
│   ├── dataset.py                    # 数据集和 DataLoader
│   ├── decoder.py                    # LSTM / Attention LSTM 解码器
│   ├── encoder.py                    # ResNet 编码器
│   ├── metrics.py                    # 评估指标 (BLEU, METEOR, CIDEr)
│   └── vocab.py                      # 词表和分词器（支持自定义/BERT）
│
├── scripts/                          # 工具脚本
│   ├── generate_annotations.py       # 任务一：多版本 prompt 标注生成
│   ├── generate_caption_v7.py        # v7 caption 专用生成（BERT tokenizer）
│   ├── run_v7_parallel.sh            # 多卡并行标注生成
│   ├── train_v7.sh                   # 训练脚本（支持 baseline/attention/续训）
│   ├── inference.py                  # 推理脚本
│   └── infer.sh                      # 推理快捷脚本
│
├── annotations/                      # 生成的标注文件
│   ├── train_captions_v7.jsonl       # 训练集标注（v7，最终版）
│   ├── val_captions_v7.jsonl         # 验证集标注（v7，最终版）
│   └── prompt_comparison_report.md   # Prompt 对比实验报告
│
├── checkpoint/                       # 模型权重（不纳入版本控制）
│   ├── Qwen3.5-9B/                   # 多模态大模型权重
│   └── v7/                           # 训练产出
│       ├── baseline/                 # Baseline 模型
│       └── attention/                # Attention 模型
│
├── data/                             # 图片数据（不纳入版本控制）
│   ├── train/                        # 训练集 (2000 张 JPG)
│   └── val/                          # 验证集 (369 张 JPG)
│
├── train.py                          # 主训练脚本
├── run_task2.sh                      # 训练入口脚本
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

**Baseline**：ResNet50 (Encoder) + LSTM (Decoder)
- Encoder：预训练 ResNet50，冻结 layer1-3，微调 layer4
- Decoder：LSTM + 全连接层，全局特征初始化隐藏状态

**Attention**：ResNet50 (Encoder) + Bahdanau Attention + LSTM (Decoder)
- Encoder：同上，提取空间特征 [B, 49, 512]
- Decoder：每个时间步通过注意力机制加权空间特征，拼接词嵌入作为 LSTM 输入

### 2.2 分词方案

采用 **BERT tokenizer**（bert-base-uncased）替代自定义词表：

| 方案 | 词表大小 | UNK 率 | 中文处理 |
|------|---------|--------|---------|
| 自定义词表 | 6831 | 42% 词只出现 1 次 | 混入词表 |
| **BERT tokenizer** | **30522** | **接近 0%** | **自动过滤** |

BERT embedding 预训练权重初始化：加载 BERT 的 word embedding [30522, 768]，投影到 [30522, 512]。

### 2.3 运行方法

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

# 使用 scripts/train_v7.sh（更多选项）
bash scripts/train_v7.sh baseline      # 从头训练
bash scripts/train_v7.sh resume-both   # 续训两者
```

### 2.4 推理

```bash
# 单张图片推理
bash scripts/infer.sh data/val/2002.jpg

# 指定模型和显卡
bash scripts/infer.sh data/val/2002.jpg attention 0

# 批量推理
bash scripts/infer.sh data/val attention 0

# 指定 checkpoint
bash scripts/infer.sh data/val/2002.jpg baseline 0 checkpoint/v7/baseline/baseline_best.pth
```

### 2.5 训练输出

```
checkpoint/v7/
├── baseline/
│   ├── baseline_best.pth          # 最优 loss 模型
│   ├── baseline_best_metric.pth   # 最优 BLEU-4 模型
│   ├── baseline_history.json      # 训练历史
│   ├── baseline_scores.json       # 评估指标
│   ├── baseline.log               # 训练日志
│   ├── vocab_baseline.json        # 词表
│   └── tb/                        # Tensorboard 日志
└── attention/
    ├── attention_best.pth
    ├── attention_best_metric.pth
    ├── attention_history.json
    ├── attention_scores.json
    ├── attention.log
    ├── vocab_attention.json
    └── tb/
```

查看 Tensorboard：
```bash
tensorboard --logdir checkpoint/v7/*/tb
```

---

## 实验结果

### Prompt 对比（任务一）

| 版本 | caption 平均词数 | objects 格式正常率 | "The image" 开头占比 |
|------|-----------------|-------------------|---------------------|
| v6 | 134.1 | 100% | 74.5% |
| **v7** | **61.9** | **100%** | **0.4%** |

### 模型对比（任务二）

使用 v7 标注 + BERT tokenizer + BERT embedding 初始化，训练 120 epoch：

| 指标 | Baseline (60ep) | **Attention (120ep)** | 提升 |
|------|----------------|----------------------|------|
| BLEU-1 | 0.4281 | **0.9395** | +120% |
| BLEU-2 | 0.1518 | **0.9129** | +501% |
| BLEU-3 | 0.0733 | **0.8998** | +1127% |
| BLEU-4 | 0.0448 | **0.8907** | +1888% |
| METEOR | 0.3007 | **0.9396** | +212% |
| ROUGE-L | 0.3045 | **0.9343** | +207% |
| CIDEr | 0.0635 | **0.9078** | +1330% |

**关键结论**：
1. **Attention 机制至关重要** — BLEU-4 从 0.045 提升到 0.891（20 倍）
2. **BERT tokenizer 解决了"胡说八道"问题** — 子词分词消除 UNK，模型生成有意义的描述
3. **BERT embedding 初始化加速收敛** — 初始 loss 降低 35%
4. **120 epoch 充分收敛** — BLEU-4 从 0.70（60ep）提升到 0.89（120ep）

### 解码策略消融实验

| 配置 | 耗时(s/张) | 重复率 | 质量 |
|------|-----------|--------|------|
| greedy | 0.31 | 34.0% | 有连续重复 |
| beam5_rp1.0 | 1.20 | 34.4% | 无连续重复 |
| **beam5_rp1.3** | **9.15** | **31.2%** | **最优** |
| beam10_rp1.3 | 17.86 | 31.5% | 最高质量但太慢 |

**推荐配置**：训练评估用 greedy（快），推理用 beam5_rp1.3（质量最优）

### Loss 收敛曲线

```
Attention Loss
6.0 |*
5.0 | *
4.0 |  *
3.0 |   * *
2.0 |       * * * *
1.0 |               * * * * * *
0.5 |                           * * * * * * * * *
0.1 |                                               * * * * * * * * * * *
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
     1  3  5  7  9  11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59
```

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
pip install pycocoevalcap  # 可选，用于 CIDEr 评估
```

## 参考资料

- [Qwen3.5 官方文档](https://qwen.ai/blog?id=qwen3.5)
- [nndl.github.io 参考书目](https://nndl.github.io/)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
