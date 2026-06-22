# 深度学习期末项目：图像描述生成与多模态学习

华中科技大学 软件学院

## 项目概述

本项目实现图像描述生成（Image Captioning）与多模态学习，包含两个核心任务：

1. **任务一 — Prompt Engineering**：使用多模态大模型 Qwen3.5-9B 为无标注图片生成多维度描述信息。
2. **任务二 — 图像描述模型训练**：基于任务一生成的标注，训练 CNN-RNN 图像描述生成模型。

## 项目结构

```
.
├── data/
│   ├── train/                  # 训练集 (2000 张 JPG)
│   └── val/                    # 验证集 (369 张 JPG)
├── checkpoint/
│   └── Qwen3.5-9B/             # 多模态大模型权重
├── scripts/
│   └── generate_annotations.py # 任务一：标注生成脚本
├── annotations/                # 生成的标注文件
│   ├── train_captions_v1.jsonl
│   ├── train_captions_v2.jsonl
│   └── ...
├── README.md
└── requirements.txt
```

---

## 任务一：图片描述生成（Prompt Engineering）

### 1.1 实现方法

#### 模型选择

使用 **Qwen3.5-9B** 多模态大语言模型，该模型：
- 支持图像+文本混合输入
- 参数量 9B，使用 bfloat16 精度加载
- 部署在本地 GPU 上，无需调用外部 API

#### Prompt 设计策略

针对任务要求的四个描述维度，分别设计了专用 prompt：

| 维度 | Prompt 设计思路 | 示例 |
|------|----------------|------|
| **caption（图片描述）** | 要求 2-3 句话描述主体、动作、场景 | "Describe this image in 2-3 sentences. Focus on the main subjects, actions, and setting." |
| **objects（物体识别）** | 要求返回纯 JSON 数组，明确排除边界框等额外信息 | "List the main objects visible in this image. Return ONLY a plain JSON array of short object name strings..." |
| **category（类别判断）** | 预定义 6 个类别，要求只返回类别名 | "Classify this image into one of: 真实照片, 表情包, 网络图片, 软件截图, 插画, 其他" |
| **short_story（短故事）** | 鼓励创意，限制 1-2 句 | "Write a 1-2 sentence short story inspired by this image." |

#### Prompt 优化技巧

1. **明确输出格式**：在 objects prompt 中强调 "ONLY a plain JSON array"，避免模型返回包含边界框的复杂结构。
2. **关闭思考模式**：通过 `enable_thinking=False` 参数禁用模型的思维链输出，获得简洁直接的回答。
3. **贪心解码**：使用 `do_sample=False` 确保输出稳定可复现。
4. **分维度调用**：对每张图片的 4 个维度分别调用模型，避免单次 prompt 过于复杂导致质量下降。

#### 多版本 Prompt 对比

脚本支持 3 个 prompt 版本（v1/v2/v3），用于对比实验：

- **v1**：英文 prompt，简洁直接
- **v2**：中文 prompt，更详细的格式要求
- **v3**：英文 prompt，更强调分析性和细节

### 1.2 输出格式

每条标注以 JSONL 格式存储，单条记录结构如下：

```json
{
  "image_id": "0.jpg",
  "split": "train",
  "prompt_version": "v1",
  "caption": "This image captures a serene sunset viewed through a car's side-view mirror...",
  "objects": ["car", "mirror", "road", "sun", "tree"],
  "category": "真实照片",
  "short_story": "The sunset painted the highway in hues of orange and gold..."
}
```

### 1.3 运行方法

#### 环境依赖

```bash
conda activate n_sam3
pip install transformers>=5.12 accelerate torch Pillow
```

#### 生成标注

```bash
# 测试前 10 张图片
python scripts/generate_annotations.py --num_images 10 --prompt_version v1

# 生成全部训练集
python scripts/generate_annotations.py --split train --prompt_version v1

# 生成验证集
python scripts/generate_annotations.py --split val --prompt_version v1

# 使用中文 prompt 版本
python scripts/generate_annotations.py --num_images 10 --prompt_version v2

# 断点续跑（中断后继续）
python scripts/generate_annotations.py --split train --prompt_version v1 --resume

# 指定 GPU（避免显存不足）
python scripts/generate_annotations.py --num_images 10 --device_map "cuda:0"
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `/home/turing1/jhcao/final-project/checkpoint/Qwen3.5-9B` | 模型权重路径 |
| `--data_dir` | `/mnt/jhcao/final-project/data` | 数据根目录 |
| `--split` | `train` | 数据集划分 (train/val) |
| `--num_images` | `-1` | 处理图片数量，-1 表示全部 |
| `--prompt_version` | `v1` | Prompt 版本 (v1/v2/v3) |
| `--output_dir` | `./annotations` | 输出目录 |
| `--resume` | `False` | 断点续跑 |
| `--device_map` | `auto` | GPU 设备映射 |

### 1.4 性能数据

- 单张图片生成 4 个维度标注约需 **10-15 秒**（RTX 4090）
- 全部 2000 张训练集预计耗时约 **5-7 小时**
- 模型显存占用约 **18 GB**（bfloat16）

---

## 任务二：图像描述模型训练

*（待实现）*

---

## 环境配置

```bash
# 创建环境
conda create -n n_sam3 python=3.12
conda activate n_sam3

# 安装依赖
pip install torch torchvision
pip install transformers>=5.12 accelerate
pip install Pillow tqdm
```

## 参考资料

- [Qwen3.5 官方文档](https://qwen.ai/blog?id=qwen3.5)
- [nndl.github.io 参考书目](https://nndl.github.io/)
