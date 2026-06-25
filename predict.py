#!/usr/bin/env python3
"""图像描述生成推理脚本

使用训练好的模型为图片生成英文描述。默认使用 CLIP ViT-L/14 + Attention 最佳权重。

用法:
  # 单张图片推理
  python predict.py --image data/val/2002.jpg

  # 指定解码策略
  python predict.py --image data/val/2002.jpg --strategy beam --beam_size 5

  # 批量推理
  python predict.py --image_dir data/val/ --output results/predictions.json

  # 使用 ResNet 模型
  python predict.py --image data/val/2002.jpg --backbone resnet50
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models import CaptionModel, Vocabulary


# 默认权重路径
DEFAULT_CLIP_CHECKPOINT = "checkpoint/v7_clip/attention_best.pth"
DEFAULT_RESNET_CHECKPOINT = "checkpoint/v7/attention/attention_best.pth"

# CLIP 预训练权重搜索路径
CLIP_PRETRAINED_CANDIDATES = [
    "checkpoint/pretrained/clip-vit-large-patch14",
    "checkpoint/pretrained/clip-vit-B-14",
    "checkpoint/clip-vit-B-14",
    "openai/clip-vit-large-patch14",
]


def find_clip_path():
    """自动查找 CLIP 预训练权重路径。"""
    for path in CLIP_PRETRAINED_CANDIDATES:
        if os.path.exists(path):
            return path
    return CLIP_PRETRAINED_CANDIDATES[0]  # 返回默认路径，让后续报错提示下载


def get_transform(backbone="clip"):
    """获取图像预处理 pipeline。"""
    if backbone == "clip":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def load_model(backbone="clip", checkpoint=None, device="cuda:0"):
    """加载模型和权重。"""
    vocab = Vocabulary.from_pretrained("bert-base-uncased")

    # 确定权重路径
    if checkpoint is None:
        if backbone == "clip":
            checkpoint = DEFAULT_CLIP_CHECKPOINT
        else:
            checkpoint = DEFAULT_RESNET_CHECKPOINT

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"找不到权重文件: {checkpoint}\n"
            f"请确认权重路径正确，或使用 --checkpoint 指定权重文件。"
        )

    # CLIP 预训练权重路径
    clip_path = find_clip_path() if backbone == "clip" else None

    # 构建模型
    model_kwargs = dict(
        arch="attention",
        backbone=backbone,
        embed_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,
        attention_dim=256,
        fine_tune_layers="none",
        pretrained=False,
    )
    if backbone == "clip":
        model_kwargs["clip_path"] = clip_path

    model = CaptionModel(**model_kwargs).to(device)

    # 加载权重
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"已加载模型: {checkpoint}")
    return model, vocab


def predict_single(model, vocab, image_path, transform, device, strategy="greedy", beam_size=5):
    """单张图片推理。"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        seqs = model.generate(
            tensor,
            strategy=strategy,
            max_len=64,
            beam_size=beam_size,
            repetition_penalty=1.2,
        )

    caption = vocab.decode(seqs[0], skip_special=True)
    return caption


def predict_batch(model, vocab, image_dir, transform, device, strategy="greedy", beam_size=5, output=None):
    """批量推理。"""
    # 获取所有图片
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
    )

    results = {}
    for img_file in tqdm(image_files, desc="推理中"):
        img_path = os.path.join(image_dir, img_file)
        caption = predict_single(model, vocab, img_path, transform, device, strategy, beam_size)
        results[img_file] = caption

    # 保存结果
    if output:
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到 {output}")

    return results


def main():
    parser = argparse.ArgumentParser(description="图像描述生成推理")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, default=None, help="批量推理图片目录")
    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "resnet50"],
                        help="视觉编码器 (默认: clip)")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "beam"],
                        help="解码策略 (默认: greedy)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search 宽度 (默认: 5)")
    parser.add_argument("--output", type=str, default=None, help="批量推理结果保存路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 设备号")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.print_help()
        print("\n错误: 请指定 --image 或 --image_dir")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model, vocab = load_model(args.backbone, args.checkpoint, device)
    transform = get_transform(args.backbone)

    if args.image:
        # 单张推理
        caption = predict_single(model, vocab, args.image, transform, device, args.strategy, args.beam_size)
        print(f"\n图片: {args.image}")
        print(f"描述: {caption}")
    else:
        # 批量推理
        results = predict_batch(model, vocab, args.image_dir, transform, device,
                                args.strategy, args.beam_size, args.output)
        print(f"\n推理完成: {len(results)} 张图片")
        # 打印前 5 个示例
        for i, (k, v) in enumerate(list(results.items())[:5]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
