"""Image Captioning 推理脚本

用法:
  # 使用 baseline 模型
  python scripts/inference.py --image data/val/2002.jpg --arch baseline --gpu 0

  # 使用 attention 模型 + beam search
  python scripts/inference.py --image data/val/2002.jpg --arch attention --gpu 0 --strategy beam --beam_size 5

  # 指定 checkpoint 目录
  python scripts/inference.py --image data/val/2002.jpg --arch baseline --exp v7/baseline --gpu 0

  # 批量推理（目录下所有图片）
  python scripts/inference.py --image_dir data/val --arch baseline --gpu 0 --output results.json
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CaptionModel, Vocabulary


def get_transform():
    """推理用图像预处理（与训练保持一致）。"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(arch: str, checkpoint_dir: str, device: torch.device, ckpt_path: str = None) -> CaptionModel:
    """加载模型和词表。"""
    # 加载词表
    vocab_path = os.path.join(checkpoint_dir, f"vocab_{arch}.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"[Vocab] Loaded {len(vocab)} words from {vocab_path}")

    # 构建模型
    model = CaptionModel(
        arch=arch,
        backbone="resnet50",
        embed_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,  # 推理时关闭 dropout
        attention_dim=256,
        fine_tune_layers="layer4",
        pretrained=False,  # 权重从 checkpoint 加载
    ).to(device)

    # 加载权重
    if ckpt_path is None:
        ckpt_path = os.path.join(checkpoint_dir, f"{arch}_best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(checkpoint_dir, f"{arch}_best_metric.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model] Loaded {arch} from {ckpt_path}")

    return model, vocab


def generate_caption(
    model: CaptionModel,
    vocab: Vocabulary,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 5,
    repetition_penalty: float = 1.2,
) -> str:
    """单张图片推理。"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        sequences = model.generate(
            tensor, strategy=strategy, beam_size=beam_size,
            repetition_penalty=repetition_penalty
        )

    tokens = sequences[0]
    caption = vocab.decode(tokens, skip_special=True)
    return caption


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "attention"])
    parser.add_argument("--exp", type=str, default="v7/baseline",
                        help="Experiment subdirectory under checkpoint/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint",
                        help="Root checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to checkpoint .pth file (overrides --exp)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty for beam search (>1.0 reduces repetition)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Must specify --image or --image_dir")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(args.checkpoint_dir, args.exp)

    model, vocab = load_model(args.arch, ckpt_dir, device, ckpt_path=args.checkpoint)
    transform = get_transform()

    # 收集图片列表
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            if os.path.splitext(os.path.basename(x))[0].isdigit() else x,
        )

    results = {}
    for img_path in image_paths:
        caption = generate_caption(
            model, vocab, img_path, transform, device,
            strategy=args.strategy, beam_size=args.beam_size,
            repetition_penalty=args.repetition_penalty
        )
        img_id = os.path.basename(img_path)
        results[img_id] = caption
        print(f"{img_id}: {caption}")

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
