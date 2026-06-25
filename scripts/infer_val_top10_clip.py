"""使用 CLIP ViT-L/14 + Attention 权重对 val 前 10 张图片生成描述

用法:
  python scripts/infer_val_top10_clip.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from models import CaptionModel, Vocabulary


def main():
    device = torch.device("cuda:0")

    # 加载词表
    vocab = Vocabulary.from_pretrained("bert-base-uncased")

    # 加载模型 (CLIP ViT-L/14 + Attention LSTM)
    clip_path = "checkpoint/pretrained/clip-vit-large-patch14"
    model = CaptionModel(
        arch="attention",
        backbone="clip",
        embed_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,
        attention_dim=256,
        fine_tune_layers="none",
        pretrained=False,
        clip_path=clip_path,
    ).to(device)

    ckpt_path = "checkpoint/v7_clip/attention_best_metric.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # CLIP 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # 推理 val 前 10 张，使用不同采样策略
    strategies = [
        ("greedy", {}),
        ("beam", {"beam_size": 3}),
        ("beam", {"beam_size": 5}),
    ]

    all_results = {}

    for strategy_name, kwargs in strategies:
        label = strategy_name if strategy_name == "greedy" else f"beam{kwargs['beam_size']}"
        print(f"\n{'='*60}")
        print(f"  Strategy: {label}")
        print(f"{'='*60}")

        results = {}
        for i in range(10):
            img_id = f"{2002 + i}.jpg"
            img_path = f"data/val/{img_id}"
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                seqs = model.generate(
                    tensor,
                    strategy=strategy_name,
                    max_len=64,
                    repetition_penalty=1.2,
                    **kwargs,
                )

            caption = vocab.decode(seqs[0], skip_special=True)
            results[img_id] = caption
            print(f"  {img_id}: {caption}")

        all_results[label] = results

    # 保存结果
    output_path = "results/inference_val_top10_clip.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
