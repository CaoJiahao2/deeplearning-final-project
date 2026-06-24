"""使用最优 attention 权重对 val 前 10 张图片生成描述

用法:
  python scripts/infer_val_top10.py
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

    # 加载模型
    model = CaptionModel(
        arch="attention",
        backbone="resnet50",
        embed_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,
        attention_dim=256,
        fine_tune_layers="layer4",
        pretrained=False,
    ).to(device)

    ckpt_path = "checkpoint/v7/attention/attention_best.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 推理 val 前 10 张
    results = {}
    for i in range(10):
        img_id = f"{2002 + i}.jpg"
        img_path = f"data/val/{img_id}"
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            seqs = model.generate(tensor, strategy="greedy")

        caption = vocab.decode(seqs[0], skip_special=True)
        results[img_id] = caption
        print(f"{img_id}: {caption}")

    # 保存结果
    output_path = "results/inference_val_top10.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
