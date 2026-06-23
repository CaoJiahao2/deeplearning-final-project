"""
v6 caption 并行生成脚本 — 仅生成 caption 字段（训练只需此项）。
用法：
  # 单卡测试
  python scripts/generate_v6_caption_only.py --split train --num_images 10 --gpu 0

  # 3 卡并行（由 shell 脚本调用）
  python scripts/generate_v6_caption_only.py --split train --gpu 0 --shard_offset 0 --shard_count 667
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

V6_CAPTION_PROMPT = (
    "You are a professional image captioner creating training data for an image captioning model. "
    "Write a factual description of this image in 2-3 sentences, 50-80 words. "
    "Cover: (1) the main subject and what is happening, "
    "(2) key visual details such as colors, lighting, and setting. "
    "Start directly with the subject — for example: "
    "'A golden retriever sits on a worn leather couch near a sunlit window' or "
    "'Two cyclists race along a coastal road at dusk, their shadows stretching across the pavement'. "
    "Do NOT begin with 'The image', 'This image', 'The picture', or any meta-phrase about the image itself. "
    "Do NOT use bullet points, markdown formatting, or line breaks. "
    "Write one plain paragraph. Avoid vague words like 'nice', 'some', 'various', 'interesting'."
)


def load_model(model_path: str, device: str = "cuda:0"):
    print(f"Loading model on {device} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Model loaded on {device}.")
    return model, processor


def generate_caption(model, processor, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": V6_CAPTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def get_image_paths(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    images = sorted(
        [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
    )
    return [os.path.join(split_dir, img) for img in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/turing1/jhcao/final-project/checkpoint/Qwen3.5-9B")
    parser.add_argument("--data_dir", default="/mnt/jhcao/final-project/data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--output_dir", default="/mnt/jhcao/final-project/annotations")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--shard_offset", type=int, default=0, help="Start from this image index")
    parser.add_argument("--shard_count", type=int, default=-1, help="Number of images to process (-1=all remaining)")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard identifier for output filename")
    parser.add_argument("--version", type=str, default="v7", help="Prompt version tag for output filename")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    all_paths = get_image_paths(args.data_dir, args.split)

    if args.num_images > 0:
        all_paths = all_paths[:args.num_images]

    # Apply shard
    paths = all_paths[args.shard_offset:]
    if args.shard_count > 0:
        paths = paths[:args.shard_count]

    # Output file
    ver = args.version
    if args.shard_count > 0:
        output_file = os.path.join(args.output_dir, f"{args.split}_captions_{ver}_shard{args.shard_id}.jsonl")
    else:
        output_file = os.path.join(args.output_dir, f"{args.split}_captions_{ver}.jsonl")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    done_ids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line.strip())["image_id"])
        print(f"Resume: {len(done_ids)} already done.")

    paths = [p for p in paths if os.path.basename(p) not in done_ids]
    if not paths:
        print("No new images to process.")
        return

    print(f"Processing {len(paths)} images on {device}, output -> {output_file}")
    model, processor = load_model(args.model_path, device)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, img_path in enumerate(paths):
            image_id = os.path.basename(img_path)
            print(f"  [{idx+1}/{len(paths)}] {image_id}", end="", flush=True)
            t0 = time.time()
            try:
                image = Image.open(img_path).convert("RGB")
                caption = generate_caption(model, processor, image)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s)")

            record = {
                "image_id": image_id,
                "split": args.split,
                "prompt_version": args.version,
                "caption": caption,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Done. Output: {output_file}")


if __name__ == "__main__":
    main()
