"""
Task 1: 使用 Qwen3.5-9B 多模态大模型为图片生成多维度标注。

生成内容：
  - caption: 图片的详细描述
  - objects: 图片中包含的物体列表
  - category: 图片类别（真实照片/表情包/网络图片/软件截图等）
  - short_story: 根据图片生成的 1-2 句简短故事

输出格式：JSONL，每行一条记录。

用法：
  # 测试前 10 张图片
  python scripts/generate_annotations.py --num_images 10

  # 生成全部训练集
  python scripts/generate_annotations.py --split train

  # 生成验证集
  python scripts/generate_annotations.py --split val

  # 指定 prompt 版本
  python scripts/generate_annotations.py --num_images 10 --prompt_version v1
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ============================================================
# Prompt 定义（多版本）
# ============================================================

PROMPTS = {
    "v1": {
        "caption": "Describe this image in 2-3 sentences. Focus on the main subjects, actions, and setting.",
        "objects": "List the main objects visible in this image. Return ONLY a plain JSON array of short object name strings, like [\"car\", \"tree\", \"person\"]. Do NOT include bounding boxes, coordinates, or any extra fields. Just the array of strings.",
        "category": "Classify this image into one of these categories: 真实照片 (real photo), 表情包 (meme), 网络图片 (web image), 软件截图 (screenshot), 插画 (illustration), 其他 (other). Return only the category name in Chinese.",
        "short_story": "Write a 1-2 sentence short story inspired by this image. Be creative and concise.",
    },
    "v2": {
        "caption": "请用2-3句话详细描述这张图片的内容，包括主要物体、场景和氛围。",
        "objects": "请列出这张图片中可以看到的主要物体名称。只返回一个纯JSON字符串数组，例如 [\"猫\", \"沙发\", \"窗户\"]。不要包含坐标、边界框或其他信息，只返回物体名称的数组。",
        "category": "请将这张图片分类为以下类别之一：真实照片、表情包、网络图片、软件截图、插画、其他。只返回类别名称。",
        "short_story": "请根据这张图片写一个1-2句话的简短故事，要有创意。",
    },
    "v3": {
        "caption": "Analyze this image carefully. Describe the scene, objects, people (if any), colors, mood, and any notable details in 2-4 sentences.",
        "objects": "What are the main objects and elements in this image? Return ONLY a JSON array of simple name strings, e.g. [\"red car\", \"traffic light\", \"pedestrian\"]. No bounding boxes, no coordinates, no extra fields.",
        "category": "What type of image is this? Choose one: 真实照片, 表情包, 网络图片, 软件截图, 插画, 其他. Return only the category.",
        "short_story": "Imagine the story behind this image. Write 1-2 sentences capturing a possible narrative or emotion.",
    },
}

# ============================================================
# 辅助函数
# ============================================================

def load_model(model_path: str, device_map: str = "auto"):
    """加载 Qwen3.5-9B 模型和 processor。"""
    print(f"正在加载模型: {model_path}")
    print(f"device_map: {device_map}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    print("模型加载完成。")
    return model, processor


def generate_response(model, processor, image: Image.Image, prompt: str) -> str:
    """对单张图片 + 单条 prompt 生成回复。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # 截取生成部分
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def parse_objects(raw: str) -> list:
    """尝试从模型输出中解析物体列表。"""
    import re

    # 尝试提取 JSON 数组
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        # 处理模型返回 {bbox_2d: ..., label: ...} 格式
                        if "label" in item:
                            result.append(str(item["label"]).strip())
                        elif "object" in item:
                            result.append(str(item["object"]).strip())
                        else:
                            # 取第一个值作为名称
                            result.append(str(list(item.values())[0]).strip())
                    else:
                        result.append(str(item).strip())
                return result
        except json.JSONDecodeError:
            pass

    # 回退：按逗号/换行分割
    items = re.split(r"[,，\n]", raw)
    return [item.strip().strip('"').strip("'") for item in items if item.strip()]


def get_image_paths(data_dir: str, split: str, num_images: int = -1) -> list:
    """获取图片路径列表。"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"数据目录不存在: {split_dir}")

    # 获取所有 jpg 文件并排序
    images = sorted(
        [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
    )

    if num_images > 0:
        images = images[:num_images]

    return [os.path.join(split_dir, img) for img in images]


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Task 1: 使用多模态大模型生成图片标注")
    parser.add_argument("--model_path", type=str,
                        default="/home/turing1/jhcao/final-project/checkpoint/Qwen3.5-9B",
                        help="模型权重路径")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/jhcao/final-project/data",
                        help="数据根目录")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="数据集划分")
    parser.add_argument("--num_images", type=int, default=-1,
                        help="处理图片数量，-1 表示全部")
    parser.add_argument("--prompt_version", type=str, default="v1",
                        choices=list(PROMPTS.keys()),
                        help="Prompt 版本")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/jhcao/final-project/annotations",
                        help="输出目录")
    parser.add_argument("--resume", action="store_true",
                        help="从已有输出文件断点续跑")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="device_map 参数，如 'auto' 或 'cuda:0' 或指定多卡 '{0: \"cuda:0\", 1: \"cuda:1\"}'")
    args = parser.parse_args()

    # 准备输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"{args.split}_captions_{args.prompt_version}.jsonl"
    )

    # 加载已完成的 image_id（断点续跑）
    done_ids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done_ids.add(record["image_id"])
        print(f"断点续跑：已完成 {len(done_ids)} 张，跳过这些图片。")

    # 获取图片列表
    image_paths = get_image_paths(args.data_dir, args.split, args.num_images)
    image_paths = [p for p in image_paths if os.path.basename(p) not in done_ids]
    print(f"待处理图片数: {len(image_paths)}")

    if len(image_paths) == 0:
        print("没有需要处理的图片，退出。")
        return

    # 加载模型
    model, processor = load_model(args.model_path, device_map=args.device_map)

    # 获取 prompt 模板
    prompts = PROMPTS[args.prompt_version]
    print(f"使用 Prompt 版本: {args.prompt_version}")

    # 逐张处理
    total = len(image_paths)
    start_time = time.time()

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, img_path in enumerate(image_paths):
            image_id = os.path.basename(img_path)
            print(f"\n[{idx + 1}/{total}] 处理: {image_id}")

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  ⚠ 无法读取图片: {e}")
                continue

            record = {
                "image_id": image_id,
                "split": args.split,
                "prompt_version": args.prompt_version,
            }

            # 生成各维度标注
            t0 = time.time()

            print("  → 生成 caption ...")
            record["caption"] = generate_response(model, processor, image, prompts["caption"])

            print("  → 生成 objects ...")
            raw_objects = generate_response(model, processor, image, prompts["objects"])
            record["objects"] = parse_objects(raw_objects)

            print("  → 生成 category ...")
            record["category"] = generate_response(model, processor, image, prompts["category"])

            print("  → 生成 short_story ...")
            record["short_story"] = generate_response(model, processor, image, prompts["short_story"])

            elapsed = time.time() - t0
            print(f"  ✓ 完成 ({elapsed:.1f}s)")

            # 写入 JSONL
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    total_time = time.time() - start_time
    print(f"\n全部完成！共处理 {total} 张图片，耗时 {total_time:.1f}s")
    print(f"结果保存至: {output_file}")


if __name__ == "__main__":
    main()
