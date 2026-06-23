"""
Task 1: 使用 Qwen3.5-9B 多模态大模型为图片生成多维度标注。

生成内容：
  - caption: 图片的详细描述
  - objects: 图片中包含的物体列表
  - category: 图片类别
  - short_story: 根据图片生成的简短故事

输出格式：JSONL，每行一条记录。

用法：
  # 测试单个 prompt 版本
  python scripts/generate_annotations.py --num_images 10 --prompt_version v1

  # 批量测试所有版本并生成对比报告
  python scripts/generate_annotations.py --num_images 10 --compare

  # 生成全部训练集
  python scripts/generate_annotations.py --split train --prompt_version v3
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
# Prompt 定义（5 组，全部英文）
#
# v1 — Baseline: 简单直接，无格式约束
# v2 — Improved: 增加具体约束（句数、格式）
# v3 — Advanced:  结构化引导，明确输出规范
# v4 — Rich:      侧重高信息量，要求细节与属性
# v5 — Best:      综合最优，兼顾质量与结构化
# ============================================================

PROMPTS = {
    # ── v1: Baseline ──────────────────────────────────────────
    # 最简单的指令，不指定格式、不指定细节要求
    "v1": {
        "caption": "Describe this image.",
        "objects": "List the objects in this image.",
        "category": "What type of image is this?",
        "short_story": "Write a short story about this image.",
    },

    # ── v2: Improved ──────────────────────────────────────────
    # 在 v1 基础上增加：句数约束、输出格式、类别选项
    "v2": {
        "caption": "Describe this image in 2-3 sentences. Include the main subjects, actions, and setting.",
        "objects": "List the main objects visible in this image. Return a JSON array of object name strings, e.g. [\"car\", \"tree\"].",
        "category": "Classify this image into one of: real photo, meme, web image, screenshot, illustration, other. Return only the category name.",
        "short_story": "Write a 1-2 sentence short story inspired by this image.",
    },

    # ── v3: Advanced ──────────────────────────────────────────
    # 结构化引导：分维度描述，明确排除无关信息，给出示例
    "v3": {
        "caption": (
            "Analyze this image and describe it in 3-4 sentences. "
            "Cover: (1) the main subjects and their actions, "
            "(2) the setting and environment, "
            "(3) colors, lighting, and overall mood. "
            "Be specific and factual."
        ),
        "objects": (
            "Identify the distinct objects, people, animals, and elements in this image. "
            "Return ONLY a plain JSON array of short name strings, e.g. [\"red car\", \"traffic light\", \"pedestrian\"]. "
            "Do NOT include bounding boxes, coordinates, or any extra structure."
        ),
        "category": (
            "Classify this image into exactly one category from this list: "
            "real photo, meme, web image, screenshot, illustration, other. "
            "Return only the category name, nothing else."
        ),
        "short_story": (
            "Imagine the story behind this image. Write 2-3 sentences that capture "
            "a possible narrative, emotion, or moment. Be creative yet grounded in what you see."
        ),
    },

    # ── v4: Rich (高信息量) ────────────────────────────────────
    # 侧重细节丰富度：要求包含属性、数量、空间关系、氛围
    "v4": {
        "caption": (
            "Provide a detailed description of this image in 4-5 sentences. "
            "Include: the number and identities of people (if any), "
            "specific objects and their attributes (color, size, material), "
            "spatial relationships between elements, "
            "the setting (indoor/outdoor, time of day, weather), "
            "and the overall atmosphere or emotional tone. "
            "Avoid vague language; use concrete, observable details."
        ),
        "objects": (
            "List every distinct object and element you can identify in this image. "
            "For important objects, include a brief attribute (color, material, or state), "
            "e.g. [\"wooden table\", \"red bicycle\", \"cloudy sky\", \"person wearing blue jacket\"]. "
            "Return ONLY a JSON array of strings. Be thorough."
        ),
        "category": (
            "Classify this image into one of: real photo, meme, web image, screenshot, illustration, other. "
            "Then briefly explain your reasoning in one sentence. "
            "Format: {\"category\": \"...\", \"reason\": \"...\"}"
        ),
        "short_story": (
            "Write a 2-3 sentence narrative inspired by this image. "
            "Incorporate specific visual details you observe — names, emotions, actions, or implied context. "
            "Aim to make the reader feel present in the scene."
        ),
    },

    # ── v5: Best Practice (综合最优) ───────────────────────────
    # 角色设定 + 结构化输出 + 质量约束
    "v5": {
        "caption": (
            "You are an expert image analyst. Describe this image comprehensively in 4-5 sentences. "
            "Structure your description as follows: "
            "Sentence 1: Overall scene summary (what is happening). "
            "Sentence 2: Key subjects and their actions or positions. "
            "Sentence 3: Environmental details (location, time, weather, lighting). "
            "Sentence 4: Visual qualities (colors, textures, composition). "
            "Sentence 5: Mood, atmosphere, or implied narrative. "
            "Be precise, vivid, and avoid repetition."
        ),
        "objects": (
            "You are a precise object detector. List all distinct objects, people, animals, "
            "text, and notable elements visible in this image. "
            "For each entry, use the format \"object (attribute)\" where attribute is color, "
            "material, state, or position — e.g. [\"red sports car\", \"elderly man (sitting)\", "
            "\"wooden fence (weathered)\", \"STOP sign\"]. "
            "Return ONLY a JSON array of strings. Aim for completeness."
        ),
        "category": (
            "Classify this image into exactly one category from: "
            "real photo, meme, web image, screenshot, illustration, other. "
            "Return a JSON object: {\"category\": \"...\", \"confidence\": \"high/medium/low\", \"reason\": \"...\"}"
        ),
        "short_story": (
            "You are a creative writer. Craft a 2-3 sentence story inspired by this image. "
            "Your story should: (1) reference specific visual elements you observe, "
            "(2) convey a clear emotion or theme, and "
            "(3) leave the reader with a memorable impression. "
            "Avoid generic descriptions; make every word count."
        ),
    },

    # ── v6: Enhanced Structured (基于 v3 增强) ─────────────────
    # 保留 v3 的格式稳定性，增强细节丰富度和物体覆盖率
    "v6": {
        "caption": (
            "Analyze this image and describe it in 3-4 sentences. "
            "Cover: (1) the main subjects, their actions, and visible attributes "
            "(color, material, size), "
            "(2) the setting, spatial layout, and environment details, "
            "(3) lighting, colors, and overall mood. "
            "Use concrete, specific language — avoid vague words like 'nice', 'some', 'various'."
        ),
        "objects": (
            "List ALL distinct objects, people, animals, plants, text, and visual elements "
            "you can identify in this image. Be thorough — aim for 15-25 items. "
            "For each item, use the format 'object' or 'adjective object' "
            "(e.g. \"red car\", \"wooden table\", \"elderly woman\", \"STOP sign\"). "
            "Return ONLY a plain JSON array of short strings. "
            "Do NOT include bounding boxes, coordinates, descriptions, or any extra structure."
        ),
        "category": (
            "Classify this image into exactly one category from this list: "
            "real photo, meme, web image, screenshot, illustration, other. "
            "Return only the category name, nothing else."
        ),
        "short_story": (
            "Imagine the story behind this image. Write 2-3 sentences that capture "
            "a possible narrative, emotion, or moment. "
            "Reference at least two specific visual details you observe. "
            "Be creative yet grounded in what you see."
        ),
    },

    # ── v7: Constrained Caption (针对 v6 问题优化) ─────────────
    # 修复 v6 的三大问题：开头单调 (68% "The image")、长度过长 (avg 134w)、含 \n (32%)
    # 遵循六条 prompt 工程原则：明确具体、设定角色、提供背景、指定格式、提供示例、设定限制
    "v7": {
        "caption": (
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
        ),
        "objects": (
            "List ALL distinct objects, people, animals, plants, text, and visual elements "
            "you can identify in this image. Be thorough — aim for 15-25 items. "
            "For each item, use the format 'object' or 'adjective object' "
            "(e.g. \"red car\", \"wooden table\", \"elderly woman\", \"STOP sign\"). "
            "Return ONLY a plain JSON array of short strings. "
            "Do NOT include bounding boxes, coordinates, descriptions, or any extra structure."
        ),
        "category": (
            "Classify this image into exactly one category from this list: "
            "real photo, meme, web image, screenshot, illustration, other. "
            "Return only the category name, nothing else."
        ),
        "short_story": (
            "Imagine the story behind this image. Write 2-3 sentences that capture "
            "a possible narrative, emotion, or moment. "
            "Reference at least two specific visual details you observe. "
            "Be creative yet grounded in what you see."
        ),
    },
}

# Prompt 设计说明（用于报告输出）
PROMPT_DESCRIPTIONS = {
    "v1": {
        "name": "Baseline",
        "strategy": "Simple, minimal instructions with no format constraints or detail requirements.",
        "techniques": ["Direct instruction", "No output format specified"],
    },
    "v2": {
        "name": "Improved",
        "strategy": "Adds specific constraints: sentence count, output format, category options.",
        "techniques": ["Length constraint", "Format specification", "Enumerated choices"],
    },
    "v3": {
        "name": "Advanced",
        "strategy": "Structured guidance with explicit dimensions and negative constraints.",
        "techniques": ["Structured breakdown (subjects/actions/mood)", "Negative constraints (no bounding boxes)", "Few-shot examples"],
    },
    "v4": {
        "name": "Rich Description",
        "strategy": "Maximizes information density: attributes, quantities, spatial relations, atmosphere.",
        "techniques": ["Attribute requirements (color, size, material)", "Spatial relationship guidance", "Concrete detail emphasis", "Reasoning request"],
    },
    "v5": {
        "name": "Best Practice",
        "strategy": "Combines role prompting, structured output template, and quality constraints.",
        "techniques": ["Role assignment (expert analyst)", "Per-sentence structure template", "Completeness instruction", "Vivid language guidance"],
    },
    "v6": {
        "name": "Enhanced Structured",
        "strategy": "Keeps v3's format stability while pushing for richer details and more thorough object coverage.",
        "techniques": ["Structured breakdown (subjects/environment/mood)", "Negative constraints (no bounding boxes)", "Explicit quantity target (15-25 items)", "Anti-vagueness constraint", "Visual detail grounding in stories"],
    },
    "v7": {
        "name": "Constrained Caption",
        "strategy": "Fixes v6's monotonous openings, excessive length, and newline issues using 6 prompt engineering principles.",
        "techniques": [
            "Role setting (professional image captioner)",
            "Context (training data for captioning model)",
            "Format spec (2-3 sentences, 50-80 words, plain paragraph)",
            "Few-shot examples (two concrete opening samples)",
            "Negative constraints (no meta-opens, no markdown, no line breaks, no vague words)",
            "Word count hard limit (50-80 words)"
        ],
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

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def parse_objects(raw: str) -> list:
    """尝试从模型输出中解析物体列表。"""
    import re

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        if "label" in item:
                            result.append(str(item["label"]).strip())
                        elif "object" in item:
                            result.append(str(item["object"]).strip())
                        else:
                            result.append(str(list(item.values())[0]).strip())
                    else:
                        result.append(str(item).strip())
                return result
        except json.JSONDecodeError:
            pass

    items = re.split(r"[,，\n]", raw)
    return [item.strip().strip('"').strip("'") for item in items if item.strip()]


def get_image_paths(data_dir: str, split: str, num_images: int = -1) -> list:
    """获取图片路径列表。"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Data directory not found: {split_dir}")

    images = sorted(
        [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
    )

    if num_images > 0:
        images = images[:num_images]

    return [os.path.join(split_dir, img) for img in images]


# ============================================================
# 单版本生成
# ============================================================

def run_single_version(model, processor, image_paths: list, prompt_version: str,
                       split: str, output_dir: str, resume: bool = False) -> str:
    """运行单个 prompt 版本的标注生成，返回输出文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}_captions_{prompt_version}.jsonl")

    done_ids = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done_ids.add(record["image_id"])
        print(f"  Resume: {len(done_ids)} already done, skipping.")

    paths = [p for p in image_paths if os.path.basename(p) not in done_ids]
    if not paths:
        print(f"  No new images to process for {prompt_version}.")
        return output_file

    prompts = PROMPTS[prompt_version]
    total = len(paths)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, img_path in enumerate(paths):
            image_id = os.path.basename(img_path)
            print(f"  [{idx + 1}/{total}] {image_id}", end="", flush=True)

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            record = {
                "image_id": image_id,
                "split": split,
                "prompt_version": prompt_version,
            }

            t0 = time.time()

            record["caption"] = generate_response(model, processor, image, prompts["caption"])
            raw_objects = generate_response(model, processor, image, prompts["objects"])
            record["objects"] = parse_objects(raw_objects)
            record["category"] = generate_response(model, processor, image, prompts["category"])
            record["short_story"] = generate_response(model, processor, image, prompts["short_story"])

            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s)")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return output_file


# ============================================================
# 对比报告生成
# ============================================================

def generate_comparison_report(output_dir: str, split: str, versions: list):
    """读取各版本结果，生成 Markdown 对比报告。"""
    report_path = os.path.join(output_dir, "prompt_comparison_report.md")

    # 加载各版本数据
    all_data = {}
    for v in versions:
        jsonl_path = os.path.join(output_dir, f"{split}_captions_{v}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"  Warning: {jsonl_path} not found, skipping.")
            continue
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        all_data[v] = {r["image_id"]: r for r in records}

    if not all_data:
        print("  No data found for comparison.")
        return

    # 获取公共 image_id 列表
    common_ids = set.intersection(*[set(d.keys()) for d in all_data.values()])
    common_ids = sorted(common_ids, key=lambda x: int(os.path.splitext(x)[0]))

    # 生成报告
    lines = []
    lines.append("# Prompt Engineering Comparison Report\n")
    lines.append(f"**Dataset**: {split} | **Images compared**: {len(common_ids)} | **Versions**: {', '.join(versions)}\n")

    # ── Prompt 策略总览 ──
    lines.append("## 1. Prompt Design Overview\n")
    lines.append("| Version | Name | Strategy | Key Techniques |")
    lines.append("|---------|------|----------|----------------|")
    for v in versions:
        desc = PROMPT_DESCRIPTIONS.get(v, {})
        name = desc.get("name", v)
        strategy = desc.get("strategy", "")
        techniques = ", ".join(desc.get("techniques", []))
        lines.append(f"| {v} | {name} | {strategy} | {techniques} |")
    lines.append("")

    # ── 各版本 Prompt 原文 ──
    lines.append("## 2. Prompt Details\n")
    for v in versions:
        lines.append(f"### {v} — {PROMPT_DESCRIPTIONS.get(v, {}).get('name', '')}\n")
        for dim in ["caption", "objects", "category", "short_story"]:
            lines.append(f"**{dim}**:")
            lines.append(f"> {PROMPTS[v][dim]}\n")
        lines.append("")

    # ── 逐图对比 ──
    lines.append("## 3. Per-Image Comparison\n")
    for img_id in common_ids:
        lines.append(f"### {img_id}\n")
        for dim in ["caption", "objects", "category", "short_story"]:
            lines.append(f"#### {dim}\n")
            lines.append("| Version | Output |")
            lines.append("|---------|--------|")
            for v in versions:
                if img_id in all_data.get(v, {}):
                    val = all_data[v][img_id].get(dim, "")
                    if isinstance(val, list):
                        val = ", ".join(str(x) for x in val)
                    # 截断过长内容
                    display = val if len(val) <= 500 else val[:500] + "..."
                    # 转义管道符
                    display = display.replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {v} | {display} |")
            lines.append("")

    # ── 统计摘要 ──
    lines.append("## 4. Summary Statistics\n")
    lines.append("| Version | Avg Caption Len (words) | Avg Objects Count | Category Distribution |")
    lines.append("|---------|------------------------|-------------------|----------------------|")

    for v in versions:
        if v not in all_data:
            continue
        records = [all_data[v][img_id] for img_id in common_ids if img_id in all_data[v]]
        # caption 平均长度
        caption_lens = [len(r.get("caption", "").split()) for r in records]
        avg_caption = sum(caption_lens) / len(caption_lens) if caption_lens else 0
        # objects 平均数量
        obj_counts = [len(r.get("objects", [])) for r in records]
        avg_objects = sum(obj_counts) / len(obj_counts) if obj_counts else 0
        # category 分布
        cats = {}
        for r in records:
            c = r.get("category", "unknown")
            # 从 JSON 对象中提取 category
            if isinstance(c, str) and c.startswith("{"):
                try:
                    c = json.loads(c).get("category", c)
                except json.JSONDecodeError:
                    pass
            cats[c] = cats.get(c, 0) + 1
        cat_str = "; ".join(f"{k}: {v}" for k, v in cats.items())
        lines.append(f"| {v} | {avg_caption:.1f} | {avg_objects:.1f} | {cat_str} |")

    lines.append("")

    # ── 结论模板 ──
    lines.append("## 5. Observations & Conclusion\n")
    lines.append("*TODO: Fill in after reviewing the comparisons.*\n")
    lines.append("- **Best version for caption quality**: ")
    lines.append("- **Best version for object detection completeness**: ")
    lines.append("- **Best version for short story creativity**: ")
    lines.append("- **Overall recommended version**: ")
    lines.append("- **Key prompt engineering insights**: ")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nComparison report saved to: {report_path}")
    return report_path


# ============================================================
# 主流程
# ============================================================

ALL_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]

def main():
    parser = argparse.ArgumentParser(description="Task 1: Multi-modal annotation generation")
    parser.add_argument("--model_path", type=str,
                        default="/home/turing1/jhcao/final-project/checkpoint/Qwen3.5-9B",
                        help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/jhcao/final-project/data",
                        help="Data root directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"], help="Dataset split")
    parser.add_argument("--num_images", type=int, default=-1,
                        help="Number of images to process (-1 for all)")
    parser.add_argument("--prompt_version", type=str, default="v1",
                        choices=ALL_VERSIONS, help="Prompt version")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/jhcao/final-project/annotations",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading")
    parser.add_argument("--compare", action="store_true",
                        help="Run ALL prompt versions and generate comparison report")
    args = parser.parse_args()

    # 获取图片列表
    image_paths = get_image_paths(args.data_dir, args.split, args.num_images)
    print(f"Total images: {len(image_paths)}")

    if args.compare:
        # 批量测试所有版本
        model, processor = load_model(args.model_path, device_map=args.device_map)

        for v in ALL_VERSIONS:
            print(f"\n{'='*60}")
            print(f"Running prompt version: {v} — {PROMPT_DESCRIPTIONS[v]['name']}")
            print(f"{'='*60}")
            run_single_version(model, processor, image_paths, v,
                               args.split, args.output_dir, args.resume)

        # 生成对比报告
        print(f"\n{'='*60}")
        print("Generating comparison report...")
        print(f"{'='*60}")
        generate_comparison_report(args.output_dir, args.split, ALL_VERSIONS)

    else:
        # 单版本模式
        model, processor = load_model(args.model_path, device_map=args.device_map)
        print(f"Using prompt version: {args.prompt_version}")
        run_single_version(model, processor, image_paths, args.prompt_version,
                           args.split, args.output_dir, args.resume)

    print("\nDone!")


if __name__ == "__main__":
    main()
