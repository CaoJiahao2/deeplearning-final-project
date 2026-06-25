"""Comprehensive Image Captioning Evaluation

Evaluates multiple model configurations across 3 metric categories:
  1. Reference-based: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, BERTScore
  2. Reference-free: CLIPScore (image-text alignment)
  3. Quality/Diversity: Distinct-1/2, avg length, repetition rate

Usage:
  python scripts/eval_comprehensive.py [--subset N] [--gpu 0]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CaptionModel, Vocabulary
from models.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def compute_bertscore(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute BERTScore (P, R, F1)."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(hyps, refs, lang="en", verbose=True,
                          model_type="roberta-large", device="cuda",
                          batch_size=64, rescale_with_baseline=True)
    return {
        "BERTScore-P": P.mean().item(),
        "BERTScore-R": R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }


def compute_clipscore(images: List[str], hyps: List[str]) -> Dict[str, float]:
    """Compute CLIPScore (reference-free image-text alignment).

    Uses transformers CLIP model directly to compute cosine similarity
    between image and text embeddings.
    """
    from transformers import CLIPProcessor, CLIPModel

    clip_path = "checkpoint/pretrained/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_path).cuda().eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_path)

    scores = []
    batch_size = 32

    for i in tqdm(range(0, len(images), batch_size), desc="CLIPScore"):
        batch_paths = images[i:i+batch_size]
        batch_hyps = hyps[i:i+batch_size]

        # Load images
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = clip_processor(
            text=batch_hyps,
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = clip_model(**inputs)
            img_embeds = outputs.image_embeds  # [B, dim]
            txt_embeds = outputs.text_embeds   # [B, dim]

            # Normalize
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity * 2.5 (CLIPScore formula)
            cosine_sim = (img_embeds * txt_embeds).sum(dim=-1)  # [B]
            batch_scores = (2.5 * cosine_sim.clamp(min=0)).cpu().tolist()
            scores.extend(batch_scores)

    del clip_model
    torch.cuda.empty_cache()

    return {"CLIPScore": sum(scores) / len(scores)}


def compute_diversity(hyps: List[str]) -> Dict[str, float]:
    """Compute diversity metrics: Distinct-1/2, avg length, repetition rate."""
    all_unigrams = []
    all_bigrams = []
    lengths = []
    repetition_counts = []

    for hyp in hyps:
        tokens = hyp.lower().split()
        lengths.append(len(tokens))
        all_unigrams.extend(tokens)

        for i in range(len(tokens) - 1):
            all_bigrams.append((tokens[i], tokens[i+1]))

        # Repetition rate: fraction of repeated n-grams
        if len(tokens) > 1:
            bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
            repeated = sum(v - 1 for v in bigram_counts.values() if v > 1)
            total_bigrams = len(tokens) - 1
            repetition_counts.append(repeated / max(total_bigrams, 1))
        else:
            repetition_counts.append(0.0)

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    avg_length = sum(lengths) / len(lengths)
    avg_repetition = sum(repetition_counts) / len(repetition_counts)

    return {
        "Distinct-1": distinct_1,
        "Distinct-2": distinct_2,
        "AvgLength": avg_length,
        "RepRate": avg_repetition,
    }


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def generate_all_captions(
    model: CaptionModel,
    image_dir: str,
    image_ids: List[str],
    transform: transforms.Compose,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 3,
) -> Dict[str, str]:
    """Generate captions for all images."""
    model.eval()
    results = {}

    for img_id in tqdm(image_ids, desc=f"Gen({strategy}{'-'+str(beam_size) if strategy=='beam' else ''})"):
        img_path = os.path.join(image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            seqs = model.generate(
                tensor,
                strategy=strategy,
                max_len=64,
                beam_size=beam_size,
                repetition_penalty=1.2,
            )

        caption = model.decoder if hasattr(model, 'decoder') else None
        # Decode using vocab
        results[img_id] = ""  # will be filled below

    return results


def generate_captions_for_config(
    arch: str,
    backbone: str,
    ckpt_path: str,
    clip_path: str,
    norm_type: str,
    image_dir: str,
    image_ids: List[str],
    vocab: Vocabulary,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 3,
) -> Dict[str, str]:
    """Load model and generate captions."""
    # Build model
    model_kwargs = dict(
        arch=arch,
        backbone=backbone,
        embed_dim=512,
        hidden_dim=512,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,
        attention_dim=256,
        fine_tune_layers="none" if backbone == "clip" else "layer4",
        pretrained=False,
    )
    if backbone == "clip":
        model_kwargs["clip_path"] = clip_path

    model = CaptionModel(**model_kwargs).to(device)

    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Image transform
    if norm_type == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Generate
    results = {}
    for img_id in tqdm(image_ids, desc=f"  {strategy}" + (f"-{beam_size}" if strategy == "beam" else "")):
        img_path = os.path.join(image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
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
        results[img_id] = caption

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=0,
                    help="Evaluate on N images (0=full val set)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    # Paths
    CLIP_PATH = "checkpoint/pretrained/clip-vit-large-patch14"
    VAL_DIR = "data/val"
    ANNOTATION = "annotations/train_captions_v7.jsonl"

    # Load vocabulary
    vocab = Vocabulary.from_pretrained("bert-base-uncased")
    print(f"[Vocab] Loaded: {len(vocab)} tokens")

    # Load references from annotation
    ref_path = "annotations/val_captions_v7.jsonl"
    if not os.path.exists(ref_path):
        ref_path = "annotations/val_captions.jsonl"
    if not os.path.exists(ref_path):
        print(f"[Ref] No val annotations found, will skip reference-based metrics")
        references = {}
    else:
        references = {}
        with open(ref_path) as f:
            for line in f:
                r = json.loads(line.strip())
                references[r["image_id"]] = r["caption"]
        print(f"[Ref] Loaded {len(references)} references from {ref_path}")

    # Val image IDs (images 2002-2370)
    all_val_ids = sorted([f for f in os.listdir(VAL_DIR) if f.endswith(".jpg")],
                         key=lambda x: int(x.split(".")[0]))
    if args.subset > 0:
        all_val_ids = all_val_ids[:args.subset]
    print(f"[Val] {len(all_val_ids)} images")

    # Check if we have references for these images
    has_refs = len(references) > 0
    if has_refs:
        common_ids = sorted(set(all_val_ids) & set(references.keys()))
        print(f"[Ref] {len(common_ids)} images with references")

    # -----------------------------------------------------------------------
    # Define configurations
    # -----------------------------------------------------------------------
    configs = [
        {
            "name": "CLIP-metric-greedy",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best_metric.pth",
            "norm": "clip", "strategy": "greedy", "beam_size": 3,
        },
        {
            "name": "CLIP-metric-beam3",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best_metric.pth",
            "norm": "clip", "strategy": "beam", "beam_size": 3,
        },
        {
            "name": "CLIP-metric-beam5",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best_metric.pth",
            "norm": "clip", "strategy": "beam", "beam_size": 5,
        },
        {
            "name": "CLIP-loss-greedy",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best.pth",
            "norm": "clip", "strategy": "greedy", "beam_size": 3,
        },
        {
            "name": "CLIP-loss-beam3",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best.pth",
            "norm": "clip", "strategy": "beam", "beam_size": 3,
        },
        {
            "name": "CLIP-loss-beam5",
            "arch": "attention", "backbone": "clip",
            "ckpt": "checkpoint/v7_clip/attention_best.pth",
            "norm": "clip", "strategy": "beam", "beam_size": 5,
        },
        {
            "name": "ResNet-metric-greedy",
            "arch": "attention", "backbone": "resnet50",
            "ckpt": "checkpoint/v7/attention/attention_best_metric.pth",
            "norm": "imagenet", "strategy": "greedy", "beam_size": 3,
        },
        {
            "name": "ResNet-metric-beam3",
            "arch": "attention", "backbone": "resnet50",
            "ckpt": "checkpoint/v7/attention/attention_best_metric.pth",
            "norm": "imagenet", "strategy": "beam", "beam_size": 3,
        },
        {
            "name": "ResNet-metric-beam5",
            "arch": "attention", "backbone": "resnet50",
            "ckpt": "checkpoint/v7/attention/attention_best_metric.pth",
            "norm": "imagenet", "strategy": "beam", "beam_size": 5,
        },
    ]

    # Filter to existing checkpoints
    configs = [c for c in configs if os.path.exists(c["ckpt"])]
    print(f"[Config] {len(configs)} valid configurations")

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    all_results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {cfg['name']}")
        print(f"  Model: {cfg['backbone']} | Weights: {cfg['ckpt']}")
        print(f"  Strategy: {cfg['strategy']}" + (f" (beam={cfg['beam_size']})" if cfg['strategy'] == 'beam' else ""))
        print(f"{'='*60}")

        t0 = time.time()

        # Generate captions
        hyps_dict = generate_captions_for_config(
            arch=cfg["arch"],
            backbone=cfg["backbone"],
            ckpt_path=cfg["ckpt"],
            clip_path=CLIP_PATH,
            norm_type=cfg["norm"],
            image_dir=VAL_DIR,
            image_ids=all_val_ids,
            vocab=vocab,
            device=device,
            strategy=cfg["strategy"],
            beam_size=cfg["beam_size"],
        )

        gen_time = time.time() - t0
        print(f"  Generation time: {gen_time:.1f}s ({gen_time/len(all_val_ids)*1000:.1f}ms/image)")

        # Compute metrics
        metrics = {}

        # 1. Reference-based metrics
        if has_refs:
            common = sorted(set(all_val_ids) & set(references.keys()) & set(hyps_dict.keys()))
            ref_list = [references[uid] for uid in common]
            hyp_list = [hyps_dict[uid] for uid in common]

            print("  Computing BLEU/METEOR/ROUGE/CIDEr...")
            traditional = compute_metrics(
                {uid: references[uid] for uid in common},
                {uid: hyps_dict[uid] for uid in common},
            )
            metrics.update(traditional)

            print("  Computing BERTScore...")
            bert = compute_bertscore(hyp_list, ref_list)
            metrics.update(bert)

        # 2. CLIPScore (reference-free)
        print("  Computing CLIPScore...")
        img_paths = [os.path.join(VAL_DIR, uid) for uid in all_val_ids if uid in hyps_dict]
        hyp_list_all = [hyps_dict[uid] for uid in all_val_ids if uid in hyps_dict]
        clip_scores = compute_clipscore(img_paths, hyp_list_all)
        metrics.update(clip_scores)

        # 3. Diversity metrics
        diversity = compute_diversity(hyp_list_all)
        metrics.update(diversity)

        metrics["gen_time_s"] = gen_time
        all_results[cfg["name"]] = metrics

        # Print summary
        print(f"\n  Results for {cfg['name']}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:15s}: {v:.4f}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*120)
    print("  COMPREHENSIVE EVALUATION SUMMARY")
    print("="*120)

    # Group metrics
    ref_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore-F1"]
    free_metrics = ["CLIPScore"]
    div_metrics = ["Distinct-1", "Distinct-2", "AvgLength", "RepRate"]

    metric_groups = [
        ("Reference-based (↑ higher = better)", ref_metrics),
        ("Image-Text Alignment (↑ higher = better)", free_metrics),
        ("Diversity & Quality", div_metrics),
    ]

    for group_name, metric_keys in metric_groups:
        print(f"\n  {group_name}:")
        # Header
        header = f"  {'Config':<25s}"
        for mk in metric_keys:
            header += f" {mk:>12s}"
        print(header)
        print("  " + "-" * len(header))

        # Data rows
        for cfg_name, scores in all_results.items():
            row = f"  {cfg_name:<25s}"
            for mk in metric_keys:
                val = scores.get(mk, 0.0)
                if isinstance(val, float):
                    row += f" {val:>12.4f}"
                else:
                    row += f" {str(val):>12s}"
            print(row)

    # Find best config per metric
    print(f"\n  Best configuration per metric:")
    for group_name, metric_keys in metric_groups:
        for mk in metric_keys:
            best_name = max(all_results.keys(), key=lambda k: all_results[k].get(mk, 0))
            best_val = all_results[best_name].get(mk, 0)
            print(f"    {mk:15s}: {best_name:<25s} ({best_val:.4f})")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # JSON
    with open("results/eval_comprehensive.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to results/eval_comprehensive.json")

    # CSV
    all_metric_keys = ref_metrics + free_metrics + div_metrics + ["gen_time_s"]
    with open("results/eval_summary.csv", "w") as f:
        header = "config," + ",".join(all_metric_keys)
        f.write(header + "\n")
        for cfg_name, scores in all_results.items():
            row = cfg_name
            for mk in all_metric_keys:
                val = scores.get(mk, "")
                row += f",{val:.4f}" if isinstance(val, float) else f",{val}"
            f.write(row + "\n")
    print(f"  Saved to results/eval_summary.csv")


if __name__ == "__main__":
    main()
