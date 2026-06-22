# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning course final project (华中科技大学 软件学院) on **image captioning and multimodal learning**. The project has two main tasks:

1. **Task 1 — Prompt Engineering**: Use a multimodal LLM to generate captions/annotations for unlabeled images. Design prompts for object recognition, short story generation, and image category classification.
2. **Task 2 — Image Captioning Model**: Train a CNN-RNN (or similar) image captioning model using the annotations from Task 1. Compare architectures with and without attention mechanisms.

## Data Layout

```
data/
  train/    # 2000 JPG images (training set)
  val/      # 369 JPG images (validation set)
```

Images are real-world scene photos with no human annotations. The `.gitignore` excludes `/data` from version control.

## Task Breakdown

The project requires implementing these components (see `final_project_agent_brief.md` for full specification):

1. **Annotation generation** — call a multimodal LLM API on each image with multiple prompt versions; save results as JSONL (one record per image with `image_id`, `caption`, `objects`, `category`, `short_story`).
2. **Model training** — build an image captioning model (CNN encoder + RNN/LSTM/GRU decoder, with optional attention); train on generated annotations.
3. **Evaluation** — compute BLEU, METEOR, CIDEr on the validation set; optionally use model-based metrics.
4. **Ablation** — compare no-attention baseline vs. attention-based model.

## Submission Requirements

- Code: prompt engineering scripts, model training/testing code, README, environment spec, deployment docs.
- Report: ≤8 pages (excluding references/appendix), academic paper structure (Abstract → Conclusion), LaTeX preferred.
- Scoring: Report 50%, Program 20%, Innovation 30%.
