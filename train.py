"""Task 2: Image Captioning — Training and Evaluation.

Two architectures:
  - baseline:  ResNet encoder + LSTM decoder (global features)
  - attention: ResNet encoder + Attention + LSTM decoder (spatial features)

Usage:
  python train.py --arch baseline --epochs 30 --gpu 0
  python train.py --arch attention --epochs 30 --gpu 0
  python train.py --eval_only --arch baseline --checkpoint checkpoint/baseline_best.pth --gpu 0
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import (
    CaptionModel,
    CaptionDataset,
    Vocabulary,
    collate_fn,
    compute_metrics,
)
from models.vocab import build_vocab_from_jsonl, PAD_IDX


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Image Captioning Training")

    # Data paths
    p.add_argument("--annotation", type=str, default="annotations/train_captions_v3.jsonl",
                    help="Path to training annotation JSONL")
    p.add_argument("--val_annotation", type=str, default=None,
                    help="Path to validation annotation JSONL (if available)")
    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--val_dir", type=str, default="data/val")

    # Model
    p.add_argument("--arch", type=str, default="baseline", choices=["baseline", "attention"],
                    help="Model architecture")
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101", "clip"])
    p.add_argument("--clip_path", type=str, default=None,
                    help="Path to local CLIP checkpoint (required when backbone=clip)")
    p.add_argument("--norm_type", type=str, default="imagenet", choices=["imagenet", "clip"],
                    help="Image normalization type (clip when using CLIP backbone)")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--attention_dim", type=int, default=256)
    p.add_argument("--fine_tune", type=str, default="layer4",
                    choices=["none", "layer4", "all"])

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_patience", type=int, default=3,
                    help="Epochs without improvement before LR reduction")
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--max_caption_len", type=int, default=64)
    p.add_argument("--min_word_freq", type=int, default=1)
    p.add_argument("--pretrained_vocab", type=str, default=None,
                    help="Pretrained tokenizer name (e.g. bert-base-uncased). "
                         "If set, overrides --min_word_freq and uses subword tokenization.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Hardware
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Experiment output dir under checkpoint/ (e.g. baseline_v7). "
                         "If set, saves checkpoints, logs, tensorboard to checkpoint/<output_dir>/")
    p.add_argument("--save_every", type=int, default=5)

    # Evaluation
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--decode_strategy", type=str, default="greedy",
                    choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=5)

    # Early stopping
    p.add_argument("--early_stopping_patience", type=int, default=10,
                    help="Stop training if val_loss doesn't improve for N epochs")
    p.add_argument("--repetition_penalty", type=float, default=1.2,
                    help="Penalty for repeated tokens during beam search (>1.0 to reduce repetition)")

    # Resume training
    p.add_argument("--resume", action="store_true",
                    help="Resume training from checkpoint")
    p.add_argument("--start_epoch", type=int, default=0,
                    help="Start epoch when resuming (0=auto from history)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args) -> Tuple[DataLoader, Optional[DataLoader], Vocabulary]:
    """Load annotations, build vocabulary, create DataLoaders."""
    # Build vocabulary from training annotations
    vocab = build_vocab_from_jsonl(args.annotation, min_freq=args.min_word_freq,
                                   pretrained=args.pretrained_vocab)

    # Training dataset
    train_dataset = CaptionDataset(
        image_dir=args.train_dir,
        annotation_path=args.annotation,
        vocab=vocab,
        max_caption_len=args.max_caption_len,
        is_train=True,
        norm_type=getattr(args, 'norm_type', 'imagenet'),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataset (if annotation exists)
    val_loader = None
    if args.val_annotation and os.path.exists(args.val_annotation):
        val_dataset = CaptionDataset(
            image_dir=args.train_dir,  # val annotations may reference train images
            annotation_path=args.val_annotation,
            vocab=vocab,
            max_caption_len=args.max_caption_len,
            is_train=False,
            norm_type=getattr(args, 'norm_type', 'imagenet'),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader, vocab


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CaptionModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    grad_clip: float,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """Train for one epoch. Returns (average_loss, global_step)."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch_idx, (images, captions, caption_lens, _) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        caption_lens = caption_lens.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images, captions, caption_lens)  # [B, max_len, vocab]
                targets = captions[:, 1:]  # [B, max_len-1]
                pred_logits = logits[:, :-1, :]  # [B, max_len-1, vocab]
                pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))  # [N, vocab]
                target_flat = targets.reshape(-1)  # [N]
                loss = criterion(pred_flat, target_flat)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, captions, caption_lens)
            targets = captions[:, 1:]
            pred_logits = logits[:, :-1, :]
            pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
            target_flat = targets.reshape(-1)
            loss = criterion(pred_flat, target_flat)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        global_step += 1

        avg_loss = total_loss / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if writer is not None:
            writer.add_scalar("train/loss_step", loss.item(), global_step)

    return total_loss / max(total_samples, 1), global_step


@torch.no_grad()
def validate(
    model: CaptionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, captions, caption_lens, _ in loader:
        images = images.to(device)
        captions = captions.to(device)
        caption_lens = caption_lens.to(device)

        logits = model(images, captions, caption_lens)
        targets = captions[:, 1:]
        pred_logits = logits[:, :-1, :]
        pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
        target_flat = targets.reshape(-1)
        loss = criterion(pred_flat, target_flat)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation (caption generation + metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_captions(
    model: CaptionModel,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 5,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate captions for all samples in the loader.

    Returns:
        references: {image_id: reference_caption}
        hypotheses: {image_id: generated_caption}
    """
    model.eval()
    references = {}
    hypotheses = {}

    for images, captions, caption_lens, image_ids in loader:
        images = images.to(device)

        # Generate
        sequences = model.generate(images, strategy=strategy, beam_size=beam_size)

        for i, (img_id, seq) in enumerate(zip(image_ids, sequences)):
            # Decode hypothesis
            hyp_text = vocab.decode(seq, skip_special=True)
            hypotheses[img_id] = hyp_text

            # Decode reference
            ref_indices = captions[i][:caption_lens[i]].tolist()
            ref_text = vocab.decode(ref_indices, skip_special=True)
            references[img_id] = ref_text

    return references, hypotheses


def evaluate_metrics(
    model: CaptionModel,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 5,
) -> Dict[str, float]:
    """Full evaluation: generate captions and compute metrics."""
    references, hypotheses = generate_captions(
        model, loader, vocab, device, strategy, beam_size
    )

    if not references:
        print("[Eval] No samples to evaluate.")
        return {}

    scores = compute_metrics(references, hypotheses)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str, arch: str, resume: bool = False) -> logging.Logger:
    """Set up file + console logging."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    mode = "a" if resume else "w"
    fh = logging.FileHandler(os.path.join(output_dir, f"{arch}.log"), mode=mode)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _init_embedding_from_pretrained(model: CaptionModel, vocab, device, logger):
    """Initialize decoder embedding with pretrained tokenizer weights.

    Loads only the embedding matrix (not the full model), projects to embed_dim if needed.
    """
    import torch.nn as nn
    from transformers import AutoModel

    tokenizer_name = vocab.tokenizer.name_or_path
    logger.info(f"[Embedding] Loading pretrained embedding from {tokenizer_name} ...")

    # Load only the embedding weights (not the full model)
    try:
        # Try loading just the embedding from safetensors/pytorch_model
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(tokenizer_name)
        embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Load weights into embedding only
        from huggingface_hub import hf_hub_download
        import os

        # Try safetensors first, then pytorch_model
        try:
            from safetensors.torch import load_file
            weight_path = hf_hub_download(tokenizer_name, "model.safetensors")
            state = load_file(weight_path)
        except Exception:
            weight_path = hf_hub_download(tokenizer_name, "pytorch_model.bin")
            state = torch.load(weight_path, map_location="cpu", weights_only=True)

        embedding.weight.data.copy_(state["embeddings.word_embeddings.weight"])
        pretrained_emb = embedding.weight.data.clone()
        del state, embedding
        logger.info(f"[Embedding] Loaded from weights file")
    except Exception as e:
        # Fallback: load full model
        logger.info(f"[Embedding] Fallback: loading full model ({e})")
        pretrained_model = AutoModel.from_pretrained(tokenizer_name)
        pretrained_emb = pretrained_model.embeddings.word_embeddings.weight.data.clone()
        del pretrained_model

    pretrained_dim = pretrained_emb.size(1)  # e.g. 768 for bert-base
    target_dim = model.decoder.embedding.weight.size(1)  # e.g. 512
    target_vocab = model.decoder.embedding.weight.size(0)

    # Truncate or pad vocabulary dimension
    if pretrained_emb.size(0) > target_vocab:
        pretrained_emb = pretrained_emb[:target_vocab]
    elif pretrained_emb.size(0) < target_vocab:
        pad = torch.zeros(target_vocab - pretrained_emb.size(0), pretrained_dim)
        pretrained_emb = torch.cat([pretrained_emb, pad], dim=0)

    # Project dimension if needed (e.g. 768 -> 512)
    if pretrained_dim != target_dim:
        projection = nn.Linear(pretrained_dim, target_dim, bias=False)
        nn.init.xavier_uniform_(projection.weight)
        with torch.no_grad():
            projected_emb = projection(pretrained_emb)  # [vocab_size, target_dim]
        del projection
    else:
        projected_emb = pretrained_emb

    # Copy into decoder embedding
    model.decoder.embedding.weight.data.copy_(projected_emb.to(device))
    logger.info(f"[Embedding] Initialized: {pretrained_dim}d -> {target_dim}d, "
                f"vocab {pretrained_emb.size(0)}/{target_vocab}")

    del pretrained_emb, projected_emb
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    # Resolve output directory
    if args.output_dir is not None:
        out = os.path.join(args.checkpoint_dir, args.output_dir)
    else:
        out = args.checkpoint_dir
    os.makedirs(out, exist_ok=True)

    # Logging
    logger = setup_logging(out, args.arch, resume=args.resume)
    logger.info(f"Output dir: {out}")

    # Tensorboard
    tb_dir = os.path.join(out, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"Tensorboard log dir: {tb_dir}")

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] {device}")

    # Data
    logger.info("[Data] Loading...")
    train_loader, val_loader, vocab = load_data(args)
    logger.info(f"[Data] Vocab size: {len(vocab)}, Train batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"[Data] Val batches: {len(val_loader)}")

    # Build model
    model = CaptionModel(
        arch=args.arch,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=len(vocab),
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_dim=args.attention_dim,
        fine_tune_layers=args.fine_tune,
        pretrained=True,
        clip_path=getattr(args, 'clip_path', None),
    ).to(device)

    # Initialize embedding from pretrained tokenizer if using one
    if args.pretrained_vocab and vocab._pretrained:
        _init_embedding_from_pretrained(model, vocab, device, logger)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Model] {args.arch} backbone={args.backbone}: {total_params:,} total params, {trainable_params:,} trainable")
    logger.info(f"[Model] Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    logger.info(f"[Model] Decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}")

    # Log hyperparams to tensorboard
    writer.add_text("hparams/json", json.dumps(vars(args), indent=2))

    # Save vocab
    vocab_path = os.path.join(out, f"vocab_{args.arch}.json")
    vocab.save(vocab_path)
    logger.info(f"[Vocab] Saved to {vocab_path}")

    # Evaluate only mode
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required with --eval_only")
        logger.info(f"[Eval] Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        if val_loader is not None:
            scores = evaluate_metrics(model, val_loader, vocab, device,
                                       strategy="greedy", beam_size=args.beam_size)
            logger.info("="*60)
            logger.info(f"  Evaluation Results ({args.arch})")
            logger.info("="*60)
            for metric, value in scores.items():
                logger.info(f"  {metric:10s}: {value:.4f}")
            logger.info("="*60)
        else:
            logger.info("[Eval] No validation annotations available.")
        writer.close()
        return

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and torch.cuda.is_available() else None

    # Resume from checkpoint
    best_val_loss = float("inf")
    best_metric_score = -1.0
    history = []
    global_step = 0
    start_epoch = 1

    if args.resume:
        # Load best checkpoint weights
        resume_ckpt = os.path.join(out, f"{args.arch}_best.pth")
        if not os.path.exists(resume_ckpt):
            resume_ckpt = os.path.join(out, f"{args.arch}_best_metric.pth")
        if os.path.exists(resume_ckpt):
            state_dict = torch.load(resume_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"[Resume] Loaded weights from {resume_ckpt}")
        else:
            logger.warning(f"[Resume] No checkpoint found in {out}, starting from scratch")

        # Load history
        history_path = os.path.join(out, f"{args.arch}_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            if history:
                # Recover best scores
                for r in history:
                    if r["val_loss"] is not None and r["val_loss"] < best_val_loss:
                        best_val_loss = r["val_loss"]
                    s = r.get("metrics", {}).get("BLEU-4", 0.0)
                    if s > best_metric_score:
                        best_metric_score = s

                if args.start_epoch > 0:
                    start_epoch = args.start_epoch
                else:
                    start_epoch = history[-1]["epoch"] + 1

                logger.info(f"[Resume] Continuing from epoch {start_epoch}, "
                            f"best_val_loss={best_val_loss:.4f}, best_BLEU-4={best_metric_score:.4f}")
        else:
            logger.warning("[Resume] No history found, starting from epoch 1")

    logger.info(f"[Train] Starting training: {args.epochs} epochs, arch={args.arch}")
    logger.info(f"[Train] Batch size: {args.batch_size}, LR: {args.lr}, Grad clip: {args.grad_clip}")
    logger.info(f"[Train] Early stopping patience: {args.early_stopping_patience}")
    logger.info("-"*60)

    # Early stopping state
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            args.grad_clip, epoch, writer=writer, global_step=global_step
        )

        # Validate loss
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)

        # LR scheduling
        monitor_loss = val_loss if val_loss is not None else train_loss
        scheduler.step(monitor_loss)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Tensorboard scalars
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        if val_loss is not None:
            writer.add_scalar("val/loss", val_loss, epoch)

        # Log
        log_msg = f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        log_msg += f" | LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        logger.info(log_msg)

        # Evaluate metrics periodically
        metric_scores = {}
        if val_loader is not None and (epoch % 10 == 0 or epoch == args.epochs):
            logger.info("  Generating captions for evaluation...")
            metric_scores = evaluate_metrics(
                model, val_loader, vocab, device,
                strategy="greedy", beam_size=args.beam_size
            )
            for k, v in metric_scores.items():
                logger.info(f"    {k}: {v:.4f}")
                writer.add_scalar(f"metrics/{k}", v, epoch)

        # Save checkpoint
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "metrics": metric_scores,
        }
        history.append(epoch_record)

        # Save best model
        current_score = metric_scores.get("BLEU-4", 0.0) if metric_scores else 0.0
        current_loss = val_loss if val_loss is not None else train_loss

        if current_loss < best_val_loss:
            best_val_loss = current_loss
            best_path = os.path.join(out, f"{args.arch}_best.pth")
            torch.save(model.state_dict(), best_path)
            patience_counter = 0  # Reset early stopping counter
            logger.info(f"  -> Saved best model (loss={current_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"  -> Early stopping at epoch {epoch} (no improvement for {args.early_stopping_patience} epochs)")
                break

        if current_score > best_metric_score and metric_scores:
            best_metric_score = current_score
            best_metric_path = os.path.join(out, f"{args.arch}_best_metric.pth")
            torch.save(model.state_dict(), best_metric_path)
            logger.info(f"  -> Saved best metric model (BLEU-4={current_score:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(out, f"{args.arch}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save training history
    history_path = os.path.join(out, f"{args.arch}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"[Train] History saved to {history_path}")

    # Final evaluation
    logger.info("="*60)
    logger.info(f"  Final Evaluation ({args.arch})")
    logger.info("="*60)

    # Load best model for final eval
    best_path = os.path.join(out, f"{args.arch}_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        logger.info(f"  Loaded best model from {best_path}")

    if val_loader is not None:
        final_scores = evaluate_metrics(
            model, val_loader, vocab, device,
            strategy="greedy", beam_size=args.beam_size
        )
        for metric, value in final_scores.items():
            logger.info(f"  {metric:10s}: {value:.4f}")

        # Save final scores
        scores_path = os.path.join(out, f"{args.arch}_scores.json")
        with open(scores_path, "w") as f:
            json.dump(final_scores, f, indent=2)
        logger.info(f"  Scores saved to {scores_path}")

    logger.info("="*60)
    logger.info("[Done]")
    writer.close()


if __name__ == "__main__":
    main()
