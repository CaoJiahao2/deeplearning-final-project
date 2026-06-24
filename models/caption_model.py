"""Caption Model: combines CNN encoder with LSTM decoder.

Two architectures:
- 'baseline': CNNEncoder (global features) + LSTMDecoder
- 'attention': CNNEncoder (spatial features) + AttentionLSTMDecoder
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .encoder import CNNEncoder
from .clip_encoder import CLIPEncoder
from .decoder import LSTMDecoder, AttentionLSTMDecoder
from .vocab import Vocabulary, PAD_IDX, START_IDX, END_IDX


class CaptionModel(nn.Module):
    """Image captioning model.

    Args:
        arch: 'baseline' or 'attention'
        backbone: CNN backbone ('resnet50', 'resnet101')
        embed_dim: embedding dimension
        hidden_dim: LSTM hidden dimension
        vocab_size: vocabulary size
        num_layers: number of LSTM layers
        dropout: dropout rate
        attention_dim: attention hidden dimension (only for 'attention' arch)
        fine_tune_layers: which CNN layers to fine-tune
        pretrained: use pretrained CNN
    """

    def __init__(
        self,
        arch: str = "baseline",
        backbone: str = "resnet50",
        embed_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        dropout: float = 0.3,
        attention_dim: int = 256,
        fine_tune_layers: str = "layer4",
        pretrained: bool = True,
        clip_path: str = None,
    ):
        super().__init__()
        self.arch = arch
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        if backbone == "clip":
            if clip_path is None:
                raise ValueError("clip_path required when backbone='clip'")
            self.encoder = CLIPEncoder(
                clip_path=clip_path,
                embed_dim=embed_dim,
                freeze=True,
            )
        else:
            self.encoder = CNNEncoder(
                backbone=backbone,
                embed_dim=embed_dim,
                pretrained=pretrained,
                fine_tune_layers=fine_tune_layers,
            )

        if arch == "baseline":
            self.decoder = LSTMDecoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif arch == "attention":
            self.decoder = AttentionLSTMDecoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout,
                attention_dim=attention_dim,
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}. Use 'baseline' or 'attention'.")

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass.

        Args:
            images: [batch, 3, 224, 224]
            captions: [batch, max_len] padded caption indices
            caption_lens: [batch] actual caption lengths

        Returns:
            logits: [batch, max_len, vocab_size]
        """
        if self.arch == "baseline":
            features = self.encoder(images)           # [B, embed_dim]
            logits = self.decoder(features, captions, caption_lens)
        else:
            spatial_features = self.encoder.forward_spatial(images)  # [B, 49, embed_dim]
            logits = self.decoder(spatial_features, captions, caption_lens)
        return logits

    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 64,
        strategy: str = "greedy",
        temperature: float = 1.0,
        beam_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> List[List[int]]:
        """Generate captions for images (inference).

        Args:
            images: [batch, 3, 224, 224]
            max_len: maximum caption length
            strategy: 'greedy' or 'beam'
            temperature: sampling temperature (for sampling strategy)
            beam_size: beam width (for beam search)
            repetition_penalty: penalty for repeated tokens (>1.0 reduces repetition)

        Returns:
            list of token index sequences
        """
        if strategy == "greedy":
            return self._greedy_decode(images, max_len)
        elif strategy == "beam":
            return self._beam_search(images, max_len, beam_size, repetition_penalty)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @torch.no_grad()
    def _greedy_decode(self, images: torch.Tensor, max_len: int) -> List[List[int]]:
        """Greedy decoding."""
        self.eval()
        batch_size = images.size(0)
        device = images.device

        if self.arch == "baseline":
            features = self.encoder(images)
            hidden = self.decoder.init_hidden(features)
            spatial_features = None
        else:
            spatial_features = self.encoder.forward_spatial(images)
            hidden = self.decoder.init_hidden(spatial_features)
            features = None

        # Start with <start> token
        input_token = torch.full((batch_size,), START_IDX, dtype=torch.long, device=device)

        all_sequences = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            if self.arch == "baseline":
                logits, hidden = self.decoder.forward_step(input_token, hidden)
            else:
                logits, hidden = self.decoder.forward_step(input_token, hidden, spatial_features)

            # Greedy: pick the highest probability token
            next_token = logits.argmax(dim=1)  # [B]

            for i in range(batch_size):
                if not finished[i]:
                    token = next_token[i].item()
                    all_sequences[i].append(token)
                    if token == END_IDX:
                        finished[i] = True

            if finished.all():
                break

            input_token = next_token

        return all_sequences

    @torch.no_grad()
    def _beam_search(self, images: torch.Tensor, max_len: int, beam_size: int,
                     repetition_penalty: float = 1.0) -> List[List[int]]:
        """Beam search decoding with repetition penalty."""
        self.eval()
        device = images.device
        batch_size = images.size(0)

        # Process each image independently
        all_sequences = []
        for i in range(batch_size):
            img = images[i:i+1]  # [1, 3, 224, 224]

            if self.arch == "baseline":
                features = self.encoder(img)
                hidden = self.decoder.init_hidden(features)
                spatial_features = None
            else:
                spatial_features = self.encoder.forward_spatial(img)
                hidden = self.decoder.init_hidden(spatial_features)
                features = None

            # Beam: list of (score, token_sequence, hidden_state)
            beams = [(0.0, [START_IDX], hidden)]
            completed = []

            for _ in range(max_len):
                all_candidates = []
                for score, seq, h in beams:
                    if seq[-1] == END_IDX:
                        completed.append((score, seq))
                        continue

                    input_token = torch.tensor([seq[-1]], device=device)
                    if self.arch == "baseline":
                        logits, new_h = self.decoder.forward_step(input_token, h)
                    else:
                        logits, new_h = self.decoder.forward_step(input_token, h, spatial_features)

                    log_probs = torch.log_softmax(logits.squeeze(0), dim=0)

                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        seen = set(seq[1:])  # Skip START_IDX
                        for token_id in seen:
                            if log_probs[token_id] > 0:
                                log_probs[token_id] /= repetition_penalty
                            else:
                                log_probs[token_id] *= repetition_penalty

                    top_scores, top_indices = log_probs.topk(beam_size)

                    for j in range(beam_size):
                        new_score = score + top_scores[j].item()
                        new_seq = seq + [top_indices[j].item()]
                        all_candidates.append((new_score, new_seq, new_h))

                if not all_candidates:
                    break

                # Keep top beams
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_size]

            # Add remaining beams
            for score, seq, _ in beams:
                completed.append((score, seq))

            # Normalize by length
            if completed:
                best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
                all_sequences.append(best[1])
            else:
                all_sequences.append([START_IDX, END_IDX])

        return all_sequences
