"""LSTM Decoders for image captioning.

Two variants:
- LSTMDecoder: baseline, uses global image features as initial hidden state
- AttentionLSTMDecoder: adds Bahdanau (additive) attention over spatial features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMDecoder(nn.Module):
    """Baseline LSTM decoder.

    Image features initialize the LSTM hidden state.
    At each timestep, the input is the word embedding.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Project image features to initial hidden/cell state
        self.init_h = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(embed_dim, hidden_dim * num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def init_hidden(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state from image features.

        Args:
            image_features: [batch, embed_dim]

        Returns:
            h_0, c_0: each [num_layers, batch, hidden_dim]
        """
        h = self.init_h(image_features)  # [B, hidden_dim * num_layers]
        c = self.init_c(image_features)  # [B, hidden_dim * num_layers]
        B = image_features.size(0)

        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single decoding step.

        Args:
            input_token: [batch] word indices
            hidden: (h, c) tuple

        Returns:
            logits: [batch, vocab_size]
            new_hidden: updated (h, c)
        """
        embedded = self.embedding(input_token).unsqueeze(1)  # [B, 1, embed_dim]
        output, hidden = self.lstm(embedded, hidden)          # [B, 1, hidden_dim]
        output = self.dropout(output.squeeze(1))              # [B, hidden_dim]
        logits = self.fc_out(output)                           # [B, vocab_size]
        return logits, hidden

    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.Tensor,
        caption_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward with teacher forcing.

        Args:
            image_features: [batch, embed_dim] from encoder
            captions: [batch, max_len] padded caption indices
            caption_lens: [batch] actual lengths

        Returns:
            logits: [batch, max_len, vocab_size]
        """
        batch_size, max_len = captions.shape
        hidden = self.init_hidden(image_features)

        # Embed all captions at once
        embedded = self.embedding(captions)  # [B, max_len, embed_dim]

        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, caption_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)

        output = self.dropout(output)
        logits = self.fc_out(output)  # [B, max_len, vocab_size]
        return logits


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention.

    score(h_t, s_i) = V^T * tanh(W_h * h_t + W_s * s_i)
    """

    def __init__(self, hidden_dim: int, embed_dim: int, attention_dim: int = 256):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_s = nn.Linear(embed_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context.

        Args:
            hidden: [batch, hidden_dim] current LSTM hidden state
            spatial_features: [batch, num_pixels, embed_dim]

        Returns:
            context: [batch, embed_dim] weighted sum
            weights: [batch, num_pixels] attention distribution
        """
        # hidden: [B, H] -> [B, 1, A]
        h_proj = self.W_h(hidden).unsqueeze(1)
        # spatial: [B, P, E] -> [B, P, A]
        s_proj = self.W_s(spatial_features)

        # scores: [B, P, 1] -> [B, P]
        scores = self.V(torch.tanh(h_proj + s_proj)).squeeze(-1)
        weights = F.softmax(scores, dim=1)  # [B, P]

        # context: [B, E]
        context = torch.bmm(weights.unsqueeze(1), spatial_features).squeeze(1)
        return context, weights


class AttentionLSTMDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention.

    At each timestep:
    1. Compute attention context from spatial features and previous hidden state
    2. Concatenate context with word embedding as LSTM input
    3. Apply LSTM
    4. Predict next word from LSTM output + context
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        dropout: float = 0.3,
        attention_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, embed_dim, attention_dim)

        # LSTM input = embedding + context
        self.lstm = nn.LSTM(
            input_size=embed_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        # Output = f(LSTM_output, context)
        self.fc_out = nn.Linear(hidden_dim + embed_dim, vocab_size)

        # Initialize hidden state from mean of spatial features
        self.init_h = nn.Linear(embed_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(embed_dim, hidden_dim * num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def init_hidden(self, spatial_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state from mean of spatial features.

        Args:
            spatial_features: [batch, num_pixels, embed_dim]

        Returns:
            h_0, c_0: each [num_layers, batch, hidden_dim]
        """
        mean_features = spatial_features.mean(dim=1)  # [B, embed_dim]
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        B = spatial_features.size(0)

        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c

    def forward(
        self,
        spatial_features: torch.Tensor,
        captions: torch.Tensor,
        caption_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward with teacher forcing.

        Args:
            spatial_features: [batch, num_pixels, embed_dim] from encoder
            captions: [batch, max_len] padded caption indices
            caption_lens: [batch] actual lengths

        Returns:
            logits: [batch, max_len, vocab_size]
        """
        batch_size, max_len = captions.shape
        hidden = self.init_hidden(spatial_features)

        # Pre-compute attention projection for spatial features
        # (done inside attention module, no need to pre-compute)

        all_logits = []
        for t in range(max_len):
            # Current input token
            input_token = captions[:, t]  # [B]

            # Compute attention
            h_t = hidden[0][-1]  # last layer hidden state [B, H]
            context, _ = self.attention(h_t, spatial_features)  # [B, E]

            # Embed token
            embedded = self.embedding(input_token)  # [B, E]

            # Concatenate embedding and context
            lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # [B, 1, 2E]

            # LSTM step
            output, hidden = self.lstm(lstm_input, hidden)  # [B, 1, H]
            output = output.squeeze(1)  # [B, H]

            # Predict: combine LSTM output with context
            combined = torch.cat([output, context], dim=1)  # [B, H+E]
            combined = self.dropout(combined)
            logits = self.fc_out(combined)  # [B, vocab_size]
            all_logits.append(logits)

        logits = torch.stack(all_logits, dim=1)  # [B, max_len, vocab_size]
        return logits

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        spatial_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single decoding step (for inference).

        Args:
            input_token: [batch] word indices
            hidden: (h, c)
            spatial_features: [batch, num_pixels, embed_dim]

        Returns:
            logits: [batch, vocab_size]
            new_hidden: (h, c)
        """
        h_t = hidden[0][-1]  # [B, H]
        context, attn_weights = self.attention(h_t, spatial_features)

        embedded = self.embedding(input_token)  # [B, E]
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        output, hidden = self.lstm(lstm_input, hidden)
        output = output.squeeze(1)

        combined = torch.cat([output, context], dim=1)
        combined = self.dropout(combined)
        logits = self.fc_out(combined)
        return logits, hidden
