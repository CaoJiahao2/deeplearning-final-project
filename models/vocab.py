"""Vocabulary for image captioning.

Supports two modes:
  1. Custom: word-level vocabulary built from training captions (legacy)
  2. Pretrained: uses a HuggingFace tokenizer (e.g. bert-base-uncased) with subword tokenization

Special token indices (consistent across both modes):
  PAD_IDX = 0
  START_IDX = 101  (BERT [CLS])
  END_IDX = 102    (BERT [SEP])
  UNK_IDX = 100    (BERT [UNK])
"""

import json
import re
from collections import Counter
from typing import List, Optional

# Special token indices — use BERT's convention
PAD_IDX = 0
START_IDX = 101   # [CLS]
END_IDX = 102     # [SEP]
UNK_IDX = 100     # [UNK]


class Vocabulary:
    """Unified vocabulary interface.

    Can be backed by either a custom word-level vocab or a pretrained tokenizer.
    """

    # Class-level constants for backward compatibility
    PAD_IDX = PAD_IDX
    START_IDX = START_IDX
    END_IDX = END_IDX
    UNK_IDX = UNK_IDX

    def __init__(self, tokenizer=None, word2idx=None, idx2word=None, word_freq=None):
        self.tokenizer = tokenizer  # HuggingFace tokenizer (if using pretrained)
        self.word2idx = word2idx or {}
        self.idx2word = idx2word or {}
        self.word_freq = word_freq or Counter()
        self._pretrained = tokenizer is not None

    @staticmethod
    def from_pretrained(model_name: str = "bert-base-uncased") -> "Vocabulary":
        """Create vocabulary from a pretrained tokenizer."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[Vocabulary] Loaded pretrained tokenizer: {model_name}, "
              f"vocab_size={tokenizer.vocab_size}")
        return Vocabulary(tokenizer=tokenizer)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization (for custom vocab)."""
        text = text.lower().strip()
        text = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", text)
        return text.split()

    def build_from_captions(self, captions: List[str], min_freq: int = 1):
        """Build custom vocabulary from a list of caption strings."""
        self.word_freq = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)

        self.word2idx = {
            "<pad>": PAD_IDX,
            "<unk>": UNK_IDX,
            "<cls>": START_IDX,
            "<sep>": END_IDX,
        }
        idx = 4
        for word, freq in self.word_freq.most_common():
            if freq >= min_freq and word not in self.word2idx:
                # Skip if it would collide with special token indices
                if idx in (PAD_IDX, UNK_IDX, START_IDX, END_IDX):
                    idx += 1
                    continue
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._pretrained = False
        return self

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text to indices. Adds START and END tokens."""
        if self._pretrained:
            # Subword tokenization
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            indices = [START_IDX] + token_ids + [END_IDX]
        else:
            # Word-level tokenization
            tokens = self.tokenize(text)
            indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
            indices = [START_IDX] + indices + [END_IDX]

        if max_len is not None:
            indices = indices[:max_len]
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode indices back to text."""
        special = {PAD_IDX, START_IDX, END_IDX}
        filtered = []
        for idx in indices:
            if skip_special and idx in special:
                if idx == END_IDX:
                    break
                continue
            filtered.append(idx)

        if self._pretrained:
            return self.tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            words = [self.idx2word.get(idx, "<unk>") for idx in filtered]
            return " ".join(words)

    def __len__(self):
        if self._pretrained:
            return self.tokenizer.vocab_size
        return len(self.word2idx)

    def save(self, path: str):
        """Save vocabulary config (not the full tokenizer)."""
        data = {
            "pretrained": self._pretrained,
            "word2idx": self.word2idx,
            "word_freq": dict(self.word_freq),
        }
        if self._pretrained:
            data["tokenizer_name"] = self.tokenizer.name_or_path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("pretrained", False):
            return cls.from_pretrained(data["tokenizer_name"])
        else:
            word2idx = data["word2idx"]
            idx2word = {int(v): k for k, v in word2idx.items()}
            return cls(word2idx=word2idx, idx2word=idx2word,
                       word_freq=Counter(data.get("word_freq", {})))


def build_vocab_from_jsonl(jsonl_path: str, min_freq: int = 1,
                           pretrained: str = None) -> Vocabulary:
    """Build vocabulary from a JSONL annotation file.

    Args:
        jsonl_path: path to annotation JSONL
        min_freq: minimum word frequency (for custom vocab)
        pretrained: pretrained tokenizer name (e.g. "bert-base-uncased").
                    If provided, uses pretrained tokenizer instead of custom vocab.
    """
    if pretrained:
        vocab = Vocabulary.from_pretrained(pretrained)
        # Still load captions to compute frequency stats
        with open(jsonl_path, "r", encoding="utf-8") as f:
            captions = [json.loads(line)["caption"] for line in f if line.strip()]
        print(f"[Vocabulary] Using pretrained tokenizer on {len(captions)} captions")
        return vocab

    captions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if "caption" in record:
                captions.append(record["caption"])
    vocab = Vocabulary()
    vocab.build_from_captions(captions, min_freq=min_freq)
    print(f"[Vocabulary] Built vocab with {len(vocab)} words from {len(captions)} captions")
    return vocab
