"""Vocabulary for image captioning.

Builds a word-to-index mapping from training captions.
Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3
"""

import json
import re
from collections import Counter
from typing import List, Optional


class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        text = text.lower().strip()
        # Separate punctuation
        text = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", text)
        return text.split()

    def build_from_captions(self, captions: List[str], min_freq: int = 1):
        """Build vocabulary from a list of caption strings."""
        self.word_freq = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)

        # Build mapping
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        idx = 4
        for word, freq in self.word_freq.most_common():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        return self

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text to indices. Optionally truncate to max_len (including <start> and <end>)."""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        # Add start and end
        indices = [self.START_IDX] + indices + [self.END_IDX]
        if max_len is not None:
            indices = indices[:max_len]
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode indices back to text."""
        special = {self.PAD_IDX, self.START_IDX, self.END_IDX}
        words = []
        for idx in indices:
            if skip_special and idx in special:
                if idx == self.END_IDX:
                    break
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "word_freq": dict(self.word_freq)}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get("word_freq", {}))
        return vocab


def build_vocab_from_jsonl(jsonl_path: str, min_freq: int = 1) -> Vocabulary:
    """Convenience: build vocabulary from a JSONL annotation file."""
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
