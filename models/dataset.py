"""Dataset and DataLoader for image captioning."""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .vocab import Vocabulary


class CaptionDataset(Dataset):
    """Image captioning dataset.

    Each sample returns:
        image: preprocessed image tensor [3, 224, 224]
        caption: encoded caption indices [max_len]
        caption_len: actual length of the caption (including <start>, <end>)
        image_id: filename string (for evaluation)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_path: str,
        vocab: Vocabulary,
        max_caption_len: int = 64,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        norm_type: str = "imagenet",
    ):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_caption_len = max_caption_len
        self.is_train = is_train
        self.transform = transform or self._default_transform(is_train, norm_type=norm_type)

        # Load annotations
        self.annotations: List[Dict] = []
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                img_path = os.path.join(image_dir, record["image_id"])
                if os.path.exists(img_path):
                    self.annotations.append(record)

        print(f"[Dataset] Loaded {len(self.annotations)} samples from {annotation_path}")

    @staticmethod
    def _default_transform(is_train: bool, norm_type: str = "imagenet") -> transforms.Compose:
        # Normalization stats
        if norm_type == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:  # imagenet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        record = self.annotations[idx]

        # Load image
        img_path = os.path.join(self.image_dir, record["image_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Encode caption
        caption_indices = self.vocab.encode(record["caption"], max_len=self.max_caption_len)
        caption_len = len(caption_indices)

        # Pad to max_caption_len
        padded = caption_indices + [Vocabulary.PAD_IDX] * (self.max_caption_len - caption_len)
        caption_tensor = torch.tensor(padded, dtype=torch.long)

        return image, caption_tensor, caption_len, record["image_id"]


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate for variable-length captions."""
    images = torch.stack([item[0] for item in batch])
    captions = torch.stack([item[1] for item in batch])
    caption_lens = torch.tensor([item[2] for item in batch], dtype=torch.long)
    image_ids = [item[3] for item in batch]
    return images, captions, caption_lens, image_ids


def get_transforms(is_train: bool = True, norm_type: str = "imagenet") -> transforms.Compose:
    """Get standard image transforms."""
    return CaptionDataset._default_transform(is_train, norm_type=norm_type)
