"""Shared CIFAR-10 data and device utilities for training scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_cifar_loader(
    data_dir: Path,
    *,
    train: bool,
    batch_size: int,
    image_range: str,
    limit: int = 0,
    offline: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, datasets.CIFAR10]:
    if batch_size < 1 or limit < 0 or num_workers < 0:
        raise ValueError("batch_size must be positive; limit and num_workers nonnegative")
    transform_list: list = [transforms.ToTensor()]
    if image_range == "minus_one_one":
        transform_list.append(transforms.Normalize((0.5,) * 3, (0.5,) * 3))
    elif image_range != "zero_one":
        raise ValueError("image_range must be zero_one or minus_one_one")
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=not offline,
        transform=transforms.Compose(transform_list),
    )
    selected = Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset
    loader = DataLoader(
        selected, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def calculate_channel_stats(dataset: datasets.CIFAR10, limit: int = 0) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calculate true per-pixel RGB mean and std from the selected training data."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    image_count = min(limit, len(dataset.data)) if limit else len(dataset.data)
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    for start in range(0, image_count, 1024):
        pixels = dataset.data[start : min(start + 1024, image_count)].astype(np.float64) / 255.0
        total += pixels.sum(axis=(0, 1, 2))
        squared += np.square(pixels).sum(axis=(0, 1, 2))
    count = image_count * 32 * 32
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-12))
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)

