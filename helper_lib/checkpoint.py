"""Checkpoint helpers shared by training scripts and the inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import DiffusionUNet, EnergyModel

FORMAT_VERSION = 1


def save_ebm_checkpoint(
    path: str | Path,
    model: EnergyModel,
    *,
    epoch: int,
    training_mode: str,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "ebm_cifar10",
            "image_shape": (3, 32, 32),
            "base_channels": model.base_channels,
            "model_state_dict": _cpu_state_dict(model),
            "epoch": epoch,
            "training_mode": training_mode,
            "history": history,
            "config": config,
        },
        path,
    )


def save_diffusion_checkpoint(
    path: str | Path,
    model: DiffusionUNet,
    ema_model: DiffusionUNet,
    *,
    epoch: int,
    training_mode: str,
    history: list[dict[str, float]],
    normalizer_mean: tuple[float, float, float],
    normalizer_std: tuple[float, float, float],
    min_signal_rate: float,
    max_signal_rate: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "diffusion_cifar10",
            "image_shape": (3, 32, 32),
            "base_width": model.base_width,
            "block_depth": model.block_depth,
            "model_state_dict": _cpu_state_dict(model),
            "ema_model_state_dict": _cpu_state_dict(ema_model),
            "normalizer_mean": tuple(float(x) for x in normalizer_mean),
            "normalizer_std": tuple(float(x) for x in normalizer_std),
            "min_signal_rate": float(min_signal_rate),
            "max_signal_rate": float(max_signal_rate),
            "epoch": epoch,
            "training_mode": training_mode,
            "history": history,
            "config": config,
        },
        path,
    )


def load_ebm_checkpoint(path: str | Path) -> tuple[EnergyModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _check_common(checkpoint, "ebm_cifar10")
    model = EnergyModel(base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, _metadata(checkpoint)


def load_diffusion_checkpoint(path: str | Path) -> tuple[DiffusionUNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _check_common(checkpoint, "diffusion_cifar10")
    model = DiffusionUNet(
        base_width=int(checkpoint["base_width"]),
        block_depth=int(checkpoint["block_depth"]),
    )
    # The EMA weights are the intended inference weights, not the live weights.
    model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
    model.eval()
    metadata = _metadata(checkpoint)
    metadata.update({
        "normalizer_mean": checkpoint["normalizer_mean"],
        "normalizer_std": checkpoint["normalizer_std"],
        "min_signal_rate": checkpoint["min_signal_rate"],
        "max_signal_rate": checkpoint["max_signal_rate"],
        "base_width": checkpoint["base_width"],
        "block_depth": checkpoint["block_depth"],
    })
    return model, metadata


def _check_common(checkpoint: dict[str, Any], kind: str) -> None:
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version")
    if checkpoint.get("kind") != kind:
        raise ValueError(f"Expected {kind} checkpoint, got {checkpoint.get('kind')!r}")
    if tuple(checkpoint.get("image_shape", ())) != (3, 32, 32):
        raise ValueError("Checkpoint image shape is not CIFAR-10 RGB 32x32")


def _metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": checkpoint.get("epoch"),
        "training_mode": checkpoint.get("training_mode"),
        "image_shape": checkpoint.get("image_shape"),
        "config": checkpoint.get("config", {}),
        "last_metrics": (checkpoint.get("history") or [{}])[-1],
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
