"""Continuous offset-cosine noise schedule used for CIFAR-10 diffusion."""

from __future__ import annotations

import math

import torch


def offset_cosine_schedule(
    diffusion_times: torch.Tensor,
    *,
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (noise_rate, signal_rate), both with the input's shape.

    At time zero, most of the image signal remains; at time one, almost all of
    it has been replaced by Gaussian noise. The two squared rates sum to one.
    """
    if not 0 < min_signal_rate < max_signal_rate < 1:
        raise ValueError("Expected 0 < min_signal_rate < max_signal_rate < 1")
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    angle = start_angle + diffusion_times * (end_angle - start_angle)
    return angle.sin(), angle.cos()


def validate_channel_stats(
    mean: torch.Tensor | tuple[float, ...] | list[float],
    std: torch.Tensor | tuple[float, ...] | list[float],
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
        raise ValueError("mean and std must be finite")
    if (std_tensor <= 0).any():
        raise ValueError("all channel standard deviations must be positive")
    return mean_tensor, std_tensor

