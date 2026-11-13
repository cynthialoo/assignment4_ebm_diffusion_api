"""Sampling functions for the GAN, energy model, and diffusion model."""

from __future__ import annotations

import math

import torch
from torchvision.utils import make_grid

from .diffusion import offset_cosine_schedule, validate_channel_stats
from .model import GAN, Generator, EnergyModel, DiffusionUNet, LATENT_DIM


def _cpu_rng(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return rng


def _randn(shape: tuple[int, ...], device: torch.device, rng: torch.Generator | None) -> torch.Tensor:
    # A CPU RNG keeps the same seed usable on CUDA, MPS, and CPU.
    if rng is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=rng, device="cpu").to(device)


def _rand(shape: tuple[int, ...], device: torch.device, rng: torch.Generator | None) -> torch.Tensor:
    if rng is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=rng, device="cpu").to(device)


def sample_images(
    model: GAN | Generator,
    device: str | torch.device = "cpu",
    num_samples: int = 10,
    seed: int | None = None,
) -> torch.Tensor:
    """Return grayscale images in [-1, 1], shaped (N, 1, 28, 28)."""
    if not 1 <= num_samples <= 64:
        raise ValueError("num_samples must be between 1 and 64")
    generator = model.generator if isinstance(model, GAN) else model
    previous_training_mode = generator.training
    generator.eval()
    if seed is None:
        noise = torch.randn(num_samples, LATENT_DIM, device=device)
    else:
        cpu_rng = torch.Generator(device="cpu")
        cpu_rng.manual_seed(seed)
        noise = torch.randn(num_samples, LATENT_DIM, generator=cpu_rng).to(device)
    with torch.inference_mode():
        images = generator(noise).detach().cpu()
    generator.train(previous_training_mode)
    return images


def generate_samples(
    model: GAN | Generator | EnergyModel | DiffusionUNet,
    device: str | torch.device = "cpu",
    num_samples: int = 10,
    *,
    seed: int | None = None,
    show: bool = False,
    diffusion_steps: int = 100,
    ebm_steps: int = 60,
    ebm_step_size: float = 10.0,
    ebm_noise_std: float = 0.005,
    normalizer_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    normalizer_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
) -> torch.Tensor:
    """Return a [0, 1] image grid; optionally display it in a notebook."""
    if isinstance(model, EnergyModel):
        images = sample_ebm_images(
            model, device=device, num_samples=num_samples, seed=seed,
            steps=ebm_steps, step_size=ebm_step_size, noise_std=ebm_noise_std,
        )
        grid = make_grid((images + 1) / 2, nrow=math.ceil(math.sqrt(num_samples)), padding=2)
    elif isinstance(model, DiffusionUNet):
        images = sample_diffusion_images(
            model, device=device, num_samples=num_samples, seed=seed,
            steps=diffusion_steps, normalizer_mean=normalizer_mean,
            normalizer_std=normalizer_std,
            min_signal_rate=min_signal_rate, max_signal_rate=max_signal_rate,
        )
        grid = make_grid(images, nrow=math.ceil(math.sqrt(num_samples)), padding=2)
    else:
        images = sample_images(model, device, num_samples, seed)
        grid = make_grid(
            images,
            nrow=math.ceil(math.sqrt(num_samples)),
            padding=2,
            normalize=True,
            value_range=(-1, 1),
        )
    if show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 7))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.show()
    return grid


def langevin_sample(
    model: EnergyModel,
    initial_images: torch.Tensor,
    *,
    steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    seed: int | None = None,
) -> torch.Tensor:
    """Modify *images* to lower energy while leaving model gradients untouched.

    ``autograd.grad`` requests gradients only for x. Each iteration detaches x
    before the next one so the sampler does not retain a growing graph.
    """
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if step_size <= 0 or noise_std < 0:
        raise ValueError("step_size must be positive and noise_std nonnegative")
    if tuple(initial_images.shape[1:]) != (3, 32, 32):
        raise ValueError("initial_images must have shape (batch, 3, 32, 32)")
    device = initial_images.device
    rng = _cpu_rng(seed)
    was_training = model.training
    model.eval()
    images = initial_images.detach().clone()
    try:
        # This explicit context allows the EBM API to work when ordinary
        # gradient tracking was disabled with no_grad (not inference_mode).
        with torch.enable_grad():
            for _ in range(steps):
                with torch.no_grad():
                    images = (images + noise_std * _randn(tuple(images.shape), device, rng)).clamp(-1, 1)
                images = images.detach().requires_grad_(True)
                energy = model(images)
                (image_grad,) = torch.autograd.grad(energy.sum(), images)
                with torch.no_grad():
                    images = (images - step_size * image_grad.clamp(-0.03, 0.03)).clamp(-1, 1)
    finally:
        model.train(was_training)
    return images.detach()


def sample_ebm_images(
    model: EnergyModel,
    device: str | torch.device = "cpu",
    num_samples: int = 1,
    *,
    steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    seed: int | None = None,
) -> torch.Tensor:
    """Return generated CIFAR-size RGB tensors in [-1, 1]."""
    if not 1 <= num_samples <= 64:
        raise ValueError("num_samples must be between 1 and 64")
    target_device = torch.device(device)
    rng = _cpu_rng(seed)
    initial = 2 * _rand((num_samples, 3, 32, 32), target_device, rng) - 1
    # Continue the same random stream for Langevin perturbations.
    if seed is None:
        return langevin_sample(model, initial, steps=steps, step_size=step_size, noise_std=noise_std)
    # The initial draw consumed 3*32*32*num_samples uniform values. A separate
    # deterministic substream is sufficient for reproducibility across calls.
    return langevin_sample(model, initial, steps=steps, step_size=step_size,
                           noise_std=noise_std, seed=seed + 1)


def sample_diffusion_images(
    model: DiffusionUNet,
    device: str | torch.device = "cpu",
    num_samples: int = 1,
    *,
    steps: int = 100,
    seed: int | None = None,
    normalizer_mean: tuple[float, float, float] | torch.Tensor = (0.5, 0.5, 0.5),
    normalizer_std: tuple[float, float, float] | torch.Tensor = (0.5, 0.5, 0.5),
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
) -> torch.Tensor:
    """Deterministic DDIM-style reverse diffusion, output in [0, 1]."""
    if not 1 <= num_samples <= 64:
        raise ValueError("num_samples must be between 1 and 64")
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    target_device = torch.device(device)
    mean, std = validate_channel_stats(normalizer_mean, normalizer_std, device=target_device)
    rng = _cpu_rng(seed)
    images = _randn((num_samples, 3, 32, 32), target_device, rng)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for step in range(steps):
                t = torch.full((num_samples, 1, 1, 1), 1 - step / steps, device=target_device)
                next_t = torch.full((num_samples, 1, 1, 1), 1 - (step + 1) / steps, device=target_device)
                noise_rate, signal_rate = offset_cosine_schedule(
                    t, min_signal_rate=min_signal_rate, max_signal_rate=max_signal_rate,
                )
                next_noise_rate, next_signal_rate = offset_cosine_schedule(
                    next_t, min_signal_rate=min_signal_rate, max_signal_rate=max_signal_rate,
                )
                predicted_noise = model(images, noise_rate.square())
                predicted_clean = (images - noise_rate * predicted_noise) / signal_rate
                predicted_clean = predicted_clean.clamp(-3, 3)
                images = next_signal_rate * predicted_clean + next_noise_rate * predicted_noise
            return (predicted_clean * std + mean).clamp(0, 1).cpu()
    finally:
        model.train(was_training)
