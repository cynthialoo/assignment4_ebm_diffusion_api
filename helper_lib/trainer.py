"""Training loops for the GAN, CIFAR-10 EBM, and CIFAR-10 diffusion model."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, Optimizer

from .diffusion import offset_cosine_schedule, validate_channel_stats
from .generator import langevin_sample
from .model import GAN, LATENT_DIM, DiffusionUNet, EnergyModel

EpochCallback = Callable[[GAN, dict[str, float]], None]


def train_gan(
    model: GAN,
    data_loader: Any,
    criterion: nn.Module | None = None,
    optimizer: tuple[Optimizer, Optimizer] | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    *,
    learning_rate: float = 0.0002,
    history: list[dict[str, float]] | None = None,
    max_batches: int | None = None,
    epoch_callback: EpochCallback | None = None,
) -> GAN:
    """Train a vanilla GAN and return the model, matching the class activity.

    The optional optimizer argument is (generator_optimizer, discriminator_optimizer).
    One optimizer cannot correctly update both networks at different training steps.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    device = torch.device(device)
    model.to(device)
    if criterion is None:
        criterion = nn.BCELoss()
    criterion.to(device)
    if optimizer is None:
        optimizer_g = Adam(
            model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        optimizer_d = Adam(
            model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
    else:
        if not isinstance(optimizer, tuple) or len(optimizer) != 2:
            raise ValueError("optimizer must be a (generator, discriminator) pair")
        optimizer_g, optimizer_d = optimizer

    for epoch in range(1, epochs + 1):
        model.train()
        total_d = 0.0
        total_g = 0.0
        seen = 0

        for batch_number, batch in enumerate(data_loader, start=1):
            real = batch[0] if isinstance(batch, (tuple, list)) else batch
            real = real.to(device)
            batch_size = real.shape[0]
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # Update D with real images and detached generated images.
            optimizer_d.zero_grad(set_to_none=True)
            prediction_real = model.discriminator(real)
            loss_real = criterion(prediction_real, real_targets)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_detached = model.generator(noise).detach()
            prediction_fake = model.discriminator(fake_detached)
            loss_fake = criterion(prediction_fake, fake_targets)
            loss_d = 0.5 * (loss_real + loss_fake)
            loss_d.backward()
            optimizer_d.step()

            # Update G to make its images receive the real-image label.
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake = model.generator(noise)
            prediction = model.discriminator(fake)
            loss_g = criterion(prediction, real_targets)
            loss_g.backward()
            optimizer_g.step()
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(True)

            total_d += float(loss_d.item()) * batch_size
            total_g += float(loss_g.item()) * batch_size
            seen += batch_size
            if max_batches is not None and batch_number >= max_batches:
                break

        if seen == 0:
            raise ValueError("data_loader produced no images")
        record = {
            "epoch": float(epoch),
            "discriminator_loss": total_d / seen,
            "generator_loss": total_g / seen,
            "examples_seen": float(seen),
        }
        if history is not None:
            history.append(record)
        if epoch_callback is not None:
            epoch_callback(model, record)

    return model


class ReplayBuffer:
    """Store old negative examples on CPU for contrastive EBM training."""

    def __init__(self, capacity: int = 2048, fresh_fraction: float = 0.05) -> None:
        if capacity < 1 or not 0 <= fresh_fraction <= 1:
            raise ValueError("capacity must be positive and fresh_fraction in [0, 1]")
        self.capacity = capacity
        self.fresh_fraction = fresh_fraction
        self.images = torch.empty((0, 3, 32, 32), dtype=torch.float32)

    def draw(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        fresh = 2 * torch.rand(batch_size, 3, 32, 32) - 1
        if len(self.images):
            use_old = torch.rand(batch_size) >= self.fresh_fraction
            old_indices = torch.randint(len(self.images), (batch_size,))
            fresh[use_old] = self.images[old_indices[use_old]]
        return fresh.to(device)

    def add(self, images: torch.Tensor) -> None:
        new_images = images.detach().cpu()
        self.images = torch.cat((new_images, self.images), dim=0)[: self.capacity]


def train_ebm(
    model: EnergyModel,
    data_loader: Any,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    *,
    alpha: float = 0.1,
    langevin_steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    buffer_size: int = 2048,
    max_batches: int | None = None,
    history: list[dict[str, float]] | None = None,
    epoch_callback: Callable[[EnergyModel, dict[str, float]], None] | None = None,
) -> EnergyModel:
    """Train an EBM with persistent negative samples and CD plus E² penalty.

    ``criterion`` is accepted for helper-library interface compatibility; the
    EBM objective is defined by contrastive energy and ``alpha`` instead.
    """
    if criterion is not None:
        raise ValueError("EBM uses its contrastive objective; do not pass criterion")
    if epochs < 1 or alpha < 0 or (max_batches is not None and max_batches < 1):
        raise ValueError("epochs must be positive, alpha nonnegative, max_batches positive")
    target_device = torch.device(device)
    model.to(target_device)
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    replay = ReplayBuffer(buffer_size)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "contrastive", "regularizer", "real_energy", "fake_energy")}
        seen = 0
        for batch_index, batch in enumerate(data_loader, start=1):
            real = batch[0] if isinstance(batch, (tuple, list)) else batch
            real = real.to(target_device)
            if tuple(real.shape[1:]) != (3, 32, 32):
                raise ValueError("EBM training images must be (batch, 3, 32, 32)")
            # The training transform maps CIFAR pixels from [0, 1] to [-1, 1].
            real = (real + noise_std * torch.randn_like(real)).clamp(-1, 1)
            fake_start = replay.draw(real.shape[0], target_device)
            fake = langevin_sample(model, fake_start, steps=langevin_steps,
                                   step_size=step_size, noise_std=noise_std)
            replay.add(fake)

            optimizer.zero_grad(set_to_none=True)
            real_energy = model(real)
            fake_energy = model(fake.detach())
            contrastive = real_energy.mean() - fake_energy.mean()
            regularizer = alpha * (real_energy.square().mean() + fake_energy.square().mean())
            loss = contrastive + regularizer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            n = real.shape[0]
            for key, value in (
                ("loss", loss), ("contrastive", contrastive), ("regularizer", regularizer),
                ("real_energy", real_energy.mean()), ("fake_energy", fake_energy.mean()),
            ):
                totals[key] += float(value.detach()) * n
            seen += n
            if max_batches is not None and batch_index >= max_batches:
                break
        if seen == 0:
            raise ValueError("data_loader produced no images")
        record = {"epoch": float(epoch), "examples_seen": float(seen)}
        record.update({key: value / seen for key, value in totals.items()})
        if history is not None:
            history.append(record)
        if epoch_callback is not None:
            epoch_callback(model, record)
    return model


def _diffusion_loss(
    model: DiffusionUNet,
    images: torch.Tensor,
    criterion: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    min_signal_rate: float,
    max_signal_rate: float,
) -> torch.Tensor:
    clean = (images - mean) / std
    noise = torch.randn_like(clean)
    t = torch.rand((len(images), 1, 1, 1), device=images.device)
    noise_rate, signal_rate = offset_cosine_schedule(
        t, min_signal_rate=min_signal_rate, max_signal_rate=max_signal_rate,
    )
    noisy = signal_rate * clean + noise_rate * noise
    prediction = model(noisy, noise_rate.square())
    return criterion(prediction, noise)


def train_diffusion(
    model: DiffusionUNet,
    data_loader: Any,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    *,
    normalizer_mean: tuple[float, float, float] | torch.Tensor = (0.5, 0.5, 0.5),
    normalizer_std: tuple[float, float, float] | torch.Tensor = (0.5, 0.5, 0.5),
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
    ema_model: DiffusionUNet | None = None,
    ema_decay: float = 0.995,
    val_loader: Any | None = None,
    max_batches: int | None = None,
    history: list[dict[str, float]] | None = None,
    epoch_callback: Callable[[DiffusionUNet, DiffusionUNet, dict[str, float]], None] | None = None,
) -> DiffusionUNet:
    """Train noise prediction and update an EMA model after each optimizer step.

    Pass ``ema_model`` when the caller needs to save and serve the smoothed
    weights. The function returns the trained live model for class-activity
    compatibility.
    """
    if epochs < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("epochs and max_batches must be positive")
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    target_device = torch.device(device)
    model.to(target_device)
    ema_model = copy.deepcopy(model) if ema_model is None else ema_model.to(target_device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    mean, std = validate_channel_stats(normalizer_mean, normalizer_std, device=target_device)
    criterion = criterion or nn.L1Loss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if not 0 < min_signal_rate < max_signal_rate < 1:
        raise ValueError("Invalid offset-cosine signal range")

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_seen = 0
        for batch_index, batch in enumerate(data_loader, start=1):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(target_device)
            if tuple(images.shape[1:]) != (3, 32, 32):
                raise ValueError("Diffusion training images must be (batch, 3, 32, 32)")
            optimizer.zero_grad(set_to_none=True)
            loss = _diffusion_loss(model, images, criterion, mean, std,
                                   min_signal_rate=min_signal_rate,
                                   max_signal_rate=max_signal_rate)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for averaged, live in zip(ema_model.parameters(), model.parameters()):
                    averaged.lerp_(live, 1 - ema_decay)
            train_total += float(loss.detach()) * len(images)
            train_seen += len(images)
            if max_batches is not None and batch_index >= max_batches:
                break
        if train_seen == 0:
            raise ValueError("data_loader produced no images")
        record = {"epoch": float(epoch), "examples_seen": float(train_seen),
                  "train_loss": train_total / train_seen}
        if val_loader is not None:
            ema_model.eval()
            val_total = 0.0
            val_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0] if isinstance(batch, (tuple, list)) else batch
                    images = images.to(target_device)
                    val_loss = _diffusion_loss(ema_model, images, criterion, mean, std,
                                                min_signal_rate=min_signal_rate,
                                                max_signal_rate=max_signal_rate)
                    val_total += float(val_loss) * len(images)
                    val_seen += len(images)
            if val_seen == 0:
                raise ValueError("val_loader produced no images")
            record["val_loss"] = val_total / val_seen
        if history is not None:
            history.append(record)
        if epoch_callback is not None:
            epoch_callback(model, ema_model, record)
    return model
