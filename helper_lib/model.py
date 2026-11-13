"""Image-generation models exposed by the course helper library.

The original MNIST GAN remains available for compatibility with Assignment 3.
Assignment 4 adds CIFAR-10 energy and diffusion models.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

LATENT_DIM = 100
IMAGE_SHAPE = (1, 28, 28)
CIFAR_IMAGE_SHAPE = (3, 32, 32)


class Generator(nn.Module):
    """Map a 100-dimensional noise vector to a grayscale 28 x 28 image."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected noise of shape (batch, {self.latent_dim}); "
                f"received {tuple(noise.shape)}."
            )
        x = self.fc(noise)
        x = x.reshape(noise.shape[0], 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        return self.tanh(self.deconv2(x))


class Discriminator(nn.Module):
    """Return a real-image probability for each 28 x 28 grayscale image."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != IMAGE_SHAPE:
            raise ValueError(
                f"Expected images of shape (batch, {IMAGE_SHAPE}); "
                f"received {tuple(image.shape)}."
            )
        x = self.activation1(self.conv1(image))
        x = self.activation2(self.bn2(self.conv2(x)))
        return self.sigmoid(self.fc(self.flatten(x)))


class GAN(nn.Module):
    """Keep the two adversarial networks together for the helper library."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise)


class EnergyModel(nn.Module):
    """Map each 3 x 32 x 32 image to one unnormalized energy score."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.base_channels = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(base_channels * 4 * 2 * 2, base_channels * 4)
        self.fc2 = nn.Linear(base_channels * 4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != CIFAR_IMAGE_SHAPE:
            raise ValueError(f"Expected (batch, 3, 32, 32), got {tuple(images.shape)}")
        x = F.silu(self.conv1(images))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = F.silu(self.fc1(torch.flatten(x, start_dim=1)))
        return self.fc2(x)


class SinusoidalEmbedding(nn.Module):
    """The noise-variance embedding used in the Module 8 practical."""

    def __init__(self, num_frequencies: int = 16) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        speeds = 2 * math.pi * torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer("angular_speeds", speeds.reshape(1, -1), persistent=False)

    def forward(self, noise_variances: torch.Tensor) -> torch.Tensor:
        if noise_variances.ndim != 4 or tuple(noise_variances.shape[1:]) != (1, 1, 1):
            raise ValueError("noise_variances must have shape (batch, 1, 1, 1)")
        phase = noise_variances.reshape(-1, 1) * self.angular_speeds
        return torch.cat([phase.sin(), phase.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.silu(self.conv1(x))) + self.projection(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = block(torch.cat((x, skips.pop()), dim=1))
        return x


class DiffusionUNet(nn.Module):
    """Compact Module 8 U-Net adapted from 64 x 64 flowers to CIFAR-10."""

    def __init__(self, base_width: int = 16, block_depth: int = 1) -> None:
        super().__init__()
        if base_width < 4 or block_depth < 1:
            raise ValueError("base_width must be >= 4 and block_depth >= 1")
        self.base_width = base_width
        self.block_depth = block_depth
        self.initial = nn.Conv2d(3, base_width, 1)
        self.embedding = SinusoidalEmbedding(16)
        self.embedding_proj = nn.Linear(32, base_width)
        self.down1 = DownBlock(2 * base_width, base_width, block_depth)
        self.down2 = DownBlock(base_width, 2 * base_width, block_depth)
        self.down3 = DownBlock(2 * base_width, 3 * base_width, block_depth)
        self.mid1 = ResidualBlock(3 * base_width, 4 * base_width)
        self.mid2 = ResidualBlock(4 * base_width, 4 * base_width)
        self.up1 = UpBlock(4 * base_width, 3 * base_width, block_depth)
        self.up2 = UpBlock(3 * base_width, 2 * base_width, block_depth)
        self.up3 = UpBlock(2 * base_width, base_width, block_depth)
        self.final = nn.Conv2d(base_width, 3, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, noisy_images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        if noisy_images.ndim != 4 or tuple(noisy_images.shape[1:]) != CIFAR_IMAGE_SHAPE:
            raise ValueError(f"Expected (batch, 3, 32, 32), got {tuple(noisy_images.shape)}")
        if noise_variances.shape != (noisy_images.shape[0], 1, 1, 1):
            raise ValueError("noise_variances must have shape (batch, 1, 1, 1)")
        embedding = self.embedding_proj(self.embedding(noise_variances))
        embedding = embedding[:, :, None, None].expand(-1, -1, 32, 32)
        skips: list[torch.Tensor] = []
        x = torch.cat((self.initial(noisy_images), embedding), dim=1)
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid2(self.mid1(x))
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


def get_model(model_name: str, **kwargs: int) -> nn.Module:
    """Construct one of the helper-library image generators."""
    name = model_name.strip().upper().replace("-", "_")
    if name == "GAN":
        return GAN(**kwargs)
    if name in {"EBM", "ENERGY", "ENERGY_MODEL"}:
        return EnergyModel(**kwargs)
    if name in {"DIFFUSION", "DIFFUSION_MODEL"}:
        return DiffusionUNet(**kwargs)
    raise ValueError(f"Unsupported model {model_name!r}; choose GAN, EBM, or Diffusion")
