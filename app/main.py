"""FastAPI service for the Assignment 4 CIFAR-10 image generators."""

from __future__ import annotations

import io
import os
from pathlib import Path
from threading import Lock
from typing import Callable

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from torchvision.utils import save_image

from helper_lib.checkpoint import load_diffusion_checkpoint, load_ebm_checkpoint
from helper_lib.generator import generate_samples
from helper_lib.model import Generator, IMAGE_SHAPE, LATENT_DIM

ROOT = Path(__file__).resolve().parents[1]
GAN_PATH = Path(os.environ.get("GAN_MODEL_PATH", ROOT / "models" / "gan_mnist.pt"))


def _checkpoint_path(env_name: str, basename: str) -> Path:
    explicit = os.environ.get(env_name)
    if explicit:
        return Path(explicit)
    full_path = ROOT / "models" / f"{basename}.pt"
    smoke_path = ROOT / "models" / f"{basename}_smoke.pt"
    return full_path if full_path.is_file() or not smoke_path.is_file() else smoke_path


EBM_PATH = _checkpoint_path("EBM_MODEL_PATH", "ebm_cifar10")
DIFFUSION_PATH = _checkpoint_path("DIFFUSION_MODEL_PATH", "diffusion_cifar10")

app = FastAPI(
    title="Assignment 4 Image Generation API",
    description="CIFAR-10 energy and diffusion image generation; retains the Assignment 3 MNIST GAN routes.",
    version="1.0.0",
)


def _load_gan(path: Path) -> tuple[Generator | None, dict, str | None]:
    if not path.is_file():
        return None, {}, f"GAN checkpoint not found: {path}"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("latent_dim") != LATENT_DIM:
            raise ValueError("GAN checkpoint latent dimension differs from 100")
        model = Generator().cpu()
        model.load_state_dict(checkpoint["generator_state_dict"])
        model.eval()
        metadata = {"epoch": checkpoint.get("epoch"), "image_shape": IMAGE_SHAPE}
        return model, metadata, None
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        return None, {}, f"Could not load GAN checkpoint at {path}: {exc}"


def _load_new(
    path: Path,
    loader: Callable,
    name: str,
) -> tuple[torch.nn.Module | None, dict, str | None]:
    if not path.is_file():
        return None, {}, f"{name} checkpoint not found: {path}. Train the model first."
    try:
        model, metadata = loader(path)
        return model, metadata, None
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        return None, {}, f"Could not load {name} checkpoint at {path}: {exc}"


GAN, GAN_META, GAN_WARNING = _load_gan(GAN_PATH)
EBM, EBM_META, EBM_WARNING = _load_new(EBM_PATH, load_ebm_checkpoint, "EBM")
DIFFUSION, DIFFUSION_META, DIFFUSION_WARNING = _load_new(
    DIFFUSION_PATH, load_diffusion_checkpoint, "Diffusion",
)
EBM_LOCK = Lock()
DIFFUSION_LOCK = Lock()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Assignment 4 CIFAR-10 image generators", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    loaded = {"gan": GAN is not None, "ebm": EBM is not None, "diffusion": DIFFUSION is not None}
    return {
        "status": "ready" if loaded["ebm"] and loaded["diffusion"] else "checkpoint_missing",
        "models_loaded": loaded,
        "training_mode": {"gan": "full" if GAN is not None else None,
                          "ebm": EBM_META.get("training_mode"),
                          "diffusion": DIFFUSION_META.get("training_mode")},
        "checkpoints": {"gan": str(GAN_PATH), "ebm": str(EBM_PATH), "diffusion": str(DIFFUSION_PATH)},
        "warnings": {"gan": GAN_WARNING, "ebm": EBM_WARNING, "diffusion": DIFFUSION_WARNING},
    }


@app.get("/gan/model-info")
def gan_model_info() -> dict:
    return {
        "noise_shape": ["batch_size", LATENT_DIM],
        "image_shape": ["batch_size", *IMAGE_SHAPE],
        "trained_checkpoint_loaded": GAN is not None,
        "metadata": GAN_META,
    }


@app.get("/ebm/model-info")
def ebm_model_info() -> dict:
    return {"image_shape": ["batch_size", 3, 32, 32],
            "trained_checkpoint_loaded": EBM is not None, "metadata": EBM_META,
            "warning": EBM_WARNING}


@app.get("/diffusion/model-info")
def diffusion_model_info() -> dict:
    return {"image_shape": ["batch_size", 3, 32, 32],
            "trained_checkpoint_loaded": DIFFUSION is not None,
            "metadata": DIFFUSION_META, "warning": DIFFUSION_WARNING}


def _png(grid: torch.Tensor, *, training_mode: str | None = None) -> Response:
    data = io.BytesIO()
    save_image(grid, data, format="PNG")
    headers = {"X-Training-Mode": training_mode or "unknown"}
    return Response(content=data.getvalue(), media_type="image/png", headers=headers)


@app.get("/ebm/generate", response_class=Response)
def generate_ebm(
    num_samples: int = Query(default=1, ge=1, le=8),
    steps: int = Query(default=64, ge=1, le=256),
    seed: int | None = Query(default=None, ge=0),
) -> Response:
    """Return a PNG grid of CIFAR-size images using input-image gradients."""
    if EBM is None:
        raise HTTPException(status_code=503, detail=EBM_WARNING)
    config = EBM_META.get("config", {})
    with EBM_LOCK:
        grid = generate_samples(
            EBM, "cpu", num_samples, seed=seed, ebm_steps=steps,
            ebm_step_size=float(config.get("step_size", 10.0)),
            ebm_noise_std=float(config.get("noise_std", 0.005)),
        )
    return _png(grid, training_mode=EBM_META.get("training_mode"))


@app.get("/diffusion/generate", response_class=Response)
def generate_diffusion(
    num_samples: int = Query(default=1, ge=1, le=8),
    steps: int = Query(default=20, ge=1, le=1000),
    seed: int | None = Query(default=None, ge=0),
) -> Response:
    """Return a PNG grid by denoising Gaussian noise with EMA U-Net weights."""
    if DIFFUSION is None:
        raise HTTPException(status_code=503, detail=DIFFUSION_WARNING)
    with DIFFUSION_LOCK:
        grid = generate_samples(
            DIFFUSION, "cpu", num_samples, seed=seed,
            diffusion_steps=steps,
            normalizer_mean=tuple(DIFFUSION_META["normalizer_mean"]),
            normalizer_std=tuple(DIFFUSION_META["normalizer_std"]),
            min_signal_rate=float(DIFFUSION_META["min_signal_rate"]),
            max_signal_rate=float(DIFFUSION_META["max_signal_rate"]),
        )
    return _png(grid, training_mode=DIFFUSION_META.get("training_mode"))


def _gan_png(num_samples: int, seed: int | None) -> Response:
    if GAN is None:
        raise HTTPException(status_code=503, detail=GAN_WARNING)
    return _png(generate_samples(GAN, "cpu", num_samples, seed=seed), training_mode="full")


@app.get("/generate-digit", response_class=Response)
def generate_digit(seed: int | None = Query(default=None, ge=0)) -> Response:
    return _gan_png(1, seed)


@app.get("/generate-digits", response_class=Response)
def generate_digits(num_samples: int = Query(default=10, ge=1, le=64),
                    seed: int | None = Query(default=None, ge=0)) -> Response:
    return _gan_png(num_samples, seed)
