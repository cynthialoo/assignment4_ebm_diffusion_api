"""Train a CIFAR-10 diffusion U-Net and export EMA weights for the API."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from helper_lib.checkpoint import save_diffusion_checkpoint
from helper_lib.generator import generate_samples
from helper_lib.model import DiffusionUNet
from helper_lib.trainer import train_diffusion
from training_common import (
    build_cifar_loader, calculate_channel_stats, choose_device, set_seed,
)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--block-depth", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--min-signal-rate", type=float, default=0.02)
    parser.add_argument("--max-signal-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-path", type=Path, default=None, help="Best validation checkpoint")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "outputs" / "diffusion")
    parser.add_argument("--preview-samples", type=int, default=8, help="0 disables epoch previews")
    parser.add_argument("--preview-steps", type=int, default=20)
    parser.add_argument("--limit-train", type=int, default=0, help="First N training images; smoke tests only")
    parser.add_argument("--limit-val", type=int, default=0, help="First N validation images; smoke tests only")
    parser.add_argument("--max-batches", type=int, default=0, help="Training batches per epoch; smoke tests only")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Use an already downloaded CIFAR-10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.preview_samples < 0 or args.preview_samples > 64:
        raise ValueError("epochs must be positive; preview-samples must be 0..64")
    set_seed(args.seed)
    device = choose_device(args.device)
    smoke = bool(args.limit_train or args.limit_val or args.max_batches)
    model_path = args.model_path or ROOT / "models" / ("diffusion_cifar10_smoke.pt" if smoke else "diffusion_cifar10.pt")
    latest_path = model_path.with_name(model_path.stem + "_latest.pt")
    val_limit = args.limit_val or (128 if smoke else 0)
    train_loader, train_dataset = build_cifar_loader(
        args.data_dir, train=True, batch_size=args.batch_size,
        image_range="zero_one", limit=args.limit_train, offline=args.offline,
        num_workers=args.num_workers,
    )
    val_loader, _ = build_cifar_loader(
        args.data_dir, train=False, batch_size=args.batch_size,
        image_range="zero_one", limit=val_limit, offline=args.offline,
        num_workers=args.num_workers,
    )
    mean, std = calculate_channel_stats(train_dataset, limit=args.limit_train)
    print(f"CIFAR-10 RGB mean={mean}, std={std}")
    model = DiffusionUNet(base_width=args.base_width, block_depth=args.block_depth).to(device)
    ema_model = copy.deepcopy(model).to(device)
    with torch.no_grad():
        shape = tuple(model(torch.zeros(2, 3, 32, 32, device=device),
                            torch.full((2, 1, 1, 1), 0.5, device=device)).shape)
    if shape != (2, 3, 32, 32):
        raise RuntimeError(f"Unexpected diffusion output shape: {shape}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "CIFAR-10 train", "dataset_size": len(train_dataset),
        "selected_training_examples": len(train_loader.dataset),
        "selected_validation_examples": len(val_loader.dataset),
        "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay, "seed": args.seed,
        "limit_train": args.limit_train, "limit_val": val_limit,
        "max_batches": args.max_batches, "loss": "L1 noise prediction",
        "schedule": "offset_cosine", "preprocessing": "ToTensor [0, 1], then per-channel standardization",
    }
    best_val = float("inf")
    print(f"Training diffusion on {device}; mode={'smoke' if smoke else 'full'}; best={model_path}")

    def save_progress(current: DiffusionUNet, averaged: DiffusionUNet,
                      record: dict[str, float]) -> None:
        nonlocal best_val
        epoch = int(record["epoch"])
        kwargs = {
            "epoch": epoch, "training_mode": "smoke" if smoke else "full",
            "history": history, "normalizer_mean": mean, "normalizer_std": std,
            "min_signal_rate": args.min_signal_rate,
            "max_signal_rate": args.max_signal_rate, "config": config,
        }
        save_diffusion_checkpoint(latest_path, current, averaged, **kwargs)
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            save_diffusion_checkpoint(model_path, current, averaged, **kwargs)
        if args.preview_samples:
            grid = generate_samples(
                averaged, device, args.preview_samples, seed=args.seed + epoch,
                diffusion_steps=args.preview_steps,
                normalizer_mean=mean, normalizer_std=std,
                min_signal_rate=args.min_signal_rate,
                max_signal_rate=args.max_signal_rate,
            )
            save_image(grid, args.samples_dir / f"epoch_{epoch:03d}.png")
        print(
            f"Epoch {epoch:02d}/{args.epochs}: train L1={record['train_loss']:.4f}, "
            f"val L1={record['val_loss']:.4f}; best={best_val:.4f}"
        )

    train_diffusion(
        model, train_loader, criterion=nn.L1Loss(), optimizer=optimizer,
        device=device, epochs=args.epochs,
        normalizer_mean=mean, normalizer_std=std,
        min_signal_rate=args.min_signal_rate,
        max_signal_rate=args.max_signal_rate,
        ema_model=ema_model, ema_decay=args.ema_decay, val_loader=val_loader,
        max_batches=args.max_batches or None,
        history=history, epoch_callback=save_progress,
    )
    (args.samples_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved best checkpoint to {model_path}; latest to {latest_path}")


if __name__ == "__main__":
    main()
