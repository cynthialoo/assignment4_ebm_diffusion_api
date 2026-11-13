"""Train a CIFAR-10 energy model and export an API-ready checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.checkpoint import save_ebm_checkpoint
from helper_lib.generator import generate_samples
from helper_lib.model import EnergyModel
from helper_lib.trainer import train_ebm
from training_common import build_cifar_loader, choose_device, set_seed

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--langevin-steps", type=int, default=60)
    parser.add_argument("--step-size", type=float, default=10.0)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "outputs" / "ebm")
    parser.add_argument("--preview-samples", type=int, default=8, help="0 disables epoch previews")
    parser.add_argument("--preview-steps", type=int, default=100)
    parser.add_argument("--limit-train", type=int, default=0, help="First N CIFAR images; smoke tests only")
    parser.add_argument("--max-batches", type=int, default=0, help="Batches per epoch; smoke tests only")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Use an already downloaded CIFAR-10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.preview_samples < 0 or args.preview_samples > 64:
        raise ValueError("epochs must be positive; preview-samples must be 0..64")
    set_seed(args.seed)
    device = choose_device(args.device)
    smoke = bool(args.limit_train or args.max_batches)
    model_path = args.model_path or ROOT / "models" / ("ebm_cifar10_smoke.pt" if smoke else "ebm_cifar10.pt")
    loader, dataset = build_cifar_loader(
        args.data_dir, train=True, batch_size=args.batch_size,
        image_range="minus_one_one", limit=args.limit_train, offline=args.offline,
        num_workers=args.num_workers,
    )
    model = EnergyModel(base_channels=args.base_channels).to(device)
    with torch.no_grad():
        shape = tuple(model(torch.zeros(2, 3, 32, 32, device=device)).shape)
    if shape != (2, 1):
        raise RuntimeError(f"Unexpected EBM output shape: {shape}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.0, 0.999))
    history: list[dict[str, float]] = []
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "CIFAR-10 train", "dataset_size": len(dataset),
        "selected_training_examples": len(loader.dataset),
        "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "alpha": args.alpha,
        "langevin_steps": args.langevin_steps, "step_size": args.step_size,
        "noise_std": args.noise_std, "buffer_size": args.buffer_size,
        "seed": args.seed, "limit_train": args.limit_train,
        "max_batches": args.max_batches, "image_range": "[-1, 1]",
    }
    print(f"Training EBM on {device}; mode={'smoke' if smoke else 'full'}; checkpoint={model_path}")

    def save_progress(current: EnergyModel, record: dict[str, float]) -> None:
        epoch = int(record["epoch"])
        save_ebm_checkpoint(model_path, current, epoch=epoch,
                            training_mode="smoke" if smoke else "full",
                            history=history, config=config)
        if args.preview_samples:
            grid = generate_samples(
                current, device, args.preview_samples,
                seed=args.seed + epoch, ebm_steps=args.preview_steps,
                ebm_step_size=args.step_size, ebm_noise_std=args.noise_std,
            )
            save_image(grid, args.samples_dir / f"epoch_{epoch:03d}.png")
        print(
            f"Epoch {epoch:02d}/{args.epochs}: loss={record['loss']:.4f}, "
            f"real={record['real_energy']:.4f}, fake={record['fake_energy']:.4f}"
        )

    train_ebm(
        model, loader, optimizer=optimizer, device=device, epochs=args.epochs,
        alpha=args.alpha, langevin_steps=args.langevin_steps,
        step_size=args.step_size, noise_std=args.noise_std,
        buffer_size=args.buffer_size, max_batches=args.max_batches or None,
        history=history, epoch_callback=save_progress,
    )
    (args.samples_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
