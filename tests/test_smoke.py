"""Small contract tests; they do not assess generated-image quality."""

from __future__ import annotations

import copy
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from helper_lib.checkpoint import (
    load_diffusion_checkpoint, load_ebm_checkpoint,
    save_diffusion_checkpoint, save_ebm_checkpoint,
)
from helper_lib.diffusion import offset_cosine_schedule
from helper_lib.generator import langevin_sample, sample_diffusion_images
from helper_lib.model import DiffusionUNet, EnergyModel, get_model
from helper_lib.trainer import train_diffusion, train_ebm


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_shapes_schedule_and_factory(self) -> None:
        energy = get_model("EBM", base_channels=4)
        diffusion = get_model("Diffusion", base_width=4, block_depth=1)
        self.assertEqual(tuple(energy(torch.zeros(2, 3, 32, 32)).shape), (2, 1))
        self.assertEqual(
            tuple(diffusion(torch.zeros(2, 3, 32, 32),
                            torch.full((2, 1, 1, 1), 0.5)).shape),
            (2, 3, 32, 32),
        )
        times = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1, 1)
        noise, signal = offset_cosine_schedule(times)
        self.assertEqual(noise.shape, times.shape)
        self.assertTrue(torch.allclose(noise.square() + signal.square(), torch.ones_like(times)))

    def test_ebm_sampler_updates_images_without_parameter_grads(self) -> None:
        torch.manual_seed(3)
        model = EnergyModel(base_channels=4)
        images = torch.rand(2, 3, 32, 32) * 2 - 1
        before = model(images).sum().item()
        generated = langevin_sample(model, images, steps=1, step_size=0.01, noise_std=0)
        after = model(generated).sum().item()
        self.assertLessEqual(after, before + 1e-6)
        self.assertFalse(generated.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_one_batch_training_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(4)
        energy = EnergyModel(base_channels=4)
        energy_before = energy.fc2.weight.detach().clone()
        energy_history: list[dict[str, float]] = []
        energy_loader = DataLoader(
            TensorDataset(torch.rand(2, 3, 32, 32) * 2 - 1), batch_size=2,
        )
        train_ebm(energy, energy_loader, epochs=1, langevin_steps=1,
                  step_size=0.1, noise_std=0, history=energy_history)
        self.assertFalse(torch.equal(energy.fc2.weight, energy_before))

        diffusion = DiffusionUNet(base_width=4, block_depth=1)
        ema = copy.deepcopy(diffusion)
        diffusion_before = diffusion.final.weight.detach().clone()
        diffusion_history: list[dict[str, float]] = []
        diffusion_loader = DataLoader(TensorDataset(torch.rand(2, 3, 32, 32)), batch_size=2)
        train_diffusion(diffusion, diffusion_loader, epochs=1, ema_model=ema,
                        normalizer_mean=(0.5,) * 3, normalizer_std=(0.5,) * 3,
                        history=diffusion_history)
        self.assertFalse(torch.equal(diffusion.final.weight, diffusion_before))
        self.assertFalse(torch.equal(ema.final.weight, diffusion_before))
        sampled = sample_diffusion_images(ema, num_samples=1, steps=2, seed=7)
        self.assertEqual(tuple(sampled.shape), (1, 3, 32, 32))
        self.assertTrue(bool(((sampled >= 0) & (sampled <= 1)).all()))

        with tempfile.TemporaryDirectory() as temp:
            ebm_path = Path(temp) / "ebm.pt"
            diff_path = Path(temp) / "diffusion.pt"
            save_ebm_checkpoint(ebm_path, energy, epoch=1, training_mode="smoke",
                                history=energy_history, config={"step_size": 0.1})
            save_diffusion_checkpoint(
                diff_path, diffusion, ema, epoch=1, training_mode="smoke",
                history=diffusion_history, normalizer_mean=(0.5,) * 3,
                normalizer_std=(0.5,) * 3, min_signal_rate=0.02,
                max_signal_rate=0.95, config={},
            )
            loaded_energy, emeta = load_ebm_checkpoint(ebm_path)
            loaded_diffusion, dmeta = load_diffusion_checkpoint(diff_path)
            self.assertEqual(emeta["training_mode"], "smoke")
            self.assertEqual(dmeta["training_mode"], "smoke")
            self.assertTrue(torch.equal(loaded_energy.fc2.weight, energy.fc2.weight))
            self.assertTrue(torch.equal(loaded_diffusion.final.weight, ema.final.weight))


class APIContractTests(unittest.TestCase):
    def test_http_png_responses_and_missing_checkpoint(self) -> None:
        import app.main as service

        torch.set_num_threads(2)
        original = (
            service.EBM, service.EBM_META, service.EBM_WARNING,
            service.DIFFUSION, service.DIFFUSION_META, service.DIFFUSION_WARNING,
        )
        try:
            # These random-weight fixtures test HTTP wiring only; they are
            # never saved or reported as trained checkpoints.
            service.EBM = EnergyModel(base_channels=4)
            service.EBM_META = {"training_mode": "smoke", "config": {"step_size": 0.1, "noise_std": 0.0}}
            service.EBM_WARNING = None
            service.DIFFUSION = DiffusionUNet(base_width=4, block_depth=1)
            service.DIFFUSION_META = {
                "training_mode": "smoke",
                "normalizer_mean": (0.5,) * 3,
                "normalizer_std": (0.5,) * 3,
                "min_signal_rate": 0.02,
                "max_signal_rate": 0.95,
            }
            service.DIFFUSION_WARNING = None
            client = TestClient(service.app)
            self.assertEqual(client.get("/health").json()["status"], "ready")
            for route in ("/ebm/generate?num_samples=2&steps=2&seed=11",
                          "/diffusion/generate?num_samples=2&steps=2&seed=11"):
                response = client.get(route)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(response.headers["x-training-mode"], "smoke")
                image = Image.open(BytesIO(response.content))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (70, 36))
                self.assertEqual(client.get(route).content, response.content)
            service.EBM = None
            service.EBM_WARNING = "test checkpoint missing"
            self.assertEqual(client.get("/ebm/generate?steps=1").status_code, 503)
        finally:
            (
                service.EBM, service.EBM_META, service.EBM_WARNING,
                service.DIFFUSION, service.DIFFUSION_META, service.DIFFUSION_WARNING,
            ) = original


if __name__ == "__main__":
    unittest.main()

