# Assignment 4: CIFAR-10 Energy and Diffusion API

For this assignment, I extended the Assignment 3/Module 6 helper-library API
with two RGB 32 × 32 image generators trained on CIFAR-10. The previous MNIST
GAN routes and checkpoint remain available. The new FastAPI routes load saved PyTorch
weights and return PNG images; they do not train a model during a request.

## Training results

I completed a notebook run on 2026-11-12 using the full CIFAR-10 training
split: the EBM reached epoch 10/10 and diffusion reached epoch 50/50.
`models/ebm_cifar10.pt` and `models/diffusion_cifar10.pt` contain these final
weights; the Docker image has been rebuilt with them. `/health` reports
`training_mode: full` for both models, and both generation routes returned
valid 32 × 32 PNGs. The saved outputs are in
`CIFAR10_EBM_Diffusion_Practical_Full_Run.ipynb`.

The `_smoke.pt` files remain as short-run functionality checks. Full-run EBM
samples still resemble colored texture noise; diffusion samples are blurry.
The final diffusion held-out noise-prediction L1 is 0.1907. This is a proxy
loss, not an image-quality score. Training loss spiked twice, and the notebook
saves the latest epoch rather than selecting the lowest held-out loss.
I did not compute FID or Inception Score or run a human study.

## Repository layout

- `helper_lib/model.py`: GAN compatibility, CIFAR-10 energy network, and a
  three-level residual U-Net with sinusoidal noise embedding.
- `helper_lib/diffusion.py`: offset-cosine signal/noise schedule and RGB
  normalization checks.
- `helper_lib/trainer.py`: persistent replay-buffer EBM training and noise
  prediction diffusion training with EMA weights.
- `helper_lib/generator.py`: Langevin image-gradient sampling, deterministic
  reverse diffusion, and image grids.
- `helper_lib/checkpoint.py`: versioned checkpoint save/load, including model
  shape, normalization, schedule, training mode, and epoch metadata.
- `train_ebm.py`, `train_diffusion.py`: CIFAR-10 command-line training entry
  points. `training_common.py` shares data loading and seed/device utilities.
- `CIFAR10_EBM_Diffusion_Practical_Full_Run.ipynb`: the completed notebook
  with saved outputs from 10 EBM epochs and 50 diffusion epochs.
- `CIFAR10_EBM_Diffusion_Practical.ipynb`: an optional short pipeline check
  using the same library and checkpoint format; it has no saved outputs.
- `app/main.py`: FastAPI inference service.

## Model and training logic

The EBM maps a normalized RGB image in `[-1, 1]` to one energy. Each negative
example starts from a replay-buffer image or fresh uniform noise. Langevin
sampling repeatedly updates **the input image**, using `autograd.grad` with
respect to the image while leaving parameter gradients untouched. Network
training then lowers energy on real CIFAR-10 images and raises it on sampled
negative images, with an energy-squared penalty. The default setup follows the
Module 8 practical: 60 Langevin steps, step size 10, perturbation standard
deviation 0.005, and regularization weight 0.1.

The diffusion model receives a standardized CIFAR-10 image corrupted at a
random continuous time and predicts the Gaussian noise that was added. It uses
an offset-cosine schedule, L1 noise-prediction loss, AdamW, and an EMA copy of
the network. The checkpoint records the training-set per-channel mean and
standard deviation and serves EMA weights. Generation uses a deterministic
DDIM-style reverse path from Gaussian noise; `steps` controls the number of
denoising calls. The compact default U-Net has three downsampling and three
upsampling levels and supports larger `--base-width`/`--block-depth` values.

## Install and train

Use Python 3.11 or newer and install local dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the interactive notebook, additionally install
`python -m pip install -r requirements-notebook.txt` and launch
`jupyter lab CIFAR10_EBM_Diffusion_Practical_Full_Run.ipynb` from this directory.
The notebooks use `data/` by default and download CIFAR-10 if it is absent.
To use an existing torchvision cache, set `CIFAR10_DATA_DIR` before launching
Jupyter. Docker inference does not need the CIFAR-10 data directory.

The first run downloads CIFAR-10 into `data/` unless `--offline` is set.
`--data-dir` may point to an existing torchvision CIFAR-10 directory. These
commands use the **entire** 50,000-image training split:

```bash
python train_ebm.py --epochs 10 --batch-size 128
python train_diffusion.py --epochs 50 --batch-size 64
```

The diffusion script evaluates noise-prediction L1 loss on the CIFAR-10 test
split after each epoch, saves the best validation checkpoint to
`models/diffusion_cifar10.pt`, and writes the latest epoch to a separate file.
Because that split is used for checkpoint selection, the resulting loss is a
development metric, not an unbiased final test score or an image-quality
measure. A separate untouched evaluation set would be needed for that claim.
The EBM script saves `models/ebm_cifar10.pt` after every epoch. Both write
preview PNGs and `training_history.json` under `outputs/`. The EBM training
loop is especially expensive because every minibatch performs many input-image
gradient steps. A GPU or Apple Silicon MPS device is recommended for full runs.

For a quick **functionality check only**, use limited data and batches. The
default output names then end in `_smoke.pt`, so the short run does not
overwrite a full checkpoint:

```bash
python train_ebm.py --epochs 1 --limit-train 128 --max-batches 1 \
  --langevin-steps 2 --preview-steps 2
python train_diffusion.py --epochs 1 --limit-train 128 --limit-val 64 \
  --max-batches 1 --preview-steps 2
```

To reuse downloaded CIFAR-10, append `--data-dir /path/to/existing/data
--offline` to either command. The data directory must contain torchvision's
`cifar-10-batches-py/` folder.

## Run the API locally

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. The important routes are:

- `GET /health`: checkpoint availability and each model's training mode.
- `GET /ebm/model-info` and `GET /diffusion/model-info`: shape, epoch, schedule,
  normalization, and checkpoint metadata.
- `GET /ebm/generate?num_samples=1&steps=64&seed=42`: PNG from Langevin
  image-gradient sampling. `num_samples` is 1–8; `steps` is 1–256.
- `GET /diffusion/generate?num_samples=1&steps=20&seed=42`: PNG from reverse
  diffusion. `num_samples` is 1–8; `steps` is 1–1000. More steps require more
  CPU time and may produce a better approximation after full training.
- `GET /generate-digit` and `GET /generate-digits`: Assignment 3 GAN
  compatibility endpoints.

For example:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS -o ebm.png 'http://127.0.0.1:8000/ebm/generate?steps=64&seed=42'
curl -fsS -o diffusion.png 'http://127.0.0.1:8000/diffusion/generate?steps=100&seed=42'
```

If a checkpoint is unavailable or invalid, its model endpoint returns HTTP
503 with a clear error. The server still starts so `/health` explains what is
missing. If you set `EBM_MODEL_PATH` or `DIFFUSION_MODEL_PATH`, that explicit
path is used without smoke fallback. Restart the server after replacing a
checkpoint.

## Docker deployment

Build **after** creating or copying the checkpoint files that should be
included in the image:

```bash
docker build -t assignment4-image-api .
docker run --rm -p 8000:8000 assignment4-image-api
```

The Docker image installs CPU-only PyTorch and copies `models/`; it requires no
CIFAR download or training at startup. Check `/health`, then save one PNG from
each new endpoint with the curl commands above. After changing checkpoints, rebuild the image. I validated the full-checkpoint
image locally; the commands above repeat the `/health` and generation checks.

## Verification

Install `requirements-test.txt`, then run
`python -m unittest discover -s tests -v`. The checks test
tensor shapes, gradients, one-batch training, checkpoint round trips, and PNG
responses. These tests verify the code path, not image quality or full training.
The written Q1–Q9 derivations are submitted separately through the course
LMS as `Assignment4_Theory_Answers.pdf`; that PDF is not part of this code
repository. The Docker image loads the full checkpoints under `models/`.
