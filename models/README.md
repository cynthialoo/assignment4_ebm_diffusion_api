The trained CIFAR-10 checkpoints are created by `train_ebm.py` and
`train_diffusion.py`. Full-data training writes `ebm_cifar10.pt` and
`diffusion_cifar10.pt`; limited smoke runs write filenames ending in `_smoke.pt`.
The API uses full checkpoints first and clearly labels any smoke fallback.
No randomly initialized CIFAR model is presented as trained.
