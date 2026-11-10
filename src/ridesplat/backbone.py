"""Locate the official 3D Gaussian Splatting code that RIDE-Splat trains on, and seed runs.

RIDE-Splat builds on graphdeco-inria/gaussian-splatting at GS_COMMIT, which
scripts/install_3dgs.sh fetches and compiles. Nothing here changes how 3DGS trains or renders.
"""
import os
import sys

GS_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gs_root():
    """Path to the 3DGS checkout: $GS_ROOT, else third_party/gaussian-splatting in this repo."""
    root = os.environ.get("GS_ROOT") or os.path.join(REPO_ROOT, "third_party", "gaussian-splatting")
    if not os.path.isfile(os.path.join(root, "scene", "dataset_readers.py")):
        sys.exit(f"3D Gaussian Splatting not found at {root}. "
                 "Run scripts/install_3dgs.sh, or point GS_ROOT at a checkout.")
    return root


def add_gs_to_path():
    """Make the 3DGS modules (scene, gaussian_renderer, ...) importable; returns the checkout path."""
    root = gs_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and every GPU)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
