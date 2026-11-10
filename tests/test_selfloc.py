import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from ridesplat.models.reflection import ReflectionModel
from ridesplat.models.selfloc import load_weight, save_weight, weight_from_reflection


def reference_weight_png(R):
    """The weight builder used in our experiments, kept verbatim as the reference."""
    R = R.float().abs()
    while R.dim() > 2:
        R = R.amax(dim=0)
    acc = R.numpy()
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    acc = gaussian_filter(acc, 8)
    return (np.clip(acc, 0, 1) * 255).astype("uint8")


def reflection_with_patch(h=64, w=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = 0.01 * torch.randn(3, h, w, generator=g)
    R[1, 20:40, 30:60] = -0.5     # a strong, signed reflection patch
    return R


def test_weight_matches_experiment_builder(tmp_path):
    R = reflection_with_patch()
    path = tmp_path / "weight.png"
    save_weight(weight_from_reflection(R, sigma=8), path)
    assert np.array_equal(np.asarray(Image.open(path)), reference_weight_png(R))


def test_weight_is_high_on_the_reflection():
    W = weight_from_reflection(reflection_with_patch(), sigma=8)
    assert W.shape == (64, 96) and W.min() >= 0 and W.max() <= 1
    assert W[30, 45] > 0.5 > W[5, 5]


def test_load_weight_rescales_and_resizes(tmp_path):
    path = tmp_path / "w.png"
    Image.fromarray(np.array([[10, 60], [110, 210]], dtype=np.uint8)).save(path)
    w = load_weight(path)
    assert w.shape == (1, 2, 2)
    assert abs(float(w.min())) < 1e-6 and abs(float(w.max()) - 1) < 1e-5
    assert load_weight(path, size=(4, 6)).shape == (1, 4, 6)


def test_weight_keeps_r_out_of_reflection_free_pixels(tmp_path):
    R = reflection_with_patch()
    path = tmp_path / "weight.png"
    save_weight(weight_from_reflection(R), path)
    m = ReflectionModel(64, 96, device="cpu", weight=load_weight(path))
    with torch.no_grad():
        m.R.copy_(R)
    T = torch.zeros(3, 64, 96)
    pred = m.predict(T).detach()
    assert float(pred[:, 0:5, 0:5].abs().max()) < 1e-3 < float(pred[:, 25:35, 40:50].abs().max())
