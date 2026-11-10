import math

import torch

from ridesplat.models.reflection import ReflectionModel

H, W = 8, 12


def model(**kw):
    return ReflectionModel(H, W, device="cpu", **kw)


def test_starts_as_plain_3dgs():
    m = model()
    T = torch.rand(3, H, W)
    assert torch.equal(m.predict(T), T)


def test_composite_is_scene_plus_weighted_reflection():
    weight = torch.rand(1, H, W)
    m = model(weight=weight)
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
    T = torch.rand(3, H, W)
    assert torch.allclose(m.predict(T), T + weight * m.R)


def test_penalty_is_mean_l1():
    m = model(l1=0.01)
    with torch.no_grad():
        m.R.fill_(-0.5)
    assert math.isclose(float(m.penalty().detach()), 0.01 * 0.5, rel_tol=1e-6)


def test_step_moves_r_toward_the_residual():
    m = model()
    T = torch.full((3, H, W), 0.2)
    ((m.predict(T) - 0.5) ** 2).mean().backward()
    m.step()
    assert float(m.R.detach().min()) > 0          # the frame is brighter than the render, so R grows
    assert m.R.grad is None


def test_state_round_trip():
    a = model()
    a.predict(torch.rand(3, H, W)).mean().add(a.penalty()).backward()
    a.step()
    b = model()
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.R, b.R)


def test_save_writes_tensor_and_images(tmp_path):
    m = model()
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
    m.save(str(tmp_path))
    assert torch.equal(torch.load(tmp_path / "refl_layer.pt"), m.R.detach())
    assert (tmp_path / "refl_layer.png").is_file() and (tmp_path / "refl_layer_abs.png").is_file()
