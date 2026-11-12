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


def test_attenuation_starts_near_zero():
    A = model(atten=True, atten_max=0.9).attenuation()
    assert torch.allclose(A, torch.full((1, H, W), 0.9 / (1 + math.exp(3))))
    assert model().attenuation() is None


def test_attenuated_composite():
    weight = torch.rand(1, H, W)
    m = model(weight=weight, atten=True, atten_max=0.9)
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
        m.atten_logit.copy_(torch.randn(1, H, W))
    T = torch.rand(3, H, W)
    A = 0.9 * torch.sigmoid(m.atten_logit)
    assert torch.allclose(m.predict(T), (1 - A) * T + weight * m.R)


def test_penalty_adds_attenuation_l1():
    m = model(l1=0.01, atten=True, atten_l1=0.02)
    with torch.no_grad():
        m.R.fill_(0.5)
    expected = 0.01 * 0.5 + 0.02 * float(m.attenuation().detach().mean())
    assert math.isclose(float(m.penalty().detach()), expected, rel_tol=1e-6)


def test_attenuation_state_round_trip_and_save(tmp_path):
    a = model(atten=True)
    ((a.predict(torch.rand(3, H, W)) - 0.3) ** 2).mean().add(a.penalty()).backward()
    a.step()
    b = model(atten=True)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.atten_logit, b.atten_logit)
    a.save(str(tmp_path))
    assert torch.allclose(torch.load(tmp_path / "atten_layer.pt"), a.attenuation().detach())
    assert (tmp_path / "atten_layer.png").is_file()
