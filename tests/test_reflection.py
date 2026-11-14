import math

import pytest
import torch

from ridesplat.models.reflection import ReflectionModel

H, W = 8, 12
FRAMES = ["train_r00000", "train_r00001"]


def model(**kw):
    return ReflectionModel(H, W, FRAMES, device="cpu", **kw)


def test_starts_as_plain_3dgs():
    m = model()
    T = torch.rand(3, H, W)
    pred, reg = m.predict(T)
    assert torch.equal(pred, T) and float(reg) == 0.0


def test_composite_is_scene_plus_weighted_reflection():
    weight = torch.rand(1, H, W)
    m = model(weight=weight)
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
    T = torch.rand(3, H, W)
    assert torch.allclose(m.predict(T)[0], T + weight * m.R)


def test_penalty_is_mean_l1():
    m = model(l1=0.01)
    with torch.no_grad():
        m.R.fill_(-0.5)
    assert math.isclose(float(m.penalty().detach()), 0.01 * 0.5, rel_tol=1e-6)


def test_step_moves_r_toward_the_residual():
    m = model()
    T = torch.full((3, H, W), 0.2)
    ((m.predict(T)[0] - 0.5) ** 2).mean().backward()
    m.step()
    assert float(m.R.detach().min()) > 0          # the frame is brighter than the render, so R grows
    assert m.R.grad is None


def test_state_round_trip():
    a = model()
    a.predict(torch.rand(3, H, W))[0].mean().add(a.penalty()).backward()
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
    assert torch.allclose(m.predict(T)[0], (1 - A) * T + weight * m.R)


def test_penalty_adds_attenuation_l1():
    m = model(l1=0.01, atten=True, atten_l1=0.02)
    with torch.no_grad():
        m.R.fill_(0.5)
    expected = 0.01 * 0.5 + 0.02 * float(m.attenuation().detach().mean())
    assert math.isclose(float(m.penalty().detach()), expected, rel_tol=1e-6)


def test_attenuation_state_round_trip_and_save(tmp_path):
    a = model(atten=True)
    ((a.predict(torch.rand(3, H, W))[0] - 0.3) ** 2).mean().add(a.penalty()).backward()
    a.step()
    b = model(atten=True)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.atten_logit, b.atten_logit)
    a.save(str(tmp_path))
    assert torch.allclose(torch.load(tmp_path / "atten_layer.pt"), a.attenuation().detach())
    assert (tmp_path / "atten_layer.png").is_file()


def test_full_composite_matches_the_formula():
    weight = torch.rand(1, H, W)
    m = model(weight=weight, atten=True, atten_max=0.9, appearance="affine")
    with torch.no_grad():
        m.R.copy_(torch.randn(3, H, W))
        m.atten_logit.copy_(torch.randn(1, H, W))
        m.gain[FRAMES[0]].copy_(torch.tensor([1.1, 0.9, 1.0]).view(3, 1, 1))
        m.bias[FRAMES[0]].copy_(torch.tensor([0.02, -0.01, 0.0]).view(3, 1, 1))
    T = torch.rand(3, H, W)
    pred, reg = m.predict(T, FRAMES[0])
    g, b = m.gain[FRAMES[0]], m.bias[FRAMES[0]]
    A = 0.9 * torch.sigmoid(m.atten_logit)
    assert torch.allclose(pred, (1 - A) * (g * T + b) + weight * m.R)
    assert torch.allclose(reg, ((g - 1) ** 2).mean() + (b ** 2).mean())


def test_appearance_starts_at_identity_and_needs_a_known_frame():
    m = model(appearance="affine")
    T = torch.rand(3, H, W)
    pred, reg = m.predict(T, FRAMES[1])
    assert torch.equal(pred, T) and float(reg.detach()) == 0.0
    with pytest.raises(KeyError):
        m.predict(T, "train_r99999")
    with pytest.raises(ValueError):
        model(appearance="mlp")


def test_penalty_weights_the_appearance_regularizer():
    m = model(l1=0.01, appearance="affine", appearance_reg=10.0)
    with torch.no_grad():
        m.R.fill_(0.5)
    assert math.isclose(float(m.penalty(torch.tensor(0.2)).detach()), 0.01 * 0.5 + 10.0 * 0.2, rel_tol=1e-6)


def test_step_moves_only_the_frame_that_was_used(tmp_path):
    m = model(appearance="affine")
    pred, reg = m.predict(torch.rand(3, H, W), FRAMES[0])
    ((pred - 0.3) ** 2).mean().add(m.penalty(reg)).backward()
    m.step()
    assert not torch.equal(m.gain[FRAMES[0]], torch.ones(3, 1, 1))
    assert torch.equal(m.gain[FRAMES[1]], torch.ones(3, 1, 1))
    b = model(appearance="affine")
    b.load_state_dict(m.state_dict())
    assert torch.equal(b.gain[FRAMES[0]], m.gain[FRAMES[0]]) and torch.equal(b.bias[FRAMES[0]], m.bias[FRAMES[0]])
    m.save(str(tmp_path))
    assert set(torch.load(tmp_path / "appearance.pt")) == set(FRAMES)
