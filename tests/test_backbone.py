import os
import re
import sys

import pytest
import torch

from ridesplat import backbone


def fake_checkout(path):
    (path / "scene").mkdir(parents=True)
    (path / "scene" / "dataset_readers.py").write_text("")
    return str(path)


def test_gs_root_follows_the_environment(tmp_path, monkeypatch):
    root = fake_checkout(tmp_path / "gs")
    monkeypatch.setenv("GS_ROOT", root)
    assert backbone.gs_root() == root


def test_missing_checkout_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(SystemExit, match="install_3dgs.sh"):
        backbone.gs_root()


def test_add_gs_to_path_adds_the_checkout_once(tmp_path, monkeypatch):
    root = fake_checkout(tmp_path / "gs")
    monkeypatch.setenv("GS_ROOT", root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    backbone.add_gs_to_path()
    backbone.add_gs_to_path()
    assert sys.path[0] == root and sys.path.count(root) == 1


def test_install_script_pins_the_same_commit():
    with open(os.path.join(backbone.REPO_ROOT, "scripts", "install_3dgs.sh")) as f:
        pinned = re.search(r"^COMMIT=(\w+)$", f.read(), re.M).group(1)
    assert pinned == backbone.GS_COMMIT


def test_seed_everything_repeats_a_draw():
    backbone.seed_everything(3)
    a = torch.rand(4)
    backbone.seed_everything(3)
    assert torch.equal(a, torch.rand(4))
