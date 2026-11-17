from argparse import ArgumentParser

import numpy as np
from PIL import Image

from ridesplat.training import SMOKE_ITERATIONS, add_ridesplat_args, apply_smoke_test, build_reflection


def parse(argv):
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30_000)       # added by 3DGS in train.py
    add_ridesplat_args(parser)
    return parser.parse_args(argv)


def test_defaults_are_the_reported_settings():
    args = parse([])
    assert (args.refl_lr, args.refl_l1, args.refl_warmup) == (0.01, 0.01, 2000)
    assert args.refl_weight is None and not args.atten and args.appearance == "none"


def test_smoke_test_shortens_the_run_and_the_warm_up():
    args = apply_smoke_test(parse(["--smoke_test"]))
    assert args.iterations == SMOKE_ITERATIONS and args.refl_warmup == SMOKE_ITERATIONS // 3
    args = apply_smoke_test(parse([]))
    assert (args.iterations, args.refl_warmup) == (30_000, 2000)


def test_build_reflection_pass_1():
    m = build_reflection(parse([]), (16, 24), ["train_r00000"], device="cpu")
    assert m.weight is None and m.atten_logit is None and m.gain == {}
    assert tuple(m.R.shape) == (3, 16, 24)


def test_build_reflection_pass_2_loads_and_resizes_the_weight(tmp_path):
    path = tmp_path / "weight.png"
    Image.fromarray(np.tile(np.arange(12, dtype=np.uint8) * 20, (8, 1))).save(path)
    args = parse(["--refl_weight", str(path), "--atten", "--appearance", "affine"])
    m = build_reflection(args, (16, 24), ["train_r00000", "train_r00001"], device="cpu")
    assert tuple(m.weight.shape) == (1, 16, 24)
    assert float(m.weight.min()) == 0.0 and abs(float(m.weight.max()) - 1.0) < 1e-5
    assert m.atten_logit is not None and set(m.gain) == {"train_r00000", "train_r00001"}
