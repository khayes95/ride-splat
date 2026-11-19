import json

import torch
from PIL import Image

import run_ridesplat


def commands(argv):
    args, extra = run_ridesplat.parse(argv)
    return run_ridesplat.pass_commands(args, extra)


def test_paper_pass_2_adds_weight_attenuation_and_appearance():
    pass1, pass2 = commands(["-s", "scene", "-m", "out"])
    for flag in ("--refl_weight", "--render_dir", "--atten", "--appearance"):
        assert flag not in pass1 and flag in pass2
    assert "--eval" in pass1 and "--eval" in pass2


def test_minimal_pass_2_keeps_only_the_weight():
    _, pass2 = commands(["-s", "scene", "-m", "out", "--minimal"])
    assert "--refl_weight" in pass2 and "--atten" not in pass2 and "--appearance" not in pass2


def test_unknown_arguments_reach_both_passes():
    pass1, pass2 = commands(["-s", "scene", "-m", "out", "--refl_l1", "0.002", "--smoke_test"])
    assert pass1[-2:] == ["--refl_l1", "0.002"] and pass2[-2:] == ["--refl_l1", "0.002"]
    assert "--smoke_test" in pass1 and "--smoke_test" in pass2


def test_rerun_resumes_after_the_last_finished_step(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_ridesplat, "run", calls.append)
    out = tmp_path / "out"
    (out / "pass1").mkdir(parents=True)
    R = torch.zeros(3, 32, 48)
    R[1, 8:16, 10:30] = -0.4
    torch.save(R, out / "pass1" / "refl_layer.pt")

    run_ridesplat.main(["-s", "scene", "-m", str(out)])          # pass 1 done: weight, then pass 2
    assert len(calls) == 1 and "--refl_weight" in calls[0]
    assert Image.open(out / "weight.png").size == (48, 32)
    assert json.loads((out / "ridesplat_run.json").read_text())["pass2"] == "paper"

    (out / "pass2").mkdir(exist_ok=True)
    torch.save(R, out / "pass2" / "refl_layer.pt")
    (out / "renders").mkdir()
    Image.new("RGB", (48, 32)).save(out / "renders" / "eval_c00000.png")
    run_ridesplat.main(["-s", "scene", "-m", str(out), "--gt", "gt"])   # only scoring is left
    assert len(calls) == 2 and calls[1][1:3] == ["-m", "ridesplat.eval.metrics"]
