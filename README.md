# RIDE-Splat

Reflection-free 3D reconstruction from driving video.

Dashcam video is recorded through a windshield, so it carries reflections of the cabin. 3D Gaussian
Splatting reconstructs those reflections as phantom geometry. RIDE-Splat models each scene as a 3DGS
reconstruction plus a single signed reflection image shared by every frame, and drops that image at
render time. It needs no clean images, reflection masks or pretrained networks.

This repository is the CMSC 473/673 Machine Learning Capstone project (University of Maryland,
Fall 2026).

## Status

Under active development. Planned contents:

| Milestone | Date |
|---|---|
| Package structure, data loaders and split tests | Oct 13 |
| Reproduction of the XLD comparison; unit tests passing | Mid October |
| Anonymized dataset release and benchmark harness | Nov 3 |
| Drifting-reflection extension; RealX3D results | Nov 24 |
| Final report, presentation and documentation | Early December |

## Layout

```
src/ridesplat/
  backbone.py   finds the pinned 3DGS checkout; seeding
  training.py   RIDE-Splat's training options and the reflection model they configure
  data/         train/held-out split: frames named eval_* are held out
  models/       the shared signed reflection layer, attenuation, per-frame appearance, and the
                self-localization weight
  eval/         PSNR, SSIM and LPIPS of held-out renders against reflection-free targets
train.py        one training pass on top of 3D Gaussian Splatting (needs a CUDA GPU)
run_ridesplat.py  the full method on one scene: pass 1, self-localization, pass 2, held-out renders
scripts/        install_3dgs.sh fetches and builds 3DGS at the pinned commit
tests/          unit tests (pytest; CPU only)
```

## Development

```
pip install -e .
pytest
```

Training needs a CUDA build of PyTorch and the 3DGS extensions:

```
pip install -e ".[train]"
bash scripts/install_3dgs.sh
python train.py -s <scene> -m <output> -r 1 --eval --smoke_test
```

The full method on one COLMAP scene whose held-out frames are named `eval_*`:

```
python run_ridesplat.py -s <scene> -m output/<scene> --gt <scene>/eval
```

This writes pass 1, the self-localization weight, pass 2, the held-out renders and `metrics.json`
under `-m`. Rerunning the same command resumes after the last finished step.

Score a folder of held-out renders against clean targets with the same file names:

```
python -m ridesplat.eval.metrics --pred <renders> --gt <targets> --out metrics.json
```

## License

MIT, see [LICENSE](LICENSE), except `train.py`, which is adapted from 3D Gaussian Splatting (Inria
and MPII) and is under the Gaussian-Splatting License in [LICENSE-3DGS.md](LICENSE-3DGS.md):
non-commercial research and evaluation use only.
