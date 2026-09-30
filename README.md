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
src/ridesplat/   library code (model, reflection layer, training, evaluation)
tests/           unit tests (pytest)
```

## Development

```
pip install -e .
pytest
```

## License

MIT, see [LICENSE](LICENSE).
