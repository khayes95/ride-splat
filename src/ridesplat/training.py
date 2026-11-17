"""RIDE-Splat's options for one 3DGS training pass, and the reflection model they configure.

Kept free of 3DGS imports so the options can be tested without a GPU; train.py adds them to the
3DGS argument parser. Defaults are the settings used for every reported result.
"""
from ridesplat.models.reflection import ReflectionModel
from ridesplat.models.selfloc import load_weight

SMOKE_ITERATIONS = 300


def add_ridesplat_args(parser):
    rs = parser.add_argument_group("RIDE-Splat")
    rs.add_argument("--refl_lr", type=float, default=0.01, help="Adam learning rate for R (eta_R).")
    rs.add_argument("--refl_l1", type=float, default=0.01, help="Weight of the L1 penalty on R (beta).")
    rs.add_argument("--refl_warmup", type=int, default=2000,
                    help="Plain 3DGS iterations before R is used (t_0).")
    rs.add_argument("--refl_weight", type=str, default=None,
                    help="Self-localization weight W (a PNG from ridesplat.models.selfloc). Pass 2 only.")
    rs.add_argument("--atten", action="store_true", help="Also fit a per-pixel attenuation A.")
    rs.add_argument("--atten_max", type=float, default=0.9, help="Upper bound on A.")
    rs.add_argument("--atten_l1", type=float, default=0.01, help="Weight of the L1 penalty on A.")
    rs.add_argument("--appearance", choices=["none", "affine"], default="none",
                    help="Per-frame RGB gain and bias on the scene render during training.")
    rs.add_argument("--appearance_lr", type=float, default=1e-3)
    rs.add_argument("--appearance_reg", type=float, default=10.0,
                    help="Weight pulling each frame's gain and bias toward identity.")
    rs.add_argument("--render_dir", type=str, default=None,
                    help="After training, render the held-out views (scene only) into this folder.")
    rs.add_argument("--smoke_test", action="store_true",
                    help=f"{SMOKE_ITERATIONS} iterations with a shortened warm-up, to check the "
                         "pipeline end to end.")
    return rs


def apply_smoke_test(args):
    """With --smoke_test, cap the run at SMOKE_ITERATIONS and the warm-up at a third of the run."""
    if args.smoke_test:
        args.iterations = min(args.iterations, SMOKE_ITERATIONS)
        args.refl_warmup = min(args.refl_warmup, args.iterations // 3)
    return args


def build_reflection(args, size, frame_names, device="cuda"):
    """The reflection model for training images of `size` (H, W) and the given frame names."""
    weight = load_weight(args.refl_weight, size) if args.refl_weight else None
    if weight is not None:
        print(f"[reflection] weight {args.refl_weight}: mean {float(weight.mean()):.3f}")
    return ReflectionModel(size[0], size[1], frame_names, device=device, lr=args.refl_lr, l1=args.refl_l1,
                           weight=weight, atten=args.atten, atten_max=args.atten_max,
                           atten_l1=args.atten_l1, appearance=args.appearance,
                           appearance_lr=args.appearance_lr, appearance_reg=args.appearance_reg)
