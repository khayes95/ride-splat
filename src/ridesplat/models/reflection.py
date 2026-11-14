"""The reflection layer: one signed RGB image R shared by every training frame.

For training frame t with 3DGS render T_t, the predicted observation is

    during the geometry warm-up:   I_t ~ T_t
    afterwards:                    I_t ~ (1 - A) * a_t(T_t) + W * R

R starts at zero and carries an L1 penalty, so ambiguous pixels are explained by the 3D scene. W is a
fixed spatial weight (all ones unless one is given). A is an optional per-pixel attenuation in
[0, atten_max), for glass that dims the scene behind it as well as adding a reflection; it starts
near zero and carries its own L1 penalty. a_t is an optional per-frame RGB gain and bias that
absorbs exposure and white-balance drift between frames, pulled toward identity. Because R and A are
shared by every frame they cannot follow parallax: they can only hold content that stays put in the
image, such as a reflection on the windshield. Held-out views render T alone: R, A and a_t never
enter evaluation.
"""
import os

import numpy as np
import torch
from PIL import Image


class ReflectionModel:
    def __init__(self, height, width, frame_names=(), device="cuda", lr=0.01, l1=0.01, weight=None,
                 atten=False, atten_max=0.9, atten_l1=0.01,
                 appearance="none", appearance_lr=1e-3, appearance_reg=10.0):
        if appearance not in ("none", "affine"):
            raise ValueError(f"appearance must be 'none' or 'affine', got {appearance!r}")
        self.l1 = l1
        self.atten_max = atten_max
        self.atten_l1 = atten_l1
        self.appearance = appearance
        self.appearance_reg = appearance_reg
        self.weight = None if weight is None else weight.to(device)        # [1,H,W] in [0,1]

        self.R = torch.nn.Parameter(torch.zeros(3, height, width, device=device))
        groups = [{"params": [self.R], "lr": lr}]
        self.atten_logit = None
        if atten:
            # A = atten_max * sigmoid(-3), about 0.04 at the start: almost no attenuation.
            self.atten_logit = torch.nn.Parameter(torch.full((1, height, width), -3.0, device=device))
            groups.append({"params": [self.atten_logit], "lr": lr})
        self.gain, self.bias = {}, {}
        if appearance == "affine":
            for name in frame_names:
                self.gain[name] = torch.nn.Parameter(torch.ones(3, 1, 1, device=device))
                self.bias[name] = torch.nn.Parameter(torch.zeros(3, 1, 1, device=device))
                groups.append({"params": [self.gain[name], self.bias[name]], "lr": appearance_lr})
        self.optimizer = torch.optim.Adam(groups)

    def attenuation(self):
        """A [1,H,W] in [0, atten_max), or None when attenuation is off."""
        if self.atten_logit is None:
            return None
        return self.atten_max * torch.sigmoid(self.atten_logit)

    def predict(self, T, name=None):
        """Predicted observation of training frame `name` from its render T [3,H,W].

        Returns (prediction, appearance regularizer for this frame); the regularizer is zero unless
        per-frame appearance is on, and goes to penalty().
        """
        R = self.R if self.weight is None else self.R * self.weight
        reg = T.new_zeros(())
        if self.appearance == "affine":
            if name not in self.gain:
                raise KeyError(f"no per-frame appearance for frame {name!r}")
            gain, bias = self.gain[name], self.bias[name]
            T = gain * T + bias
            reg = ((gain - 1.0) ** 2).mean() + (bias ** 2).mean()
        A = self.attenuation()
        if A is not None:
            T = (1.0 - A) * T
        return T + R, reg

    def penalty(self, appearance_reg=0.0):
        """Regularizers added to the photometric loss once R is active: L1 on R, L1 on A if used, and
        the appearance regularizer returned by predict() weighted by appearance_reg."""
        loss = self.l1 * self.R.abs().mean()
        if self.appearance != "none":
            loss = loss + self.appearance_reg * appearance_reg
        A = self.attenuation()
        if A is not None:
            loss = loss + self.atten_l1 * A.mean()
        return loss

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"R": self.R.detach().cpu(),
                "atten_logit": None if self.atten_logit is None else self.atten_logit.detach().cpu(),
                "gain": {k: v.detach().cpu() for k, v in self.gain.items()},
                "bias": {k: v.detach().cpu() for k, v in self.bias.items()},
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.R.copy_(state["R"])
            if self.atten_logit is not None:
                self.atten_logit.copy_(state["atten_logit"])
            for k, v in state.get("gain", {}).items():
                self.gain[k].copy_(v)
            for k, v in state.get("bias", {}).items():
                self.bias[k].copy_(v)
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, model_path):
        """Write refl_layer.pt, R and |R| as images, and the attenuation and appearance terms if fit."""
        R = self.R.detach().cpu()
        torch.save(R, os.path.join(model_path, "refl_layer.pt"))
        _save_png(R.clamp(0, 1), os.path.join(model_path, "refl_layer.png"))
        peak = float(R.abs().max())
        if peak > 1e-6:
            _save_png(R.abs() / peak, os.path.join(model_path, "refl_layer_abs.png"))
        A = self.attenuation()
        if A is not None:
            A = A.detach().cpu()
            torch.save(A, os.path.join(model_path, "atten_layer.pt"))
            _save_png(A.clamp(0, 1), os.path.join(model_path, "atten_layer.png"))
        if self.appearance != "none":
            torch.save({k: (self.gain[k].detach().cpu(), self.bias[k].detach().cpu()) for k in self.gain},
                       os.path.join(model_path, "appearance.pt"))
        print(f"[reflection] saved R (mean |R| {float(R.abs().mean()):.4f}) to {model_path}")


def _save_png(chw, path):
    """Save a [C,H,W] tensor in [0,1] (C = 1 or 3) as an 8-bit PNG."""
    a = (chw.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a[:, :, 0] if a.shape[2] == 1 else a).save(path)
