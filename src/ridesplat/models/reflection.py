"""The reflection layer: one signed RGB image R shared by every training frame.

For training frame t with 3DGS render T_t, the predicted observation is

    during the geometry warm-up:   I_t ~ T_t
    afterwards:                    I_t ~ T_t + W * R

R starts at zero and carries an L1 penalty, so ambiguous pixels are explained by the 3D scene. W is a
fixed spatial weight (all ones unless one is given). Because R is shared by every frame it cannot
follow parallax: it can only hold content that stays put in the image, such as a reflection on the
windshield. Held-out views render T alone.
"""
import os

import numpy as np
import torch
from PIL import Image


class ReflectionModel:
    def __init__(self, height, width, device="cuda", lr=0.01, l1=0.01, weight=None):
        self.l1 = l1
        self.weight = None if weight is None else weight.to(device)        # [1,H,W] in [0,1]
        self.R = torch.nn.Parameter(torch.zeros(3, height, width, device=device))
        self.optimizer = torch.optim.Adam([self.R], lr=lr)

    def predict(self, T):
        """Predicted observation of a training frame from its render T [3,H,W]."""
        return T + (self.R if self.weight is None else self.R * self.weight)

    def penalty(self):
        """L1 on R, averaged over channels and pixels; added to the photometric loss once R is active."""
        return self.l1 * self.R.abs().mean()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"R": self.R.detach().cpu(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.R.copy_(state["R"])
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, model_path):
        """Write refl_layer.pt, plus R and |R| as images for inspection."""
        R = self.R.detach().cpu()
        torch.save(R, os.path.join(model_path, "refl_layer.pt"))
        _save_png(R.clamp(0, 1), os.path.join(model_path, "refl_layer.png"))
        peak = float(R.abs().max())
        if peak > 1e-6:
            _save_png(R.abs() / peak, os.path.join(model_path, "refl_layer_abs.png"))
        print(f"[reflection] saved R (mean |R| {float(R.abs().mean()):.4f}) to {model_path}")


def _save_png(chw, path):
    Image.fromarray((chw.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)
