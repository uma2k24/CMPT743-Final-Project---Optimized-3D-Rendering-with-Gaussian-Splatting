from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


PROMPT_PALETTES = {
    "car": [[0.88, 0.18, 0.15], [0.14, 0.15, 0.18], [0.82, 0.82, 0.84]],
    "tree": [[0.20, 0.46, 0.19], [0.55, 0.69, 0.32], [0.45, 0.28, 0.14]],
    "stone": [[0.75, 0.73, 0.68], [0.58, 0.57, 0.55], [0.36, 0.35, 0.34]],
    "statue": [[0.76, 0.74, 0.69], [0.61, 0.60, 0.58], [0.31, 0.31, 0.32]],
    "ice": [[0.86, 0.93, 0.98], [0.52, 0.72, 0.90], [0.20, 0.42, 0.67]],
    "building": [[0.76, 0.75, 0.72], [0.58, 0.56, 0.54], [0.26, 0.28, 0.33]],
    "robot": [[0.77, 0.79, 0.82], [0.19, 0.23, 0.29], [0.28, 0.55, 0.86]],
    "default": [[0.79, 0.67, 0.48], [0.43, 0.56, 0.78], [0.26, 0.27, 0.31]],
}


def prompt_palette(prompt: str, device: torch.device) -> torch.Tensor:
    lower_prompt = prompt.lower()
    for keyword, palette in PROMPT_PALETTES.items():
        if keyword != "default" and keyword in lower_prompt:
            return torch.tensor(palette, dtype=torch.float32, device=device)
    return torch.tensor(PROMPT_PALETTES["default"], dtype=torch.float32, device=device)


def procedural_colors(points: torch.Tensor, normals: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    indices = (
        (points[:, 0] > 0).long()
        + (points[:, 1] > 0).long()
        + 2 * (points[:, 2] > 0).long()
    ) % palette.shape[0]
    base = palette[indices]
    shading = 0.60 + 0.40 * ((normals[:, 1:2] + 1.0) * 0.5)
    tint = 0.12 * ((normals + 1.0) * 0.5)
    return (base * shading + tint).clamp(0.02, 0.98)


@dataclass
class GaussianState:
    means: torch.Tensor
    scales: torch.Tensor
    colors: torch.Tensor
    opacity: torch.Tensor


class AnchoredGaussianModel(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        normals: torch.Tensor,
        prompt: str,
        init_scale: float = 0.075,
        jitter: float = 0.03,
    ) -> None:
        super().__init__()
        palette = prompt_palette(prompt, anchors.device)
        colors = procedural_colors(anchors, normals, palette)
        noise = jitter * torch.randn_like(anchors)

        self.register_buffer("anchor_positions", anchors)
        self.register_buffer("anchor_normals", normals)
        self.register_buffer("palette", palette)

        self.means = nn.Parameter(anchors + 0.5 * noise + 0.5 * jitter * normals)
        self.log_scales = nn.Parameter(torch.full_like(anchors, init_scale).log())
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(0.02, 0.98)))
        self.opacity_logits = nn.Parameter(torch.full((anchors.shape[0], 1), 0.70, device=anchors.device).logit())

    def state(self) -> GaussianState:
        scales = torch.exp(self.log_scales).clamp(0.01, 0.20)
        colors = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits).clamp(0.02, 0.98)
        return GaussianState(means=self.means, scales=scales, colors=colors, opacity=opacity)
