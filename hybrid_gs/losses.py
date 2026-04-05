from __future__ import annotations

import torch


def reconstruction_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = torch.mean(torch.abs(rendered - target))
    mse = torch.mean((rendered - target) ** 2)
    return l1 + 0.25 * mse


def tether_loss(means: torch.Tensor, anchors: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    offsets = means - anchors
    normal_offsets = (offsets * normals).sum(dim=-1, keepdim=True) * normals
    tangent_offsets = offsets - normal_offsets
    return tangent_offsets.pow(2).sum(dim=-1).mean() + 0.25 * normal_offsets.pow(2).sum(dim=-1).mean()


def appearance_guidance_loss(colors: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    distances = ((colors.unsqueeze(1) - palette.unsqueeze(0)) ** 2).sum(dim=-1)
    return distances.min(dim=1).values.mean()


def scale_regularization(scales: torch.Tensor) -> torch.Tensor:
    return scales.mean()


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    return torch.mean(opacity * (1.0 - opacity))
