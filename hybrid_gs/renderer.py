from __future__ import annotations

import torch

from hybrid_gs.camera import Camera
from hybrid_gs.gaussians import GaussianState


def render_gaussians(
    state: GaussianState,
    camera: Camera,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    near_plane: float = 0.05,
) -> torch.Tensor:
    device = state.means.device
    image_size = camera.image_size
    xs = torch.arange(image_size, device=device, dtype=torch.float32)
    ys = torch.arange(image_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    image = torch.ones((image_size, image_size, 3), device=device, dtype=torch.float32)
    image = image * torch.tensor(background, device=device, dtype=torch.float32).view(1, 1, 3)

    camera_points = camera.world_to_camera(state.means)
    depth = camera_points[:, 2]
    valid = depth > near_plane
    if not torch.any(valid):
        return image

    points = camera_points[valid]
    colors = state.colors[valid]
    opacity = state.opacity[valid]
    scales = state.scales[valid]

    order = torch.argsort(points[:, 2], descending=True)
    points = points[order]
    colors = colors[order]
    opacity = opacity[order]
    scales = scales[order]

    cx = image_size * 0.5
    cy = image_size * 0.5
    projected_x = camera.focal * (points[:, 0] / points[:, 2]) + cx
    projected_y = cy - camera.focal * (points[:, 1] / points[:, 2])
    sigma = (camera.focal * scales.mean(dim=-1) / points[:, 2]).clamp(0.7, image_size * 0.2)

    for index in range(points.shape[0]):
        dx = grid_x - projected_x[index]
        dy = grid_y - projected_y[index]
        dist2 = (dx * dx + dy * dy) / (sigma[index] * sigma[index] + 1e-6)
        alpha = opacity[index, 0] * torch.exp(-0.5 * dist2)
        alpha = alpha.clamp(0.0, 0.98).unsqueeze(-1)
        image = image * (1.0 - alpha) + colors[index].view(1, 1, 3) * alpha

    return image.clamp(0.0, 1.0)
