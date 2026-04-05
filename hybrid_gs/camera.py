from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class Camera:
    eye: torch.Tensor
    target: torch.Tensor
    up: torch.Tensor
    focal: float
    image_size: int

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        forward = _normalize(self.target - self.eye)
        right = _normalize(torch.cross(forward, self.up, dim=0))
        true_up = _normalize(torch.cross(right, forward, dim=0))
        translated = points - self.eye.unsqueeze(0)
        basis = torch.stack((right, true_up, forward), dim=1)
        return translated @ basis


def orbit_cameras(
    num_views: int,
    radius: float,
    elevation_degrees: float,
    image_size: int,
    fov_degrees: float,
    device: torch.device,
) -> list[Camera]:
    elevation = math.radians(elevation_degrees)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    focal = 0.5 * image_size / math.tan(math.radians(fov_degrees) * 0.5)
    cameras: list[Camera] = []

    for index in range(num_views):
        azimuth = (2.0 * math.pi * index) / max(num_views, 1)
        eye = torch.tensor(
            [
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.sin(elevation),
                radius * math.cos(elevation) * math.sin(azimuth),
            ],
            dtype=torch.float32,
            device=device,
        )
        cameras.append(Camera(eye=eye, target=target, up=up, focal=focal, image_size=image_size))

    return cameras
