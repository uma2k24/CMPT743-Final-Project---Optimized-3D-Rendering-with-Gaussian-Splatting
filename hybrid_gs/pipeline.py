from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

from hybrid_gs.camera import orbit_cameras
from hybrid_gs.gaussians import AnchoredGaussianModel, GaussianState, procedural_colors, prompt_palette
from hybrid_gs.losses import (
    appearance_guidance_loss,
    opacity_regularization,
    reconstruction_loss,
    scale_regularization,
    tether_loss,
)
from hybrid_gs.mesh import Mesh, load_obj_mesh, primitive_mesh_from_prompt, sample_surface
from hybrid_gs.renderer import render_gaussians


@dataclass
class HybridConfig:
    prompt: str
    mesh_path: str | None
    reference_image_path: str | None
    reference_mask_path: str | None
    out_dir: Path
    num_splats: int
    steps: int
    num_views: int
    image_size: int
    lr: float
    seed: int
    device: torch.device
    lambda_tether: float = 3.0
    lambda_appearance: float = 0.15
    lambda_scale: float = 0.02
    lambda_opacity: float = 0.01
    lambda_mask: float = 0.20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_image(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_gaussian_state(path: Path, state: GaussianState) -> None:
    np.savez(
        path,
        means=state.means.detach().cpu().numpy(),
        scales=state.scales.detach().cpu().numpy(),
        colors=state.colors.detach().cpu().numpy(),
        opacity=state.opacity.detach().cpu().numpy(),
    )


def load_mesh(cfg: HybridConfig) -> Mesh:
    if cfg.mesh_path:
        return load_obj_mesh(cfg.mesh_path, cfg.device)
    return primitive_mesh_from_prompt(cfg.prompt, cfg.device)


def build_proxy_targets(mesh: Mesh, cameras, prompt: str, image_size: int) -> list[torch.Tensor]:
    samples, normals = sample_surface(mesh, 4 * 512)
    palette = prompt_palette(prompt, mesh.vertices.device)
    colors = procedural_colors(samples, normals, palette)
    teacher = GaussianState(
        means=samples,
        scales=torch.full_like(samples, 0.035),
        colors=colors,
        opacity=torch.full((samples.shape[0], 1), 0.35, device=samples.device),
    )
    return [render_gaussians(teacher, camera) for camera in cameras]


def _load_resized_rgb(path: str | Path, image_size: int) -> torch.Tensor:
    # Real image supervision is resized into the same square canvas used by the renderer.
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _load_resized_mask(path: str | Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((image_size, image_size), Resampling.BILINEAR)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def maybe_load_reference_supervision(cfg: HybridConfig) -> tuple[torch.Tensor, torch.Tensor] | None:
    # Optional real-image supervision: use a single image plus a foreground mask
    # to anchor appearance and silhouette for the front-most camera view.
    if not cfg.reference_image_path:
        return None

    rgb = _load_resized_rgb(cfg.reference_image_path, cfg.image_size).to(cfg.device)
    if cfg.reference_mask_path:
        mask = _load_resized_mask(cfg.reference_mask_path, cfg.image_size).to(cfg.device)
    else:
        mask = torch.ones((cfg.image_size, cfg.image_size), device=cfg.device, dtype=torch.float32)
    return rgb, mask


def optimize(cfg: HybridConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(cfg)
    anchors, normals = sample_surface(mesh, cfg.num_splats)
    model = AnchoredGaussianModel(anchors=anchors, normals=normals, prompt=cfg.prompt).to(cfg.device)
    cameras = orbit_cameras(
        num_views=cfg.num_views,
        radius=2.8,
        elevation_degrees=20.0,
        image_size=cfg.image_size,
        fov_degrees=45.0,
        device=cfg.device,
    )
    targets = build_proxy_targets(mesh, cameras, cfg.prompt, cfg.image_size)
    reference_supervision = maybe_load_reference_supervision(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Running baseline with {cfg.num_splats} anchored splats on {cfg.device}.")
    print("Pipeline: mesh prior -> surface samples -> Gaussian init -> refinement -> multi-view render")

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()
        state = model.state()

        reconstruction = torch.zeros((), device=cfg.device)
        mask_loss = torch.zeros((), device=cfg.device)
        for camera, target in zip(cameras, targets):
            rendered = render_gaussians(state, camera)
            reconstruction = reconstruction + reconstruction_loss(rendered, target)
        reconstruction = reconstruction / len(cameras)

        if reference_supervision is not None:
            # Use the first orbit camera as the "observed" view and add both RGB and
            # silhouette supervision from the actual source image.
            reference_rgb, reference_mask = reference_supervision
            reference_render, reference_alpha = render_gaussians(state, cameras[0], return_alpha=True)
            reconstruction = reconstruction + reconstruction_loss(reference_render, reference_rgb)
            mask_loss = torch.mean(torch.abs(reference_alpha - reference_mask))

        tether = tether_loss(state.means, model.anchor_positions, model.anchor_normals)
        appearance = appearance_guidance_loss(state.colors, model.palette)
        scale_penalty = scale_regularization(state.scales)
        opacity_penalty = opacity_regularization(state.opacity)

        total = (
            reconstruction
            + cfg.lambda_tether * tether
            + cfg.lambda_appearance * appearance
            + cfg.lambda_scale * scale_penalty
            + cfg.lambda_opacity * opacity_penalty
            + cfg.lambda_mask * mask_loss
        )
        total.backward()
        optimizer.step()

        if step == 1 or step % max(cfg.steps // 10, 1) == 0 or step == cfg.steps:
            print(
                f"[{step:04d}/{cfg.steps}] "
                f"total={total.item():.4f} "
                f"recon={reconstruction.item():.4f} "
                f"tether={tether.item():.4f} "
                f"appearance={appearance.item():.4f} "
                f"mask={mask_loss.item():.4f}"
            )

    final_state = model.state()
    save_gaussian_state(cfg.out_dir / "gaussian_state.npz", final_state)
    for index, (camera, target) in enumerate(zip(cameras, targets)):
        rendered = render_gaussians(final_state, camera)
        save_image(cfg.out_dir / f"view_{index:02d}_render.png", rendered)
        save_image(cfg.out_dir / f"view_{index:02d}_target.png", target)


def parse_args() -> HybridConfig:
    parser = argparse.ArgumentParser(description="Hybrid mesh + Gaussian splatting baseline.")
    parser.add_argument("--prompt", default="stone statue", help="Semantic prompt used for mesh choice and appearance prior.")
    parser.add_argument("--mesh", dest="mesh_path", default=None, help="Optional OBJ mesh path exported from Fantasia3D or another generator.")
    parser.add_argument("--reference-image", dest="reference_image_path", default=None, help="Optional real RGB image used to supervise the front view.")
    parser.add_argument("--reference-mask", dest="reference_mask_path", default=None, help="Optional mask aligned with --reference-image for silhouette supervision.")
    parser.add_argument("--out-dir", default="outputs/demo", help="Directory for rendered outputs.")
    parser.add_argument("--num-splats", type=int, default=384, help="Number of anchored Gaussian splats.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--num-views", type=int, default=6, help="Number of training/rendering views.")
    parser.add_argument("--image-size", type=int, default=96, help="Square render resolution.")
    parser.add_argument("--lr", type=float, default=0.05, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--lambda-mask", type=float, default=0.20, help="Weight for optional mask supervision from a real image.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return HybridConfig(
        prompt=args.prompt,
        mesh_path=args.mesh_path,
        reference_image_path=args.reference_image_path,
        reference_mask_path=args.reference_mask_path,
        out_dir=Path(args.out_dir),
        num_splats=args.num_splats,
        steps=args.steps,
        num_views=args.num_views,
        image_size=args.image_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        lambda_mask=args.lambda_mask,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    optimize(cfg)
