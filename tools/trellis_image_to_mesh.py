from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRELLIS image-to-mesh export.")
    parser.add_argument("--input-image", required=True, help="RGBA or RGB input image.")
    parser.add_argument("--output-mesh", required=True, help="OBJ output path.")
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="TRELLIS model name or local path.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--sparse-steps", type=int, default=12, help="Sparse structure sampler steps.")
    parser.add_argument("--slat-steps", type=int, default=12, help="SLAT sampler steps.")
    parser.add_argument("--sparse-cfg-strength", type=float, default=7.5, help="Sparse structure CFG strength.")
    parser.add_argument("--slat-cfg-strength", type=float, default=3.0, help="SLAT CFG strength.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument("--attention-backend", default=None, choices=["flash-attn", "xformers"], help="Optional attention backend override.")
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray:
    # TRELLIS may return tensors or numpy-like arrays depending on internals.
    if value is None:
        raise ValueError("Cannot convert a missing mesh attribute to numpy.")
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _save_obj(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    # Export a minimal triangle OBJ because the existing GS code already knows
    # how to ingest OBJ meshes through hybrid_gs.mesh.load_obj_mesh.
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    if faces.min() < 0:
        raise ValueError("Mesh faces contain negative indices.")

    with output_path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def export_mesh(mesh: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer TRELLIS/native export when the mesh object supports it.
    if hasattr(mesh, "export"):
        mesh.export(str(output_path))
        return

    # Otherwise fall back to common mesh container conventions used by Python 3D libs.
    if isinstance(mesh, dict):
        vertices = mesh.get("vertices")
        if vertices is None:
            vertices = mesh.get("verts")
        faces = mesh.get("faces")
        if faces is None:
            faces = mesh.get("triangles")
    else:
        vertices = getattr(mesh, "vertices", None)
        if vertices is None:
            vertices = getattr(mesh, "verts", None)
        faces = getattr(mesh, "faces", None)
        if faces is None:
            faces = getattr(mesh, "triangles", None)

    if vertices is None or faces is None:
        raise RuntimeError("TRELLIS mesh export format is unsupported by this helper.")

    _save_obj(_to_numpy(vertices), _to_numpy(faces), output_path)


def main() -> None:
    args = parse_args()

    # These environment variables mirror common TRELLIS setup guidance and let
    # the caller choose an attention backend without editing the script.
    os.environ.setdefault("SPCONV_ALGO", "native")
    if args.attention_backend:
        os.environ["ATTN_BACKEND"] = args.attention_backend

    from trellis.pipelines import TrellisImageTo3DPipeline

    # The pretrained TRELLIS pipeline consumes a single object-centric image.
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for TRELLIS, but no CUDA device is available.")
        pipeline.cuda()

    image = Image.open(args.input_image)

    # TRELLIS exposes separate sampling controls for sparse structure and SLAT.
    # This script forwards them directly so experiments do not require code edits.
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.sparse_steps,
            "cfg_strength": args.sparse_cfg_strength,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg_strength,
        },
    )

    meshes = outputs.get("mesh", [])
    if not meshes:
        raise RuntimeError("TRELLIS did not return any mesh outputs.")

    # We currently take the first mesh candidate and normalize it into an OBJ file
    # that the Gaussian baseline can load as its structural prior.
    export_mesh(meshes[0], Path(args.output_mesh))


if __name__ == "__main__":
    main()
