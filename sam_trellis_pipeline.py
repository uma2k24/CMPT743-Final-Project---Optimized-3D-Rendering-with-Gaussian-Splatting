from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from hybrid_gs.pipeline import HybridConfig, optimize, set_seed
from sam_stage import generate_masks, save_mask_artifacts, select_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a SAM -> TRELLIS -> Gaussian Splatting pipeline."
    )
    parser.add_argument("--input-image", required=True, help="Input image path.")
    parser.add_argument("--prompt", required=True, help="Prompt used for TRELLIS and Gaussian appearance priors.")
    parser.add_argument("--out-dir", default="outputs/sam_trellis", help="Output directory.")

    parser.add_argument("--sam-checkpoint", required=True, help="Path to the SAM checkpoint.")
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type.")
    parser.add_argument("--sam-device", default="auto", choices=["auto", "cpu", "cuda"], help="SAM device.")
    parser.add_argument("--sam-selection", default="largest", choices=["largest", "best"], help="How to pick the final SAM mask.")
    parser.add_argument("--sam-points-per-side", type=int, default=32, help="SAM automatic mask sampling density.")
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.88, help="SAM predicted IoU threshold.")
    parser.add_argument("--sam-stability-thresh", type=float, default=0.95, help="SAM stability score threshold.")
    parser.add_argument("--sam-min-mask-region-area", type=int, default=0, help="SAM minimum mask region area.")
    parser.add_argument("--skip-sam", action="store_true", help="Skip SAM and send the original image to TRELLIS.")

    parser.add_argument("--trellis-python", default=sys.executable, help="Python executable from the TRELLIS environment.")
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-image-large", help="TRELLIS model name or local model path.")
    parser.add_argument("--trellis-device", default="cuda", choices=["cpu", "cuda"], help="TRELLIS device.")
    parser.add_argument("--trellis-seed", type=int, default=1, help="TRELLIS random seed.")
    parser.add_argument("--trellis-sparse-steps", type=int, default=12, help="TRELLIS sparse structure sampler steps.")
    parser.add_argument("--trellis-slat-steps", type=int, default=12, help="TRELLIS SLAT sampler steps.")
    parser.add_argument("--trellis-sparse-cfg-strength", type=float, default=7.5, help="TRELLIS sparse structure CFG strength.")
    parser.add_argument("--trellis-slat-cfg-strength", type=float, default=3.0, help="TRELLIS SLAT CFG strength.")
    parser.add_argument("--trellis-attention-backend", default=None, choices=["flash-attn", "xformers"], help="Optional TRELLIS attention backend override.")
    parser.add_argument("--mesh-name", default="trellis_mesh.obj", help="Output mesh filename.")

    parser.add_argument("--skip-gs", action="store_true", help="Stop after exporting the TRELLIS mesh.")
    parser.add_argument("--num-splats", type=int, default=384, help="Gaussian splat count.")
    parser.add_argument("--steps", type=int, default=200, help="Gaussian optimization steps.")
    parser.add_argument("--num-views", type=int, default=6, help="Rendered training views.")
    parser.add_argument("--image-size", type=int, default=96, help="Render resolution.")
    parser.add_argument("--lr", type=float, default=0.05, help="Gaussian optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Gaussian optimization seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for Gaussian optimization.")

    return parser.parse_args()


def run_trellis(
    *,
    image_path: Path,
    output_mesh_path: Path,
    python_executable: str,
    model_name: str,
    seed: int,
    sparse_steps: int,
    slat_steps: int,
    sparse_cfg_strength: float,
    slat_cfg_strength: float,
    device: str,
    attention_backend: str | None,
) -> Path:
    # TRELLIS lives in a separate environment, so this repo treats it as an external
    # tool boundary and calls one helper script using the TRELLIS Python executable.
    helper_script = Path(__file__).resolve().parent / "tools" / "trellis_image_to_mesh.py"
    command = [
        python_executable,
        str(helper_script),
        "--input-image",
        str(image_path),
        "--output-mesh",
        str(output_mesh_path),
        "--model",
        model_name,
        "--seed",
        str(seed),
        "--sparse-steps",
        str(sparse_steps),
        "--slat-steps",
        str(slat_steps),
        "--sparse-cfg-strength",
        str(sparse_cfg_strength),
        "--slat-cfg-strength",
        str(slat_cfg_strength),
        "--device",
        device,
    ]

    if attention_backend:
        command.extend(["--attention-backend", attention_backend])

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "TRELLIS subprocess failed."
        raise RuntimeError(message)
    if not output_mesh_path.exists():
        raise RuntimeError(f"TRELLIS did not create the expected mesh at {output_mesh_path}.")
    return output_mesh_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sam_dir = out_dir / "sam"
    mesh_dir = out_dir / "trellis"
    gs_dir = out_dir / "gaussians"
    mesh_path = mesh_dir / args.mesh_name

    # This path is what we will hand to TRELLIS. It starts as the original image
    # and gets replaced with SAM's cropped RGBA foreground if SAM is enabled.
    trellis_input_path = Path(args.input_image)

    # The manifest is a lightweight handoff log so you can inspect what image,
    # mask, mesh, and GS output were used without re-reading the console.
    manifest: dict[str, object] = {
        "input_image": str(Path(args.input_image).resolve()),
        "prompt": args.prompt,
    }

    if not args.skip_sam:
        # Stage 1: isolate one foreground object from the source image.
        masks = generate_masks(
            image_path=args.input_image,
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            points_per_side=args.sam_points_per_side,
            pred_iou_thresh=args.sam_pred_iou_thresh,
            stability_score_thresh=args.sam_stability_thresh,
            min_mask_region_area=args.sam_min_mask_region_area,
        )
        chosen_mask = select_mask(masks, strategy=args.sam_selection)
        artifacts = save_mask_artifacts(args.input_image, chosen_mask, sam_dir)
        trellis_input_path = artifacts.cropped_rgba_path
        manifest["sam"] = {
            "mask_path": str(artifacts.mask_path.resolve()),
            "masked_rgba_path": str(artifacts.masked_rgba_path.resolve()),
            "cropped_rgba_path": str(artifacts.cropped_rgba_path.resolve()),
            "predicted_iou": artifacts.score,
            "area": artifacts.area,
            "selection": args.sam_selection,
        }

    # Stage 2: run image-to-3D in TRELLIS and export the first returned mesh as OBJ.
    exported_mesh = run_trellis(
        image_path=trellis_input_path,
        output_mesh_path=mesh_path,
        python_executable=args.trellis_python,
        model_name=args.trellis_model,
        seed=args.trellis_seed,
        sparse_steps=args.trellis_sparse_steps,
        slat_steps=args.trellis_slat_steps,
        sparse_cfg_strength=args.trellis_sparse_cfg_strength,
        slat_cfg_strength=args.trellis_slat_cfg_strength,
        device=args.trellis_device,
        attention_backend=args.trellis_attention_backend,
    )
    manifest["trellis"] = {
        "input_image": str(trellis_input_path.resolve()),
        "mesh_path": str(exported_mesh.resolve()),
        "model": args.trellis_model,
    }

    if not args.skip_gs:
        # Stage 3: treat the TRELLIS OBJ exactly like any other coarse mesh prior
        # for the existing Gaussian splatting baseline in hybrid_gs.pipeline.
        device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = HybridConfig(
            prompt=args.prompt,
            mesh_path=str(exported_mesh),
            out_dir=gs_dir,
            num_splats=args.num_splats,
            steps=args.steps,
            num_views=args.num_views,
            image_size=args.image_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        set_seed(cfg.seed)
        optimize(cfg)
        manifest["gaussian_splatting"] = {
            "out_dir": str(gs_dir.resolve()),
            "num_splats": args.num_splats,
            "steps": args.steps,
            "device": str(device),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the whole chain so later experiments can trace which intermediate
    # artifacts produced a given Gaussian output directory.
    with (out_dir / "pipeline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
