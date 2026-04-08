from __future__ import annotations

# argparse defines the CLI for the end-to-end SAM -> TripoSR -> GS pipeline.
import argparse
# json is used to save a lightweight manifest of the handoff artifacts.
import json
# subprocess runs TripoSR in its own Python environment.
import subprocess
# sys provides a fallback interpreter path when one is not supplied explicitly.
import sys
# Path keeps filesystem handling consistent across Windows and Linux.
from pathlib import Path

# torch is used here only to pick CPU vs CUDA for the Gaussian stage.
import torch

# Reuse the existing Gaussian optimization pipeline from this repo.
from hybrid_gs.pipeline import HybridConfig, optimize, set_seed
# Reuse the existing standalone SAM helpers so we do not duplicate segmentation code.
from sam_stage import generate_masks, save_mask_artifacts, select_mask


def parse_args() -> argparse.Namespace:
    # This file is the dedicated image-based hybrid pipeline:
    # input image -> optional SAM crop -> TripoSR OBJ -> Gaussian splats.
    parser = argparse.ArgumentParser(
        description="Run a SAM -> TripoSR -> Gaussian Splatting pipeline."
    )
    parser.add_argument("--input-image", required=True, help="Input image path.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt used for the Gaussian appearance prior and experiment tracking.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/sam_triposr",
        help="Directory for SAM, TripoSR, Gaussian outputs, and the manifest.",
    )

    parser.add_argument("--sam-checkpoint", required=True, help="Path to the SAM checkpoint.")
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type.")
    parser.add_argument("--sam-device", default="auto", choices=["auto", "cpu", "cuda"], help="SAM device.")
    parser.add_argument("--sam-selection", default="largest", choices=["largest", "best"], help="How to pick the final SAM mask.")
    parser.add_argument("--sam-points-per-side", type=int, default=32, help="SAM automatic mask sampling density.")
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.88, help="SAM predicted IoU threshold.")
    parser.add_argument("--sam-stability-thresh", type=float, default=0.95, help="SAM stability score threshold.")
    parser.add_argument("--sam-min-mask-region-area", type=int, default=0, help="SAM minimum mask region area.")
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip SAM and send the original image directly to TripoSR.",
    )

    # Everything below this point configures the external TripoSR subprocess.
    # The main repo does not import TripoSR as a Python library. Instead, it runs
    # TripoSR through its official CLI entry point inside a separate environment.
    parser.add_argument(
        "--triposr-python",
        default=sys.executable,
        help="Python executable from the TripoSR environment.",
    )
    parser.add_argument(
        "--triposr-workdir",
        required=True,
        help="Path to the cloned TripoSR repository.",
    )
    parser.add_argument(
        "--triposr-model",
        default="stabilityai/TripoSR",
        help="TripoSR pretrained model name or local model path.",
    )
    parser.add_argument(
        "--triposr-device",
        default="cuda:0",
        help="Device string forwarded to TripoSR, such as cuda:0 or cpu.",
    )
    parser.add_argument(
        "--triposr-output-dir",
        default=None,
        help="Optional output directory for TripoSR. Defaults to <out-dir>/triposr.",
    )
    parser.add_argument(
        "--triposr-mc-resolution",
        type=int,
        default=256,
        help="Marching cubes resolution forwarded to TripoSR.",
    )
    parser.add_argument(
        "--triposr-chunk-size",
        type=int,
        default=8192,
        help="Chunk size forwarded to TripoSR to trade off VRAM vs speed.",
    )
    parser.add_argument(
        "--triposr-foreground-ratio",
        type=float,
        default=0.85,
        help="Foreground ratio used by TripoSR when TripoSR handles background removal.",
    )
    parser.add_argument(
        "--triposr-no-remove-bg",
        action="store_true",
        help="Disable TripoSR background removal. Useful when SAM already produced a clean crop.",
    )
    parser.add_argument(
        "--triposr-render",
        action="store_true",
        help="Ask TripoSR to render preview views alongside the exported mesh.",
    )
    parser.add_argument(
        "--mesh-name",
        default="triposr_mesh.obj",
        help="Final OBJ filename copied into this repo's pipeline output directory.",
    )

    # Everything below this point configures the Gaussian splatting stage that
    # already exists in this repository.
    parser.add_argument("--skip-gs", action="store_true", help="Stop after exporting the TripoSR mesh.")
    parser.add_argument("--num-splats", type=int, default=384, help="Gaussian splat count.")
    parser.add_argument("--steps", type=int, default=200, help="Gaussian optimization steps.")
    parser.add_argument("--num-views", type=int, default=6, help="Rendered training views.")
    parser.add_argument("--image-size", type=int, default=96, help="Render resolution.")
    parser.add_argument("--lr", type=float, default=0.05, help="Gaussian optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Gaussian optimization seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for Gaussian optimization.")

    return parser.parse_args()


def run_triposr(
    *,
    image_path: Path,
    pipeline_output_mesh_path: Path,
    python_executable: str,
    workdir: str,
    model_name: str,
    device: str,
    output_dir: Path,
    mc_resolution: int,
    chunk_size: int,
    foreground_ratio: float,
    no_remove_bg: bool,
    render: bool,
) -> tuple[Path, dict[str, object]]:
    # Resolve all paths up front because the TripoSR subprocess runs with cwd set
    # to the external TripoSR repo, not this repository.
    image_path = image_path.resolve()
    pipeline_output_mesh_path = pipeline_output_mesh_path.resolve()
    output_dir = output_dir.resolve()

    # TripoSR's official entry point is run.py, which takes one or more images and
    # writes mesh.<format> into numbered subdirectories below --output-dir.
    #
    # We intentionally call the script as a subprocess rather than importing its
    # internals. That keeps this repo loosely coupled to the exact TripoSR codebase
    # layout and lets the TripoSR environment manage its own dependencies.
    command = [
        # Use the Python interpreter from the TripoSR environment, not this repo's env.
        python_executable,
        # TripoSR's official CLI script.
        "run.py",
        # Input image path. For this pipeline we pass exactly one image.
        str(image_path),
        # Forward the requested TripoSR device string.
        "--device",
        device,
        # Forward the pretrained model id or local model path.
        "--pretrained-model-name-or-path",
        model_name,
        # Smaller chunk size reduces VRAM pressure at the cost of more runtime.
        "--chunk-size",
        str(chunk_size),
        # Marching cubes resolution controls the extracted mesh detail level.
        "--mc-resolution",
        str(mc_resolution),
        # TripoSR writes into its own numbered subdirectories under this root.
        "--output-dir",
        str(output_dir),
        # Force OBJ so the downstream Gaussian stage can load the result directly.
        "--model-save-format",
        "obj",
        # This only matters when TripoSR is allowed to do its own background removal.
        "--foreground-ratio",
        str(foreground_ratio),
    ]

    # If SAM already isolated the object well, background removal inside TripoSR is
    # usually redundant and can be disabled with this flag.
    if no_remove_bg:
        command.append("--no-remove-bg")

    # Preview rendering is optional and can be expensive, so keep it opt-in.
    if render:
        command.append("--render")

    completed = subprocess.run(
        # Run the command exactly as assembled above.
        command,
        # We want to control error formatting ourselves.
        check=False,
        # Capture stderr/stdout so failures are easy to surface to the user.
        capture_output=True,
        # Decode output as text instead of raw bytes.
        text=True,
        # Run inside the cloned TripoSR repository because run.py lives there.
        cwd=workdir,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "TripoSR subprocess failed."
        raise RuntimeError(message)

    # For a single input image, TripoSR writes output/0/mesh.obj by default.
    # That "0" is TripoSR's per-image index for the first image in the batch.
    generated_mesh_path = output_dir / "0" / "mesh.obj"
    if not generated_mesh_path.exists():
        raise RuntimeError(f"TripoSR completed but the expected mesh was not found at {generated_mesh_path}.")

    # Copy the mesh into the pipeline-owned location so later code does not depend
    # on TripoSR's internal folder structure.
    #
    # This matters for team handoff: partners integrating the GS side only need to
    # care about one stable path inside this repo's output directory.
    pipeline_output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_output_mesh_path.write_text(generated_mesh_path.read_text(encoding="utf-8"), encoding="utf-8")

    return pipeline_output_mesh_path, {
        # Store the exact image TripoSR consumed. This may be the original input or
        # the SAM-cropped RGBA image depending on whether SAM was enabled.
        "input_image": str(image_path.resolve()),
        # Record TripoSR's own output root so intermediate artifacts can be inspected.
        "output_dir": str(output_dir.resolve()),
        # Record the original OBJ path created by TripoSR itself.
        "generated_mesh_path": str(generated_mesh_path.resolve()),
        # Record the normalized mesh path that the rest of this repo actually uses.
        "mesh_path": str(pipeline_output_mesh_path.resolve()),
        # Track which TripoSR model was used for this run.
        "model": model_name,
        # Track which device TripoSR used.
        "device": device,
        # Track reconstruction-detail settings that affect mesh quality.
        "mc_resolution": mc_resolution,
        "chunk_size": chunk_size,
        "foreground_ratio": foreground_ratio,
        # Track whether TripoSR background removal was disabled.
        "no_remove_bg": no_remove_bg,
        # Track whether preview renders were requested.
        "render": render,
        # Track the external repo path for reproducibility.
        "workdir": str(Path(workdir).resolve()),
    }


def main() -> None:
    # Read all pipeline configuration from the CLI.
    args = parse_args()
    # The root output directory contains every artifact produced by this wrapper.
    out_dir = Path(args.out_dir).resolve()
    # SAM artifacts are kept separate so it is easy to inspect the segmentation stage.
    sam_dir = out_dir / "sam"
    # TripoSR intermediates are kept under their own directory tree.
    triposr_dir = Path(args.triposr_output_dir) if args.triposr_output_dir else (out_dir / "triposr")
    # Final Gaussian renders and targets are written here by the baseline optimizer.
    gs_dir = out_dir / "gaussians"
    # This is the repo-owned OBJ path handed from TripoSR into the GS stage.
    mesh_path = out_dir / "triposr" / args.mesh_name

    # TripoSR sees either the original image or the cleaned SAM crop.
    triposr_input_path = Path(args.input_image).resolve()

    # The manifest is the handoff contract for your team: it records what image was
    # used, where the mesh came from, and what settings produced the output.
    manifest: dict[str, object] = {
        "backend": "triposr",
        "input_image": str(Path(args.input_image).resolve()),
        "prompt": args.prompt,
    }

    if not args.skip_sam:
        # Stage 1: isolate the foreground object so TripoSR sees an object-centric image.
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
        # Once SAM finishes, the TripoSR stage should consume the cropped RGBA image
        # instead of the raw input image.
        triposr_input_path = artifacts.cropped_rgba_path
        manifest["sam"] = {
            "mask_path": str(artifacts.mask_path.resolve()),
            "masked_rgba_path": str(artifacts.masked_rgba_path.resolve()),
            "cropped_rgba_path": str(artifacts.cropped_rgba_path.resolve()),
            "predicted_iou": artifacts.score,
            "area": artifacts.area,
            "selection": args.sam_selection,
        }

    # Stage 2: run TripoSR and retrieve its OBJ mesh.
    #
    # This is the key difference from TripoSG: TripoSR can already write OBJ, so
    # we do not need a GLB-to-OBJ conversion step in this pipeline.
    exported_mesh, triposr_manifest = run_triposr(
        image_path=triposr_input_path,
        pipeline_output_mesh_path=mesh_path,
        python_executable=args.triposr_python,
        workdir=args.triposr_workdir,
        model_name=args.triposr_model,
        device=args.triposr_device,
        output_dir=triposr_dir,
        mc_resolution=args.triposr_mc_resolution,
        chunk_size=args.triposr_chunk_size,
        foreground_ratio=args.triposr_foreground_ratio,
        no_remove_bg=args.triposr_no_remove_bg,
        render=args.triposr_render,
    )
    manifest["triposr"] = triposr_manifest

    if not args.skip_gs:
        # Stage 3: feed TripoSR's OBJ directly into the existing Gaussian baseline.
        #
        # The Gaussian stage is intentionally unchanged. That means your partners can
        # merge this file and the OBJ handoff without having to redesign GS itself.
        gs_device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
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
            device=gs_device,
        )
        set_seed(cfg.seed)
        optimize(cfg)
        manifest["gaussian_splatting"] = {
            "out_dir": str(gs_dir.resolve()),
            "num_splats": args.num_splats,
            "steps": args.steps,
            "device": str(gs_device),
        }

    # Save a compact manifest so experiments can be reproduced later.
    #
    # Even when --skip-gs is used, the manifest still captures the SAM and TripoSR
    # outputs so that the mesh-only stage can be validated independently.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pipeline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
