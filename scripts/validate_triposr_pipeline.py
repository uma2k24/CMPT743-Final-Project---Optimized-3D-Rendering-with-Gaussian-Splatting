from __future__ import annotations

# argparse provides a small CLI for preflight and output verification.
import argparse
# json is used to inspect the saved pipeline manifest.
import json
# Path simplifies filesystem checks and path joining.
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # This script serves two lightweight testing roles:
    # 1. preflight validation before running TripoSR
    # 2. output validation after the pipeline finishes
    parser = argparse.ArgumentParser(description="Validate TripoSR pipeline setup and outputs.")
    parser.add_argument("--triposr-workdir", default=None, help="Path to the cloned TripoSR repository.")
    parser.add_argument("--triposr-python", default=None, help="Python executable from the TripoSR environment.")
    parser.add_argument("--input-image", default=None, help="Input image used for the pipeline.")
    parser.add_argument("--sam-checkpoint", default=None, help="SAM checkpoint path.")
    parser.add_argument(
        "--verify-output",
        default=None,
        help="Pipeline output directory to verify after a run.",
    )
    return parser.parse_args()


def require_exists(path_text: str | None, label: str) -> Path:
    # This helper makes the error messages shorter and more uniform.
    if not path_text:
        raise RuntimeError(f"Missing required argument for {label}.")
    path = Path(path_text)
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def validate_preflight(args: argparse.Namespace) -> None:
    # Check the external TripoSR repo layout before attempting the full pipeline.
    triposr_workdir = require_exists(args.triposr_workdir, "TripoSR workdir")
    triposr_python = require_exists(args.triposr_python, "TripoSR python")
    input_image = require_exists(args.input_image, "Input image")
    sam_checkpoint = require_exists(args.sam_checkpoint, "SAM checkpoint")

    run_py = triposr_workdir / "run.py"
    requirements = triposr_workdir / "requirements.txt"

    if not run_py.exists():
        raise RuntimeError(f"TripoSR entry point was not found at {run_py}")
    if not requirements.exists():
        raise RuntimeError(f"TripoSR requirements file was not found at {requirements}")

    print("Preflight check passed.")
    print(f"TripoSR repo: {triposr_workdir.resolve()}")
    print(f"TripoSR python: {triposr_python.resolve()}")
    print(f"Input image: {input_image.resolve()}")
    print(f"SAM checkpoint: {sam_checkpoint.resolve()}")


def verify_obj(obj_path: Path) -> None:
    # A minimal OBJ validity check is enough for this handoff: the Gaussian stage
    # only needs vertices and faces to be present.
    has_vertex = False
    has_face = False
    with obj_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                has_vertex = True
            elif line.startswith("f "):
                has_face = True
            if has_vertex and has_face:
                break

    if not has_vertex or not has_face:
        raise RuntimeError(f"OBJ file is missing vertices or faces: {obj_path}")


def validate_output(output_dir_text: str) -> None:
    # Check the files that matter to the downstream Gaussian branch.
    output_dir = require_exists(output_dir_text, "Pipeline output directory")
    manifest_path = output_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Pipeline manifest was not found at {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("backend") != "triposr":
        raise RuntimeError("Manifest backend is not 'triposr'.")

    triposr_info = manifest.get("triposr")
    if not isinstance(triposr_info, dict):
        raise RuntimeError("Manifest does not contain a valid 'triposr' section.")

    mesh_path_text = triposr_info.get("mesh_path")
    mesh_path = require_exists(mesh_path_text, "TripoSR OBJ mesh")
    verify_obj(mesh_path)

    print("Output verification passed.")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"OBJ mesh: {mesh_path.resolve()}")


def main() -> None:
    args = parse_args()

    # If the caller requests output verification, do that mode only.
    if args.verify_output:
        validate_output(args.verify_output)
        return

    # Otherwise run the preflight checks.
    validate_preflight(args)


if __name__ == "__main__":
    main()
