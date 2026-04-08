from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class SamArtifacts:
    # A small record of the three files this stage produces for downstream use.
    mask_path: Path
    masked_rgba_path: Path
    cropped_rgba_path: Path
    score: float
    area: int


def _resolve_device(device: str) -> str:
    # "auto" keeps the CLI simple: use CUDA when available, otherwise CPU.
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    # PIL keeps the implementation dependency-light and gives us RGB directly.
    return np.array(Image.open(image_path).convert("RGB"))


def generate_masks(
    image_path: str | Path,
    checkpoint_path: str | Path,
    model_type: str = "vit_h",
    device: str = "auto",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 0,
) -> list[dict[str, Any]]:
    # SAM is optional for the repo as a whole, so import it only when this stage runs.
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment_anything is not installed. Install it before running the SAM stage."
        ) from exc

    image = load_rgb_image(image_path)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=_resolve_device(device))

    # AutomaticMaskGenerator returns many candidate object masks for one image.
    # The thresholds here are exposed in the CLI so you can tune noisy inputs.
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return mask_generator.generate(image)


def select_mask(masks: list[dict[str, Any]], strategy: str = "largest") -> dict[str, Any]:
    # TRELLIS expects one foreground image. This stage reduces SAM's many proposals
    # to one mask using either a size-first or quality-first heuristic.
    if not masks:
        raise ValueError("SAM did not produce any masks.")

    if strategy == "best":
        return max(
            masks,
            key=lambda item: (
                float(item.get("predicted_iou", 0.0)),
                float(item.get("stability_score", 0.0)),
                int(item.get("area", 0)),
            ),
        )

    if strategy != "largest":
        raise ValueError(f"Unknown mask selection strategy: {strategy}")

    return max(masks, key=lambda item: int(item.get("area", 0)))


def save_mask_artifacts(
    image_path: str | Path,
    mask_data: dict[str, Any],
    out_dir: str | Path,
    prefix: str = "sam",
) -> SamArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb_image(image_path)
    mask = np.asarray(mask_data["segmentation"], dtype=bool)
    alpha = mask.astype(np.uint8) * 255

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Selected SAM mask is empty.")

    # We preserve the original RGB pixels but replace the alpha channel with the
    # selected mask, which gives TRELLIS an object image on transparent background.
    rgba = np.dstack((rgb, alpha))
    rgba_image = Image.fromarray(rgba, mode="RGBA")

    # Cropping to the mask bounding box removes empty border area and usually gives
    # better object-centric input for image-to-3D generation.
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1

    mask_path = out_dir / f"{prefix}_mask.png"
    masked_rgba_path = out_dir / f"{prefix}_masked.png"
    cropped_rgba_path = out_dir / f"{prefix}_cropped.png"

    Image.fromarray(alpha, mode="L").save(mask_path)
    rgba_image.save(masked_rgba_path)
    rgba_image.crop((left, top, right, bottom)).save(cropped_rgba_path)

    return SamArtifacts(
        mask_path=mask_path,
        masked_rgba_path=masked_rgba_path,
        cropped_rgba_path=cropped_rgba_path,
        score=float(mask_data.get("predicted_iou", 0.0)),
        area=int(mask_data.get("area", int(mask.sum()))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM foreground extraction on an image.")
    parser.add_argument("--input-image", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="SAM checkpoint path.")
    parser.add_argument("--out-dir", default="outputs/sam", help="Directory for SAM artifacts.")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--selection", default="largest", choices=["largest", "best"], help="How to select the final mask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: produce candidate masks from the input image.
    masks = generate_masks(
        image_path=args.input_image,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )

    # Step 2: collapse the SAM output to one foreground mask for downstream stages.
    chosen_mask = select_mask(masks, strategy=args.selection)

    # Step 3: write the files other stages actually consume.
    artifacts = save_mask_artifacts(args.input_image, chosen_mask, args.out_dir)

    print(f"Mask saved to {artifacts.mask_path}")
    print(f"Masked image saved to {artifacts.masked_rgba_path}")
    print(f"Cropped image saved to {artifacts.cropped_rgba_path}")


if __name__ == "__main__":
    main()
