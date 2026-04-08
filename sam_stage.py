import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def run_sam(image_path):
    """
    Runs SAM segmentation on an input image.

    Args:
        image_path (str): path to input image

    Returns:
        masks (list): list of segmentation masks
    """

    # -----------------------------
    # 1. Load image
    # -----------------------------
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 2. Load SAM model
    # -----------------------------
    model_type = "vit_h"
    checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)

    # -----------------------------
    # 3. Generate masks automatically
    # -----------------------------
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    print(f"Generated {len(masks)} masks")

    return masks


if __name__ == "__main__":
    masks = run_sam("data/input.jpg")