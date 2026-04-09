import numpy as np


def get_camera_intrinsics(
    image_width: int,
    image_height: int,
    fx: float = 800.0,
    fy: float = 800.0
) -> np.ndarray:
    """
    Create a basic camera intrinsic matrix.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def get_camera_extrinsics(
    camera_position: np.ndarray = np.array([0.0, 0.0, 3.0], dtype=np.float32)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Very simple camera extrinsics:
    - Identity rotation
    - Translation based on camera position

    Assumes the mesh is near the origin and the camera looks toward -Z.
    """
    R = np.eye(3, dtype=np.float32)
    t = -camera_position.reshape(3, 1)
    return R, t


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
):
    """
    Project 3D world points into 2D image coordinates.

    Returns:
        pixels: (N, 2) projected pixel coordinates
        depths: (N,) depth values in camera space
        valid_mask: (N,) mask for points in front of the camera
    """
    points_cam = (R @ points_3d.T) + t
    points_cam = points_cam.T

    depths = points_cam[:, 2]
    valid_mask = depths < 0  # camera looking toward negative Z

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = -depths  # use positive depth for projection

    z = np.clip(z, 1e-6, None)

    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]

    pixels = np.stack([u, v], axis=1)
    return pixels, z, valid_mask