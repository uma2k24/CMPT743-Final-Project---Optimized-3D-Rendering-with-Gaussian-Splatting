import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _rotation_matrix_xyz(
    rotation_x_deg: float = 0.0,
    rotation_y_deg: float = 0.0,
    rotation_z_deg: float = 0.0
) -> np.ndarray:
    """
    Create a camera-local Euler rotation matrix using X -> Y -> Z order.
    """
    rx = np.deg2rad(rotation_x_deg)
    ry = np.deg2rad(rotation_y_deg)
    rz = np.deg2rad(rotation_z_deg)

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x],
    ], dtype=np.float32)

    rot_y = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ], dtype=np.float32)

    rot_z = np.array([
        [cos_z, -sin_z, 0.0],
        [sin_z, cos_z, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return (rot_z @ rot_y @ rot_x).astype(np.float32)


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
    camera_position: np.ndarray | None = None,
    target: np.ndarray | None = None,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    radius: float = 3.0,
    rotation_x_deg: float = 0.0,
    rotation_y_deg: float = 0.0,
    rotation_z_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build world-to-camera extrinsics for a camera that looks at a target.

    If ``camera_position`` is omitted, the camera orbits around ``target``
    using azimuth/elevation angles and the provided radius. Additional
    X/Y/Z Euler rotations are then applied in camera-local coordinates.
    """
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    else:
        target = np.asarray(target, dtype=np.float32)

    if camera_position is None:
        azimuth_rad = np.deg2rad(azimuth_deg)
        elevation_rad = np.deg2rad(elevation_deg)
        camera_offset = np.array([
            radius * np.sin(azimuth_rad) * np.cos(elevation_rad),
            radius * np.sin(elevation_rad),
            radius * np.cos(azimuth_rad) * np.cos(elevation_rad),
        ], dtype=np.float32)
        camera_position = target + camera_offset
    else:
        camera_position = np.asarray(camera_position, dtype=np.float32)

    forward = _normalize(target - camera_position)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = _normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)

    look_at_rotation = np.stack([right, true_up, -forward], axis=0).astype(np.float32)
    local_rotation = _rotation_matrix_xyz(
        rotation_x_deg=rotation_x_deg,
        rotation_y_deg=rotation_y_deg,
        rotation_z_deg=rotation_z_deg,
    )

    R = (local_rotation @ look_at_rotation).astype(np.float32)
    t = (-R @ camera_position.reshape(3, 1)).astype(np.float32)
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
