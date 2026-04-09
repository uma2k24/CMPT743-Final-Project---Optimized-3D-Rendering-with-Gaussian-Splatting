import numpy as np
import trimesh


def sample_surface_points(mesh: trimesh.Trimesh, num_points: int = 5000):
    """
    Uniformly sample points from the surface of a mesh.

    Returns:
        points: (N, 3) array of sampled 3D points
        face_indices: (N,) array of face indices corresponding to sampled points
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    return points, face_indices


def sample_point_colors(mesh: trimesh.Trimesh, face_indices: np.ndarray):
    """
    Assign colors to sampled points.

    If the mesh has per-face or per-vertex color information, use it.
    Otherwise assign a default gray color.
    """
    default_color = np.array([180, 180, 180], dtype=np.uint8)

    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        if len(mesh.visual.face_colors) > 0:
            colors = mesh.visual.face_colors[face_indices][:, :3]
            return colors.astype(np.uint8)

    colors = np.tile(default_color, (len(face_indices), 1))
    return colors