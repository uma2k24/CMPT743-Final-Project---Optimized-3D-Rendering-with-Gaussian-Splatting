import os
import cv2
import numpy as np

from load_mesh import load_mesh
from sample_points import sample_surface_points, sample_point_colors
from camera import get_camera_intrinsics, get_camera_extrinsics, project_points


def render_splats(
    points_3d: np.ndarray,
    colors: np.ndarray,
    image_width: int = 800,
    image_height: int = 800,
    point_radius: int = 2,
    output_path: str = "../outputs/splat_render.png"
):
    """
    Render projected 3D points as simple 2D splats.
    """
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    K = get_camera_intrinsics(image_width, image_height)
    R, t = get_camera_extrinsics()

    pixels, depths, valid_mask = project_points(points_3d, K, R, t)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError("No valid projected points were found.")

    valid_pixels = pixels[valid_indices]
    valid_depths = depths[valid_indices]
    valid_colors = colors[valid_indices]

    sort_order = np.argsort(valid_depths)[::-1]
    valid_pixels = valid_pixels[sort_order]
    valid_colors = valid_colors[sort_order]

    for (u, v), color in zip(valid_pixels, valid_colors):
        x = int(round(u))
        y = int(round(v))

        if 0 <= x < image_width and 0 <= y < image_height:
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(image, (x, y), point_radius, bgr, thickness=-1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved render to: {output_path}")


if __name__ == "__main__":
    mesh_path = "../data/meshes/test.obj"
    mesh = load_mesh(mesh_path)

    points, face_indices = sample_surface_points(mesh, num_points=10000)
    colors = sample_point_colors(mesh, face_indices)

    render_splats(
        points_3d=points,
        colors=colors,
        image_width=800,
        image_height=800,
        point_radius=2,
        output_path="../outputs/splat_render.png"
    )