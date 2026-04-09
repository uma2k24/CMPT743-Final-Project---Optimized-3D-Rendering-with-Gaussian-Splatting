import argparse
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
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    camera_distance: float = 3.0,
    rotation_x_deg: float = 0.0,
    rotation_y_deg: float = 0.0,
    rotation_z_deg: float = 0.0,
    output_path: str = "/outputs/splat_render.png"
):
    """
    Render projected 3D points as simple 2D splats.
    """
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    K = get_camera_intrinsics(image_width, image_height)
    R, t = get_camera_extrinsics(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        radius=camera_distance,
        rotation_x_deg=rotation_x_deg,
        rotation_y_deg=rotation_y_deg,
        rotation_z_deg=rotation_z_deg,
    )

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


def parse_args():
    parser = argparse.ArgumentParser(description="Render sampled mesh points as 2D splats.")
    parser.add_argument(
        "--mesh-path",
        default="outputs/horse_test_gs/mesh_obj_folder_results/triposr_mesh.obj",
        help="Path to the mesh OBJ file.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of surface points to sample from the mesh.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Radius of each rendered splat in pixels.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Horizontal orbit angle in degrees. 0 keeps the original front view.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Vertical orbit angle in degrees.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=3.0,
        help="Distance from the camera to the origin.",
    )
    parser.add_argument(
        "--rot-x",
        type=float,
        default=0.0,
        help="Extra camera-local rotation around the X axis in degrees.",
    )
    parser.add_argument(
        "--rot-y",
        type=float,
        default=0.0,
        help="Extra camera-local rotation around the Y axis in degrees.",
    )
    parser.add_argument(
        "--rot-z",
        type=float,
        default=0.0,
        help="Extra camera-local rotation around the Z axis in degrees.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/splat_render.png",
        help="Where to save the rendered image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = load_mesh(args.mesh_path)

    points, face_indices = sample_surface_points(mesh, num_points=args.num_points)
    colors = sample_point_colors(mesh, face_indices)

    render_splats(
        points_3d=points,
        colors=colors,
        image_width=args.width,
        image_height=args.height,
        point_radius=args.point_radius,
        azimuth_deg=args.azimuth,
        elevation_deg=args.elevation,
        camera_distance=args.distance,
        rotation_x_deg=args.rot_x,
        rotation_y_deg=args.rot_y,
        rotation_z_deg=args.rot_z,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
