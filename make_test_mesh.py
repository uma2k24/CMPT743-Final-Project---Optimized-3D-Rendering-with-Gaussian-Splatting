from __future__ import annotations

# argparse is used to define the command-line interface for this script.
import argparse
# Path is used to create the output directory and normalize the output path.
from pathlib import Path

# trimesh provides simple primitive meshes and OBJ export support.
import trimesh


def parse_args() -> argparse.Namespace:
    # Create the top-level CLI parser for the test mesh generator.
    parser = argparse.ArgumentParser(description="Generate a simple test mesh as an OBJ file.")
    # Let the user choose between the two supported primitive mesh types.
    parser.add_argument(
        "--shape",
        choices=["cube", "sphere"],
        required=True,
        help="Primitive shape to export.",
    )
    # Output OBJ file path.
    parser.add_argument(
        "--output",
        required=True,
        help="Output OBJ path.",
    )
    # Sphere-specific radius parameter.
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius.",
    )
    # Cube-specific edge length parameter.
    parser.add_argument(
        "--size",
        type=float,
        default=2.0,
        help="Cube edge length.",
    )
    # Parse and return the CLI arguments.
    return parser.parse_args()


def build_mesh(shape: str, radius: float, size: float) -> trimesh.Trimesh:
    # Build a sphere primitive when requested.
    if shape == "sphere":
        return trimesh.primitives.Sphere(radius=radius)
    # Otherwise build a cube primitive using the same size along all three axes.
    return trimesh.primitives.Box(extents=(size, size, size))


def main() -> None:
    # Read command-line arguments.
    args = parse_args()
    # Construct the requested primitive mesh.
    mesh = build_mesh(args.shape, args.radius, args.size)
    # Normalize the output path into a Path object.
    output_path = Path(args.output)
    # Create the parent directory if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Export the mesh as an OBJ file to the requested location.
    mesh.export(output_path)
    # Print the saved path so the caller can confirm where the file went.
    print(f"Saved {args.shape} mesh to {output_path}")


if __name__ == "__main__":
    # Run the script when executed directly.
    main()
