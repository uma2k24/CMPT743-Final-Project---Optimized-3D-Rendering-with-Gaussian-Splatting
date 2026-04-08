from __future__ import annotations

# argparse is used to define the command-line interface for this script.
import argparse
# sys lets us detect whether the script was launched with CLI arguments.
import sys
# Path is used to create the output directory and normalize the output path.
from pathlib import Path

# trimesh provides simple primitive meshes and OBJ export support.
import trimesh


# Example usage:
# python make_test_mesh.py --shape sphere --output outputs/test_sphere.obj
# python make_test_mesh.py --shape cube --output outputs/test_cube.obj


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


def prompt_for_args() -> argparse.Namespace:
    # Ask the user which primitive mesh to generate when no CLI args were given.
    while True:
        shape = input("Choose a shape (sphere/cube): ").strip().lower()
        if shape in {"sphere", "cube"}:
            break
        print("Please enter 'sphere' or 'cube'.")

    # Match the example output filenames by default.
    default_output = f"outputs/test_{shape}.obj"
    output = input(f"Output path [{default_output}]: ").strip() or default_output

    # Provide shape-specific defaults while still letting the user override them.
    if shape == "sphere":
        default_radius = "1.0"
        radius_text = input(f"Sphere radius [{default_radius}]: ").strip() or default_radius
        return argparse.Namespace(shape=shape, output=output, radius=float(radius_text), size=2.0)

    default_size = "2.0"
    size_text = input(f"Cube edge length [{default_size}]: ").strip() or default_size
    return argparse.Namespace(shape=shape, output=output, radius=1.0, size=float(size_text))


def build_mesh(shape: str, radius: float, size: float) -> trimesh.Trimesh:
    # Build a sphere primitive when requested.
    if shape == "sphere":
        return trimesh.primitives.Sphere(radius=radius)
    # Otherwise build a cube primitive using the same size along all three axes.
    return trimesh.primitives.Box(extents=(size, size, size))


def main() -> None:
    # Use CLI args when provided, otherwise fall back to an interactive prompt.
    args = parse_args() if len(sys.argv) > 1 else prompt_for_args()
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
