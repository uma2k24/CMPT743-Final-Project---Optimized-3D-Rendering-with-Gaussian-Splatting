from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hybrid_gs.gaussians import procedural_colors, prompt_palette
from hybrid_gs.mesh import load_obj_mesh, primitive_mesh_from_prompt, sample_surface


def _load_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required for the interactive viewer. "
            "Install it with: & .\\.venv\\Scripts\\python.exe -m pip install plotly"
        ) from exc
    return go, pio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive 3D Gaussian splat viewer.")
    parser.add_argument(
        "--state",
        default=None,
        help="Path to a saved gaussian_state.npz file from hybrid_gs.pipeline.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Optional OBJ mesh path. Used to sample a fallback Gaussian cloud if --state is not provided.",
    )
    parser.add_argument(
        "--prompt",
        default="stone statue",
        help="Prompt used for procedural colors when sampling a fallback cloud.",
    )
    parser.add_argument(
        "--num-splats",
        type=int,
        default=384,
        help="Number of sampled fallback splats when loading from a mesh or primitive prompt.",
    )
    parser.add_argument(
        "--max-splats",
        type=int,
        default=3000,
        help="Optional cap for displayed splats. Larger values make the browser heavier.",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=38.0,
        help="Multiplier applied to Gaussian scale to produce marker sizes.",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=3.0,
        help="Minimum displayed marker size in pixels.",
    )
    parser.add_argument(
        "--output-html",
        default="outputs/interactive_splat_viewer.html",
        help="Where to save the interactive HTML viewer.",
    )
    parser.add_argument(
        "--title",
        default="Interactive Gaussian Splat Viewer",
        help="Title shown in the HTML viewer.",
    )
    return parser.parse_args()


def load_state_from_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"means", "scales", "colors", "opacity"}
    missing = required.difference(data.files)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"State file {path} is missing required arrays: {missing_text}")

    return {
        "means": np.asarray(data["means"], dtype=np.float32),
        "scales": np.asarray(data["scales"], dtype=np.float32),
        "colors": np.asarray(data["colors"], dtype=np.float32),
        "opacity": np.asarray(data["opacity"], dtype=np.float32),
    }


def build_fallback_state(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    if args.mesh:
        mesh = load_obj_mesh(args.mesh, device)
    else:
        mesh = primitive_mesh_from_prompt(args.prompt, device)

    anchors, normals = sample_surface(mesh, args.num_splats)
    palette = prompt_palette(args.prompt, device)
    colors = procedural_colors(anchors, normals, palette)
    scales = torch.full_like(anchors, 0.05)
    opacity = torch.full((anchors.shape[0], 1), 0.75, dtype=torch.float32, device=device)

    return {
        "means": anchors.cpu().numpy(),
        "scales": scales.cpu().numpy(),
        "colors": colors.cpu().numpy(),
        "opacity": opacity.cpu().numpy(),
    }


def maybe_subsample(state: dict[str, np.ndarray], max_splats: int) -> dict[str, np.ndarray]:
    num_splats = state["means"].shape[0]
    if max_splats <= 0 or num_splats <= max_splats:
        return state

    opacity = state["opacity"].reshape(-1)
    order = np.argsort(opacity)[::-1][:max_splats]
    return {key: value[order] for key, value in state.items()}


def state_to_figure(state: dict[str, np.ndarray], title: str, size_scale: float, min_size: float):
    go, _ = _load_plotly()

    means = state["means"]
    scales = state["scales"]
    colors = state["colors"].clip(0.0, 1.0)
    opacity = state["opacity"].reshape(-1).clip(0.05, 1.0)

    marker_sizes = np.clip(scales.mean(axis=1) * size_scale, min_size, None)
    rgba_colors = [
        f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, {alpha:.4f})"
        for rgb, alpha in zip(colors, opacity)
    ]

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=means[:, 0],
                y=means[:, 1],
                z=means[:, 2],
                mode="markers",
                marker={
                    "size": marker_sizes,
                    "color": rgba_colors,
                    "sizemode": "diameter",
                },
                customdata=np.concatenate(
                    [
                        scales.mean(axis=1, keepdims=True),
                        opacity[:, None],
                        colors,
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "x=%{x:.3f}<br>"
                    "y=%{y:.3f}<br>"
                    "z=%{z:.3f}<br>"
                    "scale=%{customdata[0]:.4f}<br>"
                    "opacity=%{customdata[1]:.3f}<br>"
                    "color=(%{customdata[2]:.2f}, %{customdata[3]:.2f}, %{customdata[4]:.2f})"
                    "<extra></extra>"
                ),
            )
        ]
    )

    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
    )
    return figure


def main() -> None:
    args = parse_args()
    if args.state:
        state = load_state_from_npz(args.state)
    else:
        state = build_fallback_state(args)

    state = maybe_subsample(state, args.max_splats)
    figure = state_to_figure(
        state=state,
        title=args.title,
        size_scale=args.size_scale,
        min_size=args.min_size,
    )

    _, pio = _load_plotly()
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(figure, file=str(output_path), auto_open=False, include_plotlyjs=True)
    print(f"Saved interactive viewer to: {output_path}")


if __name__ == "__main__":
    main()
