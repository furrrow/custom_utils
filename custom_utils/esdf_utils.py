"""
https://github.com/gershom96/VLA_DataGeneration/blob/main/tests/esdf-test.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from custom_utils.io_utils import overlay_path, plot_bbox
import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from matplotlib.lines import Line2D
# The application chooses its backend before importing this module. Forcing a
# GUI backend here prevents the ROS inference node from starting headlessly.
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
custom_cmap_old = LinearSegmentedColormap.from_list(
    "esdf",
    [
        (0.00, "royalblue"),
        (0.45, "lightblue"),
        (0.50, "black"),       # zero
        (0.55, "salmon"),
        (1.00, "crimson"),
    ],
)
custom_cmap = LinearSegmentedColormap.from_list(
    "esdf_red_focused",
    [
        (0.00, "navy"),
        (0.45, "lightskyblue"),
        (0.50, "black"),        # Zero marker
        (0.55, "darkred"),
        (0.68, "crimson"),
        (0.82, "darkorange"),
        (1.00, "yellow"),       # Peak positive value
    ],
)


@dataclass(frozen=True)
class BevView:
    """Display convention for top-down maps: lateral on x, forward on y."""

    extent: tuple[float, float, float, float]
    sensor_xy: tuple[float, float]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BevView":
        return cls(
            extent=(args.y_min, args.y_max, args.x_min, args.x_max),
            sensor_xy=(args.sensor_x, args.sensor_y),
        )


@dataclass(frozen=True)
class PathOverlay:
    points: np.ndarray
    color: str
    label: str | None = None
    linewidth: float = 5.0


def semantic_bev_image(
    occupied_mask: np.ndarray,
    visible_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> np.ndarray:
    semantic = np.ones((*occupied_mask.shape, 3), dtype=np.float32)
    semantic[unknown_mask] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
    semantic[visible_free_mask] = np.asarray([0.75, 0.92, 1.0], dtype=np.float32)
    semantic[occupied_mask] = np.asarray([0.88, 0.20, 0.20], dtype=np.float32)
    return semantic


def finite_percentile_abs(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    return max(vmax, 1e-6)


def plot_scalar_map(
    ax: plt.Axes,
    values: np.ndarray,
    extent: Sequence[float] | None,
    title: str,
    sensor_xy: tuple[float, float],
    cmap: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    show_sensor: bool = True,
) -> None:
    image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="nearest")

    if show_sensor:
        ax.scatter([sensor_xy[0]], [sensor_xy[1]], marker="X", s=180, c="yellow",
                   edgecolors="black", linewidths=1.5, zorder=10)

    # ax.set_title(title, fontsize=20, pad=14, fontweight="bold")
    # ax.set_xlabel("Image x / horizontal position", fontsize=18, labelpad=10)
    # ax.set_ylabel("Image y / vertical position", fontsize=18, labelpad=10)

    ax.tick_params(axis="both", which="major", labelsize=15, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_aspect("equal", adjustable="box")

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025, shrink=0.50,)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label, fontsize=17, labelpad=10)
    cbar.ax.tick_params(labelsize=14, width=1.3, length=5)
    cbar.outline.set_linewidth(1.3)

def plot_scalar_map_flipped(
        ax: plt.Axes,
        values: np.ndarray,
        extent: Sequence[float],
        title: str,
        sensor_xy: tuple[float, float],
        cmap: str,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar_label: str | None = None,
) -> None:
    image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")

    ax.scatter([sensor_xy[1]], [sensor_xy[0]], marker="X", s=180, c="yellow",
               edgecolors="black", linewidths=1.5, zorder=10, label="Robot")

    # ax.set_title(title, fontsize=20, pad=14, fontweight="bold")
    ax.set_ylabel("Forward position, x (m)", fontsize=18, labelpad=10)
    ax.set_xlabel("Lateral position, y (m)", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.5, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025, shrink=0.50,)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label, fontsize=17, labelpad=10)
    cbar.ax.tick_params(labelsize=14, width=1.3, length=5)
    cbar.outline.set_linewidth(1.3)

def figure_to_rgb(fig: plt.Figure, *, tight_layout: bool = False) -> np.ndarray:
    """Render a figure to RGB and always release its Matplotlib resources."""
    try:
        if tight_layout:
            fig.tight_layout(pad=0.8)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(fig)


def plot_paths(ax: plt.Axes, paths: Sequence[PathOverlay]) -> list[Line2D]:
    lines = []
    for path in paths:
        line, = ax.plot(
            path.points[:, 1],
            path.points[:, 0],
            color=path.color,
            linewidth=path.linewidth,
            label=path.label,
            zorder=8,
        )
        lines.append(line)
    return lines


def plot_motion(
    ax: plt.Axes,
    movements: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    publication_style: bool = False,
) -> None:
    color = "cyan" if publication_style else "yellow"
    for previous, current in movements:
        ax.annotate(
            "",
            xy=(current[1], current[0]),
            xytext=(previous[1], previous[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                edgecolor="black" if publication_style else color,
                linewidth=3.0 if publication_style else 2.0,
                mutation_scale=22 if publication_style else 10,
            ),
            zorder=9,
        )
        if publication_style:
            ax.scatter(previous[1], previous[0], marker="s", s=90, c=color,
                       edgecolors="black", linewidths=1.2, zorder=10)
            ax.scatter(current[1], current[0], marker="X", s=110, c=color,
                       edgecolors="black", linewidths=1.2, zorder=10)
        else:
            ax.scatter(previous[1], previous[0], marker="X", c="red", s=20)
            ax.scatter(current[1], current[0], marker="P", c="green", s=20)


def plot_esdf_view(
    ax: plt.Axes,
    esdf: np.ndarray,
    view: BevView,
    *,
    paths: Sequence[PathOverlay] = (),
    movements: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    cmap: Any = custom_cmap,
    colorbar_label: str | None = None,
    publication_style: bool = False,
) -> list[Line2D]:
    scale = finite_percentile_abs(esdf, percentile=99.0)
    plot_scalar_map_flipped(
        ax, esdf, view.extent, "ESDF (m)",
        view.sensor_xy, cmap=cmap,
        vmin=-scale, vmax=scale, colorbar_label=colorbar_label,
    )
    ax.invert_xaxis()
    plot_motion(ax, movements, publication_style=publication_style)
    return plot_paths(ax, paths)


def add_esdf_legend(
    ax: plt.Axes,
    path_lines: Sequence[Line2D],
    *,
    has_movements: bool,
) -> None:
    robot = Line2D([], [], marker="X", linestyle="None", markersize=11,
                   markerfacecolor="yellow", markeredgecolor="black", label="Robot")
    handles: list[Line2D] = [robot, *path_lines]
    if has_movements:
        handles.extend((
            Line2D([], [], marker="s", linestyle="None", markersize=10,
                   markerfacecolor="cyan", markeredgecolor="black",
                   label="Dynamic obstacle"),
            Line2D([], [], marker=r"$\rightarrow$", linestyle="None", markersize=18,
                   color="cyan", markeredgecolor="black",
                   label="Direction of travel"),
        ))
    ax.legend(handles=handles, loc="upper right", fontsize=13, framealpha=0.9)


def plot_depth_view(ax: plt.Axes, depth: np.ndarray) -> None:
    scale = finite_percentile_abs(depth, percentile=99.0)
    plot_scalar_map(
        ax, depth[::-1, :], None, "Depth", (0.0, 0.0),
        cmap="coolwarm", vmin=0, vmax=scale,
        colorbar_label="Depth (m)", show_sensor=False,
    )


def plot_occupancy_view(
    ax: plt.Axes,
    esdf_result: dict[str, np.ndarray],
    view: BevView,
    *,
    movements: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    publication_style: bool = False,
    show_sensor: bool = True,
    title: str = "Red: occupied, Grey: unknown",
) -> None:
    image = semantic_bev_image(
        esdf_result["occupied_mask"],
        esdf_result["visible_free_mask"],
        esdf_result["unknown_mask"],
    ).swapaxes(0, 1)
    ax.imshow(image, origin="lower", extent=view.extent, interpolation="nearest")
    ax.invert_xaxis()
    if show_sensor:
        ax.scatter(
            [view.sensor_xy[1]], [view.sensor_xy[0]], marker="x", s=36,
            c="yellow", linewidths=1.5,
        )
    plot_motion(ax, movements, publication_style=publication_style)
    ax.set_title(title)
    ax.set_xlabel("Lateral position, y (m)")
    ax.set_ylabel("Forward position, x (m)")
    ax.set_aspect("equal", adjustable="box")


def plot_camera_path_view(
    ax: plt.Axes,
    rgb: np.ndarray,
    paths: Sequence[np.ndarray],
    cam_matrix: np.ndarray,
    T_cam_from_base: np.ndarray,
    *,
    bbox_result: dict[str, np.ndarray] | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    image = rgb
    if bbox_result is not None:
        image = plot_bbox(image, bbox_result, show_plot=False, return_img=True)
    if image_size is not None:
        image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    if paths:
        image = overlay_path(
            trajectories=np.stack(paths),
            img=image,
            cam_matrix=cam_matrix,
            T_cam_from_base=T_cam_from_base,
        )
    ax.imshow(image)
    ax.axis("off")


def ground_alignment_summary(result: dict[str, Any]) -> str:
    alignment = result.get("ground_alignment")
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return ""
    source = alignment.get("normal_source", "unknown")
    applied_tilt = alignment.get("applied_tilt_deg")
    if applied_tilt is None:
        return f" | ground align: {source}"
    return f" | ground align: {source} {float(applied_tilt):.1f} deg"


def visualize_path_debug(
    depth: np.ndarray,
    rgb: np.ndarray,
    esdf_result: dict[str, np.ndarray],
    bbox_result: dict[str, np.ndarray],
    cam_matrix: np.ndarray,
    T_cam_from_base: np.ndarray,
    before_path: np.ndarray,
    after_path: np.ndarray,
    point_movement_bev: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> np.ndarray:
    view = BevView.from_args(args)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        (
            f"frame: {args.frame_preset} | res: {args.resolution:.2f} m"
        ),
        fontsize=14,
    )
    plot_camera_path_view(
        axes[0, 0], rgb, [before_path, after_path], cam_matrix, T_cam_from_base,
        bbox_result=bbox_result, image_size=(args.img_w, args.img_h),
    )
    axes[0, 0].set_title("Img with Bounding Boxes and paths")
    plot_esdf_view(
        axes[0, 1], esdf_result["esdf"], view,
        paths=(
            PathOverlay(before_path, "red"),
            PathOverlay(after_path, "green"),
        ),
        movements=point_movement_bev,
    )
    plot_depth_view(axes[1, 0], depth)
    plot_occupancy_view(
        axes[1, 1], esdf_result, view, movements=point_movement_bev,
    )
    return figure_to_rgb(fig)

def visualize_depth_only(
    depth: np.ndarray,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    plot_depth_view(ax, depth)
    ax.axis('off')
    return figure_to_rgb(fig, tight_layout=True)


def visualize_bbox_only(
    rgb: np.ndarray,
    bbox_result: dict[str, np.ndarray],
) -> np.ndarray:
    return plot_bbox(rgb, bbox_result, show_plot=False, return_img=True)


def visualize_occupy_only(
    esdf_result: dict[str, np.ndarray],
    point_movement_bev: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    plot_occupancy_view(
        ax, esdf_result, BevView.from_args(args), movements=point_movement_bev,
        publication_style=True, show_sensor=False,
    )
    ax.set_xlabel("Lateral position, y (m)", fontsize=18, labelpad=10)
    ax.set_ylabel("Forward position, x (m)", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    return figure_to_rgb(fig, tight_layout=True)


def visualize_esdf(
    esdf_result: dict[str, np.ndarray],
    args: argparse.Namespace,
    before_path: np.ndarray | None = None,
    after_path: np.ndarray | None = None,
    point_movement_bev: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> np.ndarray:
    paths = []
    if before_path is not None:
        paths.append(PathOverlay(before_path, "red", "Original path"))
    if after_path is not None:
        paths.append(PathOverlay(after_path, "lime", "Optimized path"))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    path_lines = plot_esdf_view(
        ax,
        esdf_result["esdf"],
        BevView.from_args(args),
        paths=paths,
        movements=point_movement_bev,
        cmap="viridis",
        colorbar_label="ESDF (m)",
        publication_style=True,
    )

    add_esdf_legend(ax, path_lines, has_movements=len(point_movement_bev) > 0)

    return figure_to_rgb(fig, tight_layout=True)


def visualize_camera_esdf(
    rgb: np.ndarray,
    esdf_result: dict[str, np.ndarray],
    bbox_result: dict[str, np.ndarray],
    cam_matrix: np.ndarray,
    T_cam_from_base: np.ndarray,
    args: argparse.Namespace,
    before_path: np.ndarray | None = None,
    after_path: np.ndarray | None = None,
    point_movement_bev: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> np.ndarray:
    """Render the camera detections and regular ESDF view side by side."""
    camera_paths = [path for path in (before_path, after_path) if path is not None]
    esdf_paths = []
    if before_path is not None:
        esdf_paths.append(PathOverlay(before_path, "red", "Original path"))
    if after_path is not None:
        esdf_paths.append(PathOverlay(after_path, "lime", "Optimized path"))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=180)
    plot_camera_path_view(
        axes[0], rgb, camera_paths, cam_matrix, T_cam_from_base,
        bbox_result=bbox_result, image_size=(args.img_w, args.img_h),
    )
    axes[0].set_title("Camera detections and paths")

    path_lines = plot_esdf_view(
        axes[1], esdf_result["esdf"], BevView.from_args(args),
        paths=esdf_paths, movements=point_movement_bev, cmap="viridis",
        colorbar_label="ESDF (m)", publication_style=True,
    )
    add_esdf_legend(
        axes[1], path_lines, has_movements=len(point_movement_bev) > 0,
    )
    return figure_to_rgb(fig, tight_layout=True)


def visualize_static_dynamic_paths(
        depth: np.ndarray,
        rgb: np.ndarray,
        dynamic_esdf_result: dict[str, np.ndarray],
        static_esdf_result: dict[str, np.ndarray],
        bbox_result: dict[str, np.ndarray],
        cam_matrix: np.ndarray,
        T_cam_from_base: np.ndarray,
        before_path: np.ndarray,
        static_path: np.ndarray,
        dynamic_path: np.ndarray,
        point_movement_bev: list[tuple[np.ndarray, np.ndarray]],
        idx: int,
        args: argparse.Namespace,
) -> np.ndarray:
    view = BevView.from_args(args)
    filtered = dynamic_esdf_result["points_filtered"]
    alignment_text = ground_alignment_summary(dynamic_esdf_result)

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle(
        (
            f"frame {idx} | "
            f"filtered points: {filtered.shape[0]} | "
            f"frame: {args.frame_preset} | res: {args.resolution:.2f} m"
            f"{alignment_text}"
        ),
        fontsize=14,
    )
    plot_camera_path_view(
        axes[0, 0], rgb, [before_path, dynamic_path], cam_matrix, T_cam_from_base,
        bbox_result=bbox_result, image_size=(args.img_w, args.img_h),
    )
    axes[0, 0].set_title("Img with Bounding Boxes and paths")

    plot_esdf_view(
        axes[0, 1], dynamic_esdf_result["esdf"], view,
        paths=(PathOverlay(before_path, "red", linewidth=2.2),
               PathOverlay(dynamic_path, "green", linewidth=2.2)),
        movements=point_movement_bev,
    )
    axes[0, 1].set_title("dynamic ESDF (m)")
    plot_esdf_view(
        axes[0, 2], static_esdf_result["esdf"], view,
        paths=(PathOverlay(before_path, "red", linewidth=2.2),
               PathOverlay(static_path, "green", linewidth=2.2)),
        movements=point_movement_bev,
    )
    axes[0, 2].set_title("static ESDF (m)")

    plot_depth_view(axes[1, 0], depth)
    plot_occupancy_view(
        axes[1, 1], dynamic_esdf_result, view,
        movements=point_movement_bev,
        title="Dynamic: Red: occupied, Grey: unknown",
    )
    plot_occupancy_view(
        axes[1, 2], static_esdf_result, view,
        movements=point_movement_bev,
        title="Static: Red: occupied, Grey: unknown",
    )
    return figure_to_rgb(fig)
