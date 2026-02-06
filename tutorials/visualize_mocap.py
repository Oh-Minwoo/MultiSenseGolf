import argparse
import os
import h5py
import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Tuple, Union

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def hierarchy_to_edges(hierarchy: Iterable[Iterable[int]]) -> Tuple[Tuple[int, int], ...]:
    """Convert joint hierarchy chains into unique pairwise edges."""
    edges: List[Tuple[int, int]] = []
    seen = set()
    for chain in hierarchy:
        nodes = list(chain)
        for a, b in zip(nodes[:-1], nodes[1:]):
            edge = (a, b)
            if edge not in seen and a != b:
                seen.add(edge)
                edges.append(edge)
    return tuple(edges)

JOINT_HIERARCHY: Tuple[Tuple[int, ...], ...] = (
    (0, 1),
    (0, 1, 2),
    (0, 1, 2, 3),
    (0, 4),
    (0, 4, 5),
    (0, 4, 5, 6),
    (0, 7),
    (0, 7, 8),
    (0, 7, 8, 9),
    (0, 7, 8, 9, 10),
    (0, 7, 8, 9, 10, 11),
    (0, 7, 8, 9, 10, 11, 12),
    (0, 7, 8, 9, 13),
    (0, 7, 8, 9, 13, 14),
    (0, 7, 8, 9, 13, 14, 15),
    (0, 7, 8, 9, 13, 14, 15, 16),
    (0, 7, 8, 9, 17),
    (0, 7, 8, 9, 17, 18),
    (0, 7, 8, 9, 17, 18, 19),
    (0, 7, 8, 9, 17, 18, 19, 20),
)

DEFAULT_CONNECTIONS: Tuple[Tuple[int, int], ...] = hierarchy_to_edges(JOINT_HIERARCHY)



def flatten_joint_positions(position_array: np.ndarray) -> np.ndarray:
    """Reshape joint position array (frames, 63) -> (frames, 21, 3)."""
    if position_array.ndim != 2 or position_array.shape[1] % 3 != 0:
        raise ValueError("Expected shape (frames, 63) representing 21 joints with xyz coordinates.")
    joint_count = position_array.shape[1] // 3
    return position_array.reshape(position_array.shape[0], joint_count, 3)


def load_pns_global_positions(
    h5_path: Union[str, Path],
    *,
    stream: str = "pns-joint-position",
    value_key: str = "cm-values",
) -> Tuple[np.ndarray, np.ndarray]:
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"not_found: {h5_path}")
    data_key = f"{stream}/{value_key}/data"
    time_key = f"{stream}/{value_key}/time_s"
    with h5py.File(h5_path, "r") as h5:
        if data_key not in h5 or time_key not in h5:
            raise KeyError(f"missing_dataset: {data_key} or {time_key}")
        position_data = np.asarray(h5[data_key][()], dtype=np.float32)
        timestamps = np.asarray(h5[time_key][()], dtype=np.float64).reshape(-1)
    return position_data, timestamps


def _resolve_h5_path(root: Union[str, Path], subject: str, swing: str) -> str:
    subject = str(subject)
    swing = str(swing)
    if not subject.lower().startswith("sub"):
        subject = f"Sub{int(subject):02d}"
    if not swing.lower().startswith("swing"):
        swing = f"Swing{int(swing):02d}"
    fname = f"{subject.lower()}_{swing}_stream_data.hdf5"
    return str(Path(root) / subject / swing / fname)


def visualize_pns_skeleton(
    h5_path: Union[str, Path],
    *,
    joint_connections: Optional[Iterable[Tuple[int, int]]] = None,
    interval_ms: Optional[float] = None,
    speed_multiplier: float = 1.0,
    min_interval_ms: float = 2.0,
) -> FuncAnimation:
    position_data, timestamps = load_pns_global_positions(h5_path)
    if position_data.size == 0 or timestamps.size == 0:
        raise ValueError("Selected swing contains no position data or timestamps.")

    joints = flatten_joint_positions(position_data)
    frame_count, joint_count, _ = joints.shape

    connections = tuple(joint_connections) if joint_connections is not None else DEFAULT_CONNECTIONS
    connections = tuple(edge for edge in connections if max(edge) < joint_count)

    measured_interval_s: Optional[float] = None
    measured_fps: Optional[float] = None
    if timestamps.size > 1:
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 0]
        if positive.size:
            measured_interval_s = float(np.median(positive))
            if measured_interval_s > 0:
                measured_fps = 1.0 / measured_interval_s

    if interval_ms is None:
        if measured_fps is None or measured_fps <= 0.0:
            base_fps = 96.0  # fallback when timestamps are unusable
        else:
            base_fps = max(1e-6, measured_fps)
        effective_fps = base_fps * max(speed_multiplier, 1e-6)
        interval_ms = 1000.0 / effective_fps
    else:
        interval_ms = interval_ms / max(speed_multiplier, 1e-6)

    interval_ms = max(min_interval_ms, interval_ms)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"PNS Skeleton – {Path(h5_path).stem}")

    xs, ys, zs = joints[0].T
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rotated = (rotation_matrix @ joints[0].T).T
    xs, ys, zs = rotated.T
    scatter = ax.scatter(xs, ys, zs, c="tab:blue", s=20)
    lines = []
    for idx_a, idx_b in connections:
        line, = ax.plot(
            [xs[idx_a], xs[idx_b]],
            [ys[idx_a], ys[idx_b]],
            [zs[idx_a], zs[idx_b]],
            color="tab:orange",
            lw=2,
        )
        lines.append(line)

    # Set dynamic axis limits with a small margin.
    rotated_all = (rotation_matrix @ joints.reshape(-1, 3).T).T.reshape(joints.shape)
    min_vals = rotated_all.min(axis=(0, 1))
    max_vals = rotated_all.max(axis=(0, 1))
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    center = (min_vals + max_vals) / 2.0
    span = ranges.max() * 0.6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes) 

    def update(frame_idx: int):
        frame = (rotation_matrix @ joints[frame_idx].T).T
        xs, ys, zs = frame.T
        scatter._offsets3d = (xs, ys, zs)
        for line, (idx_a, idx_b) in zip(lines, connections):
            line.set_data([xs[idx_a], xs[idx_b]], [ys[idx_a], ys[idx_b]])
            line.set_3d_properties([zs[idx_a], zs[idx_b]])
        if time_text is not None:
            current_time = timestamps[min(frame_idx, timestamps.size - 1)]
            time_text.set_text(f"t = {current_time:.3f}s")
            return [scatter, time_text, *lines]
        return [scatter, *lines]

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=max(min_interval_ms, interval_ms),
        blit=False,
        repeat=False,
    )
    plt.tight_layout()

    plt.show()
    return anim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--swing", required=True)
    args = parser.parse_args()

    h5_path = _resolve_h5_path(args.data_root, args.subject, args.swing)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"not_found: {h5_path}")
    visualize_pns_skeleton(h5_path)


if __name__ == "__main__":
    main()