from typing import Optional, Tuple, Union

from pathlib import Path

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import h5py


def _load_pressure_from_h5(
    h5_path: Union[str, Path],
    *,
    left_key: str = "wireless-insole-left/pressure_data/data",
    right_key: str = "wireless-insole-right/pressure_data/data",
    left_time_key: str = "wireless-insole-left/pressure_data/time_s",
    right_time_key: str = "wireless-insole-right/pressure_data/time_s",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"not_found: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        if left_key not in h5 or right_key not in h5:
            raise KeyError(f"missing_dataset: {left_key} or {right_key}")
        if left_time_key not in h5 or right_time_key not in h5:
            raise KeyError(f"missing_dataset: {left_time_key} or {right_time_key}")
        left = np.asarray(h5[left_key][()], dtype=np.float32)
        right = np.asarray(h5[right_key][()], dtype=np.float32)
        left_time = np.asarray(h5[left_time_key][()], dtype=np.float64)
        right_time = np.asarray(h5[right_time_key][()], dtype=np.float64)
    return left, right, left_time, right_time


def _ensure_grid_sequence(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        return data
    if data.ndim == 2:
        frame_count, feature_count = data.shape
        grid_size = int(round(np.sqrt(feature_count)))
        if grid_size * grid_size != feature_count:
            raise ValueError(
                f"Cannot reshape pressure data of shape {data.shape} into a square grid."
            )
        return data.reshape(frame_count, grid_size, grid_size)

    if data.ndim == 4 and 1 in data.shape:
        squeezed = np.squeeze(data)
        if squeezed.ndim == 3:
            return squeezed

    raise ValueError(f"Unsupported pressure data shape: {data.shape}")


def compute_fps(time_values: np.ndarray) -> Optional[float]:
    if time_values.size < 2:
        return None
    deltas = np.diff(time_values)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return None
    return float(1.0 / np.mean(positive_deltas))


def _resolve_h5_path(root: Union[str, Path], subject: str, swing: str) -> str:
    subject = str(subject)
    swing = str(swing)
    if not subject.lower().startswith("sub"):
        subject = f"Sub{int(subject):02d}"
    if not swing.lower().startswith("swing"):
        swing = f"Swing{int(swing):02d}"
    fname = f"{subject.lower()}_{swing}_stream_data.hdf5"
    return str(Path(root) / subject / swing / fname)


def plot_pressure_heatmap_segment(
    h5_path: Union[str, Path],
    cmap: str = "inferno",
) -> animation.FuncAnimation:
    left, right, left_time, right_time = _load_pressure_from_h5(h5_path)
    left_time_segment = np.asarray(left_time, dtype=np.float64).reshape(-1)
    right_time_segment = np.asarray(right_time, dtype=np.float64).reshape(-1)
    left_segment = _ensure_grid_sequence(left)
    right_segment = _ensure_grid_sequence(right)

    if left_segment.shape[0] == 0 or right_segment.shape[0] == 0:
        raise ValueError("Selected time window does not contain any pressure samples.")

    vmin = min(left_segment.min(), right_segment.min())
    vmax = max(left_segment.max(), right_segment.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Pressure heatmaps")

    left_im = axes[0].imshow(left_segment[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    axes[0].set_title("Left Insole")
    axes[0].axis("off")

    right_im = axes[1].imshow(right_segment[0], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    axes[1].set_title("Right Insole")
    axes[1].axis("off")

    fig.subplots_adjust(bottom=0.2, wspace=0.1)

    cbar = fig.colorbar(left_im, ax=axes, shrink=0.75, location="bottom", pad=0.08)
    cbar.set_label("Pressure")

    time_text = axes[0].text(0.5, -0.15, "", transform=axes[0].transAxes, ha="center")

    left_fps = compute_fps(left_time_segment)
    interval_ms = 100 if left_fps is None else max(10, int(round(1000.0 / left_fps)))

    def update(frame_idx: int):
        left_im.set_data(left_segment[frame_idx])
        right_im.set_data(right_segment[min(frame_idx, right_segment.shape[0] - 1)])
        current_time = float(np.asarray(left_time_segment[frame_idx], dtype=np.float64).ravel()[0])
        time_text.set_text(f"t = {current_time:.3f}s")
        return left_im, right_im, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=left_segment.shape[0],
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    plt.show()
    return ani


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--swing", required=True)
    args = parser.parse_args()

    h5_path = _resolve_h5_path(args.data_root, args.subject, args.swing)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"not_found: {h5_path}")
    plot_pressure_heatmap_segment(h5_path)


if __name__ == "__main__":
    main()
