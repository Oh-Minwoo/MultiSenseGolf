import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np


def _resolve_subject_swing(subject: str, swing: str) -> Tuple[str, str]:
    subject = str(subject)
    swing = str(swing)
    if not subject.lower().startswith("sub"):
        subject = f"Sub{int(subject):02d}"
    if not swing.lower().startswith("swing"):
        swing = f"Swing{int(swing):02d}"
    return subject, swing


def _resolve_h5_path(root: Path, subject: str, swing: str) -> Path:
    subject, swing = _resolve_subject_swing(subject, swing)
    fname = f"{subject.lower()}_{swing}_stream_data.hdf5"
    return root / subject / swing / fname


def _resolve_video_path(root: Path, subject: str, swing: str) -> Path:
    subject, swing = _resolve_subject_swing(subject, swing)
    return root / subject / swing / "FPV_RGB.mp4"


def load_pupil_gaze_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    gaze_key = "pupil_gaze_xy/gaze_values/data"
    time_key = "pupil_gaze_xy/gaze_values/time_s"
    with h5py.File(h5_path, "r") as h5:
        if gaze_key not in h5 or time_key not in h5:
            raise KeyError(f"missing_dataset: {gaze_key} or {time_key}")
        raw = np.asarray(h5[gaze_key][()], dtype=np.float32)
        t = np.asarray(h5[time_key][()], dtype=np.float64).reshape(-1)

    if raw.ndim == 1:
        if raw.size % 2 != 0:
            raise ValueError("raw gaze must be Nx2 or flattenable to Nx2")
        raw = raw.reshape(-1, 2)
    elif raw.ndim >= 2:
        if raw.shape[1] >= 2:
            raw = raw[:, :2]
        else:
            raw = raw.reshape(-1, 2)

    if raw.shape[0] != t.shape[0]:
        n = min(raw.shape[0], t.shape[0])
        raw = raw[:n]
        t = t[:n]

    return t.astype(np.float64), raw.astype(np.float32)


def play_gaze_overlay(
    h5_path: str,
    *,
    canvas_size=None,
    trail_len: int = 20,
    point_radius: int = 6,
    line_thickness: int = 2,
    window_name: str = "Gaze Overlay",
    world_video_path: Optional[str] = None,
    world_start_time: Optional[float] = None,
):
    t, g = load_pupil_gaze_from_h5(Path(h5_path))


    fps = len(g) / (t[-1] - t[0])
    x, y = g[:, 0], g[:, 1]

    normalized = (np.nanmax(np.abs(x)) <= 1.5) and (np.nanmax(np.abs(y)) <= 1.5)

    use_video = world_video_path is not None

    if use_video:
        cap = cv2.VideoCapture(world_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open world video: {world_video_path}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        base_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if base_W <= 0 or base_H <= 0:
            ret_tmp, tmp_frame = cap.read()
            if not ret_tmp:
                cap.release()
                raise RuntimeError("Failed to read first frame from world video.")
            base_H, base_W = tmp_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if normalized:
            x_disp_video = (x * base_W).astype(np.float32)
            y_disp_video = (y * base_H).astype(np.float32)
        else:
            x_disp_video = x.astype(np.float32)
            y_disp_video = y.astype(np.float32)

        frame_base = world_start_time if world_start_time is not None else t[0]
        start_idx_float = (t[0] - frame_base) * fps_video
        start_idx = int(np.floor(start_idx_float)) if start_idx_float > 0 else 0
        start_idx = max(0, min(total_frames - 1, start_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        delay_ms = max(1, int(round(1000.0 / max(1e-3, fps_video))))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        paused = False
        step_once = False
        frame_buffer = None
        current_frame_idx = start_idx
        frame_ts = frame_base + current_frame_idx / fps_video
        gaze_ptr = -1

        while True:
            need_new_frame = frame_buffer is None or not paused or step_once
            if need_new_frame:
                if current_frame_idx >= total_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_ts = frame_base + current_frame_idx / fps_video
                current_frame_idx += 1
                frame_buffer = frame
                display_H, display_W = frame_buffer.shape[:2]
                display_frame = frame_buffer.copy()
                step_once = False
            else:
                if frame_buffer is None:
                    break
                display_frame = frame_buffer.copy()
                display_H, display_W = display_frame.shape[:2]

            while (gaze_ptr + 1) < len(t) and t[gaze_ptr + 1] <= frame_ts:
                gaze_ptr += 1

            if gaze_ptr >= 0:
                trail_start = max(0, gaze_ptr - trail_len + 1)
                xs = x_disp_video[trail_start:gaze_ptr + 1]
                ys = y_disp_video[trail_start:gaze_ptr + 1]
                pts = []
                for xx, yy in zip(xs, ys):
                    if 0 <= xx < display_W and 0 <= yy < display_H:
                        pts.append([int(xx), int(yy)])
                if len(pts) >= 2:
                    cv2.polylines(display_frame, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), line_thickness)

                xi, yi = int(x_disp_video[gaze_ptr]), int(y_disp_video[gaze_ptr])
                if 0 <= xi < display_W and 0 <= yi < display_H:
                    cv2.circle(display_frame, (xi, yi), point_radius, (0, 0, 255), -1)

            frame_number_display = max(0, current_frame_idx - 1)
            info = f"frame={frame_number_display}/{total_frames}  time={frame_ts:.3f}s  normalized={normalized}"
            cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                step_once = True
                paused = True
                continue
            elif key == ord('a'):
                jump = max(1, int(round(fps_video)))
                current_frame_idx = max(start_idx, current_frame_idx - jump)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                frame_buffer = None
                step_once = True
                paused = True
                gaze_ptr = max(-1, np.searchsorted(t, frame_base + current_frame_idx / fps_video, side="right") - 1)
                continue
            elif key == ord('d'):
                jump = max(1, int(round(fps_video)))
                current_frame_idx = min(total_frames, current_frame_idx + jump)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                frame_buffer = None
                step_once = True
                paused = True
                continue

        cap.release()
        cv2.destroyAllWindows()
        return

    if canvas_size is None:
        valid = (x > 0) & (y > 0)
        if np.any(valid):
            xmax = int(np.percentile(x[valid], 99)) + 2
            ymax = int(np.percentile(y[valid], 99)) + 2
        else:
            xmax = 1088
            ymax = 1080
        W, H = max(320, xmax), max(240, ymax)
    else:
        W, H = canvas_size

    if normalized:
        x_disp = (x * W).astype(np.float32)
        y_disp = (y * H).astype(np.float32)
    else:
        x_disp = x.astype(np.float32)
        y_disp = y.astype(np.float32)

    delay_ms = max(1, int(1000.0 / min(120.0, fps)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    paused = False
    i = 0
    n = len(t)

    while i < n:
        frame = np.full((H, W, 3), 255, dtype=np.uint8)

        start_idx = max(0, i - trail_len + 1)
        xs = x_disp[start_idx:i + 1]
        ys = y_disp[start_idx:i + 1]

        pts = []
        for xx, yy in zip(xs, ys):
            if 0 <= xx < W and 0 <= yy < H:
                pts.append([int(xx), int(yy)])
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 0, 0), line_thickness)

        xi, yi = int(x_disp[i]), int(y_disp[i])
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(frame, (xi, yi), point_radius, (0, 0, 255), -1)

        window_info = f"{t[0]:.3f}-{t[-1]:.3f}s"
        info = f"i={i+1}/{n}  fps~{fps:.1f}  normalized={normalized}  (W,H)=({W},{H})  window={window_info}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            i = min(i + 1, n - 1)
            continue
        elif key == ord('a'):
            i = max(0, i - max(1, trail_len))
            continue
        elif key == ord('d'):
            i = min(n - 1, i + max(1, trail_len))
            continue

        if not paused:
            i += 1

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--swing", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    h5_path = _resolve_h5_path(root, args.subject, args.swing)
    if not h5_path.is_file():
        raise FileNotFoundError(f"not_found: {h5_path}")

    video_path = _resolve_video_path(root, args.subject, args.swing)
    world_video = str(video_path) if video_path.is_file() else None

    play_gaze_overlay(
        str(h5_path),
        world_video_path=world_video,
    )


if __name__ == "__main__":
    main()
