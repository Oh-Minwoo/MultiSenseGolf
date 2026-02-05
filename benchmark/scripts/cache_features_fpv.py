import argparse
import csv
import os
from typing import List, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_FEATURE_ROOT = os.path.join(_REPO_ROOT, "fpv_features")


def _load_frames_cv2_by_index(video_path: str, frame_indices: np.ndarray) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video: {video_path}")

    frames: List[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def _load_frames_torchvision_by_index(video_path: str, frame_indices: np.ndarray) -> List[np.ndarray]:
    from torchvision.io import read_video

    video, _, info = read_video(video_path, pts_unit="sec")
    video = video.numpy()
    idx = np.clip(frame_indices.astype(int), 0, video.shape[0] - 1)
    frames = [video[i] for i in idx]
    return frames


def _load_frames(video_path: str, frame_indices: np.ndarray) -> List[np.ndarray]:
    try:
        return _load_frames_cv2_by_index(video_path, frame_indices)
    except Exception:
        return _load_frames_torchvision_by_index(video_path, frame_indices)


def _load_clip(model_name: str, pretrained: str, device: str):
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = None
        return model, preprocess, device, tokenizer
    except Exception:
        import clip

        model, preprocess = clip.load(model_name, device=device, jit=False)
        tokenizer = None
        return model, preprocess, device, tokenizer


def _encode_frames(frames: List[np.ndarray], model, preprocess, device: str, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image

    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            pil = [Image.fromarray(f) for f in batch]
            inputs = torch.stack([preprocess(p) for p in pil], dim=0).to(device)
            feats = model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0).astype(np.float32)


def _iter_swings(root: str, video_name: str, ts_name: str) -> List[Tuple[str, str, str]]:
    swings: List[Tuple[str, str, str]] = []
    for sub in sorted(os.listdir(root)):
        if not sub.lower().startswith("sub"):
            continue
        sub_dir = os.path.join(root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for swing in sorted(os.listdir(sub_dir)):
            swing_dir = os.path.join(sub_dir, swing)
            if not os.path.isdir(swing_dir):
                continue
            video_path = os.path.join(swing_dir, video_name)
            ts_path = os.path.join(swing_dir, ts_name)
            if not os.path.isfile(video_path) or not os.path.isfile(ts_path):
                continue
            swings.append((swing_dir, video_path, ts_path))
    return swings


def _feature_out_path(feature_root: str, swing_dir: str, output_name: str) -> str:
    swing = os.path.basename(swing_dir)
    sub = os.path.basename(os.path.dirname(swing_dir))
    sub_dir = os.path.join(feature_root, sub.lower())
    fname = f"{swing.lower()}_{output_name}"
    return os.path.join(sub_dir, fname)


def _load_fpv_timestamps(ts_path: str) -> Tuple[np.ndarray, np.ndarray]:
    indices: List[int] = []
    times: List[float] = []
    with open(ts_path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                indices.append(int(parts[0]))
                times.append(float(parts[1]))
            except Exception:
                continue
    return np.asarray(indices, dtype=np.int64), np.asarray(times, dtype=np.float64)


def _select_indices_by_fps(
    indices: np.ndarray, times: np.ndarray, target_fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    if len(times) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    step = 1.0 / target_fps
    target_times = np.arange(times[0], times[-1] + 1e-9, step, dtype=np.float64)
    pos = np.searchsorted(times, target_times, side="left")
    pos = np.clip(pos, 0, len(times) - 1)
    prev = np.clip(pos - 1, 0, len(times) - 1)
    choose = np.where(
        np.abs(times[prev] - target_times) <= np.abs(times[pos] - target_times),
        prev,
        pos,
    )
    return indices[choose], times[choose]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--video-name", default="FPV_RGB.mp4")
    parser.add_argument("--timestamp-name", default="FPV_Timestamps.csv")
    parser.add_argument("--output-name", default="FPV_RGB.clip_vit_b16.npz")
    parser.add_argument(
        "--feature-root",
        default=None,
        help="Where to store FPV feature npz files (default: <repo>/fpv_features).",
    )
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    swings = _iter_swings(args.root, args.video_name, args.timestamp_name)
    if len(swings) == 0:
        print("no_videos_found")
        return

    feature_root = args.feature_root or _DEFAULT_FEATURE_ROOT

    model, preprocess, device, _ = _load_clip(args.model, args.pretrained, args.device)
    model = model.to(device)

    total = 0
    skipped = 0
    for swing_dir, video_path, ts_path in swings:
        out_path = _feature_out_path(feature_root, swing_dir, args.output_name)
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        frame_indices, frame_times = _load_fpv_timestamps(ts_path)
        sel_indices, sel_times = _select_indices_by_fps(
            frame_indices, frame_times, args.target_fps
        )
        if len(sel_indices) == 0:
            print(f"empty_timestamps={ts_path}")
            continue
        frames = _load_frames(video_path, sel_indices)
        if len(frames) == 0:
            print(f"empty_video={video_path}")
            continue
        features = _encode_frames(frames, model, preprocess, args.device, args.batch_size)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, features=features, time_s=sel_times)
        total += 1
        if total % 50 == 0:
            print(f"processed={total} skipped={skipped}")

    print(f"processed={total} skipped={skipped} total={len(swings)}")


if __name__ == "__main__":
    main()
