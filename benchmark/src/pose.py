from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from data import COCO_INDEX, COCO_KEYPOINTS, ensure_output_dirs, interpolate_short_gaps, sha256_file


POSE_SETTINGS = {
    "estimator": "ultralytics_yolo11n_pose",
    "checkpoint": "yolo11n-pose.pt",
    "checkpoint_sha256": "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
    "image_size": 640,
    "detector_confidence": 0.25,
    "detector_iou": 0.50,
    "keypoint_confidence": 0.25,
    "max_interpolation_gap_frames": 3,
    "min_valid_frame_ratio": 0.60,
    "min_valid_frames": 20,
}


def pose_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = cfg.get("front_view_pose", {})
    return {
        **POSE_SETTINGS,
        "device": runtime.get("device", "cpu"),
        "half_precision": bool(runtime.get("half_precision", False)),
    }


def pose_cache_path(cache_dir: Path, participant: int, swing: int) -> Path:
    return cache_dir / "front_view_pose" / f"P{participant:02d}" / f"Swing{swing:02d}.npz"


def load_front_timestamps(path: Path, frame_count: int) -> np.ndarray:
    try:
        df = pd.read_csv(path)
        for candidate in ("timestamp_color_unix_s", "timestamp_unix_s"):
            if candidate in df.columns:
                times = df[candidate].to_numpy(float)
                break
        else:
            raise KeyError("timestamp column not found")
    except Exception:
        times = np.arange(frame_count, dtype=float) / 30.0
    if len(times) >= frame_count:
        return times[:frame_count]
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
        extra = times[-1] + dt * np.arange(1, frame_count - len(times) + 1)
        return np.concatenate([times, extra])
    return np.arange(frame_count, dtype=float) / 30.0


def preprocess_pose(
    xy_pixels: np.ndarray,
    confidence: np.ndarray,
    times: np.ndarray,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    xy = np.asarray(xy_pixels, dtype=float).copy()
    conf = np.asarray(confidence, dtype=float).copy()
    times = np.asarray(times, dtype=float).reshape(-1)
    n = min(len(xy), len(conf), len(times))
    xy, conf, times = xy[:n], conf[:n], times[:n]
    settings = pose_settings(cfg)
    threshold = float(settings["keypoint_confidence"])
    max_gap = int(settings["max_interpolation_gap_frames"])
    low = conf < threshold
    xy[low] = np.nan
    li = COCO_INDEX["left_hip"]
    ri = COCO_INDEX["right_hip"]
    ls = COCO_INDEX["left_shoulder"]
    rs = COCO_INDEX["right_shoulder"]
    pelvis = (xy[:, li] + xy[:, ri]) / 2.0
    shoulder_mid = (xy[:, ls] + xy[:, rs]) / 2.0
    shoulder_width = np.linalg.norm(xy[:, ls] - xy[:, rs], axis=1)
    torso_length = np.linalg.norm(shoulder_mid - pelvis, axis=1)
    scale = shoulder_width.copy()
    invalid_shoulder = ~np.isfinite(scale) | (scale < 2.0)
    scale[invalid_shoulder] = torso_length[invalid_shoulder]
    frame_valid = np.isfinite(pelvis).all(axis=1) & np.isfinite(scale) & (scale >= 2.0)
    normalized = (xy - pelvis[:, None, :]) / scale[:, None, None]
    normalized[~frame_valid] = np.nan
    flat = normalized.reshape(n, -1)
    flat = interpolate_short_gaps(flat, max_gap)
    normalized = flat.reshape(n, len(COCO_KEYPOINTS), 2)
    velocity = np.full_like(normalized, np.nan)
    if n > 1:
        dt = np.diff(times)
        valid_dt = np.isfinite(dt) & (dt > 1e-6)
        delta = normalized[1:] - normalized[:-1]
        velocity[1:][valid_dt] = delta[valid_dt] / dt[valid_dt, None, None]
        velocity[0] = velocity[1]
    return {
        "normalized_xy": normalized.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "frame_valid": frame_valid,
        "valid_frame_ratio": float(frame_valid.mean()) if n else 0.0,
        "mean_keypoint_confidence": float(np.nanmean(conf)) if n else 0.0,
        "mean_confidence_by_keypoint": np.nanmean(conf, axis=0) if n else np.full(17, np.nan),
        "missing_ratio_by_keypoint": np.mean(~np.isfinite(normalized[..., 0]), axis=0) if n else np.ones(17),
        "times": times,
    }


def extract_front_view_pose(
    availability: pd.DataFrame,
    cfg: Mapping[str, Any],
    base_dir: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    from ultralytics import YOLO

    cache_dir, output_dir = ensure_output_dirs(cfg, base_dir)
    pose_cfg = pose_settings(cfg)
    checkpoint = cache_dir / "models" / pose_cfg["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pose checkpoint is missing: {checkpoint}")
    model = YOLO(str(checkpoint))
    quality_rows: List[Dict[str, Any]] = []
    started = time.time()
    candidates = availability[availability["front_view_rgb_valid"]].reset_index(drop=True)
    for position, row in candidates.iterrows():
        participant = int(row["participant_id"])
        swing = int(row["swing_id"])
        out_path = pose_cache_path(cache_dir, participant, swing)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_file() and not rebuild:
            data = np.load(out_path, allow_pickle=True)
            xy = data["xy"]
            conf = data["confidence"]
            times = data["times"]
            width = int(data["width"])
            height = int(data["height"])
            runtime = float(data.get("runtime_seconds", np.nan))
        else:
            video = Path(row["front_view_video_path"])
            timestamps = video.parent / "Front_View_Timestamps.csv"
            t0 = time.time()
            xy_list: List[np.ndarray] = []
            conf_list: List[np.ndarray] = []
            width = height = 0
            results = model.predict(
                source=str(video), stream=True, imgsz=int(pose_cfg["image_size"]),
                conf=float(pose_cfg["detector_confidence"]), iou=float(pose_cfg["detector_iou"]),
                device=pose_cfg["device"], half=bool(pose_cfg["half_precision"]), verbose=False,
                save=False, save_txt=False, project=str(output_dir / "pose_inference"),
            )
            for result in results:
                height, width = result.orig_shape
                if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
                    xy_list.append(np.full((17, 2), np.nan, dtype=np.float32))
                    conf_list.append(np.zeros(17, dtype=np.float32))
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
                selected = int(np.argmax(areas))
                xy_list.append(result.keypoints.xy[selected].detach().cpu().numpy().astype(np.float32))
                conf_list.append(result.keypoints.conf[selected].detach().cpu().numpy().astype(np.float32))
            xy = np.asarray(xy_list, dtype=np.float32)
            conf = np.asarray(conf_list, dtype=np.float32)
            times = load_front_timestamps(timestamps, len(xy))
            runtime = time.time() - t0
            np.savez_compressed(
                out_path, xy=xy, confidence=conf, times=times, width=width, height=height,
                runtime_seconds=runtime, checkpoint_sha256=sha256_file(checkpoint),
                estimator=pose_cfg["estimator"],
            )
        processed = preprocess_pose(xy, conf, times, cfg)
        valid = (
            len(xy) >= int(pose_cfg["min_valid_frames"])
            and processed["valid_frame_ratio"] >= float(pose_cfg["min_valid_frame_ratio"])
        )
        wrist_conf = float(
            np.nanmean(processed["mean_confidence_by_keypoint"][[COCO_INDEX["left_wrist"], COCO_INDEX["right_wrist"]]])
        )
        quality: Dict[str, Any] = {
            "swing_key": row["swing_key"], "participant_id": participant, "swing_id": swing,
            "skill_group": row["skill_group"], "frame_count": int(len(xy)),
            "detected_frame_count": int(np.sum(np.isfinite(xy).any(axis=(1, 2)))) if len(xy) else 0,
            "valid_frame_ratio": processed["valid_frame_ratio"],
            "mean_keypoint_confidence": processed["mean_keypoint_confidence"],
            "mean_wrist_confidence": wrist_conf, "pose_quality_valid": bool(valid),
            "runtime_seconds": runtime, "cache_path": str(out_path),
        }
        for index, name in enumerate(COCO_KEYPOINTS):
            quality[f"{name}_mean_confidence"] = processed["mean_confidence_by_keypoint"][index]
            quality[f"{name}_missing_ratio"] = processed["missing_ratio_by_keypoint"][index]
        quality_rows.append(quality)
        if (position + 1) % 50 == 0 or position + 1 == len(candidates):
            elapsed = time.time() - started
            print(
                f"[pose] {position + 1}/{len(candidates)} valid={sum(bool(q['pose_quality_valid']) for q in quality_rows)} "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )
            pd.DataFrame(quality_rows).to_csv(output_dir / "front_view_pose_quality.csv", index=False)
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(output_dir / "front_view_pose_quality.csv", index=False)
    return quality_df

