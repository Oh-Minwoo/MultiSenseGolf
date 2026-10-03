from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


JOINT_NAMES = [
    "Hip", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot",
    "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
]


JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


COCO_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


REGRESSION_TARGET = "Ball Speed"


CLASSIFICATION_TARGETS = ["Spin Axis", "Launch Direction"]


ALL_TARGETS = [REGRESSION_TARGET, *CLASSIFICATION_TARGETS]


def save_json(path: str | os.PathLike[str], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_output_dirs(cfg: Mapping[str, Any], base_dir: Path) -> Tuple[Path, Path]:
    cache_dir = (base_dir / cfg["paths"]["cache_dir"]).resolve()
    output_dir = (base_dir / cfg["paths"]["output_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, output_dir


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def skill_group(participant: int) -> str:
    if 1 <= participant <= 8:
        return "Beginner"
    if 9 <= participant <= 16:
        return "Intermediate"
    if 17 <= participant <= 24:
        return "Professional"
    return "Unknown"


def swing_key(participant: int, swing: int) -> str:
    return f"P{participant:02d}_Swing{swing:02d}"


def swing_dir(data_root: str | os.PathLike[str], participant: int, swing: int) -> Path:
    return Path(data_root) / f"P{participant:02d}" / f"Swing{swing:02d}"


def participant_number(value: Any) -> int:
    """Read numeric annotation IDs or released PXX participant identifiers."""
    text = str(value).strip()
    if re.fullmatch(r"[Pp]\d+", text):
        return int(text[1:])
    number = float(text)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid participant identifier: {value!r}")
    return int(number)


def read_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read CSV or XLSX metadata while preserving timestamp precision."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Expected a CSV or XLSX metadata file: {path}")


def find_h5(path: Path) -> Path | None:
    matches = sorted(path.glob("*_stream_data.hdf5"))
    return matches[0] if matches else None


def h5_first_key(h5: Any, candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in h5:
            return key
    return None


def pressure_path(side: str, leaf: str) -> List[str]:
    return [
        f"wireless-insole-{side}/pressure-data/{leaf}",
        f"wireless-insole-{side}/pressure_data/{leaf}",
    ]


def finite_number(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "-", "nan", "NaN"}:
            return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def signed_lr_number(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        if text in {"", "-"}:
            return float("nan")
        sign = 1.0
        if text[:1].upper() == "R":
            text = text[1:]
        elif text[:1].upper() == "L":
            sign = -1.0
            text = text[1:]
        try:
            return sign * float(text)
        except Exception:
            return float("nan")
    return finite_number(value)


def class3(value: float, threshold: float) -> int:
    if not np.isfinite(value):
        return -1
    if value < -threshold:
        return 0
    if value > threshold:
        return 2
    return 1


def build_label_frame(annotation: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    out = annotation.copy()
    ball = out["Ball Speed (m/s)"].map(finite_number).to_numpy(float)
    sidespin = out["Sidespin (rpm)"].map(finite_number).to_numpy(float)
    backspin = out["Backspin (rpm)"].map(finite_number).to_numpy(float)
    spin_angle = np.degrees(np.arctan2(sidespin, backspin))
    spin_angle[~(np.isfinite(sidespin) & np.isfinite(backspin))] = np.nan
    launch_angle = out["Horizontal Launch Angle (deg)"].map(signed_lr_number).to_numpy(float)
    spin_thr = launch_thr = 2.0
    out["ball_speed"] = ball
    out["spin_axis_angle_deg"] = spin_angle
    out["spin_axis_class"] = [class3(v, spin_thr) for v in spin_angle]
    out["launch_direction_angle_deg"] = launch_angle
    out["launch_direction_class"] = [class3(v, launch_thr) for v in launch_angle]
    out["Participant ID"] = out["Participant Number"].map(participant_number)
    out["Swing ID"] = out["Swing Number"].astype(int)
    out["Skill Group"] = out["Participant ID"].map(skill_group)
    return out


def target_array(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == REGRESSION_TARGET:
        return frame["ball_speed"].to_numpy(float)
    if target == "Spin Axis":
        return frame["spin_axis_class"].to_numpy(int)
    if target == "Launch Direction":
        return frame["launch_direction_class"].to_numpy(int)
    raise KeyError(target)


def normalize_quaternions_wxyz(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=np.float64).copy()
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    if q.ndim == 2:
        q = q[:, None, :]
        squeeze = True
    else:
        squeeze = False
    for j in range(q.shape[1]):
        for t in range(1, q.shape[0]):
            if np.dot(q[t - 1, j], q[t, j]) < 0:
                q[t, j] *= -1.0
    return q[:, 0] if squeeze else q


def mean_quaternion_wxyz(quats: np.ndarray) -> np.ndarray:
    q = normalize_quaternions_wxyz(quats)
    if q.ndim == 1:
        return q
    ref = q[0]
    aligned = q.copy()
    aligned[np.einsum("ij,j->i", aligned, ref) < 0] *= -1.0
    a = aligned.T @ aligned
    values, vectors = np.linalg.eigh(a)
    mean_q = vectors[:, int(np.argmax(values))]
    if np.dot(mean_q, ref) < 0:
        mean_q *= -1
    return mean_q / max(np.linalg.norm(mean_q), 1e-12)


def quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    q = normalize_quaternions_wxyz(quats)
    w, x, y, z = [q[..., i] for i in range(4)]
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def address_relative_rotations(quats_wxyz: np.ndarray, times: np.ndarray, address_ms: int) -> np.ndarray:
    q = normalize_quaternions_wxyz(quats_wxyz)
    times = np.asarray(times, dtype=float).reshape(-1)
    cutoff = times[0] + address_ms / 1000.0
    address_mask = times <= cutoff
    if int(address_mask.sum()) < 2:
        address_mask[: min(len(times), 2)] = True
    refs = np.stack([mean_quaternion_wxyz(q[address_mask, j]) for j in range(q.shape[1])])
    r_ref = quat_wxyz_to_rotmat(refs)
    r = quat_wxyz_to_rotmat(q)
    return np.einsum("jik,tjkl->tjil", r_ref, r)


def rotation_angle(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=float).reshape(-1)
    target = np.asarray(target_times, dtype=float).reshape(-1)
    idx = np.searchsorted(source, target)
    idx = np.clip(idx, 0, len(source) - 1)
    left = np.clip(idx - 1, 0, len(source) - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(source[idx] - target)
    return np.where(choose_left, left, idx)


def interpolate_short_gaps(series: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(series, dtype=float).copy()
    if out.ndim == 1:
        out = out[:, None]
        squeeze = True
    else:
        squeeze = False
    n = len(out)
    for c in range(out.shape[1]):
        finite = np.isfinite(out[:, c])
        i = 0
        while i < n:
            if finite[i]:
                i += 1
                continue
            start = i
            while i < n and not finite[i]:
                i += 1
            end = i
            gap = end - start
            if gap <= max_gap and start > 0 and end < n and finite[start - 1] and finite[end]:
                out[start:end, c] = np.linspace(
                    out[start - 1, c], out[end, c], gap + 2, dtype=float
                )[1:-1]
    return out[:, 0] if squeeze else out


def angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cosine = np.sum(ba * bc, axis=-1) / np.maximum(denom, 1e-12)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    angle[~(np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1) & np.isfinite(c).all(axis=-1))] = np.nan
    return angle


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return float("nan")
    valid_pair = np.isfinite(points[:-1]).all(axis=1) & np.isfinite(points[1:]).all(axis=1)
    if not valid_pair.any():
        return float("nan")
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1)[valid_pair].sum())


def confidence_interval_t(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    from scipy.stats import t

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(x))
    half = float(t.ppf((1 + confidence) / 2.0, len(x) - 1) * np.std(x, ddof=1) / math.sqrt(len(x)))
    return mean - half, mean + half


def _valid_dataset(h5: h5py.File, path: str | None, minimum: int = 2) -> bool:
    if path is None or path not in h5:
        return False
    ds = h5[path]
    return bool(ds.shape and ds.shape[0] >= minimum)


def _metadata_map(path: str | os.PathLike[str]) -> Dict[int, Dict[str, Any]]:
    df = read_table(path)
    result: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        participant = participant_number(row["Participant Number"])
        result[participant] = row.to_dict()
    return result


def build_availability(cfg: Mapping[str, Any], base_dir: Path, include_pose: bool = True) -> pd.DataFrame:
    _, output_dir = ensure_output_dirs(cfg, base_dir)
    annotation = read_table(cfg["paths"]["annotation"])
    labels = build_label_frame(annotation, cfg)
    metadata = _metadata_map(cfg["paths"]["participant_metadata"])
    data_root = cfg["paths"]["data_root"]
    pose_quality_path = output_dir / "front_view_pose_quality.csv"
    pose_quality: Dict[str, Dict[str, Any]] = {}
    if include_pose and pose_quality_path.is_file():
        quality = pd.read_csv(pose_quality_path)
        pose_quality = {str(row["swing_key"]): row.to_dict() for _, row in quality.iterrows()}

    rows = []
    for _, row in labels.iterrows():
        participant = int(row["Participant ID"])
        swing = int(row["Swing ID"])
        key = swing_key(participant, swing)
        sd = swing_dir(data_root, participant, swing)
        h5_path = find_h5(sd)
        fpv_path = sd / "FPV_RGB.mp4"
        fpv_ts_path = sd / "FPV_Timestamps.csv"
        front_path = sd / "Front_View" / "RGB_Video.mp4"
        front_ts_path = sd / "Front_View" / "Front_View_Timestamps.csv"
        flags = {
            "pns_lead_hand_valid": False,
            "pns_bilateral_hands_valid": False,
            "pns_full_body_valid": False,
            "head_imu_valid": False,
            "pressure_valid": False,
        }
        stream_error = ""
        h5_paths: Dict[str, str] = {}
        if h5_path is not None:
            try:
                with h5py.File(h5_path, "r") as h5:
                    acc = "pns-joint-synthetic-accel/acceleration-values/data"
                    quat = "pns-joint-quaternion/angle-values/data"
                    head_acc = "pupil-imu-accel/acceleration-values/data"
                    head_gyro = "pupil-imu-gyro/velocity-values/data"
                    left_pressure = h5_first_key(h5, pressure_path("left", "data"))
                    right_pressure = h5_first_key(h5, pressure_path("right", "data"))
                    h5_paths = {
                        "pns_acceleration_path": acc if acc in h5 else "",
                        "pns_quaternion_path": quat if quat in h5 else "",
                        "head_acceleration_path": head_acc if head_acc in h5 else "",
                        "head_gyro_path": head_gyro if head_gyro in h5 else "",
                        "left_pressure_path": left_pressure or "",
                        "right_pressure_path": right_pressure or "",
                    }
                    pns_ok = _valid_dataset(h5, acc) and _valid_dataset(h5, quat)
                    if pns_ok:
                        pns_ok = h5[acc].shape[1] == 63 and h5[quat].shape[1] == 84
                    flags["pns_lead_hand_valid"] = bool(pns_ok)
                    flags["pns_bilateral_hands_valid"] = bool(pns_ok)
                    flags["pns_full_body_valid"] = bool(pns_ok)
                    flags["head_imu_valid"] = bool(
                        _valid_dataset(h5, head_acc) and _valid_dataset(h5, head_gyro)
                        and h5[head_acc].shape[1] == 3 and h5[head_gyro].shape[1] == 3
                    )
                    flags["pressure_valid"] = bool(
                        _valid_dataset(h5, left_pressure) and _valid_dataset(h5, right_pressure)
                    )
            except Exception as exc:
                stream_error = f"{type(exc).__name__}: {exc}"

        q = pose_quality.get(key, {})
        front_pose_valid = bool(q.get("pose_quality_valid", False)) if include_pose else False
        meta = metadata.get(participant, {})
        swing_start = finite_number(row["Swing Start Time (s)"])
        swing_end = finite_number(row["Swing End Time (s)"])
        front_impact = finite_number(row["Front-View Impact Timestamp (s)"])
        fpv_impact = finite_number(row["FPV Impact Timestamp (s)"])
        impact_valid = np.isfinite(front_impact) and np.isfinite(fpv_impact)
        bounds_valid = np.isfinite(swing_start) and np.isfinite(swing_end) and swing_end > swing_start
        record: Dict[str, Any] = {
            "swing_key": key,
            "participant_id": participant,
            "swing_id": swing,
            "skill_group": skill_group(participant),
            "gender": meta.get("Gender", "Unknown"),
            "body_mass_kg": finite_number(meta.get("Body Mass (kg)")),
            "ball_speed": row["ball_speed"],
            "ball_speed_valid": bool(np.isfinite(row["ball_speed"])),
            "spin_axis_angle_deg": row["spin_axis_angle_deg"],
            "spin_axis_class": int(row["spin_axis_class"]),
            "spin_axis_valid": bool(int(row["spin_axis_class"]) >= 0),
            "launch_direction_angle_deg": row["launch_direction_angle_deg"],
            "launch_direction_class": int(row["launch_direction_class"]),
            "launch_direction_valid": bool(int(row["launch_direction_class"]) >= 0),
            **flags,
            "fpv_video_valid": bool(fpv_path.is_file() and fpv_ts_path.is_file()),
            "front_view_rgb_valid": bool(front_path.is_file() and front_ts_path.is_file()),
            "front_view_pose_valid": front_pose_valid,
            "front_view_pose_valid_frame_ratio": q.get("valid_frame_ratio", np.nan),
            "front_view_pose_mean_confidence": q.get("mean_keypoint_confidence", np.nan),
            "impact_timestamp_valid": bool(impact_valid),
            "swing_bounds_valid": bool(bounds_valid),
            "swing_start_timestamp": swing_start,
            "swing_end_timestamp": swing_end,
            "front_view_impact_timestamp": front_impact,
            "fpv_impact_timestamp": fpv_impact,
            "h5_path": str(h5_path) if h5_path else "",
            "fpv_video_path": str(fpv_path),
            "front_view_video_path": str(front_path),
            "stream_inspection_error": stream_error,
            **h5_paths,
        }
        rows.append(record)

    availability = pd.DataFrame(rows)
    required_columns = [
        "ball_speed_valid", "spin_axis_valid", "launch_direction_valid",
        "pns_lead_hand_valid", "head_imu_valid", "pressure_valid",
        "fpv_video_valid", "front_view_rgb_valid",
        "impact_timestamp_valid", "swing_bounds_valid",
    ]
    if include_pose:
        required_columns.append("front_view_pose_valid")
    availability["common_cohort"] = availability[required_columns].all(axis=1)

    return availability
