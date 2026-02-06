from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch may not be installed yet
    Dataset = object

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_FPV_FEATURE_ROOT = os.path.join(_REPO_ROOT, "fpv_features")


JOINT_NAMES = [
    "Hip",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
]

IMU_JOINTS = [
    "LeftForeArm",
    "RightForeArm",
    "LeftLeg",
    "RightLeg",
    "Head",
    "Hip",
]

RIGHT_MASK_24x10 = np.array([
    [0,1,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,0],
    [0,0,0,1,1,1,1,1,1,0],
    [0,0,0,0,1,1,1,1,0,0],
], dtype=bool)

LEFT_MASK_24x10 = np.fliplr(RIGHT_MASK_24x10)

subject_weights = [
    47, 51, 63, 60, 65, 56, 85, 83, 80, 89, 
    64, 80, 54, 72, 71, 83, 73, 65, 80, 77, 
    60, 77, 67, 64
]


def _get_subject_weight(subject: str) -> float:
    digits = "".join(ch for ch in str(subject) if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot_parse_subject_id: {subject!r}")
    subject_id = int(digits)
    if not (1 <= subject_id <= len(subject_weights)):
        raise ValueError(f"subject_id_out_of_range: {subject_id}")
    return float(subject_weights[subject_id - 1])


def _resolve_fpv_feature_path(
    swing_dir: str,
    subject: str,
    swing: str,
    fpv_feature_name: str,
    fpv_feature_root: Optional[str] = None,
) -> str:
    legacy = os.path.join(swing_dir, fpv_feature_name)
    if os.path.isfile(legacy):
        return legacy
    root = fpv_feature_root or _DEFAULT_FPV_FEATURE_ROOT
    sub_dir = os.path.join(root, subject.lower())
    fname = f"{swing.lower()}_{fpv_feature_name}"
    return os.path.join(sub_dir, fname)


@dataclass
class SwingSample:
    subject: str
    swing: str
    h5_path: str
    fpv_feat_path: Optional[str]
    fpv_ts_path: Optional[str]


def _resample_linear(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    # values: (T, D), times: (T,)
    if len(times) == 0 or len(target_times) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    out = np.empty((len(target_times), values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(target_times, times, values[:, i]).astype(np.float32)
    return out


def _resample_cubic(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if len(times) == 0 or len(target_times) == 0:
        if values.ndim == 1:
            d = 1
        else:
            d = int(values.shape[1])
        return np.zeros((len(target_times), d), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    values = np.asarray(values, dtype=np.float32)
    if len(times) < 4:
        return _resample_linear(times, values, target_times)
    try:
        from scipy.interpolate import interp1d

        f = interp1d(
            times,
            values,
            kind="cubic",
            axis=0,
            bounds_error=False,
            fill_value=(values[0], values[-1]),
            assume_sorted=False,
        )
        return f(target_times).astype(np.float32)
    except Exception:
        return _resample_linear(times, values, target_times)


def _resample_hold(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    # zero-order hold: use last observed sample
    if len(times) == 0 or len(target_times) == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    idx = np.searchsorted(times, target_times, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return values[idx].astype(np.float32)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window - 1
    x_pad = np.pad(x, ((pad, 0), (0, 0)), mode="edge")
    cumsum = np.cumsum(x_pad, axis=0)
    out = (cumsum[window:] - cumsum[:-window]) / float(window)
    return out


def _common_time_range(times_list: List[np.ndarray]) -> Optional[Tuple[float, float]]:
    if not times_list:
        return None
    if any(t is None or len(t) == 0 for t in times_list):
        return None
    start = max(float(t[0]) for t in times_list)
    end = min(float(t[-1]) for t in times_list)
    if end <= start:
        return None
    return start, end


def _make_target_times(start: float, end: float, fps: float) -> np.ndarray:
    step = 1.0 / fps
    # include end if close due to floating point
    return np.arange(start, end + 1e-9, step, dtype=np.float64)


def _pos_to_accel(
    x: np.ndarray,
    dt: float,
    n: int = 4,
    boundary: str = "nan",
) -> np.ndarray:
    """
    TransPose Eq. (11) acceleration synthesis.

    Parameters
    ----------
    x : np.ndarray
        Positions of IMUs. Expected shape: (T, M, 3).
        T: number of frames, M: number of IMUs (e.g., 6), 3: xyz.
    dt : float
        Time interval between consecutive frames (seconds).
    n : int
        Smoothing factor in Eq. (11). Paper uses n=4.
    boundary : str
        How to handle boundary frames where t-n < 0 or t+n >= T.
        - "nan": set those accelerations to NaN (recommended for training masks)
        - "zero": set them to 0
        - "trim": return only valid range [n, T-n)

    Returns
    -------
    a : np.ndarray
        Synthesized accelerations with shape:
        - (T, M, 3) for boundary in {"nan","zero"}
        - (T-2n, M, 3) for boundary == "trim"
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must have shape (T, M, 3).")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    T = x.shape[0]
    if T <= 2 * n:
        raise ValueError(f"Need T > 2n. Got T={T}, n={n}.")

    denom = (n * dt) ** 2
    a_valid = (x[:-2*n] + x[2*n:] - 2.0 * x[n:-n]) / denom  # shape: (T-2n, M, 3)

    if boundary == "trim":
        return a_valid

    a = np.empty_like(x, dtype=np.float64)
    if boundary == "nan":
        a[:] = np.nan
    elif boundary == "zero":
        a[:] = 0.0
    else:
        raise ValueError('boundary must be one of {"nan","zero","trim"}.')

    a[n:-n] = a_valid
    return a


def _quat_to_rotmat(q_xyzw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert quaternions (x, y, z, w) to rotation matrices.

    Args:
        q_xyzw: (T, M, 4) quaternion array where q[...,0:3]=(x,y,z), q[...,3]=w.
        eps: numerical stability for normalization.

    Returns:
        R: (T, M, 3, 3) rotation matrices.
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"q must have shape (T, M, 4). Got {q.shape}.")

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Normalize to unit quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    norm = np.maximum(norm, eps)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)

    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)

    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)

    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R

def _normalize_rotations_to_root(R: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """
    DIP Eq. (9): Rbar_s(t) = R_root(t)^(-1) * R_s(t)

    Args:
        R: (T, M, 3, 3) rotation matrices.
        root_idx: root sensor index along M.

    Returns:
        R_bar: (T, M, 3, 3) root-normalized rotations.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 4 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (T, M, 3, 3). Got {R.shape}.")
    T, M = R.shape[0], R.shape[1]
    if not (0 <= root_idx < M):
        raise ValueError(f"root_idx out of range. root_idx={root_idx}, M={M}")

    R_root = R[:, root_idx]                      # (T, 3, 3)
    R_root_inv = np.swapaxes(R_root, -1, -2)     # inverse = transpose
    R_bar = R_root_inv[:, None, :, :] @ R        # (T, M, 3, 3)
    return R_bar


def _normalize_accelerations_to_root(a: np.ndarray, R: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """
    DIP Eq. (10): abar_s(t) = R_root(t)^(-1) * (a_s(t) - a_root(t))

    Args:
        a: (T, M, 3) accelerations.
        R: (T, M, 3, 3) rotations (used to obtain per-frame root inverse rotation).
        root_idx: root sensor index along M.

    Returns:
        a_bar: (T, M, 3) root-normalized accelerations.
    """
    a = np.asarray(a, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"a must have shape (T, M, 3). Got {a.shape}.")
    if R.ndim != 4 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (T, M, 3, 3). Got {R.shape}.")
    if a.shape[0] != R.shape[0] or a.shape[1] != R.shape[1]:
        raise ValueError(f"a and R must share (T, M). Got a={a.shape}, R={R.shape}.")

    T, M = a.shape[0], a.shape[1]
    if not (0 <= root_idx < M):
        raise ValueError(f"root_idx out of range. root_idx={root_idx}, M={M}")

    a_root = a[:, root_idx]                      # (T, 3)
    a_rel = a - a_root[:, None, :]               # (T, M, 3)

    R_root = R[:, root_idx]                      # (T, 3, 3)
    R_root_inv = np.swapaxes(R_root, -1, -2)     # (T, 3, 3)

    a_bar = (R_root_inv[:, None, :, :] @ a_rel[..., None]).squeeze(-1)  # (T, M, 3)
    return a_bar


    

def _make_imu_accel_rot_features(
    pos_sel: np.ndarray,
    quat_sel: np.ndarray,
    dt: float,
    root_idx: int,
    accel_n: int = 4,
) -> np.ndarray:
    rot_mat = _quat_to_rotmat(quat_sel)
    if pos_sel.shape[0] <= 2 * accel_n:
        accel = np.zeros_like(pos_sel, dtype=np.float64)
    else:
        accel = _pos_to_accel(pos_sel, dt, n=accel_n, boundary="zero")

    norm_rot_mat = _normalize_rotations_to_root(rot_mat, root_idx=root_idx)
    norm_accel = _normalize_accelerations_to_root(accel, rot_mat, root_idx=root_idx)

    norm_rot_mat = norm_rot_mat.reshape(norm_rot_mat.shape[0], norm_rot_mat.shape[1], 9)
    combined = np.concatenate([norm_accel, norm_rot_mat], axis=-1)  # (T, M, 12)
    return combined.reshape(combined.shape[0], -1)




def _load_pressure(h5: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = h5["wireless-insole-left/pressure_data/data"][()]
    right = h5["wireless-insole-right/pressure_data/data"][()]
    left_time = h5["wireless-insole-left/pressure_data/time_s"][()]
    right_time = h5["wireless-insole-right/pressure_data/time_s"][()]

    left_time = np.asarray(left_time, dtype=np.float64).reshape(-1)
    right_time = np.asarray(right_time, dtype=np.float64).reshape(-1)

    def _flatten(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        t = int(x.shape[0]) if x.ndim >= 1 else 0
        d = int(np.prod(x.shape[1:])) if x.ndim >= 2 else 0
        return x.reshape(t, d).astype(np.float32)

    left = _flatten(left)
    right = _flatten(right)

    t = min(int(left.shape[0]), int(right.shape[0]), int(left_time.size), int(right_time.size))
    left = left[:t]
    right = right[:t]
    time = left_time[:t]

    if left.shape[1] != 240 or right.shape[1] != 240:
        raise ValueError(f"expected 240 dims per foot (24x10), got left={left.shape} right={right.shape}")
    return left, right, time


def _pressure_baseline_clip(x: np.ndarray, q: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    base = float(np.percentile(finite, q))
    y = x - base
    return np.maximum(y, 0.0).astype(np.float32)


def _apply_foot_mask(x_240: np.ndarray, mask_24x10: np.ndarray) -> np.ndarray:
    if x_240.ndim != 2 or x_240.shape[1] != 240:
        raise ValueError(f"expected (T,240) for foot pressure, got {x_240.shape}")
    grid = x_240.reshape(x_240.shape[0], 24, 10)
    grid = grid * mask_24x10[None, :, :]
    return grid.reshape(x_240.shape[0], 240).astype(np.float32)


def _load_gt(h5: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    data = h5["pns-joint-position/cm-values/data"][()].astype(np.float32)
    time = h5["pns-joint-position/cm-values/time_s"][()].astype(np.float64).squeeze()
    return data, time


def _load_quaternion(h5: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
    data = h5["pns-joint-quaternion/angle-values/data"][()].astype(np.float32)
    time = h5["pns-joint-quaternion/angle-values/time_s"][()].astype(np.float64).squeeze()
    return data, time


def _load_fpv_features(path: str) -> np.ndarray:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    return features


def _load_fpv_time_from_csv(path: str) -> np.ndarray:
    times: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[1]))
            except Exception:
                continue
    return np.asarray(times, dtype=np.float64)




def _load_fpv_time(path: str) -> np.ndarray:
    return _load_fpv_time_from_csv(path)



def scan_dataset(
    root: str,
    include_pressure: bool,
    include_fpv: bool,
    fpv_feature_name: str = "FPV_RGB.clip_vit_b16.npz",
    fpv_timestamp_name: str = "FPV_Timestamps.csv",
    fpv_feature_root: Optional[str] = None,
    target_type: str = "position",
    imu_source: str = "position",
    allowed_subjects: Optional[set] = None,
    verbose: bool = False,
) -> Tuple[List[SwingSample], Dict[str, int]]:
    if target_type != "position":
        raise ValueError("target_type_quaternion_no_longer_supported")
    samples: List[SwingSample] = []
    counts: Dict[str, int] = {}
    for sub in sorted(os.listdir(root)):
        if not sub.lower().startswith("sub"):
            continue
        if allowed_subjects is not None and sub not in allowed_subjects:
            continue
        sub_dir = os.path.join(root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for swing in sorted(os.listdir(sub_dir)):
            swing_dir = os.path.join(sub_dir, swing)
            if not os.path.isdir(swing_dir):
                continue
            h5_candidates = [
                p for p in os.listdir(swing_dir) if p.endswith("_stream_data.hdf5")
            ]
            if not h5_candidates:
                continue
            h5_path = os.path.join(swing_dir, h5_candidates[0])
            fpv_feat_path = _resolve_fpv_feature_path(
                swing_dir=swing_dir,
                subject=sub,
                swing=swing,
                fpv_feature_name=fpv_feature_name,
                fpv_feature_root=fpv_feature_root,
            )
            fpv_ts_path = os.path.join(swing_dir, fpv_timestamp_name)
            if include_fpv and (not os.path.isfile(fpv_feat_path) or not os.path.isfile(fpv_ts_path)):
                continue
            if include_pressure:
                try:
                    with h5py.File(h5_path, "r") as h5:
                        if "wireless-insole-left/pressure_data/data" not in h5:
                            continue
                        if "wireless-insole-right/pressure_data/data" not in h5:
                            continue
                        if "pns-joint-quaternion/angle-values/data" not in h5:
                            continue
                        left = h5["wireless-insole-left/pressure_data/data"]
                        right = h5["wireless-insole-right/pressure_data/data"]
                        if left.shape[0] == 0 or right.shape[0] == 0:
                            continue
                        gt_time = h5["pns-joint-position/cm-values/time_s"][()].astype(np.float64).squeeze()
                        quat_time = h5["pns-joint-quaternion/angle-values/time_s"][()].astype(np.float64).squeeze()
                        pressure_time = h5["wireless-insole-left/pressure_data/time_s"][()].astype(np.float64)
                except Exception:
                    continue
            else:
                with h5py.File(h5_path, "r") as h5:
                    gt_time = h5["pns-joint-position/cm-values/time_s"][()].astype(np.float64).squeeze()
                    if "pns-joint-quaternion/angle-values/data" not in h5:
                        continue
                    quat_time = h5["pns-joint-quaternion/angle-values/time_s"][()].astype(np.float64).squeeze()

            times_list = [gt_time]
            if quat_time is not None:
                times_list.append(quat_time)
            if include_pressure:
                times_list.append(pressure_time)
            if include_fpv:
                try:
                    fpv_time = _load_fpv_time(fpv_ts_path)
                except Exception:
                    continue
                times_list.append(fpv_time)
            if _common_time_range(times_list) is None:
                continue
            sample = SwingSample(
                sub,
                swing,
                h5_path,
                fpv_feat_path if include_fpv else None,
                fpv_ts_path if include_fpv else None,
            )
            samples.append(sample)
            counts[sub] = counts.get(sub, 0) + 1
    if verbose:
        total = len(samples)
        print(f"total_samples={total}")
        for sub in sorted(counts.keys()):
            print(f"{sub}={counts[sub]}")
    return samples, counts


class SwingDataset(Dataset):
    def __init__(
        self,
        root: str,
        include_pressure: bool,
        include_fpv: bool,
        target_fps: float = 48.0,
        fpv_feature_name: str = "FPV_RGB.clip_vit_b16.npz",
        fpv_default_fps: float = 20.0,
        fpv_timestamp_name: str = "FPV_Timestamps.csv",
        fpv_feature_root: Optional[str] = None,
        imu_joints: Optional[List[str]] = None,
        target_type: str = "position",
        imu_source: str = "position",
        split_path: Optional[str] = None,
        split_name: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        transform=None,
    ) -> None:
        self.root = root
        self.include_pressure = include_pressure
        self.include_fpv = include_fpv
        self.target_fps = target_fps
        self.fpv_feature_name = fpv_feature_name
        self.fpv_feature_root = fpv_feature_root
        self.fpv_default_fps = fpv_default_fps
        self.imu_joints = imu_joints or IMU_JOINTS
        if target_type != "position":
            raise ValueError("target_type_quaternion_no_longer_supported")
        self.imu_source = imu_source
        self.split_path = split_path
        self.split_name = split_name
        self.subjects = subjects
        self.transform = transform

        allowed_subjects = None
        if subjects is not None:
            allowed_subjects = set(subjects)
        elif split_path is not None:
            if split_name is None:
                raise ValueError("split_name_required_when_split_path_is_set")
            from src.data.splits import load_split

            split = load_split(split_path)
            allowed_subjects = set(getattr(split, split_name))

        self.samples, self.counts = scan_dataset(
            root,
            include_pressure=include_pressure,
            include_fpv=include_fpv,
            fpv_feature_name=fpv_feature_name,
            fpv_timestamp_name=fpv_timestamp_name,
            fpv_feature_root=fpv_feature_root,
            target_type=target_type,
            imu_source=imu_source,
            allowed_subjects=allowed_subjects,
            verbose=False,
        )

        self.joint_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        self.imu_joint_idx = [self.joint_to_idx[name] for name in self.imu_joints]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        with h5py.File(sample.h5_path, "r") as h5:
            gt, gt_time = _load_gt(h5)
            gt = gt.reshape(gt.shape[0], len(JOINT_NAMES), 3)

            quat, quat_time = _load_quaternion(h5)
            quat = quat.reshape(quat.shape[0], len(JOINT_NAMES), 4)

            pressure = None
            pressure_time = None
            if self.include_pressure:
                left_p, right_p, pressure_time = _load_pressure(h5)
                left_p = _pressure_baseline_clip(left_p, q=1.0)
                right_p = _pressure_baseline_clip(right_p, q=1.0)
                weight = _get_subject_weight(sample.subject)
                denom = np.float32(max(weight, 1e-6))
                left_p = (left_p / denom).astype(np.float32)
                right_p = (right_p / denom).astype(np.float32)
                pressure = (left_p, right_p)

        fpv = None
        fpv_time = None
        if self.include_fpv and sample.fpv_feat_path and sample.fpv_ts_path:
            fpv = _load_fpv_features(sample.fpv_feat_path)
            fpv_time = _load_fpv_time(sample.fpv_ts_path)
            if len(fpv) != len(fpv_time):
                n = min(len(fpv), len(fpv_time))
                fpv = fpv[:n]
                fpv_time = fpv_time[:n]

        times_list = [gt_time]
        if quat_time is not None:
            times_list.append(quat_time)
        if pressure_time is not None:
            times_list.append(pressure_time)
        if fpv_time is not None:
            times_list.append(fpv_time)

        time_range = _common_time_range(times_list)
        if time_range is None:
            raise ValueError(f"no_overlapping_time_range: {sample.h5_path}")

        target_times = _make_target_times(time_range[0], time_range[1], self.target_fps)

        # Resample GT to target times
        gt_flat = gt.reshape(gt.shape[0], -1)
        gt_resampled = _resample_linear(gt_time, gt_flat, target_times)
        gt_resampled = gt_resampled.reshape(len(target_times), len(JOINT_NAMES), 3)
        gt_for_imu = gt_resampled

        # Root-relative: Hip is index 0
        hip = gt_resampled[:, 0:1, :]
        gt_resampled = gt_resampled - hip

        # Low-pass filter GT (root-relative)
        gt_flat = gt_resampled.reshape(len(target_times), -1)
        # gt_flat = _lowpass_filter(gt_flat, self.gt_cutoff_hz, self.target_fps)
        gt_resampled = gt_flat.reshape(len(target_times), len(JOINT_NAMES), 3)

        gt_target = gt_resampled.reshape(len(target_times), -1)

        quat_flat = quat.reshape(quat.shape[0], -1)
        quat_resampled = _resample_linear(quat_time, quat_flat, target_times)
        quat_resampled = quat_resampled.reshape(len(target_times), len(JOINT_NAMES), 4)

        # IMU uses selected joints
        if self.imu_source == "quaternion":
            imu = quat_resampled[:, self.imu_joint_idx, :].reshape(len(target_times), -1)
        else:
            pos_sel = gt_for_imu[:, self.imu_joint_idx, :]  # (T, M, 3)
            quat_sel = quat_resampled[:, self.imu_joint_idx, :]  # (T, M, 4)
            hip_global_idx = self.joint_to_idx.get("Hip", 0)
            root_idx = self.imu_joint_idx.index(hip_global_idx) if hip_global_idx in self.imu_joint_idx else 0
            imu = _make_imu_accel_rot_features(
                pos_sel=pos_sel,
                quat_sel=quat_sel,
                dt=1.0 / float(self.target_fps),
                root_idx=root_idx,
            )

        if pressure is not None:
            left_p, right_p = pressure
            left_p = np.maximum(_resample_cubic(pressure_time, left_p, target_times), 0.0)
            right_p = np.maximum(_resample_cubic(pressure_time, right_p, target_times), 0.0)

            left_p = _apply_foot_mask(left_p, LEFT_MASK_24x10)
            right_p = _apply_foot_mask(right_p, RIGHT_MASK_24x10)
            pressure = np.concatenate([left_p, right_p], axis=1).astype(np.float32)

        if fpv is not None:
            fpv = _resample_hold(fpv_time, fpv, target_times)


        item = {
            "subject": sample.subject,
            "swing": sample.swing,
            "time_s": target_times.astype(np.float64),
            "imu": imu.astype(np.float32),
            "gt": gt_target.astype(np.float32),
        }
        if self.include_pressure:
            item["pressure"] = pressure.astype(np.float32)
        if self.include_fpv:
            item["fpv"] = fpv.astype(np.float32)

        if self.transform is not None:
            item = self.transform(item)
        return item
