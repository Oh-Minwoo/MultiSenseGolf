from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.data import build_availability, participant_number, save_json, set_seed, sha256_file, target_array
from src.models import calculate_metrics
from src.pose import POSE_SETTINGS, extract_front_view_pose

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "outputs"
CONFIG_DEFAULT = SCRIPT_DIR / "config.yaml"
OUTER_FOLDS = (1, 2, 3, 4)
MODALITIES = ["PNS_LeadHand", "Head_IMU", "Pressure", "FPV", "FrontView_2D"]
DISPLAY_NAMES = {"PNS_LeadHand": "PNS LeadHand", "Head_IMU": "Head IMU", "Pressure": "Pressure", "FPV": "FPV", "FrontView_2D": "Front-view 2D", "NoSensorBaseline": "No-sensor baseline"}
TARGETS = ["Ball Speed", "Spin Axis", "Launch Direction"]
METRICS = {"Ball Speed": ["r2", "rmse", "mae"], "Spin Axis": ["auc_macro_ovr", "balanced_accuracy", "macro_f1"], "Launch Direction": ["auc_macro_ovr", "balanced_accuracy", "macro_f1"]}
FEATURE_DIMS = {"PNS_LeadHand": 10, "Head_IMU": 16, "Pressure": 18, "FPV": 34, "FrontView_2D": 18}

def load_cohort():
    return pd.read_csv(BASE_DIR / "cohort.csv").sort_values(["participant_id", "swing_id"]).reset_index(drop=True)

def load_folds():
    folds = pd.read_csv(SCRIPT_DIR / "folds.csv")
    folds["participant_id"] = folds["participant_id"].map(participant_number)
    return folds
def crop_to_bounds(values: np.ndarray, times: np.ndarray, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    times = np.asarray(times, dtype=float).reshape(-1)
    n = min(len(values), len(times))
    values, times = values[:n], times[:n]
    mask = np.isfinite(times) & (times >= start) & (times <= end)
    if int(mask.sum()) >= 2:
        return values[mask], times[mask]
    return values, times



def finite_stat(values: np.ndarray, statistic: str) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if statistic == "rms":
        return float(np.sqrt(np.mean(x * x)))
    if statistic == "range":
        return float(np.max(x) - np.min(x))
    if statistic == "mean":
        return float(np.mean(x))
    if statistic == "std":
        return float(np.std(x, ddof=0))
    if statistic == "median":
        return float(np.median(x))
    raise KeyError(statistic)



def summarize_channels(values: np.ndarray, channel_names: Sequence[str], statistics: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    vector: List[float] = []
    names: List[str] = []
    for index, channel in enumerate(channel_names):
        for statistic in statistics:
            vector.append(finite_stat(values[:, index], statistic))
            names.append(f"{channel}.{statistic}")
    return np.asarray(vector, dtype=np.float32), names



def extract_pns_lead_hand(h5: h5py.File, row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    from data import JOINT_INDEX, address_relative_rotations, rotation_angle

    acc_path = "pns-joint-synthetic-accel/acceleration-values/data"
    quat_path = "pns-joint-quaternion/angle-values/data"
    acc = np.asarray(h5[acc_path][:], dtype=float).reshape(-1, 21, 3)
    acc_times = np.asarray(h5[acc_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    quat = np.asarray(h5[quat_path][:], dtype=float).reshape(-1, 21, 4)
    quat_times = np.asarray(h5[quat_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    start, end = float(row["swing_start_timestamp"]), float(row["swing_end_timestamp"])
    acc, acc_times = crop_to_bounds(acc, acc_times, start, end)
    quat, quat_times = crop_to_bounds(quat, quat_times, start, end)
    lead = JOINT_INDEX["LeftHand"]
    lead_acc = acc[:, lead]
    acc_values = np.column_stack([lead_acc, np.linalg.norm(lead_acc, axis=1)])
    acc_vector, acc_names = summarize_channels(
        acc_values,
        ["synthetic_acceleration_x", "synthetic_acceleration_y", "synthetic_acceleration_z", "synthetic_acceleration_magnitude"],
        ["rms", "range"],
    )
    relative_rotation = address_relative_rotations(
        quat[:, lead:lead + 1], quat_times, 200
    )
    angle = np.degrees(rotation_angle(relative_rotation)[:, 0])
    angle_vector, angle_names = summarize_channels(angle, ["address_relative_rotation_angle_deg"], ["mean", "range"])
    return np.concatenate([acc_vector, angle_vector]), acc_names + angle_names



def extract_head_imu(h5: h5py.File, row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    acc_path = "pupil-imu-accel/acceleration-values/data"
    gyro_path = "pupil-imu-gyro/velocity-values/data"
    acc = np.asarray(h5[acc_path][:], dtype=float)
    gyro = np.asarray(h5[gyro_path][:], dtype=float)
    acc_times = np.asarray(h5[acc_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    gyro_times = np.asarray(h5[gyro_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    start, end = float(row["swing_start_timestamp"]), float(row["swing_end_timestamp"])
    acc, _ = crop_to_bounds(acc, acc_times, start, end)
    gyro, _ = crop_to_bounds(gyro, gyro_times, start, end)
    blocks = [acc[:, 0], acc[:, 1], acc[:, 2], gyro[:, 0], gyro[:, 1], gyro[:, 2], np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)]
    names = ["raw_acceleration_x", "raw_acceleration_y", "raw_acceleration_z", "raw_angular_velocity_x", "raw_angular_velocity_y", "raw_angular_velocity_z", "acceleration_magnitude", "angular_velocity_magnitude"]
    vector: List[float] = []
    feature_names: List[str] = []
    for block, name in zip(blocks, names):
        for statistic in ("rms", "range"):
            vector.append(finite_stat(block, statistic))
            feature_names.append(f"{name}.{statistic}")
    return np.asarray(vector, dtype=np.float32), feature_names



def extract_pressure(h5: h5py.File, row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    from data import nearest_indices, path_length, pressure_path

    left_path = next(path for path in pressure_path("left", "data") if path in h5)
    right_path = next(path for path in pressure_path("right", "data") if path in h5)
    left = np.asarray(h5[left_path][:], dtype=float)
    right = np.asarray(h5[right_path][:], dtype=float)
    left_times = np.asarray(h5[left_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    right_times = np.asarray(h5[right_path.rsplit("/", 1)[0] + "/time_s"][:], dtype=float).reshape(-1)
    start, end = float(row["swing_start_timestamp"]), float(row["swing_end_timestamp"])
    left, left_times = crop_to_bounds(left, left_times, start, end)
    right, right_times = crop_to_bounds(right, right_times, start, end)
    right = right[nearest_indices(right_times, left_times)]
    left_total = np.nansum(left, axis=(1, 2))
    right_total = np.nansum(right, axis=(1, 2))
    bilateral = left_total + right_total
    scale = np.nanmax(bilateral)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nan

    def center_of_pressure(grid: np.ndarray, total: np.ndarray) -> np.ndarray:
        height, width = grid.shape[1:]
        x = np.arange(width, dtype=float)[None, None, :]
        y = np.arange(height, dtype=float)[None, :, None]
        cx = np.divide(np.sum(grid * x, axis=(1, 2)), total, out=np.full(len(grid), np.nan), where=total > 0) / max(width - 1, 1)
        cy = np.divide(np.sum(grid * y, axis=(1, 2)), total, out=np.full(len(grid), np.nan), where=total > 0) / max(height - 1, 1)
        return np.column_stack([cx, cy])

    left_cop = center_of_pressure(left, left_total)
    right_cop = center_of_pressure(right, right_total)
    zero = bilateral <= 0
    left_cop[zero] = np.nan
    right_cop[zero] = np.nan
    ratio = np.divide(left_total, bilateral, out=np.full_like(left_total, np.nan), where=bilateral > 0)
    values = np.column_stack([
        left_total / scale,
        right_total / scale,
        bilateral / scale,
        ratio,
        left_cop,
        right_cop,
    ])
    names = ["left_total_adc_relative", "right_total_adc_relative", "bilateral_total_adc_relative", "left_load_ratio", "left_cop_x", "left_cop_y", "right_cop_x", "right_cop_y"]
    vector, feature_names = summarize_channels(values, names, ["mean", "range"])
    vector = np.concatenate([vector, np.asarray([path_length(left_cop), path_length(right_cop)], dtype=np.float32)])
    feature_names.extend(["left_cop_path_length", "right_cop_path_length"])
    return vector, feature_names



def load_video_times(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    for name in ("timestamp_unix_s", "timestamp_color_unix_s"):
        if name in frame:
            return frame[name].to_numpy(float)
    raise KeyError(f"Timestamp column missing in {path}")



def extract_fpv(row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    video_path = Path(row["fpv_video_path"])
    times = load_video_times(video_path.with_name("FPV_Timestamps.csv"))
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(frame_count, len(times))
    cutoff = float(row["fpv_impact_timestamp"])
    count = min(count, int(np.searchsorted(times[:count], cutoff, side="left")))
    size = 160
    bins = 8
    histograms: List[np.ndarray] = []
    magnitudes: List[float] = []
    previous = None
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        resized_height = max(32, int(round(height * size / max(width, 1))))
        gray = cv2.cvtColor(cv2.resize(frame, (size, resized_height)), cv2.COLOR_BGR2GRAY)
        if previous is None:
            previous = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        orientation = np.floor((angle % (2 * np.pi)) * bins / (2 * np.pi)).astype(int) % bins
        histogram = np.bincount(orientation.ravel(), weights=magnitude.ravel(), minlength=bins).astype(float)
        if histogram.sum() > 0:
            histogram /= histogram.sum()
        histograms.append(histogram)
        magnitudes.append(float(np.nanmean(magnitude)))
        previous = gray
    capture.release()
    histogram = np.asarray(histograms, dtype=float)
    if len(histogram) < 2:
        raise ValueError("Fewer than two strictly pre-impact optical-flow frames")
    delta = np.vstack([np.zeros((1, bins), dtype=float), np.diff(histogram, axis=0)])
    values = np.concatenate([histogram, delta, np.asarray(magnitudes, dtype=float)[:, None]], axis=1)
    names = [f"hoof_bin_{index}" for index in range(bins)] + [f"hoof_delta_bin_{index}" for index in range(bins)] + ["optical_flow_magnitude"]
    return summarize_channels(values, names, ["mean", "std"])



def extract_front_view(row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    from data import COCO_INDEX, angle_three_points, interpolate_short_gaps

    pose_root = BASE_DIR / "cache" / "front_view_pose"
    pose_path = pose_root / f"P{int(row['participant_id']):02d}" / f"Swing{int(row['swing_id']):02d}.npz"
    data = np.load(pose_path, allow_pickle=True)
    xy = np.asarray(data["xy"], dtype=float)
    confidence = np.asarray(data["confidence"], dtype=float)
    times = np.asarray(data["times"], dtype=float).reshape(-1)
    n = min(len(xy), len(confidence), len(times))
    xy, confidence, times = xy[:n], confidence[:n], times[:n]
    start, end = float(row["swing_start_timestamp"]), float(row["swing_end_timestamp"])
    mask = np.isfinite(times) & (times >= start) & (times <= end)
    if int(mask.sum()) >= 2:
        xy, confidence, times = xy[mask], confidence[mask], times[mask]
    xy[confidence < 0.25] = np.nan
    flat = interpolate_short_gaps(xy.reshape(len(xy), -1), 3)
    xy = flat.reshape(len(xy), 17, 2)

    idx = COCO_INDEX
    ls, rs = xy[:, idx["left_shoulder"]], xy[:, idx["right_shoulder"]]
    lh, rh = xy[:, idx["left_hip"]], xy[:, idx["right_hip"]]
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    address_mask = times <= times[0] + 200.0 / 1000.0
    if int(address_mask.sum()) < 2:
        address_mask[: min(len(times), 2)] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        address_pose = np.nanmedian(xy[address_mask], axis=0)
    address_shoulder_width = float(np.linalg.norm(address_pose[idx["left_shoulder"]] - address_pose[idx["right_shoulder"]]))
    if not np.isfinite(address_shoulder_width) or address_shoulder_width < 2.0:
        address_shoulder_width = float("nan")
    address_nose = address_pose[idx["nose"]]
    address_hip_mid = (address_pose[idx["left_hip"]] + address_pose[idx["right_hip"]]) / 2.0

    shoulder_vector = rs - ls
    pelvis_vector = rh - lh
    torso_vector = shoulder_mid - hip_mid
    shoulder_angle = np.degrees(np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0]))
    pelvis_angle = np.degrees(np.arctan2(pelvis_vector[:, 1], pelvis_vector[:, 0]))
    torso_tilt = np.degrees(np.arctan2(torso_vector[:, 0], -torso_vector[:, 1]))
    left_elbow = np.degrees(angle_three_points(ls, xy[:, idx["left_elbow"]], xy[:, idx["left_wrist"]]))
    right_elbow = np.degrees(angle_three_points(rs, xy[:, idx["right_elbow"]], xy[:, idx["right_wrist"]]))
    right_knee = np.degrees(angle_three_points(rh, xy[:, idx["right_knee"]], xy[:, idx["right_ankle"]]))
    head_displacement = np.linalg.norm(xy[:, idx["nose"]] - address_nose, axis=1) / address_shoulder_width
    pelvis_displacement = np.linalg.norm(hip_mid - address_hip_mid, axis=1) / address_shoulder_width
    current_shoulder_width = np.linalg.norm(rs - ls, axis=1)
    ankle_distance = np.linalg.norm(xy[:, idx["left_ankle"]] - xy[:, idx["right_ankle"]], axis=1)
    stance_ratio = np.divide(ankle_distance, current_shoulder_width, out=np.full(len(xy), np.nan), where=current_shoulder_width >= 2.0)
    descriptors = np.column_stack([
        shoulder_angle,
        pelvis_angle,
        torso_tilt,
        left_elbow,
        right_elbow,
        right_knee,
        head_displacement,
        pelvis_displacement,
        stance_ratio,
    ])
    names = [
        "shoulder_line_angle_deg",
        "pelvis_line_angle_deg",
        "torso_tilt_deg",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "right_knee_angle_deg",
        "normalized_head_displacement",
        "normalized_pelvis_displacement",
        "stance_ratio",
    ]
    return summarize_channels(descriptors, names, ["median", "range"])



def extract_one(row: pd.Series) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    arrays: Dict[str, np.ndarray] = {}
    names: Dict[str, List[str]] = {}
    with h5py.File(row["h5_path"], "r") as h5:
        arrays["PNS_LeadHand"], names["PNS_LeadHand"] = extract_pns_lead_hand(h5, row)
        arrays["Head_IMU"], names["Head_IMU"] = extract_head_imu(h5, row)
        arrays["Pressure"], names["Pressure"] = extract_pressure(h5, row)
    arrays["FPV"], names["FPV"] = extract_fpv(row)
    arrays["FrontView_2D"], names["FrontView_2D"] = extract_front_view(row)
    for modality in MODALITIES:
        if len(arrays[modality]) != FEATURE_DIMS[modality]:
            raise AssertionError(f"{modality} dimension {len(arrays[modality])} != {FEATURE_DIMS[modality]}")
    return arrays, names



def run_features(rebuild: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    started = time.time()
    cohort = load_cohort()
    cache_dir = BASE_DIR / "cache"
    row_dir = cache_dir / "feature_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    matrices: Dict[str, List[np.ndarray]] = {modality: [] for modality in MODALITIES}
    feature_names: Dict[str, List[str]] = {}
    errors: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(cohort.iterrows(), start=1):
        row_path = row_dir / f"{row['swing_key']}.npz"
        try:
            if row_path.is_file() and not rebuild:
                cached = np.load(row_path, allow_pickle=True)
                arrays = {modality: cached[modality].astype(np.float32) for modality in MODALITIES}
                names = {modality: list(cached[f"{modality}_names"].astype(str)) for modality in MODALITIES}
            else:
                arrays, names = extract_one(row)
                payload: Dict[str, Any] = {}
                for modality in MODALITIES:
                    payload[modality] = arrays[modality].astype(np.float32)
                    payload[f"{modality}_names"] = np.asarray(names[modality], dtype=str)
                np.savez_compressed(row_path, **payload)
            for modality in MODALITIES:
                if len(arrays[modality]) != FEATURE_DIMS[modality]:
                    raise AssertionError(f"Cached {modality} dimension mismatch")
                if modality not in feature_names:
                    feature_names[modality] = names[modality]
                elif feature_names[modality] != names[modality]:
                    raise AssertionError(f"Feature names changed for {modality}")
                matrices[modality].append(np.asarray(arrays[modality], dtype=np.float32))
        except Exception as exc:
            errors.append({"swing_key": row["swing_key"], "error_type": type(exc).__name__, "error": str(exc)})
            pd.DataFrame(errors).to_csv(BASE_DIR / "feature_extraction_errors.csv", index=False)
            raise RuntimeError(f"Feature extraction failed for {row['swing_key']}: {exc}") from exc
        if position % 25 == 0 or position == len(cohort):
            print(f"[features] {position}/{len(cohort)} elapsed={(time.time()-started)/60:.1f} min", flush=True)

    arrays = {modality: np.vstack(matrices[modality]).astype(np.float32) for modality in MODALITIES}
    cache_file = cache_dir / "features.npz"
    np.savez_compressed(cache_file, **arrays)
    save_json(cache_dir / "feature_names.json", feature_names)
    index = cohort[["swing_key", "participant_id", "swing_id"]].copy()
    index.insert(0, "feature_row_index", np.arange(len(index), dtype=int))
    index.to_csv(cache_dir / "rows.csv", index=False)
    return arrays, feature_names


def load_features() -> Dict[str, np.ndarray]:
    path = BASE_DIR / "cache" / "features.npz"
    if not path.is_file():
        raise FileNotFoundError("Run --stage features first")
    archive = np.load(path)
    arrays = {modality: archive[modality].astype(np.float32) for modality in MODALITIES}
    for modality in MODALITIES:
        if arrays[modality].shape[1] != FEATURE_DIMS[modality]:
            raise AssertionError(f"{modality} cached dimension mismatch")
    return arrays



def tune_family(
    matrix: np.ndarray,
    cohort: pd.DataFrame,
    target: str,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    family: str,
    cfg: Mapping[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    from data import target_array
    from benchmark.src.models import BlockPreprocessor, _candidate_list, _fit, _make_model, _predict, _selection_value, calculate_metrics

    y = target_array(cohort, target)
    prepared = []
    for fit_idx, val_idx in splits:
        preprocessor = BlockPreprocessor()
        x_fit = preprocessor.fit_transform([matrix[fit_idx]])
        x_val = preprocessor.transform([matrix[val_idx]])
        prepared.append((fit_idx, val_idx, x_fit, x_val))
    best_score = -np.inf
    best_params: Dict[str, Any] | None = None
    for candidate_index, params in enumerate(_candidate_list(cfg, target, family)):
        scores = []
        for inner_fold, (fit_idx, val_idx, x_fit, x_val) in enumerate(prepared, start=1):
            model = _make_model(cfg, target, family, params, seed + candidate_index * 17 + inner_fold)
            _fit(model, x_fit, y[fit_idx], target, family)
            prediction, probabilities = _predict(model, x_val, target)
            metrics = calculate_metrics(y[val_idx], prediction, probabilities, target)
            scores.append(_selection_value(metrics, target))
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_params = dict(params)
    if best_params is None:
        raise RuntimeError("No hyperparameter candidate was selected")
    return best_params, best_score



def aggregate_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    from data import confidence_interval_t

    rows = []
    for keys, group in per_fold.groupby(["condition", "target", "metric"], sort=False):
        values = group["value"].to_numpy(float)
        low, high = confidence_interval_t(values, 0.95)
        rows.append({"condition": keys[0], "display_name": DISPLAY_NAMES[keys[0]], "target": keys[1], "metric": keys[2], "mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1)), "ci95_low": low, "ci95_high": high, "outer_fold_count": int(np.isfinite(values).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(BASE_DIR / "metrics.csv", index=False)
    return result



def run_evaluation(cfg: Mapping[str, Any], resume: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from data import target_array
    from benchmark.src.models import BlockPreprocessor, _fit, _inner_splits, _make_model, _predict, calculate_metrics

    started = time.time()
    cohort = load_cohort()
    arrays = load_features()
    folds = load_folds()
    assignment = dict(zip(folds["participant_id"].astype(int), folds["outer_fold"].astype(int)))
    sample_fold = cohort["participant_id"].map(assignment).to_numpy(int)

    def existing_records(filename: str) -> List[Dict[str, Any]]:
        path = BASE_DIR / filename
        if not (resume and path.is_file()):
            return []
        frame = pd.read_csv(path)
        return frame[frame["condition"] != "NoSensorBaseline"].to_dict("records")

    metric_rows = existing_records("fold_metrics.csv")
    oof_rows = existing_records("predictions.csv")
    selected_rows = existing_records("selected_models.csv")
    completed = set()
    if metric_rows:
        frame = pd.DataFrame(metric_rows)
        for keys, group in frame.groupby(["condition", "target", "outer_fold"]):
            if group["metric"].nunique() == len(METRICS[str(keys[1])]):
                completed.add((str(keys[0]), str(keys[1]), int(keys[2])))

    split_cache: Dict[Tuple[int, str], Sequence[Tuple[np.ndarray, np.ndarray]]] = {}
    total_jobs = len(MODALITIES) * len(TARGETS) * len(OUTER_FOLDS)
    finished = len(completed)

    def persist() -> None:
        pd.DataFrame(metric_rows).to_csv(BASE_DIR / "fold_metrics.csv", index=False)
        pd.DataFrame(oof_rows).to_csv(BASE_DIR / "predictions.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(BASE_DIR / "selected_models.csv", index=False)

    for modality in MODALITIES:
        matrix = arrays[modality]
        for target in TARGETS:
            y = target_array(cohort, target)
            for outer_fold in OUTER_FOLDS:
                key = (modality, target, outer_fold)
                if key in completed:
                    print(f"[evaluation] resume skip {modality} | {target} | fold={outer_fold}", flush=True)
                    continue
                job_started = time.time()
                test_idx = np.flatnonzero(sample_fold == outer_fold)
                train_idx = np.flatnonzero(sample_fold != outer_fold)
                train_ids = set(cohort.iloc[train_idx]["participant_id"].astype(int))
                test_ids = set(cohort.iloc[test_idx]["participant_id"].astype(int))
                overlap = train_ids & test_ids
                assert not overlap
                split_key = (outer_fold, target)
                if split_key not in split_cache:
                    split_cache[split_key] = _inner_splits(cohort, train_idx, outer_fold, target, int(cfg["seed"]))
                splits = split_cache[split_key]
                for fit_idx, validation_idx in splits:
                    assert set(cohort.iloc[fit_idx].participant_id).isdisjoint(set(cohort.iloc[validation_idx].participant_id))

                family_results: Dict[str, Dict[str, Any]] = {}
                for family in ("linear", "xgboost"):
                    seed = int(cfg["seed"]) + outer_fold * 10000 + TARGETS.index(target) * 1000 + (0 if family == "linear" else 500)
                    params, inner_score = tune_family(matrix, cohort, target, splits, family, cfg, seed)
                    family_results[family] = {"parameters": params, "inner_score": inner_score, "seed": seed}

                selected_family = max(family_results, key=lambda family: family_results[family]["inner_score"])
                selected = family_results[selected_family]
                preprocessor = BlockPreprocessor()
                x_train = preprocessor.fit_transform([matrix[train_idx]])
                x_test = preprocessor.transform([matrix[test_idx]])
                model = _make_model(cfg, target, selected_family, selected["parameters"], int(selected["seed"]))
                _fit(model, x_train, y[train_idx], target, selected_family)
                prediction, probabilities = _predict(model, x_test, target)
                calculated = calculate_metrics(y[test_idx], prediction, probabilities, target)
                runtime = time.time() - job_started
                for metric in METRICS[target]:
                    metric_rows.append({"condition": modality, "display_name": DISPLAY_NAMES[modality], "target": target, "outer_fold": outer_fold, "metric": metric, "value": float(calculated[metric]), "selected_model_family": selected_family, "runtime_seconds": runtime, "train_sample_count": len(train_idx), "test_sample_count": len(test_idx), "train_participant_count": len(train_ids), "test_participant_count": len(test_ids), "raw_feature_dimension": int(matrix.shape[1]), "retained_feature_dimension": int(x_train.shape[1]), "random_seed": int(selected["seed"])})
                selected_rows.append({"condition": modality, "display_name": DISPLAY_NAMES[modality], "target": target, "outer_fold": outer_fold, "selected_model_family": selected_family, "selected_parameters": json.dumps(selected["parameters"], sort_keys=True), "selected_inner_score": selected["inner_score"], "linear_best_parameters": json.dumps(family_results["linear"]["parameters"], sort_keys=True), "linear_best_inner_score": family_results["linear"]["inner_score"], "xgboost_best_parameters": json.dumps(family_results["xgboost"]["parameters"], sort_keys=True), "xgboost_best_inner_score": family_results["xgboost"]["inner_score"]})
                for local_index, sample_index in enumerate(test_idx):
                    record = {"condition": modality, "display_name": DISPLAY_NAMES[modality], "target": target, "outer_fold": outer_fold, "selected_model_family": selected_family, "swing_key": cohort.iloc[sample_index]["swing_key"], "participant_id": int(cohort.iloc[sample_index]["participant_id"]), "swing_id": int(cohort.iloc[sample_index]["swing_id"]), "observed": float(y[sample_index]), "predicted": float(prediction[local_index]), "probability_class_0": np.nan, "probability_class_1": np.nan, "probability_class_2": np.nan}
                    if probabilities is not None:
                        for class_id in range(3):
                            record[f"probability_class_{class_id}"] = float(probabilities[local_index, class_id])
                    oof_rows.append(record)
                finished += 1
                persist()
                print(f"[evaluation] {finished}/{total_jobs} {modality} | {target} | fold={outer_fold} runtime={runtime:.1f}s total={(time.time()-started)/60:.1f}min", flush=True)

    persist()
    per_fold = pd.DataFrame(metric_rows)
    return per_fold, pd.DataFrame(oof_rows)


def dummy_predictions(cfg):
    cohort = load_cohort()
    folds = load_folds()
    assignment = dict(zip(folds.participant_id, folds.outer_fold))
    sample_fold = cohort.participant_id.map(assignment).to_numpy(int)
    metrics, predictions = [], []
    for target in TARGETS:
        y = target_array(cohort, target)
        for fold in OUTER_FOLDS:
            train = np.flatnonzero(sample_fold != fold)
            test = np.flatnonzero(sample_fold == fold)
            model = DummyRegressor(strategy="mean") if target == "Ball Speed" else DummyClassifier(strategy="prior", random_state=int(cfg["seed"]))
            model.fit(np.zeros((len(train), 1)), y[train])
            predicted = model.predict(np.zeros((len(test), 1)))
            probabilities = None
            if target != "Ball Speed":
                if set(model.classes_) != {0, 1, 2}:
                    raise ValueError(f"{target}, fold {fold}: training data must contain all three classes.")
                probabilities = model.predict_proba(np.zeros((len(test), 1)))
            values = calculate_metrics(y[test], predicted, probabilities, target)
            for metric in METRICS[target]:
                metrics.append({"condition": "NoSensorBaseline", "target": target, "outer_fold": fold, "metric": metric, "value": values[metric]})
            for j, i in enumerate(test):
                record = {"condition": "NoSensorBaseline", "target": target, "outer_fold": fold, "swing_key": cohort.iloc[i].swing_key, "participant_id": int(cohort.iloc[i].participant_id), "observed": float(y[i]), "predicted": float(predicted[j])}
                for c in range(3):
                    record[f"probability_class_{c}"] = float(probabilities[j, c]) if probabilities is not None else np.nan
                predictions.append(record)
    return pd.DataFrame(metrics), pd.DataFrame(predictions)

def resolve_metadata(root, filename):
    for folder in (root / "Documentation", root / "Documentation" / "Documentation"):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"{filename} not found under {root / 'Documentation'}")

def initialize(args):
    global BASE_DIR
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    root = args.data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    BASE_DIR = args.output_dir.resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    cfg["paths"] = {"data_root": str(root), "annotation": str(resolve_metadata(root, "Annotation Data.csv")), "participant_metadata": str(resolve_metadata(root, "Participant Metadata.csv")), "cache_dir": "cache", "output_dir": "."}
    cfg["models"].update(xgboost_device=args.device, xgboost_n_jobs=8)
    cfg["front_view_pose"] = {"device": "cpu" if args.device == "cpu" else 0, "half_precision": args.device == "cuda"}
    # Cache reuse requires the same metadata, participant folds and settings.
    inputs = {key: sha256_file(cfg["paths"][key]) for key in ("annotation", "participant_metadata")}
    inputs["folds"] = sha256_file(SCRIPT_DIR / "folds.csv")
    settings = {"config": cfg, "inputs": inputs}
    saved = BASE_DIR / "settings.json"
    if saved.is_file() and json.loads(saved.read_text(encoding="utf-8")) != settings:
        raise ValueError("Inputs or settings changed; use a new --output-dir.")
    save_json(saved, settings)
    set_seed(int(cfg["seed"]))
    return cfg

def prepare(cfg):
    from ultralytics.utils.downloads import attempt_download_asset

    availability = build_availability(cfg, BASE_DIR, include_pose=False)
    checkpoint = BASE_DIR / "cache" / "models" / POSE_SETTINGS["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        attempt_download_asset(str(checkpoint))
    if sha256_file(checkpoint) != POSE_SETTINGS["checkpoint_sha256"]:
        raise ValueError(f"Unexpected pose checkpoint: {checkpoint}")
    extract_front_view_pose(availability, cfg, BASE_DIR)
    availability = build_availability(cfg, BASE_DIR, include_pose=True)
    cohort = availability.loc[availability.common_cohort].sort_values(["participant_id", "swing_id"]).reset_index(drop=True)
    if cohort.empty:
        raise ValueError("No swings satisfy the benchmark requirements.")
    folds = load_folds()
    unknown = set(cohort.participant_id) - set(folds.participant_id)
    if unknown:
        raise ValueError(f"Participants without a fold assignment: {sorted(unknown)}")
    present = folds[folds.participant_id.isin(cohort.participant_id)]
    if set(present.outer_fold) != set(OUTER_FOLDS):
        raise ValueError("Participants from all four folds are required.")
    cohort.to_csv(BASE_DIR / "cohort.csv", index=False)
    print(f"[prepare] {len(cohort)} swings, {cohort.participant_id.nunique()} participants", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR.parent / "Data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stage", choices=["prepare", "features", "evaluate", "all"], default="all")
    args = parser.parse_args()
    cfg = initialize(args)
    if args.stage in ("prepare", "all"):
        prepare(cfg)
    if args.stage in ("features", "all"):
        run_features()
    if args.stage in ("evaluate", "all"):
        metrics, predictions = run_evaluation(cfg)
        dummy_metrics, dummy_oof = dummy_predictions(cfg)
        metrics = pd.concat([metrics, dummy_metrics], ignore_index=True)
        predictions = pd.concat([predictions, dummy_oof], ignore_index=True)
        metrics.to_csv(BASE_DIR / "fold_metrics.csv", index=False)
        predictions.to_csv(BASE_DIR / "predictions.csv", index=False)
        aggregate_metrics(metrics)
    print(f"[done] {args.stage}: {BASE_DIR}", flush=True)

if __name__ == "__main__":
    main()
