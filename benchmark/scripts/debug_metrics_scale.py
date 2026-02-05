import argparse
import numpy as np
import torch

from src.data.collate import collate_batch
from src.data.dataset import SwingDataset
from src.data.transforms import DictTransform, StandardScaler
from src.models.build import build_model
from src.train.metrics import mpjpe, pa_mpjpe
from src.utils.io import load_yaml


def _fit_scalers(ds: SwingDataset, keys):
    buffers = {k: [] for k in keys}
    for i in range(len(ds)):
        item = ds[i]
        for k in keys:
            buffers[k].append(item[k])
    scalers = {k: StandardScaler.fit(buffers[k]) for k in keys}
    return scalers


def _stats(x: np.ndarray, name: str) -> None:
    print(f"{name}_min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} std={x.std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split-name", default="val")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    if model_cfg.get("target_type", "position") != "position":
        raise ValueError("target_type_quaternion_no_longer_supported")

    ds = SwingDataset(
        root=data_cfg["root"],
        include_pressure=data_cfg.get("include_pressure", False),
        include_fpv=data_cfg.get("include_fpv", False),
        target_fps=data_cfg.get("target_fps", 48.0),
        imu_source=data_cfg.get("imu_source", "position"),
        split_path=data_cfg.get("split_path"),
        split_name=args.split_name,
    )

    if data_cfg.get("normalize", False):
        train_ds = SwingDataset(
            root=data_cfg["root"],
            include_pressure=data_cfg.get("include_pressure", False),
            include_fpv=data_cfg.get("include_fpv", False),
            target_fps=data_cfg.get("target_fps", 48.0),
            imu_source=data_cfg.get("imu_source", "position"),
            split_path=data_cfg.get("split_path"),
            split_name="train",
        )
        keys = ["imu", "gt"]
        if data_cfg.get("include_pressure", False):
            keys.append("pressure")
        if data_cfg.get("include_fpv", False):
            keys.append("fpv")
        scalers = _fit_scalers(train_ds, keys)
        ds.transform = DictTransform(
            imu=scalers.get("imu"),
            pressure=scalers.get("pressure"),
            fpv=scalers.get("fpv"),
            gt=scalers.get("gt"),
        )
        gt_scaler = scalers.get("gt")
        if gt_scaler is not None:
            _stats(gt_scaler.mean, "gt_scaler_mean")
            _stats(gt_scaler.std, "gt_scaler_std")
    else:
        gt_scaler = None

    item = ds[args.index]
    batch = collate_batch([item])
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        pred = model({k: v for k, v in batch.items() if torch.is_tensor(v)})
    pred_np = pred.numpy()
    gt_np = batch["gt"].numpy()

    _stats(gt_np, "gt_norm")
    _stats(pred_np, "pred_norm")
    if gt_np.std() > 0:
        print(f"pred_gt_std_ratio={pred_np.std() / gt_np.std():.3f}")

    if gt_scaler is not None:
        gt_den = gt_scaler.inverse_transform(gt_np)
        pred_den = gt_scaler.inverse_transform(pred_np)
        _stats(gt_den, "gt_denorm")
        _stats(pred_den, "pred_denorm")

        mp = mpjpe(torch.from_numpy(pred_den), torch.from_numpy(gt_den)).item()
        pa = pa_mpjpe(torch.from_numpy(pred_den), torch.from_numpy(gt_den)).item()
        print(f"mpjpe_denorm={mp:.6f} pa_mpjpe_denorm={pa:.6f}")
    else:
        mp = mpjpe(torch.from_numpy(pred_np), torch.from_numpy(gt_np)).item()
        pa = pa_mpjpe(torch.from_numpy(pred_np), torch.from_numpy(gt_np)).item()
        print(f"mpjpe={mp:.6f} pa_mpjpe={pa:.6f}")


if __name__ == "__main__":
    main()
