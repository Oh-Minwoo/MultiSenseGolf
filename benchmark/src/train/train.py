from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.dataset import SwingDataset
from src.data.transforms import DictTransform, StandardScaler
from src.models.build import build_model
from src.train.eval import run_eval
from src.train.optim import build_optimizer
from src.train.sched import build_scheduler
from src.utils.ckpt import save_checkpoint
from src.utils.device import get_device
from src.utils.io import load_yaml, save_yaml
from src.utils.logging import SimpleLogger
from src.utils.seed import set_seed
from src.models.losses import (
    fourth_derivative_smooth_loss,
    mpjpe_loss,
    mpjve_loss,
)


def _fit_scalers(ds: SwingDataset, keys: List[str]) -> Dict[str, StandardScaler]:
    buffers = {k: [] for k in keys}
    for i in range(len(ds)):
        item = ds[i]
        for k in keys:
            buffers[k].append(item[k])
    scalers = {k: StandardScaler.fit(buffers[k]) for k in keys}
    return scalers


def _make_loader(ds: SwingDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(args.device)

    imu_joints = data_cfg.get("imu_joints")
    train_ds = SwingDataset(
        root=data_cfg["root"],
        include_pressure=data_cfg.get("include_pressure", False),
        include_fpv=data_cfg.get("include_fpv", False),
        target_fps=data_cfg.get("target_fps", 48.0),
        imu_joints=imu_joints,
        target_type=model_cfg.get("target_type", "position"),
        imu_source=data_cfg.get("imu_source", "position"),
        split_path=data_cfg.get("split_path"),
        split_name="train",
    )

    val_ds = SwingDataset(
        root=data_cfg["root"],
        include_pressure=data_cfg.get("include_pressure", False),
        include_fpv=data_cfg.get("include_fpv", False),
        target_fps=data_cfg.get("target_fps", 48.0),
        imu_joints=imu_joints,
        target_type=model_cfg.get("target_type", "position"),
        imu_source=data_cfg.get("imu_source", "position"),
        split_path=data_cfg.get("split_path"),
        split_name="val",
    )
    test_ds = None
    if args.run_test:
        test_ds = SwingDataset(
            root=data_cfg["root"],
            include_pressure=data_cfg.get("include_pressure", False),
            include_fpv=data_cfg.get("include_fpv", False),
            target_fps=data_cfg.get("target_fps", 48.0),
            imu_joints=imu_joints,
            target_type=model_cfg.get("target_type", "position"),
            imu_source=data_cfg.get("imu_source", "position"),
            split_path=data_cfg.get("split_path"),
            split_name="test",
        )

    scalers = None
    if data_cfg.get("normalize", False):
        keys = ["imu", "gt"]
        if data_cfg.get("include_pressure", False):
            keys.append("pressure")
        if data_cfg.get("include_fpv", False):
            keys.append("fpv")
        scalers = _fit_scalers(train_ds, keys)
        transform = DictTransform(
            imu=scalers.get("imu"),
            pressure=scalers.get("pressure"),
            fpv=scalers.get("fpv"),
            gt=scalers.get("gt"),
        )
        train_ds.transform = transform
        val_ds.transform = transform
        if test_ds is not None:
            test_ds.transform = transform

    train_loader = _make_loader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=True,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = _make_loader(
            test_ds,
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            shuffle=False,
        )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model.parameters(), train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}))

    out_dir = args.output_dir or train_cfg.get("output_dir", "benchmark/outputs/exp_name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, f"{os.path.basename(out_dir)}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    logger = SimpleLogger(os.path.join(out_dir, "logs"))
    save_yaml(os.path.join(out_dir, "config.yaml"), cfg)

    # Print run summary
    modalities = ["imu"]
    if data_cfg.get("include_pressure", False):
        modalities.append("pressure")
    if data_cfg.get("include_fpv", False):
        modalities.append("fpv")
    backbone_name = model_cfg.get("backbone", {}).get("name", "tcn")
    target_type = model_cfg.get("target_type", "position")
    print(
        f"modalities={','.join(modalities)} backbone={backbone_name} "
        f"target={target_type} output_dir={out_dir}"
    )

    target_fps = float(data_cfg.get("target_fps", 48.0))
    best = np.inf
    patience = int(train_cfg.get("patience", 10))
    min_delta = float(train_cfg.get("min_delta", 0.01))
    bad_epochs = 0
    best_path = os.path.join(out_dir, "checkpoints", "best.pt")
    for epoch in range(int(train_cfg.get("epochs", 30))):
        model.train()
        total_loss = 0.0
        pos_loss_total = 0.0
        vel_loss_total = 0.0
        jerk_loss_total = 0.0
        n = 0
        for batch in train_loader:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)
            pred = model(batch)
            target = batch["gt"]
            lengths = batch.get("lengths")
            if target_type == "quaternion":
                pass
                # loss = quaternion_l2(pred, target)
            else:
                lam_pos = float(train_cfg.get("lam_pos", 1.0))
                lam_mpjve = float(train_cfg.get("lam_mpjve", 0.1))
                lam_jerk = float(train_cfg.get("lam_jerk", 0.1))

                pos_loss = mpjpe_loss(pred, target, lengths=lengths)
                vel_loss = mpjve_loss(pred, target, dt=1.0 / target_fps)
                jerk_loss = fourth_derivative_smooth_loss(pred)
                loss = lam_pos * pos_loss + lam_mpjve * vel_loss + lam_jerk * jerk_loss
                # loss = lam_pos * pos_loss + vel_loss + jerk_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pos_loss_total += pos_loss.item()
            vel_loss_total += vel_loss.item()
            jerk_loss_total += jerk_loss.item()

            n += 1
        if scheduler is not None:
            scheduler.step()

        val_metrics = run_eval(
            model,
            val_loader,
            device,
            gt_scaler=scalers.get("gt") if scalers is not None else None,
            target_fps=target_fps
        )
        avg_loss = total_loss / max(n, 1)
        log_data = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
        logger.log(log_data)
        if target_type == "quaternion":
            print(
                f"epoch={epoch} train_loss={avg_loss:.6f} "
                f"val_quat_l2={val_metrics['quat_l2']:.6f} "
                f"val_mpjae={val_metrics['mpjae']:.6f}"
            )
        else:
            print(
                f"epoch={epoch} train_loss={avg_loss:.6f} "
                # f"pos_loss={pos_loss_total / max(n, 1)} vel_loss={vel_loss_total / max(n, 1)} jerk_loss={jerk_loss_total / max(n, 1)} "
                f"val_mpjpe={val_metrics['mpjpe']:.6f} "
                f"val_mpjve={val_metrics['mpjve']:.6f} "
                f"val_jitter={val_metrics['jitter']:.6f} "
            )

        if target_type == "quaternion":
            current = val_metrics["quat_l2"]
        else:
            current = val_metrics["mpjpe"]
        if current + min_delta < best:
            best = current
            bad_epochs = 0
            save_checkpoint(
                best_path,
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metric": best,
                },
            )
        else:
            if patience > 0:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early_stop_epoch={epoch}")
                    break

    if test_loader is not None:
        if os.path.isfile(best_path):
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state["model"])
        test_metrics = run_eval(
            model,
            test_loader,
            device,
            gt_scaler=scalers.get("gt") if scalers is not None else None,
            target_fps=target_fps
        )
        logger.log({"epoch": "final", **test_metrics})
        if target_type == "quaternion":
            print(f"test_quat_l2={test_metrics['quat_l2']:.6f}")
            print(f"test_mpjae={test_metrics['mpjae']:.6f}")
            print(f"best_val_quat_l2={best:.6f}")
        else:
            print(
                f"test_mpjpe={test_metrics['mpjpe']:.6f} "
                f"test_mpjve={test_metrics['mpjve']:.6f} "
                f"test_jitter={test_metrics['jitter']:.6f} "
            )
            print(f"best_val_mpjpe={best:.6f}")


if __name__ == "__main__":
    main()
