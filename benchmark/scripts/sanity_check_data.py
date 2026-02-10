import argparse

from src.data.dataset import scan_dataset, SwingDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. D:\\caddie_final\\Data")
    parser.add_argument("--include-pressure", action="store_true")
    parser.add_argument("--include-fpv", action="store_true")
    parser.add_argument("--fpv-feature-name", default="FPV_RGB.clip_vit_b16.npz")
    args = parser.parse_args()

    samples, counts = scan_dataset(
        args.root,
        include_pressure=args.include_pressure,
        include_fpv=args.include_fpv,
        fpv_feature_name=args.fpv_feature_name,
        verbose=True,
    )

    print(f"usable_samples={len(samples)}")
    print(f"subjects={len(counts)}")

    if len(samples) == 0:
        return

    # Show one sample shapes
    ds = SwingDataset(
        args.root,
        include_pressure=args.include_pressure,
        include_fpv=args.include_fpv,
        fpv_feature_name=args.fpv_feature_name,
    )
    item = ds[0]
    for key in ["imu", "pressure", "fpv", "gt"]:
        if key in item:
            print(f"{key}_shape={item[key].shape}")
    print(f"time_shape={item['time_s'].shape}")
    print(f"sample_subject={item['subject']}")
    print(f"sample_swing={item['swing']}")


if __name__ == "__main__":
    main()
