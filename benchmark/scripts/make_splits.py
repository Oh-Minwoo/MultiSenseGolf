import argparse
import os
from typing import Dict, List

from src.data.dataset import scan_dataset
from src.data.splits import SubjectSplit, make_subject_split, save_split


def _list_subjects(root: str) -> List[str]:
    subjects = []
    for name in sorted(os.listdir(root)):
        if name.lower().startswith("sub") and os.path.isdir(os.path.join(root, name)):
            subjects.append(name)
    return subjects


def _count_samples_by_subject(root: str, include_pressure: bool, include_fpv: bool) -> Dict[str, int]:
    _, counts = scan_dataset(
        root,
        include_pressure=include_pressure,
        include_fpv=include_fpv,
        verbose=False,
    )
    return counts


def _print_split_stats(split: SubjectSplit, counts: Dict[str, int]) -> None:
    def group_total(subs: List[str]) -> int:
        return sum(counts.get(s, 0) for s in subs)

    print(f"train_subjects={len(split.train)} train_samples={group_total(split.train)}")
    print(f"val_subjects={len(split.val)} val_samples={group_total(split.val)}")
    print(f"test_subjects={len(split.test)} test_samples={group_total(split.test)}")
    for name in ["train", "val", "test"]:
        subs = getattr(split, name)
        for s in subs:
            print(f"{name}:{s}={counts.get(s, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. D:\\caddie_final\\Data")
    parser.add_argument("--out", required=True, help="Output JSON path, e.g. configs\\split.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-pressure", action="store_true")
    parser.add_argument("--include-fpv", action="store_true")
    args = parser.parse_args()

    subjects = _list_subjects(args.root)
    if len(subjects) == 0:
        print("no_subjects_found")
        return

    split = make_subject_split(
        subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_split(args.out, split)

    counts = _count_samples_by_subject(
        args.root, include_pressure=args.include_pressure, include_fpv=args.include_fpv
    )
    _print_split_stats(split, counts)
    print(f"saved_split={args.out}")


if __name__ == "__main__":
    main()
