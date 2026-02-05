from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SubjectSplit:
    train: List[str]
    val: List[str]
    test: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def make_subject_split(
    subjects: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SubjectSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("ratios_must_sum_to_1")
    subjects = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train = subjects[:n_train]
    val = subjects[n_train : n_train + n_val]
    test = subjects[n_train + n_val :]
    return SubjectSplit(train=train, val=val, test=test)


def save_split(path: str, split: SubjectSplit) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_split(path: str) -> SubjectSplit:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SubjectSplit(train=data["train"], val=data["val"], test=data["test"])


def count_by_subject(
    samples: Sequence[Tuple[str, str]]
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for subject, _ in samples:
        counts[subject] = counts.get(subject, 0) + 1
    return counts
