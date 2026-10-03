from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier, XGBRegressor

from data import ALL_TARGETS, REGRESSION_TARGET, target_array

warnings.filterwarnings(
    "ignore", message=".*Falling back to prediction using DMatrix.*", category=UserWarning
)



def _inner_splits(cohort: pd.DataFrame, outer_train_idx: np.ndarray, outer_fold: int, target: str, seed: int):
    train = cohort.iloc[outer_train_idx]
    ids_by_skill = {
        skill: sorted(group["participant_id"].astype(int).unique())
        for skill, group in train.groupby("skill_group")
    }
    y = target_array(train, target)
    best_score, best_assign = np.inf, None
    rng = np.random.default_rng(seed + outer_fold * 101 + ALL_TARGETS.index(target) * 1009)
    for _ in range(2000):
        assignment: Dict[int, int] = {}
        for ids in ids_by_skill.values():
            shuffled = rng.permutation(ids)
            for i, participant in enumerate(shuffled):
                assignment[int(participant)] = i % 3
        score = 0.0
        valid = True
        global_counts = np.bincount(y.astype(int), minlength=3) / 3 if target != REGRESSION_TARGET else None
        for inner_fold in range(3):
            mask = train["participant_id"].map(assignment).to_numpy(int) == inner_fold
            if target != REGRESSION_TARGET:
                counts = np.bincount(y[mask].astype(int), minlength=3)
                if np.any(counts == 0) or np.any(np.bincount(y[~mask].astype(int), minlength=3) == 0):
                    valid = False
                    break
                score += float(np.sum(((counts - global_counts) / np.maximum(global_counts, 1)) ** 2))
            else:
                score += float((mask.sum() - len(mask) / 3) ** 2 / max(len(mask), 1))
        if valid and score < best_score:
            best_score, best_assign = score, assignment
    if best_assign is None:
        raise RuntimeError(f"Could not create class-complete inner folds for outer={outer_fold}, target={target}")
    result = []
    participant_fold = train["participant_id"].map(best_assign).to_numpy(int)
    for inner_fold in range(3):
        val_local = np.flatnonzero(participant_fold == inner_fold)
        fit_local = np.flatnonzero(participant_fold != inner_fold)
        fit_idx, val_idx = outer_train_idx[fit_local], outer_train_idx[val_local]
        fit_ids = set(cohort.iloc[fit_idx]["participant_id"].astype(int))
        val_ids = set(cohort.iloc[val_idx]["participant_id"].astype(int))
        assert not (fit_ids & val_ids)
        result.append((fit_idx, val_idx))
    return result


class BlockPreprocessor:
    def __init__(self):
        self.steps: List[Tuple[SimpleImputer, VarianceThreshold | None, StandardScaler]] = []
        self.output_dimensions: List[int] = []

    def fit_transform(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        transformed = []
        for block in blocks:
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x = imputer.fit_transform(block)
            selector: VarianceThreshold | None = VarianceThreshold(0.0)
            try:
                x = selector.fit_transform(x)
            except ValueError:
                selector = None
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            self.steps.append((imputer, selector, scaler))
            self.output_dimensions.append(int(x.shape[1]))
            transformed.append(x.astype(np.float32))
        return np.concatenate(transformed, axis=1)

    def transform(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        transformed = []
        for block, (imputer, selector, scaler) in zip(blocks, self.steps):
            x = imputer.transform(block)
            if selector is not None:
                x = selector.transform(x)
            transformed.append(scaler.transform(x).astype(np.float32))
        return np.concatenate(transformed, axis=1)


def _candidate_list(cfg: Mapping[str, Any], target: str, family: str):
    if family == "linear":
        key = "ridge_alpha" if target == REGRESSION_TARGET else "logistic_c"
        param = "alpha" if target == REGRESSION_TARGET else "C"
        return [{param: float(value)} for value in cfg["models"][key]]
    return [dict(x) for x in cfg["models"]["xgboost_candidates"]]


def _make_model(cfg: Mapping[str, Any], target: str, family: str, params: Mapping[str, Any], seed: int):
    if target == REGRESSION_TARGET and family == "linear":
        return Ridge(
            alpha=float(params["alpha"]), solver="lsqr",
            max_iter=5000,
            tol=1e-5,
        )
    if target != REGRESSION_TARGET and family == "linear":
        return LogisticRegression(
            C=float(params["C"]), class_weight="balanced", solver="lbfgs",
            max_iter=5000,
            tol=1e-3, random_state=seed,
        )
    common = dict(params)
    common.update({
        "random_state": seed, "n_jobs": int(cfg["models"]["xgboost_n_jobs"]),
        "tree_method": "hist", "device": cfg["models"]["xgboost_device"],
    })
    if target == REGRESSION_TARGET:
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **common)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=3)
    weights = len(y) / (3 * np.maximum(counts, 1))
    return weights[y.astype(int)]


def _fit(model: Any, x: np.ndarray, y: np.ndarray, target: str, family: str):
    if target != REGRESSION_TARGET and family == "xgboost":
        model.fit(x, y, sample_weight=_balanced_sample_weights(y))
    else:
        model.fit(x, y)
    return model


def _predict(model: Any, x: np.ndarray, target: str):
    if target == REGRESSION_TARGET:
        return model.predict(x), None
    probabilities = model.predict_proba(x)
    if probabilities.shape[1] != 3:
        full = np.zeros((len(x), 3), dtype=float)
        full[:, np.asarray(model.classes_, dtype=int)] = probabilities
        probabilities = full
    return np.argmax(probabilities, axis=1), probabilities


def calculate_metrics(y: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray | None, target: str):
    if target == REGRESSION_TARGET:
        return {
            "r2": float(r2_score(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
        }
    binary = label_binarize(y, classes=[0, 1, 2])
    try:
        auc = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {
        "auc_macro_ovr": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def _selection_value(metrics: Mapping[str, float], target: str) -> float:
    if target == REGRESSION_TARGET:
        return -float(metrics["rmse"])
    return float(np.nanmean([metrics["auc_macro_ovr"], metrics["balanced_accuracy"], metrics["macro_f1"]]))
