import argparse
import os
import json
from typing import Dict, Any

import h5py
import numpy as np


def _resolve_h5_path(root: str, subject: str, swing: str) -> str:
    subject = str(subject)
    swing = str(swing)
    if not subject.lower().startswith("sub"):
        subject = f"Sub{int(subject):02d}"
    if not swing.lower().startswith("swing"):
        swing = f"Swing{int(swing):02d}"
    fname = f"{subject.lower()}_{swing}_stream_data.hdf5"
    return os.path.join(root, subject, swing, fname)


def _summarize_dataset(name: str, ds: h5py.Dataset) -> str:
    shape = ds.shape
    dtype = ds.dtype
    return f"{name}: shape={shape} dtype={dtype}"


def _load_all_datasets(h5: h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()]

    h5.visititems(_visit)
    return out


def _pick_sample_keys(keys: list[str]) -> list[str]:
    def _first_match(substrings: list[str]) -> str | None:
        for k in keys:
            lk = k.lower()
            if any(s in lk for s in substrings):
                return k
        return None

    picks = []
    for group in [
        ["pns-joint", "pns_joint"],
        ["pressure"],
        ["gaze", "pupil", "eye"],
    ]:
        m = _first_match(group)
        if m and m not in picks:
            picks.append(m)
    return picks


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"S", "O"}:
            return _to_jsonable(x.tolist())
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--swing", required=True)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    path = _resolve_h5_path(args.data_root, args.subject, args.swing)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"not_found: {path}")
    
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        print(f"h5_path={path}")
        print(f"root_keys={keys}")

        all_data = _load_all_datasets(h5)
        for name in sorted(all_data.keys()):
            ds = h5[name]
            print(_summarize_dataset(name, ds))

        print("Samples:")
        sample_keys = _pick_sample_keys(sorted(all_data.keys()))
        if not sample_keys:
            sample_keys = sorted(all_data.keys())[:3]
        for k in sample_keys:
            v = all_data[k]
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape} dtype={v.dtype} sample={v.ravel()[:6].tolist()}")
            else:
                print(f"  {k}: {v}")

        if args.save_dir is not None:
            payload = {k: _to_jsonable(v) for k, v in all_data.items()}
            os.makedirs(args.save_dir or ".", exist_ok=True)
            filename = f"{args.subject}_{args.swing}.json"
            save_path = os.path.join(args.save_dir, filename)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
