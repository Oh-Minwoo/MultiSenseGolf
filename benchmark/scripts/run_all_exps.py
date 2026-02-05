import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


def _read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    if not os.path.isfile(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _extract_metrics(records: List[Dict]) -> Dict[str, Optional[float]]:
    best_val_mpjpe = None
    best_val_mpjve = None
    best_val_jitter = None
    for r in records:
        if isinstance(r.get("epoch"), int):
            if "mpjpe" in r:
                if best_val_mpjpe is None or r["mpjpe"] < best_val_mpjpe:
                    best_val_mpjpe = r["mpjpe"]
            if "mpjve" in r:
                if best_val_mpjve is None or r["mpjve"] < best_val_mpjve:
                    best_val_mpjve = r["mpjve"]
            if "jitter" in r:
                if best_val_jitter is None or r["jitter"] < best_val_jitter:
                    best_val_jitter = r["jitter"]
    test_mpjpe = None
    test_mpjve = None
    test_jitter = None
    test_vel = None
    for r in records:
        if r.get("epoch") == "final":
            test_mpjpe = r.get("mpjpe")
            test_mpjve = r.get("mpjve")
            test_jitter = r.get("jitter")
    return {
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_mpjve": best_val_mpjve,
        "best_val_jitter": best_val_jitter,
        "test_mpjpe": test_mpjpe,
        "test_mpjve": test_mpjve,
        "test_jitter": test_jitter,
    }


def _iter_configs(config_dir: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith(".yaml"):
            continue
        paths.append(os.path.join(config_dir, name))
    return paths


def _run_one(config_path: str, device: Optional[str], run_test: bool, output_root: str) -> Dict:
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    out_dir = os.path.join(output_root, exp_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = [
        sys.executable,
        "benchmark/src/train/train.py",
        "--config",
        config_path,
        "--output-dir",
        out_dir,
    ]
    if device:
        cmd += ["--device", device]
    if run_test:
        cmd += ["--run-test"]
    print(f"running={exp_name}")
    subprocess.run(cmd, check=True, env=env)
    metrics_path = os.path.join(out_dir, "logs", "metrics.jsonl")
    metrics = _extract_metrics(_read_jsonl(metrics_path))
    return {"exp": exp_name, "output_dir": out_dir, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="benchmark/configs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--output-root", default="benchmark/outputs")
    parser.add_argument("--summary-path", default="benchmark/outputs/summary.json")
    args = parser.parse_args()

    configs = _iter_configs(args.config_dir)
    if len(configs) == 0:
        print("no_configs_found")
        return

    results = []
    for cfg in configs:
        results.append(_run_one(cfg, args.device, args.run_test, args.output_root))

    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"summary_saved={args.summary_path}")


if __name__ == "__main__":
    main()
