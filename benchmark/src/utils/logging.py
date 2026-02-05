from __future__ import annotations

import json
import os
from typing import Any, Dict


class SimpleLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "metrics.jsonl")

    def log(self, data: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
