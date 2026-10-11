"""Paths and small helpers shared by the experiment scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "results" / "figures"
METRIC_DIR = ROOT / "results" / "metrics"
CKPT_DIR = ROOT / "checkpoints"
for d in (FIG_DIR, METRIC_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def save_json(obj, name: str) -> Path:
    path = METRIC_DIR / name
    path.write_text(json.dumps(_to_builtin(obj), indent=2))
    return path


def load_json(name: str):
    return json.loads((METRIC_DIR / name).read_text())
