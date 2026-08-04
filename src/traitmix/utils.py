"""Seeding, config handling, run identity, run registry. Single source of truth."""
import hashlib, json, random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)

def load_config(path, overrides=None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("inherit", None)
    if base:
        cfg = _deep_merge(load_config(Path(path).parent / base), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]

def run_id(cfg: dict, seed: int) -> str:
    return f"{cfg['name']}_{config_hash(cfg)}_s{seed}"

class Registry:
    """Record of run status; enables skip-if-done resumption across notebook restarts."""
    def __init__(self, path=None):
        self.path = Path(path or RESULTS / "registry.json")
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}
    def done(self, rid): return self.state.get(rid, {}).get("status") == "done"
    def mark(self, rid, status="done", **kw):
        self.state[rid] = {"status": status, **kw}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1))

def append_row(row: dict, path=None):
    path = Path(path or RESULTS / "raw_results.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")

def read_rows(path=None):
    import pandas as pd
    path = Path(path or RESULTS / "raw_results.jsonl")
    if path.exists():
        return pd.read_json(path, lines=True)
    legacy = path.with_suffix(".csv")
    return pd.read_csv(legacy) if legacy.exists() else pd.DataFrame()
