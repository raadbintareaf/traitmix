"""Mid-run checkpointing: resume an interrupted simulation at the last completed round."""
import pickle, random
from pathlib import Path
import numpy as np
from .utils import CKPT

KEEP_LAST = 2

def ckpt_dir(rid):
    d = CKPT / rid; d.mkdir(parents=True, exist_ok=True); return d

def save(rid, round_idx, payload: dict):
    payload = dict(payload, _py_rng=random.getstate(), _np_rng=np.random.get_state(), _round=round_idx)
    p = ckpt_dir(rid) / f"round_{round_idx:04d}.ckpt"
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    for old in sorted(ckpt_dir(rid).glob("round_*.ckpt"))[:-KEEP_LAST]:
        old.unlink()

def latest(rid):
    d = CKPT / rid
    files = sorted(d.glob("round_*.ckpt")) if d.exists() else []
    if not files:
        return None
    with open(files[-1], "rb") as f:
        payload = pickle.load(f)
    random.setstate(payload.pop("_py_rng")); np.random.set_state(payload.pop("_np_rng"))
    return payload

def clear(rid):
    d = CKPT / rid
    if d.exists():
        for f in d.glob("round_*.ckpt"): f.unlink()
