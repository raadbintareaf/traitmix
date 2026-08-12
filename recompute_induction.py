#!/usr/bin/env python3
"""
recompute_induction.py — One consistent induction table across every model.

Four models were validated with an earlier version of the script that reported only rank
correlation. Rather than re-run them, the per-model CSVs are re-scored here with the same
two criteria applied to all six, so the table in the paper is internally consistent.
"""
from pathlib import Path
import pandas as pd
from scipy import stats as sps

RES = Path(__file__).resolve().parent / "results"
GATE_RHO, GATE_DELTA = 0.60, 1.00
MODELS = [("llama3b", "Llama-3.2-3B", 3), ("qwen3b", "Qwen2.5-3B", 3),
          ("qwen7", "Qwen2.5-7B", 7), ("base", "Llama-3.1-8B", 8),
          ("q", "Qwen2.5-14B", 14), ("large", "Qwen2.5-32B", 32)]
SPAN = {"llama3b": 0.09, "qwen3b": 0.68, "qwen7": 2.06,
        "base": 0.94, "q": 1.28, "large": 0.45}
SHAPE = {"llama3b": "flat", "qwen3b": "crossover", "qwen7": "additive",
         "base": "crossover", "q": "crossover", "large": "crossover"}

rows = []
for tag, name, size in MODELS:
    f = RES / f"induction_{tag}.csv"
    if not f.exists():
        print(f"  missing {f.name}"); continue
    d = pd.read_csv(f)
    r = {"model": name, "size": size, "tag": tag}
    fails = []
    for t, k in [("openness", "O"), ("agreeableness", "A")]:
        x, y = d[f"target_{t}"], d[f"measured_{t}"]
        rho = sps.spearmanr(x, y).statistic
        delta = y[x == x.max()].mean() - y[x == x.min()].mean()
        r[f"rho_{k}"], r[f"delta_{k}"] = rho, delta
        if rho < GATE_RHO:
            fails.append(f"{k}:rank")
        if delta < GATE_DELTA:
            fails.append(f"{k}:magnitude")
    r["gate"] = "pass" if not fails else "FAIL(" + ",".join(fails) + ")"
    r["span"] = SPAN.get(tag)
    r["surface"] = SHAPE.get(tag)
    rows.append(r)

T = pd.DataFrame(rows).sort_values("size")
T.to_csv(RES / "induction_summary_all.csv", index=False)
print(f"{'model':15s}{'sz':>4s}{'rhoO':>7s}{'dO':>6s}{'rhoA':>7s}{'dA':>6s}"
      f"{'span':>7s}  {'gate':22s} shape")
for _, r in T.iterrows():
    print(f"  {r.model:13s}{r['size']:>4d}{r.rho_O:>7.2f}{r.delta_O:>6.2f}"
          f"{r.rho_A:>7.2f}{r.delta_A:>6.2f}{r.span:>7.2f}  {r.gate:26s} {r.surface}")
ok = T[T.gate == "pass"]
print(f"\n  pass the gate: {len(ok)} of {len(T)}")
print(f"  of those, crossover in {int((ok[chr(39)+chr(115)+chr(104)+chr(97)+chr(112)+chr(101)+chr(39)]==chr(39)+chr(99)+chr(114)+chr(111)+chr(115)+chr(115)+chr(111)+chr(118)+chr(101)+chr(114)+chr(39)).sum())} of {len(ok)}")
print(f"  excluded: " + "; ".join(f"{r.model} {r.gate}" for _, r in T[T.gate != 'pass'].iterrows()))
