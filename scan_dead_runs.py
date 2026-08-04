#!/usr/bin/env python3
"""
scan_dead_runs.py — Detect runs that completed but produced no real agent behaviour.

WHY: the LLM client used to catch API errors and return "PASS (LLM_ERROR: ...)", so a run
against a misconfigured server (e.g. wrong --served-model-name -> HTTP 404) would finish
normally and write a results row in which every agent did nothing. Such rows look valid but
contain no simulation. The client now raises instead (see llm.py MAX_ERROR_RATE), but runs
completed before that fix must be screened.

SIGNATURES OF A DEAD RUN
  * distinct2 / self_bleu3 missing or NaN     -> fewer than ~5 posts were ever written
  * crosscut_rate NaN                         -> no replies across opinion camps
  * trait_drift_mean approximately equal to mean|0.5 - theta|  -> the trait scorer saw empty
    text and returned its 0.5 default, so "drift" is just the design distance from neutral
  * llm_errors > 0 (only recorded for runs made after the fix)

Usage (from repo root):  python scan_dead_runs.py
Optionally:              python scan_dead_runs.py --delete   (removes flagged rows, with backup)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "results" / "raw_results.jsonl"


def expected_dead_drift(row, realized: pd.DataFrame | None) -> float:
    """mean|0.5 - theta| for this run's composition: what drift equals when nothing is posted."""
    if realized is None:
        return np.nan
    m = realized[(realized.config == row["config"]) & (realized.seed == row["seed"])]
    if m.empty:
        return np.nan
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    mus = [m.iloc[0].get(f"real_mu_{t}", np.nan) for t in traits]
    sds = [m.iloc[0].get(f"real_sd_{t}", np.nan) for t in traits]
    # E|X-0.5| for X~N(mu,sd) approximated by |mu-0.5| plus the half-normal term
    vals = [abs(mu - 0.5) + 0.7979 * sd * np.exp(-((mu - 0.5) ** 2) / (2 * sd ** 2 + 1e-9))
            if np.isfinite(mu) and np.isfinite(sd) else np.nan for mu, sd in zip(mus, sds)]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="remove flagged rows (keeps a backup)")
    ap.add_argument("--drift-tol", type=float, default=0.02)
    args = ap.parse_args()

    if not RAW.exists():
        sys.exit(f"{RAW} not found")
    df = pd.read_json(RAW, lines=True)
    print(f"scanning {len(df)} runs across {df.config.nunique()} configs\n")

    rp = ROOT / "results" / "realized_traits.csv"
    realized = pd.read_csv(rp) if rp.exists() else None
    if realized is None:
        print("(results/realized_traits.csv missing - run diagnose_confounds.py for the "
              "drift-signature check; structural checks still apply)\n")

    flags = []
    for _, r in df.iterrows():
        reasons = []
        if "llm_errors" in df.columns and pd.notna(r.get("llm_errors")) and r.get("llm_errors", 0) > 0:
            reasons.append(f"llm_errors={int(r['llm_errors'])}")
        if pd.isna(r.get("distinct2")):
            reasons.append("no posts (distinct2 NaN)")
        if pd.isna(r.get("crosscut_rate")):
            reasons.append("no cross-camp replies")
        exp = expected_dead_drift(r, realized)
        if np.isfinite(exp) and pd.notna(r.get("trait_drift_mean")):
            if abs(r["trait_drift_mean"] - exp) < args.drift_tol:
                reasons.append(f"drift={r['trait_drift_mean']:.3f} matches empty-text "
                               f"prediction {exp:.3f}")
        if reasons:
            flags.append({"run_id": r["run_id"], "config": r["config"], "seed": r["seed"],
                          "runtime_s": r.get("runtime_s"), "reasons": "; ".join(reasons)})

    if not flags:
        print("CLEAN - no runs show signs of silent LLM failure.")
        return
    f = pd.DataFrame(flags)
    print(f"FLAGGED {len(f)} of {len(df)} runs:\n")
    print(f.to_string(index=False))
    print("\nBy config:")
    print(f.config.value_counts().to_string())

    if args.delete:
        shutil.copy(RAW, RAW.with_suffix(".jsonl.bak"))
        bad = set(f.run_id)
        kept = [l for l in RAW.read_text().splitlines()
                if l.strip() and json.loads(l)["run_id"] not in bad]
        RAW.write_text("\n".join(kept) + "\n")
        reg = ROOT / "results" / "registry.json"
        if reg.exists():
            state = json.loads(reg.read_text())
            for rid in bad:
                state.pop(rid, None)
            reg.write_text(json.dumps(state, indent=1))
        print(f"\nremoved {len(bad)} rows -> {len(kept)} remain "
              f"(backup: {RAW.with_suffix('.jsonl.bak').name}); registry entries cleared "
              f"so they can be re-run.")
    else:
        print("\nRe-run with --delete to remove these rows (a backup is kept).")


if __name__ == "__main__":
    main()
