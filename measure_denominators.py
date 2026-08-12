#!/usr/bin/env python3
"""
measure_denominators.py — Counts behind the segregation rates.

Reviewer point R9 observes that cross-cutting interaction and the E-I index are ratios
whose denominators are never reported, and that when a society converges the denominator
can become very small. A rate of zero computed over three opposing-sign pairs is a
different statement from a rate of zero computed over nine hundred, and the article should
not present them identically.

This script re-runs a small set of conditions with checkpoints retained and recovers, from
the actual post and opinion records:

  reply_total          replies exchanged between agents, of any kind
  reply_cross          replies between agents holding opposing-sign opinions
  reply_within         replies between agents holding same-sign opinions
  pairs_opposing       ordered pairs of agents holding opposing-sign opinions
  pairs_possible       ordered pairs of agents holding non-zero opinions
  frac_nonzero         share of agents holding a non-zero opinion at the end

The conditions chosen span the range that matters: the baseline, the most converged
condition in the study, and the most divided one. If the denominator collapses anywhere,
it collapses under high Agreeableness.

Usage:  python measure_denominators.py
        python measure_denominators.py --runs 2 --conditions e1_norm_baseline,e1_agreeableness_high
"""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

DEFAULT = ["e1_norm_baseline", "e1_agreeableness_high", "e1_neuroticism_high",
           "e2_homog", "e2_diverse"]
WHERE = {"e1_norm_baseline": "e1", "e1_agreeableness_high": "e1",
         "e1_neuroticism_high": "e1", "e2_homog": "e2", "e2_diverse": "e2"}


def counts_from_state(st):
    """Denominators, computed from the same records the rates are computed from."""
    ops = st.get("opinions", {})
    topics = [t for t in ops if "filler" not in str(t).lower()]
    posts = st.get("posts", [])
    by_id = {p["id"]: p for p in posts}

    reply_total = reply_cross = reply_within = 0
    for p in posts:
        par = p.get("reply_to")
        if par is None or par not in by_id:
            continue
        a, b = p.get("author"), by_id[par].get("author")
        top = p.get("topic")
        if a is None or b is None or a == b or top not in ops:
            continue
        oa, ob = ops[top].get(a), ops[top].get(b)
        if oa is None or ob is None:
            continue
        reply_total += 1
        if np.sign(oa) != 0 and np.sign(ob) != 0:
            if np.sign(oa) != np.sign(ob):
                reply_cross += 1
            else:
                reply_within += 1

    pairs_opp = pairs_poss = 0
    nonzero = []
    for t in topics:
        vals = ops[t]
        nz = [a for a, v in vals.items() if v != 0]
        nonzero.append(len(nz) / max(len(vals), 1))
        for a, b in permutations(nz, 2):
            pairs_poss += 1
            if np.sign(vals[a]) != np.sign(vals[b]):
                pairs_opp += 1
    return dict(reply_total=reply_total, reply_cross=reply_cross,
                reply_within=reply_within, pairs_opposing=pairs_opp,
                pairs_possible=pairs_poss,
                frac_nonzero=float(np.mean(nonzero)) if nonzero else np.nan,
                n_posts=len(posts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--conditions", default=",".join(DEFAULT))
    ap.add_argument("--start-seed", type=int, default=911,
                    help="seeds outside the experimental design")
    args = ap.parse_args()

    from traitmix.utils import load_config
    from traitmix.runner import run_one
    from traitmix.classifier import make_scorer
    from traitmix import checkpointing as ck

    scorer = make_scorer()
    rows = []
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cond in conds:
        sub = WHERE.get(cond, cond.split("_")[0])
        path = ROOT / "configs" / sub / f"{cond}.yaml"
        if not path.exists():
            print(f"  {cond}: config not found, skipped"); continue
        cfg = load_config(path)
        for k in range(args.runs):
            seed = args.start_seed + k
            print(f"  {cond} seed {seed} ...", flush=True)
            r = run_one(cfg, seed=seed, force=True, trait_scorer=scorer, keep_ckpt=True)
            reg = json.loads((RES / "registry.json").read_text())
            rid = max((key for key in reg if key.startswith(cond) and key.endswith(f"_s{seed}")),
                      default=None)
            st = ck.latest(rid) if rid else None
            if st is None:
                print("     checkpoint not recoverable; skipped"); continue
            c = counts_from_state(st)
            c.update(config=cond, seed=seed,
                     crosscut_rate=r.get("crosscut_rate"), pol_ei=r.get("pol_ei"),
                     pol_var=r.get("pol_var"))
            rows.append(c)

    if not rows:
        sys.exit("no runs produced a recoverable checkpoint")
    D = pd.DataFrame(rows)
    G = D.groupby("config").agg(
        runs=("seed", "count"), pol_var=("pol_var", "mean"),
        crosscut_rate=("crosscut_rate", "mean"),
        reply_total=("reply_total", "mean"), reply_cross=("reply_cross", "mean"),
        pairs_opposing=("pairs_opposing", "mean"), pairs_possible=("pairs_possible", "mean"),
        frac_nonzero=("frac_nonzero", "mean")).reset_index()
    G["pct_pairs_opposing"] = 100 * G.pairs_opposing / G.pairs_possible.replace(0, np.nan)
    D.to_csv(RES / "denominators_runs.csv", index=False)
    G.to_csv(RES / "denominators.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== DENOMINATORS BEHIND THE SEGREGATION RATES ===")
    print(G.round(3).to_string(index=False))
    print("\n  reply_cross is the numerator of the cross-cutting rate; reply_total its")
    print("  denominator. pairs_opposing counts the ordered pairs that could in principle")
    print("  have produced a cross-cutting exchange. Where the latter is small, a rate")
    print("  near zero carries little information, and the article should say so.")
    print(f"\nwrote {RES/'denominators.csv'} and {RES/'denominators_runs.csv'}")
    print("These runs use seeds outside the design and are excluded from the analysis.")


if __name__ == "__main__":
    main()
