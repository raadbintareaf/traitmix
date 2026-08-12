#!/usr/bin/env python3
"""
validate_induction_per_model.py — Did the personality manipulation take effect in this model?

The induction gate (E0) was administered only to the primary model. Every claim about a
second, third or fourth model presupposes that its agents actually carried the traits they
were given, and that presupposition is currently untested. It matters most for the one model
whose response surface has a different shape: if trait induction is weak there, that is the
explanation, and it is a far better one than an unexplained exception.

The script administers the full IPIP-NEO-120 to the nine trait configurations used in the
response surface, scores them by the published keying, and reports the convergent validity
against the targeted levels. It runs no simulation, so it takes minutes rather than hours.

Usage, with the model already being served:
    python validate_induction_per_model.py --tag qwen7
    python validate_induction_per_model.py --tag qwen7 --reps 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="condition prefix without 'e3', e.g. qwen7 for e3qwen7_*")
    ap.add_argument("--reps", type=int, default=1,
                    help="administrations per configuration; 2 gives a reliability check")
    args = ap.parse_args()

    from traitmix.utils import load_config
    from traitmix import personality as pers
    from traitmix.questionnaire import administer, convergent_validity
    from traitmix.data import load_ipip
    from traitmix.llm import make_llm

    cfg_dir = (ROOT / "configs" / "e3") if args.tag == "base" \
        else (ROOT / "configs" / f"e3_{args.tag}")
    if not cfg_dir.is_dir():
        avail = sorted(p.name for p in (ROOT / "configs").glob("e3*") if p.is_dir())
        sys.exit(f"{cfg_dir} not found. Available: {avail}")
    cfgs = sorted(cfg_dir.glob("*.yaml"))
    if not cfgs:
        sys.exit(f"{cfg_dir} contains no configurations")

    items, inventory_name = load_ipip()
    print(f"inventory : {inventory_name}, {len(items)} items")
    if len(items) < 100:
        sys.exit("Loaded the demo battery, not the IPIP-NEO-120. Check "
                 "data/ipip/ipip_neo_120.csv exists before validating induction.")
    first = load_config(cfgs[0])
    llm = make_llm(first.get("llm", {}))
    model_id = first["llm"]["model"]
    print(f"directory : {cfg_dir.name}")
    print(f"model     : {model_id}")
    print(f"configs   : {len(cfgs)}  reps: {args.reps}\n")

    targets, measured, rows = [], [], []
    for c in cfgs:
        cfg = load_config(c)
        mu = cfg["composition"]["mu"]
        theta = np.array([mu[t] for t in pers.TRAITS], dtype=float)
        persona = {"name": "Alex_0", "age": 34, "occ": "nurse"}
        got = administer(llm, theta, persona, items, reps=args.reps)
        m = np.array([got[t] for t in pers.TRAITS], dtype=float)
        targets.append(theta); measured.append(m)
        rows.append({"config": cfg["name"],
                     **{f"target_{t}": theta[k] for k, t in enumerate(pers.TRAITS)},
                     **{f"measured_{t}": m[k] for k, t in enumerate(pers.TRAITS)}})
        print(f"  {cfg['name']:22s} targeted O={theta[0]:.1f} A={theta[3]:.1f}  "
              f"measured O={got['openness']:.2f} A={got['agreeableness']:.2f}")

    from scipy import stats as sps
    D = pd.DataFrame(rows)
    GATE_RHO, GATE_DELTA = 0.60, 1.00
    print(f"\n{'trait':18s}{'rho':>8s}{'delta':>8s}{'lo':>7s}{'hi':>7s}   verdict")
    summary, failures = {}, []
    for t in pers.TRAITS:
        x, y = D[f"target_{t}"], D[f"measured_{t}"]
        if x.nunique() < 2:
            continue                       # trait not manipulated in this design
        rho = sps.spearmanr(x, y).statistic
        lo, hi = y[x == x.min()].mean(), y[x == x.max()].mean()
        delta = hi - lo
        ok = (rho >= GATE_RHO) and (delta >= GATE_DELTA)
        why = ("passes" if ok else
               "FAILS magnitude" if rho >= GATE_RHO else
               "FAILS rank" if delta >= GATE_DELTA else "FAILS both")
        if not ok:
            failures.append(t)
        summary.update({f"rho_{t}": rho, f"delta_{t}": delta})
        print(f"  {t:16s}{rho:>8.3f}{delta:>8.2f}{lo:>7.2f}{hi:>7.2f}   {why}")

    print(f"\n  gate: rho >= {GATE_RHO:.2f} AND delta >= {GATE_DELTA:.2f} scale points, "
          f"applied per manipulated trait")
    print("  (rho tests whether the configurations are ordered correctly; delta tests "
          "whether\n   they are separated enough to change behaviour. Rank alone is not "
          "sufficient.)")
    if failures:
        print(f"\n  RESULT: induction inadequate for {', '.join(failures)}.")
        print("  A flat or anomalous response surface is then most likely a consequence of")
        print("  weak induction rather than a property of composition.")
    else:
        print("\n  RESULT: induction adequate on every manipulated trait. An anomalous")
        print("  response surface here is a substantive finding, not an artifact.")

    out = RES / f"induction_{args.tag}.csv"
    D.to_csv(out, index=False)
    summ = RES / "induction_by_model.csv"
    rec = {"tag": args.tag, "model": model_id, "reps": args.reps, **summary,
           "failed_traits": ",".join(failures) or "none"}
    if summ.exists():
        prev = pd.read_csv(summ)
        prev = prev[prev.tag != args.tag]
        pd.concat([prev, pd.DataFrame([rec])], ignore_index=True).to_csv(summ, index=False)
    else:
        pd.DataFrame([rec]).to_csv(summ, index=False)
    print(f"\nwrote {out} and {summ}")


if __name__ == "__main__":
    main()
