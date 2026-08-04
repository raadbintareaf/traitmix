#!/usr/bin/env python3
"""
audit_sign_stability.py — E5 robustness audit, done correctly.

Each anchor condition must be compared against the baseline UNDER THE SAME PERTURBATION
(e5_<anchor>__<pert> vs e5_norm_baseline__<pert>), not against the unperturbed baseline -
otherwise the perturbation's own main effect contaminates the contrast.

For every (anchor, metric) it reports the effect direction under each perturbation plus the
unperturbed reference, and a sign-stability fraction. A finding is called ROBUST only if the
sign holds in every cell.

Outputs: results/audit_sign_stability.csv and paper/tables/table_e5_audit.tex

Usage (from repo root):  python audit_sign_stability.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, TAB = ROOT / "results", ROOT / "paper" / "tables"

ANCHOR_MAIN = {"norm_baseline": "e1_norm_baseline",
               "openness_high": "e1_openness_high",
               "neuroticism_high": "e1_neuroticism_high"}
METRICS = ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei",
           "CI_accuracy_z", "CI_hidden_profile_rate"]
LABEL = {"pol_var": "Opinion var.", "pol_extremity": "Extremity",
         "crosscut_rate": "Cross-cut rate", "pol_ei": "Echo closure",
         "CI_accuracy_z": "Collective acc.", "CI_hidden_profile_rate": "Hidden profile"}
PERT_LABEL = {"ws": "small-world", "er": "random graph", "rho_lo": "activation 0.2",
              "rho_hi": "activation 0.6", "mem": "memory 20", "feed": "feed 15",
              "temp": "temperature 1.0", "qwen": "Qwen-14B", "gemma": "Gemma-2-9B"}


def main() -> None:
    p = RES / "derived_results.csv"
    if not p.exists():
        sys.exit("run analysis/aggregate_results.py first")
    d = pd.read_csv(p)
    e5 = d[d.config.str.startswith("e5_")].copy()
    if e5.empty:
        sys.exit("no E5 runs found")
    e5[["anchor", "pert"]] = e5.config.str.replace("e5_", "", 1).str.split("__", expand=True)

    rows = []
    for anchor, main_cfg in ANCHOR_MAIN.items():
        if anchor == "norm_baseline":
            continue                      # the baseline is the comparison, not an anchor
        for metric in METRICS:
            if metric not in d.columns:
                continue
            # unperturbed reference: main-run anchor vs main-run baseline
            ref = (d.loc[d.config == main_cfg, metric].mean()
                   - d.loc[d.config == ANCHOR_MAIN["norm_baseline"], metric].mean())
            cells = {"unperturbed": ref}
            for pert in sorted(e5.pert.unique()):
                a = e5[(e5.anchor == anchor) & (e5.pert == pert)][metric]
                b = e5[(e5.anchor == "norm_baseline") & (e5.pert == pert)][metric]
                cells[pert] = (a.mean() - b.mean()) if len(a) and len(b) else np.nan
            vals = np.array([v for v in cells.values() if np.isfinite(v)])
            if not len(vals):
                continue
            same = int((np.sign(vals) == np.sign(ref)).sum())
            rows.append({"anchor": anchor, "metric": metric, "reference_effect": ref,
                         "n_cells": len(vals), "same_sign": same,
                         "stability": same / len(vals),
                         "min_effect": vals.min(), "max_effect": vals.max(),
                         **{f"d_{k}": v for k, v in cells.items()}})

    out = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    out.to_csv(RES / "audit_sign_stability.csv", index=False)

    print("=" * 92)
    print("E5 SIGN-STABILITY AUDIT (each anchor vs baseline under the SAME perturbation)")
    print("=" * 92)
    show = ["anchor", "metric", "reference_effect", "same_sign", "n_cells", "stability",
            "min_effect", "max_effect"]
    print(out[show].round(3).to_string(index=False))

    robust = out[out.stability == 1.0]
    fragile = out[out.stability < 1.0]
    print(f"\nROBUST (sign holds in every cell): {len(robust)} of {len(out)}")
    for _, r in robust.iterrows():
        print(f"   {r.anchor:18s} {LABEL.get(r.metric, r.metric):16s} "
              f"effect {r.reference_effect:+.3f}  range [{r.min_effect:+.3f}, {r.max_effect:+.3f}]")
    if len(fragile):
        print(f"\nFRAGILE (sign flips somewhere): {len(fragile)} - report as exploratory")
        for _, r in fragile.iterrows():
            print(f"   {r.anchor:18s} {LABEL.get(r.metric, r.metric):16s} "
                  f"stability {r.same_sign}/{r.n_cells}")

    # ---- LaTeX table for the manuscript ----
    TAB.mkdir(parents=True, exist_ok=True)
    perts = [c[2:] for c in out.columns if c.startswith("d_") and c != "d_unperturbed"]
    lines = ["\\begin{table}[t]\\centering", "\\small",
             "\\caption{Robustness audit. Each cell is the difference between the anchor "
             "condition and the baseline \\emph{under the same perturbation}. A finding is "
             "robust if the sign is preserved throughout.}",
             "\\label{tab:e5audit}",
             "\\begin{tabular}{ll" + "c" * (len(perts) + 2) + "}", "\\toprule",
             "Anchor & Metric & Unpert. & "
             + " & ".join(PERT_LABEL.get(p, p).replace("_", "\\_") for p in perts)
             + " & Stab. \\\\", "\\midrule"]
    for _, r in out.iterrows():
        cells = [f"{r['d_unperturbed']:+.3f}"] + [f"{r.get('d_'+p, float('nan')):+.3f}" for p in perts]
        stab = f"{r.same_sign}/{r.n_cells}"
        if r.stability == 1.0:
            stab = f"\\textbf{{{stab}}}"
        lines.append(f"{r.anchor.replace('_',' ')} & {LABEL.get(r.metric, r.metric)} & "
                     + " & ".join(cells) + f" & {stab} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_e5_audit.tex").write_text("\n".join(lines))
    print(f"\nwrote {RES/'audit_sign_stability.csv'}")
    print(f"wrote {TAB/'table_e5_audit.tex'}")


if __name__ == "__main__":
    main()
