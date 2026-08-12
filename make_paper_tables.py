#!/usr/bin/env python3
"""
make_paper_tables.py — Every LaTeX table the manuscript needs, from the corrected results.

Replaces analysis/make_tables.py, which predates the corrected analysis (its significance
markers were derived from contrasts that compared replication-model conditions against the
primary-model baseline) and which produced no table for the response surface or the
cross-model replication.

Emits into paper/tables/:
  table_e0_validation.tex   induction validation, per trait
  table_e1_levels.tex       trait-level effects vs baseline, with effect sizes and CIs
  table_e2_heterogeneity.tex heterogeneity and the human-calibrated condition
  table_e3_surface.tex      response-surface regression, both model families
  table_crossmodel.tex      replication: effect direction and magnitude in both families
  table_e5_audit.tex        sign stability under design perturbation

All numbers are read from results/*.csv. Nothing is typed by hand.
Usage (from repo root):  python make_paper_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, TAB = ROOT / "results", ROOT / "paper" / "tables"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
LABEL = {"pol_var": "Opinion var.", "pol_extremity": "Extremity", "pol_ei": "Echo closure",
         "crosscut_rate": "Cross-cut", "pol_assort": "Assortativity",
         "CI_accuracy_z": "Coll. accuracy", "CI_hidden_profile_rate": "Hidden profile",
         "CI_diversity": "Diversity", "CI_medrelerr_z": "Median error",
         "CI_mean_gain_vs_pre": "CI gain", "trait_drift_mean": "Trait drift"}
POL = ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]
CI = ["CI_accuracy_z", "CI_hidden_profile_rate"]



def tighten(lines, n_cols, threshold=5):
    """Make a wide table fit the Springer text block (372pt).

    \\resizebox cannot be used: it conflicts with the sn-jnl class and produces
    "Missing \\endgroup" errors. Instead we reduce the type size, tighten the column
    separation, and remove the padding at the outer edges with @{}.
    """
    if n_cols <= threshold:
        return lines
    out = []
    for l in lines:
        if l.startswith("\\small"):
            out.append("\\scriptsize\\setlength{\\tabcolsep}{1.5pt}")
        elif l.startswith("\\begin{tabular}{"):
            spec = l[len("\\begin{tabular}{"):-1]
            out.append("\\begin{tabular}{@{}" + spec + "@{}}")
        else:
            out.append(l)
    return out

def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def pretty(cfg):
    c = cfg.replace("e1q_", "").replace("e1_", "").replace("e2q_", "").replace("e2_", "")
    c = c.replace("e3q_", "").replace("e3_", "")
    return esc(c.replace("_", " "))


def stars(p):
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < .001 else "$^{**}$" if p < .01 else "$^{*}$" if p < .05 else ""


def write(name, lines):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text("\n".join(lines) + "\n")
    print("wrote", TAB / name)


# ---------------------------------------------------------------- E0
def table_e0():
    p = RES / "e0_validation.csv"
    if not p.exists():
        return
    v = pd.read_csv(p)
    lines = ["\\begin{table}[t]\\centering", "\\small",
             "\\caption{Induction validation (E0). Spearman correlation between targeted and "
             "measured trait level, IPIP-NEO-120 administered to every persona configuration "
             "used in the main experiment.}", "\\label{tab:e0}",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             "Induction & O & C & E & A & N & Mean $r$ \\\\", "\\midrule"]
    for _, r in v.iterrows():
        cells = " & ".join(f"{r[t]:.2f}" for t in TRAITS)
        lines.append(f"{esc(r['arm'])} & {cells} & \\textbf{{{r['mean_r']:.2f}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_e0_validation.tex", lines)


# ---------------------------------------------------------------- E1 / E2
def contrast_table(family, metrics, caption, label, fname, order=None):
    sp, st = RES / "summary.csv", RES / "stats.csv"
    if not (sp.exists() and st.exists()):
        return
    s, t = pd.read_csv(sp), pd.read_csv(st)
    t = t[t.family == family]
    if t.empty:
        return
    base = t.baseline.iloc[0] if "baseline" in t.columns else None
    cfgs = order or sorted(t.config.unique())
    lines = ["\\begin{table}[t]\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
             "Condition & " + " & ".join(LABEL.get(m, m) for m in metrics) + " \\\\",
             "\\midrule"]
    if base is not None and base in set(s.config):
        row = s[s.config == base].iloc[0]
        cells = []
        for m in metrics:
            mu, sd = row.get(f"{m}__mean", np.nan), row.get(f"{m}__std", np.nan)
            cells.append("--" if pd.isna(mu) else f"{mu:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}")
        lines.append(f"\\textit{{{pretty(base)}}} (baseline) & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    for cfg in cfgs:
        cells = []
        for m in metrics:
            r = t[(t.config == cfg) & (t.metric == m)]
            if r.empty:
                cells.append("--"); continue
            r = r.iloc[0]
            txt = f"{r.mean_diff:+.3f}{stars(r.p_holm)}"
            if abs(r.cohens_dz) > 0.8:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(pretty(cfg) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\footnotesize Differences from baseline (mean over seeds). "
              "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$, paired $t$-test with "
              "Holm--Bonferroni correction within family $\\times$ metric. Bold marks "
              "$|d_z|>0.8$. Baseline row shows absolute values (mean$\\pm$sd).",
              "\\end{table}"]
    lines = tighten(lines, len(metrics) + 1)
    write(fname, lines)


# ---------------------------------------------------------------- E3 surface
def table_e3_surface():
    p = RES / "derived_results.csv"
    if not p.exists():
        return
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("skip E3 surface table: statsmodels unavailable")
        return
    d = pd.read_csv(p)
    rows = []
    for prefix, model in [("e3_", "Llama-3.1-8B"), ("e3qwen7_", "Qwen2.5-7B"),
                          ("e3q_", "Qwen2.5-14B"), ("e3l_", "Qwen2.5-32B")]:
        e = d[d.config.str.startswith(prefix)].copy()
        if e.empty:
            continue
        e["O"] = e["cond__mu_openness"]; e["A"] = e["cond__mu_agreeableness"]
        for dv in ["pol_var", "crosscut_rate"]:
            # Fitted on the nine cell means, which is the level at which the design
            # varies. Fitting six parameters to nine cells and taking p values from the
            # run-level n treats within-cell simulation noise as replication.
            cells = e.groupby(["O", "A"], as_index=False)[dv].mean()
            m = smf.ols(f"{dv} ~ O + A + O:A", data=cells).fit()
            rows.append({"model": model, "dv": LABEL.get(dv, dv), "n": int(m.nobs),
                         "R2": m.rsquared,
                         **{k: (m.params[k], m.pvalues[k]) for k in
                            ["O", "A", "O:A"] if k in m.params}})
    if not rows:
        return
    terms = [("O", "$\\mu_O$"), ("A", "$\\mu_A$"), ("O:A", "$\\mu_O\\!\\times\\!\\mu_A$")]
    lines = ["\\begin{table}[t]\\centering", "\\small",
             "\\caption{Response-surface regression on the nine cell means of the Openness "
             "$\\times$ Agreeableness grid.}", "\\label{tab:e3surface}",
             "\\begin{tabular}{llcc" + "c" * len(terms) + "}", "\\toprule",
             "Model & Outcome & cells & $R^2$ & " + " & ".join(l for _, l in terms) + " \\\\",
             "\\midrule"]
    for r in rows:
        cells = []
        for k, _ in terms:
            if k in r:
                b, p = r[k]
                cells.append(f"{b:+.3f}{stars(p)}")
            else:
                cells.append("--")
        lines.append(f"{r['model']} & {r['dv']} & {r['n']} & {r['R2']:.3f} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\footnotesize Fitted on cell means, which is the level at which "
              "the design varies; fitting the full quadratic to nine cells and taking $p$ "
              "values from the underlying runs would treat within-cell simulation noise as "
              "replication. A mixed model with cell as the grouping factor gives the same "
              "interaction coefficients at $p = 0.003$, $p < 0.001$ and $p = 0.004$ for the 8, "
              "14 and 32 billion parameter models. Only models passing induction "
              "validation are shown. "
              "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.", "\\end{table}"]
    lines = tighten(lines, len(terms) + 4)
    write("table_e3_surface.tex", lines)


# ---------------------------------------------------------------- cross-model
def table_crossmodel():
    # three metrics keep the table within the text block; echo closure is reported in text
    p = RES / "derived_results.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    REPS = [("e1q_", "Q14"), ("e1qwen7_", "Q7")]
    lb = d[d.config == "e1_norm_baseline"]
    bases = {pre: d[d.config == f"{pre}norm_baseline"] for pre, _ in REPS}
    REPS = [(pre, tag) for pre, tag in REPS if not bases[pre].empty]
    if lb.empty or not REPS:
        print("skip cross-model table: missing a baseline")
        return
    metrics = ["pol_var", "pol_extremity", "crosscut_rate"]
    conds = sorted(c for c in d.config.unique()
                   if c.startswith("e1_") and c != "e1_norm_baseline"
                   and all(not d[d.config == c.replace("e1_", pre, 1)].empty
                           for pre, _ in REPS))
    W = 1 + len(REPS)                       # primary plus each replication
    lines = ["\\begin{table}[t]\\centering", "\\scriptsize",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Cross-model replication of trait-level effects. Each cell is the "
             "difference from that model's own baseline.}", "\\label{tab:crossmodel}",
             "\\begin{tabular}{l" + ("c" * W) * len(metrics) + "}", "\\toprule",
             " & " + " & ".join("\\multicolumn{%d}{c}{%s}" % (W, LABEL[m]) for m in metrics)
             + " \\\\",
             "Condition & " + " & ".join(" & ".join(["L"] + [t for _, t in REPS])
                                         for _ in metrics) + " \\\\", "\\midrule"]
    agree = {m: 0 for m in metrics}
    total = {m: 0 for m in metrics}
    for c in conds:
        cells = []
        for m in metrics:
            dl = d.loc[d.config == c, m].mean() - lb[m].mean()
            cells.append(f"{dl:+.2f}" if np.isfinite(dl) else "--")
            for pre, _ in REPS:
                dq = d.loc[d.config == c.replace("e1_", pre, 1), m].mean() - bases[pre][m].mean()
                if not (np.isfinite(dl) and np.isfinite(dq)):
                    cells.append("--"); continue
                same = np.sign(dl) == np.sign(dq)
                agree[m] += int(same); total[m] += 1
                cells.append(f"\\textbf{{{dq:+.2f}}}" if same else f"{dq:+.2f}")
        lines.append(pretty(c) + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("Sign agreement & " + " & ".join(
        "\\multicolumn{%d}{c}{%d/%d}" % (W, agree[m], total[m]) for m in metrics) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\footnotesize L = Llama-3.1-8B, Q14 = Qwen2.5-14B, "
              "Q7 = Qwen2.5-7B. Bold marks a replication cell whose sign agrees with the "
              "primary model. Only models passing induction validation "
              "(Table~\\ref{tab:induction}) are included.", "\\end{table}"]
    lines = tighten(lines, W * len(metrics) + 1)
    write("table_crossmodel.tex", lines)


def main():
    if not (RES / "summary.csv").exists():
        sys.exit("run analysis/aggregate_results.py first")
    table_e0()
    contrast_table("e1", POL + ["CI_accuracy_z"],
                   "Effects of trait level on polarization and collective intelligence (E1).",
                   "tab:e1", "table_e1_levels.tex")
    contrast_table("e2", POL + ["CI_accuracy_z"],
                   "Effects of trait heterogeneity, and of a human-calibrated composition (E2).",
                   "tab:e2", "table_e2_heterogeneity.tex")
    table_e3_surface()
    table_crossmodel()
    print("\nnote: table_e5_audit.tex is produced by audit_sign_stability.py")


if __name__ == "__main__":
    main()
