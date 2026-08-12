#!/usr/bin/env python3
"""
make_results_ledger.py — One source for every number the manuscript reports.

The reviewer was unable to reconcile several values with the supplement, and found three
quantities circulating under one label. The root cause was that the collective-intelligence
composites were standardised against the whole dataset, so a value depended on which other
runs were present; tables and the supplement, regenerated at different times against
datasets of different sizes, captured different reference populations. Standardisation is
now within model, which fixes the instability, but a stable value is not the same as a
traceable one.

This script produces two artefacts.

  results_ledger.csv / .tex
      One row per number quoted in the manuscript: the quantity, where it is cited, the
      value, and the file, column and aggregation that produced it. A reader can recompute
      any of them from the released run-level data.

  run_accounting.csv / .tex
      The total run count reconciled across experiment families, so the headline figure can
      be reconstructed from the supplement (reviewer point C21).

Usage:  python make_results_ledger.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"

FAMILY_LABEL = {
    "e1": ("E1 trait levels", "Llama-3.1-8B", "2 topics"),
    "e2": ("E2 heterogeneity", "Llama-3.1-8B", "2 topics"),
    "e3": ("E3 response surface", "Llama-3.1-8B", "2 topics"),
    "e5": ("E5 robustness audit", "Llama-3.1-8B", "2 topics"),
    "scale": ("Scale ablation", "Llama-3.1-8B", "2 topics"),
    "e1t": ("E1 six-topic replication", "Llama-3.1-8B", "6 topics"),
    "e2t": ("E2 six-topic replication", "Llama-3.1-8B", "6 topics"),
    "e1q": ("E1 replication", "Qwen2.5-14B", "2 topics"),
    "e2q": ("E2 replication", "Qwen2.5-14B", "2 topics"),
    "e3q": ("E3 replication", "Qwen2.5-14B", "2 topics"),
    "e1qwen7": ("E1 replication", "Qwen2.5-7B", "2 topics"),
    "e2qwen7": ("E2 replication", "Qwen2.5-7B", "2 topics"),
    "e3qwen7": ("E3 replication", "Qwen2.5-7B", "2 topics"),
    "e3l": ("E3 replication", "Qwen2.5-32B", "2 topics"),
    "e3qwen3b": ("E3 replication", "Qwen2.5-3B (excluded)", "2 topics"),
    "e3llama3b": ("E3 replication", "Llama-3.2-3B (excluded)", "2 topics"),
    "abpr": ("R1 probe-anchor ablation", "Llama-3.1-8B", "2 topics"),
    "abwi": ("R2 recommender ablation", "Llama-3.1-8B", "2 topics"),
    "abex": ("R2 expressed-stance variant", "Llama-3.1-8B", "2 topics"),
}


def load():
    d = pd.read_csv(RES / "derived_results.csv")
    d = d[~d.config.str.startswith("e3mistral7_")]          # discarded, see Section 4.1
    d = d[~((d.config.str.startswith("e3l_")) & (d.seed == 99))]   # benchmark run
    d = d[d.seed < 900]                                      # exposure-measurement seeds
    return d


def family(cfg: str) -> str:
    return cfg.split("_")[0]


def paired(d, cfg, base, metric):
    a = d[d.config == cfg][["seed", metric]]
    b = d[d.config == base][["seed", metric]]
    m = a.merge(b, on="seed", suffixes=("", "_b")).dropna()
    if len(m) < 3:
        return np.nan, np.nan, 0
    diff = m[metric] - m[f"{metric}_b"]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    p = sps.ttest_rel(m[metric], m[f"{metric}_b"]).pvalue
    return float(diff.mean()), float(dz), len(m)


def main() -> None:
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)
    L = []

    def add(quantity, cited, value, source, aggregation):
        L.append(dict(quantity=quantity, cited_in=cited, value=value,
                      source=source, aggregation=aggregation))

    # ---------- run counts ----------
    add("Total analysed runs", "Abstract; Sec. 4; Sec. 5", f"{len(d)}",
        "derived_results.csv", "row count after excluding the discarded model, "
        "the benchmark run and the exposure seeds")
    prim = d[d.config.str.match(r"^(e1|e2|e3|e5|scale)_") & ~d.config.str.contains("__qwen")]
    add("Primary-model runs", "Sec. 5.7", f"{len(prim)}",
        "derived_results.csv", "runs of the primary model on the two-topic design")

    # ---------- headline contrasts ----------
    for cfg, base, metric, where in [
            ("e1_agreeableness_high", "e1_norm_baseline", "crosscut_rate", "Sec. 5.2"),
            ("e1_agreeableness_high", "e1_norm_baseline", "pol_var", "Sec. 5.2"),
            ("e1_neuroticism_high", "e1_norm_baseline", "pol_extremity", "Sec. 5.2"),
            ("e2_homog", "e1_norm_baseline", "pol_var", "Sec. 5.4"),
            ("e2_human_norm", "e1_norm_baseline", "pol_ei", "Sec. 5.4"),
            ("e2_human_norm", "e1_norm_baseline", "CI_accuracy_z", "Sec. 5.4"),
            ("e2_diverse", "e1_norm_baseline", "pol_var", "Sec. 5.4")]:
        v, dz, n = paired(d, cfg, base, metric)
        if np.isfinite(v):
            add(f"{cfg} vs baseline, {metric}", where, f"{v:+.3f} (dz {dz:+.2f}, n {n})",
                "derived_results.csv", "mean paired difference over seeds, matched by seed")

    # ---------- correlations ----------
    s = prim[["crosscut_rate", "CI_accuracy_z"]].dropna()
    r, p = sps.pearsonr(s.crosscut_rate, s.CI_accuracy_z)
    add("Cross-cutting vs collective accuracy", "Abstract; Sec. 5.7", f"r = {r:+.3f}",
        "derived_results.csv",
        "Pearson over primary-model runs; accuracy standardised within model")
    f = prim[["filler_variance", "pol_extremity"]].dropna()
    rf, pf = sps.pearsonr(f.filler_variance, f.pol_extremity)
    add("Filler-topic control", "Sec. 5.1; Fig. 4", f"r = {rf:+.3f}, p = {pf:.3f}, n = {len(f)}",
        "derived_results.csv", "Pearson, primary model only")
    real = RES / "realized_traits.csv"
    if real.exists():
        rt = pd.read_csv(real)
        mm = prim.merge(rt[["config", "seed", "realized_sd_mean"]], on=["config", "seed"])
        s2 = mm[["realized_sd_mean", "pol_var"]].dropna()
        r2, p2 = sps.pearsonr(s2.realized_sd_mean, s2.pol_var)
        add("Realised dispersion vs opinion variance", "Sec. 5.1; Fig. 4",
            f"r = {r2:+.3f}, n = {len(s2)}", "realized_traits.csv + derived_results.csv",
            "Pearson across primary-model runs")
        e2 = mm[mm.config.str.startswith("e2_") & ~mm.config.str.contains("human")]
        s3 = e2[["realized_sd_mean", "pol_var"]].dropna()
        if len(s3) > 5:
            r3, _ = sps.pearsonr(s3.realized_sd_mean, s3.pol_var)
            add("Heterogeneity dose-response", "Sec. 5.4", f"r = {r3:+.3f}, n = {len(s3)}",
                "realized_traits.csv + derived_results.csv",
                "Pearson within the heterogeneity family, human-calibrated excluded")

    # ---------- response surface ----------
    try:
        import statsmodels.formula.api as smf
        for pre, name, where in [("e3_", "Llama-3.1-8B", "Sec. 5.5"),
                                 ("e3qwen7_", "Qwen2.5-7B", "Sec. 5.9"),
                                 ("e3q_", "Qwen2.5-14B", "Sec. 5.9"),
                                 ("e3l_", "Qwen2.5-32B", "Sec. 5.9")]:
            e = d[d.config.str.startswith(pre)].copy()
            if e.empty:
                continue
            e["O"] = e["cond__mu_openness"]; e["A"] = e["cond__mu_agreeableness"]
            cells = e.groupby(["O", "A"], as_index=False).pol_var.mean()
            fit = smf.ols("pol_var ~ O + A + O:A", data=cells).fit()
            add(f"O x A interaction, {name}", where,
                f"b = {fit.params['O:A']:+.3f}, p = {fit.pvalues['O:A']:.4f}, "
                f"R2 = {fit.rsquared:.3f}",
                "derived_results.csv",
                "OLS on the nine cell means; reduced model without quadratic terms")
    except ImportError:
        pass

    # ---------- ablations ----------
    for pre, lab in [("abpr_", "R1 probe-anchor ablation"), ("abwi_", "R2 recommender ablation")]:
        base = f"{pre}e1_norm_baseline"
        if d[d.config == base].empty:
            continue
        conds = ["e1_agreeableness_high", "e1_neuroticism_high", "e1_openness_high",
                 "e2_homog", "e2_mid", "e2_diverse", "e3_O2_A2", "e3_O8_A8",
                 "e1_agreeableness_low"]
        rows_pub, rows_abl = [], []
        for m in ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]:
            for c in conds:
                v0, _, _ = paired(d, c, "e1_norm_baseline", m)
                v1, _, _ = paired(d, pre + c, base, m)
                if np.isfinite(v0) and np.isfinite(v1):
                    rows_pub.append(v0); rows_abl.append(v1)
        if rows_pub:
            rho = sps.spearmanr(rows_pub, rows_abl).statistic
            agree = sum(np.sign(a) == np.sign(b) for a, b in zip(rows_pub, rows_abl))
            add(f"{lab}: agreement with published", "Sec. 5.1",
                f"rank rho = {rho:+.3f}, signs {agree}/{len(rows_pub)}",
                "derived_results.csv",
                "condition effects computed against each variant's own baseline")

    # ---------- exposure ----------
    ex = RES / "ci_exposure.csv"
    if ex.exists():
        e = pd.read_csv(ex)
        add("Estimation-item exposure", "Sec. 3.4; Sec. 5.6",
            f"{e.announcements.mean():.0f} announcements, {e.replies.mean():.1f} replies, "
            f"{e.agents_engaged.mean():.0f}/100 agents engaged",
            "ci_exposure.csv", "mean over three baseline runs with checkpoints retained")

    # ---------- write the ledger ----------
    D = pd.DataFrame(L)
    D.to_csv(RES / "results_ledger.csv", index=False)
    lines = [r"\begin{table}[t]\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Results ledger: every quantity quoted in the article, with the file, "
             r"column and aggregation that produced it.}", r"\label{tab:ledger}",
             r"\begin{tabular}{@{}p{4.0cm}p{2.0cm}p{3.1cm}p{4.2cm}@{}}", r"\toprule",
             r"Quantity & Cited in & Value & Source and aggregation \\", r"\midrule"]
    for _, r in D.iterrows():
        esc = lambda t: str(t).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        lines.append(f"{esc(r.quantity)} & {esc(r.cited_in)} & {esc(r.value)} & "
                     f"{esc(r.source)}; {esc(r.aggregation)} \\\\[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_ledger.tex").write_text("\n".join(lines) + "\n")

    # ---------- run accounting ----------
    d2 = d.copy()
    d2["fam"] = d2.config.map(family)
    acc = []
    for fam, n in d2.fam.value_counts().items():
        exp, model, topics = FAMILY_LABEL.get(fam, (fam, "?", "?"))
        acc.append(dict(prefix=fam, experiment=exp, model=model, topics=topics,
                        conditions=d2[d2.fam == fam].config.nunique(), runs=n))
    A = pd.DataFrame(acc).sort_values(["model", "experiment"])
    A.to_csv(RES / "run_accounting.csv", index=False)
    lines = [r"\begin{table}[t]\centering", r"\small",
             r"\caption{Run accounting. Condition prefixes map to experiment, model and "
             r"topic set; the run counts sum to the total reported in the article.}",
             r"\label{tab:runaccounting}",
             r"\begin{tabular}{@{}llllrr@{}}", r"\toprule",
             r"Prefix & Experiment & Model & Topics & Conditions & Runs \\", r"\midrule"]
    for _, r in A.iterrows():
        lines.append(f"\\texttt{{{r.prefix}}} & {r.experiment} & {r.model} & {r.topics} & "
                     f"{r.conditions} & {r.runs} \\\\")
    lines += [r"\midrule", f"& & & & \\textbf{{{A.conditions.sum()}}} & "
              f"\\textbf{{{A.runs.sum()}}} \\\\", r"\bottomrule", r"\end{tabular}",
              r"\end{table}"]
    (OUT / "table_runaccounting.tex").write_text("\n".join(lines) + "\n")

    print(f"ledger: {len(D)} quantities -> results_ledger.csv, table_ledger.tex")
    print(f"run accounting: {len(A)} families, {A.runs.sum()} runs -> table_runaccounting.tex\n")
    print(A.to_string(index=False))
    print(f"\n  TOTAL {A.runs.sum()} runs")


if __name__ == "__main__":
    main()
