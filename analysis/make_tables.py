"""summary.csv + stats.csv -> paper/tables/*.tex (booktabs). Numbers are never typed by hand:
the manuscript \\input{}s these files."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "paper" / "tables"
LABELS = {"pol_var": "Opinion var.", "pol_extremity": "Extremity", "pol_assort": "Assortativity",
          "pol_ei": "Echo closure", "crosscut_rate": "Cross-cut rate",
          "CI_mean_gain_vs_pre": "CI gain", "CI_hidden_profile_rate": "HP solve",
          "trait_drift_mean": "Trait drift"}

def fmt(mean, std, bold=False, underline=False, dagger=False):
    if pd.isna(mean): return "--"
    s = f"{mean:.3f}{{\\small$\\pm${std:.3f}}}" if pd.notna(std) else f"{mean:.3f}"
    if bold: s = f"\\textbf{{{s}}}"
    elif underline: s = f"\\underline{{{s}}}"
    return s + ("$^{\\dagger}$" if dagger else "")

def table(df_sum, stats, configs, metrics, caption, label, fname, better_high=None):
    better_high = better_high or {}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = df_sum[df_sum.config.isin(configs)].set_index("config").reindex(configs)
    lines = ["\\begin{table}[t]\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
             "Condition & " + " & ".join(LABELS.get(m, m) for m in metrics) + " \\\\", "\\midrule"]
    ranks = {}
    for m in metrics:
        col = rows.get(f"{m}__mean")
        if col is None: ranks[m] = ([], [])
        else:
            asc = not better_high.get(m, True)
            order = col.sort_values(ascending=asc).index.tolist()
            ranks[m] = (order[:1], order[1:2])
    for cfg in configs:
        cells = []
        for m in metrics:
            mean = rows.at[cfg, f"{m}__mean"] if f"{m}__mean" in rows else np.nan
            std = rows.at[cfg, f"{m}__std"] if f"{m}__std" in rows else np.nan
            dag = False
            if stats is not None and not stats.empty:
                hit = stats[(stats.config == cfg) & (stats.metric == m) & (stats.p_holm < 0.05)]
                dag = not hit.empty
            cells.append(fmt(mean, std, bold=cfg in ranks[m][0], underline=cfg in ranks[m][1], dagger=dag))
        lines.append(cfg.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\footnotesize $^{\\dagger}$ $p<0.05$ vs.\\ norm baseline "
              "(Wilcoxon, Holm--Bonferroni corrected). Mean$\\pm$std over seeds.",
              "\\end{table}"]
    (OUT / fname).write_text("\n".join(lines))
    print("wrote", OUT / fname)

def main():
    if not (RES / "summary.csv").exists():
        print("summary.csv missing — run aggregate_results.py first"); return
    s = pd.read_csv(RES / "summary.csv")
    st = pd.read_csv(RES / "stats.csv") if (RES / "stats.csv").exists() else pd.DataFrame()
    e1 = [c for c in s.config if c.startswith("e1_")]
    if e1:
        table(s, st, sorted(e1), ["pol_var", "pol_assort", "crosscut_rate",
                                  "CI_mean_gain_vs_pre", "CI_hidden_profile_rate", "trait_drift_mean"],
              "E1: trait-level effects on polarization and collective intelligence.",
              "tab:e1", "table_e1_main.tex",
              better_high={"pol_var": False, "pol_assort": False, "crosscut_rate": True,
                           "CI_mean_gain_vs_pre": True, "CI_hidden_profile_rate": True,
                           "trait_drift_mean": False})
    e2 = [c for c in s.config if c.startswith("e2_")]
    if e2:
        table(s, st, sorted(e2), ["pol_var", "pol_extremity", "CI_mean_gain_vs_pre",
                                  "CI_hidden_profile_rate", "distinct2"],
              "E2: heterogeneity effects.", "tab:e2", "table_e2_heterogeneity.tex")
    if (RES / "e0_validation.csv").exists():
        v = pd.read_csv(RES / "e0_validation.csv")
        lines = ["\\begin{table}[t]\\centering", "\\caption{E0 validation gate: convergent validity "
                 "(Spearman $r$, targeted vs.\\ measured trait level).}", "\\label{tab:e0}", "\\small",
                 "\\begin{tabular}{lccccc c}", "\\toprule",
                 "Arm & O & C & E & A & N & mean $r$ \\\\", "\\midrule"]
        for _, r in v.iterrows():
            lines.append(f"{r['arm']} & " + " & ".join(f"{r[t]:.2f}" for t in
                        ["openness","conscientiousness","extraversion","agreeableness","neuroticism"])
                        + f" & \\textbf{{{r['mean_r']:.2f}}} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (OUT / "table_e0_validation.tex").write_text("\n".join(lines))
        print("wrote", OUT / "table_e0_validation.tex")

if __name__ == "__main__":
    main()
