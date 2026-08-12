#!/usr/bin/env python3
"""make_sixmodel_figure.py — Response surfaces for every model, with induction alongside."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 6.8, "axes.labelsize": 7.2, "legend.fontsize": 6.0,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "pdf.fonttype": 42,
                     "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv(RES / "derived_results.csv")
d = d[~d.config.str.startswith("e3mistral7_")]
d = d[~((d.config.str.startswith("e3l_")) & (d.seed == 99))]
ind = pd.read_csv(RES / "induction_summary_all.csv").set_index("tag")

PANELS = [("e3llama3b_", "Llama-3.2-3B", "llama3b"), ("e3qwen3b_", "Qwen2.5-3B", "qwen3b"),
          ("e3qwen7_", "Qwen2.5-7B", "qwen7"), ("e3_", "Llama-3.1-8B", "base"),
          ("e3q_", "Qwen2.5-14B", "q"), ("e3l_", "Qwen2.5-32B", "large")]

fig, axes = plt.subplots(2, 3, figsize=(6.69, 4.1), constrained_layout=True)
import statsmodels.formula.api as smf
for k, (pre, name, tag) in enumerate(PANELS):
    ax = axes[k // 3, k % 3]
    e = d[d.config.str.startswith(pre)].copy()
    if e.empty:
        ax.axis("off"); continue
    e["O"] = e["cond__mu_openness"]; e["A"] = e["cond__mu_agreeableness"]
    g = e.groupby(["A", "O"]).pol_var
    m, s = g.mean().unstack(), g.std().unstack()
    passed = ind.loc[tag, "gate"] == "pass"
    for j, A in enumerate(sorted(m.index)):
        ax.errorbar(m.columns, m.loc[A], yerr=s.loc[A], color=[BLUE, ORANGE, GREEN][j],
                    marker="osD"[j], ms=3.6, lw=1.3, capsize=2,
                    alpha=1.0 if passed else 0.40, label=f"$\\mu_A$={A:g}")
    c = e.groupby(["O", "A"], as_index=False).pol_var.mean()
    fit = smf.ols("pol_var ~ O + A + O:A", data=c).fit()
    ttl = name if passed else f"{name}  (excluded)"
    ax.set_title(ttl, fontsize=7.3, pad=3,
                 color="black" if passed else GREY)
    note = (f"$b={fit.params['O:A']:+.2f}$, $p={fit.pvalues['O:A']:.3f}$" if passed
            else f"induction {'magnitude' if 'magnitude' in ind.loc[tag,'gate'] else 'rank'} failure")
    ax.annotate(note, xy=(0.04, 0.05), xycoords="axes fraction", fontsize=5.8,
                color=GREY if passed else VERM)
    if k // 3 == 1:
        ax.set_xlabel("Openness $\\mu$")
    if k % 3 == 0:
        ax.set_ylabel("Opinion variance")
    if k == 0:
        ax.legend(frameon=False, loc="upper left", fontsize=5.6)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "figE_sixmodel.pdf"); plt.close(fig)
print("wrote", FIG / "figE_sixmodel.pdf")
