#!/usr/bin/env python3
"""
make_extra_figures.py — Two further figures for the manuscript.

figA_traitforest.pdf   Trait-level effects with bootstrap confidence intervals (E1).
                       A forest plot conveys direction, magnitude and precision at once,
                       which a table of the same numbers does not.

figB_alignment.pdf     The central result of the paper made visible: polarization measures
                       plotted against collective accuracy across all primary-model runs,
                       showing that cross-cutting interaction is positively associated with
                       accuracy and that segregation and extremity are negatively associated
                       with it. Reported in the text only as correlations, which understates
                       how clear the pattern is.

Both are sized to the journal's full-page figure width of 170 mm.

Usage:  python make_extra_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
BLUE, ORANGE, GREEN, VERM, PURPLE, GREY = ("#0072B2", "#E69F00", "#009E73",
                                           "#D55E00", "#CC79A7", "#666666")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
FULL = 6.69   # 170 mm


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# ---------------------------------------------------------------- forest plot
def fig_forest():
    st = pd.read_csv(RES / "stats.csv")
    e1 = st[st.family == "e1"]
    if e1.empty:
        return
    panels = [("pol_var", "Opinion variance"), ("pol_extremity", "Extremity"),
              ("crosscut_rate", "Cross-cutting rate"), ("CI_accuracy_z", "Collective accuracy")]
    order = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    labels, keys = [], []
    for t in order:
        for lv in ("high", "low"):
            keys.append(f"e1_{t}_{lv}")
            labels.append(f"{t[:5].capitalize()}. {lv}")

    fig, axes = plt.subplots(1, len(panels), figsize=(FULL, 2.9),
                             sharey=True, constrained_layout=True)
    for ax, (metric, title) in zip(axes, panels):
        sub = e1[e1.metric == metric].set_index("config")
        y = np.arange(len(keys))[::-1]
        for yi, k in zip(y, keys):
            if k not in sub.index:
                continue
            r = sub.loc[k]
            sig = r.p_holm < 0.05
            col = VERM if r.mean_diff > 0 else BLUE
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=1.6 if sig else 0.9,
                    alpha=1.0 if sig else 0.55, solid_capstyle="butt", zorder=3)
            ax.plot(r.mean_diff, yi, marker="o" if sig else "o", ms=4.2 if sig else 3.0,
                    color=col, alpha=1.0 if sig else 0.55,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.axvline(0, color="k", lw=0.8, zorder=2)
        ax.set_yticks(y, labels)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel("difference from baseline")
    h = [plt.Line2D([], [], color=VERM, lw=1.6, marker="o", ms=4.2),
         plt.Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4.2),
         plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=3.0, alpha=0.55)]
    fig.legend(h, ["increase (corrected $p<0.05$)", "decrease (corrected $p<0.05$)",
                   "not significant"], frameon=False, ncol=3, fontsize=6.3,
               loc="outside lower center")
    save(fig, "figA_traitforest.pdf")


# ---------------------------------------------------------------- alignment
def fig_alignment():
    d = pd.read_csv(RES / "derived_results.csv")
    d = d[~d.config.str.startswith("e3mistral7_")]
    L = d[d.config.str.match(r"^(e1|e2|e3|e5|scale)_") & ~d.config.str.contains("__qwen")]
    panels = [("crosscut_rate", "Cross-cutting rate", GREEN),
              ("pol_var", "Opinion variance", BLUE),
              ("pol_ei", "Echo-chamber closure", ORANGE),
              ("pol_extremity", "Extremity", VERM)]
    # Only cross-cutting survives partialling; the rest are shown to make that visible.
    SURVIVES = {"crosscut_rate": True}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.15), sharey=True,
                             constrained_layout=True)
    for ax, (metric, lab, col) in zip(axes, panels):
        s = L[[metric, "CI_accuracy_z"]].dropna()
        if len(s) < 5:
            continue
        surv = SURVIVES.get(metric, False)
        ax.scatter(s[metric], s.CI_accuracy_z, s=9, color=col,
                   alpha=0.45 if surv else 0.22, edgecolor="none", zorder=3)
        b, a = np.polyfit(s[metric], s.CI_accuracy_z, 1)
        xs = np.linspace(s[metric].min(), s[metric].max(), 50)
        ax.plot(xs, a + b * xs, color="k", lw=1.2 if surv else 0.8,
                ls="-" if surv else (0, (3, 2)), zorder=4)
        r, p = sps.pearsonr(s[metric], s.CI_accuracy_z)
        ptxt = "$p<10^{-4}$" if p < 1e-4 else f"$p={p:.3f}$"
        ax.annotate(f"$r={r:+.2f}$, {ptxt}", xy=(0.04, 0.93), xycoords="axes fraction",
                    fontsize=6.3, fontweight="bold" if surv else "normal")
        ax.annotate("survives partialling" if surv else "does not survive",
                    xy=(0.04, 0.84), xycoords="axes fraction", fontsize=5.7,
                    color=col if surv else GREY, style="italic")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("Collective accuracy ($z$)")
    save(fig, "figB_alignment.pdf")


if __name__ == "__main__":
    fig_forest()
    fig_alignment()
