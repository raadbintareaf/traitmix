#!/usr/bin/env python3
"""
make_paper_figures.py — The four figures the manuscript actually needs.

The original make_figures.py was written before the results existed and still emits a
Pareto-frontier panel premised on a polarization/collective-intelligence trade-off. No such
trade-off is present in the data (the two outcome families are aligned, not opposed), so
that panel is replaced here. This script produces:

  fig1_crossover.pdf     Openness x Agreeableness response surface for BOTH model families,
                         side by side -- the headline interaction and its replication.
  fig2_heterogeneity.pdf Realised trait dispersion against dispersion- and segregation-type
                         polarization measures -- the E2 dose-response, showing that the two
                         construct families move in OPPOSITE directions.
  fig3_crossmodel.pdf    Effect of each trait condition in Llama vs Qwen, per metric --
                         which effects replicate across model families and which do not.
  fig4_controls.pdf      The two artifact controls: filler-topic response-style check by
                         model family, and realised trait SD against opinion variance.

Journal-ready: vector PDF, Okabe-Ito palette, 8pt type, no in-figure titles (captions carry
them), error bands from seed standard deviations.

Usage (from repo root):  python make_paper_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def _sample_label(d):
    """Which analysis sample a run belongs to, so figures and text cannot diverge."""
    import numpy as np
    lab = np.where(d.config.str.contains(r"^e\dq_|__qwen|qwen|llama3b|e3l_", regex=True),
                   "replication",
                   np.where(d.config.str.startswith(("abpr_", "abwi_", "abex_")), "ablation",
                            np.where(d.config.str.match(r"^(e1t|e2t)_"), "six-topic",
                                     "primary")))
    return lab


def load():
    p = RES / "derived_results.csv"
    if not p.exists():
        sys.exit("run analysis/aggregate_results.py first")
    d = pd.read_csv(p).copy()
    # Exactly the exclusions the analysis applies, so a figure and the text can never
    # be computed on different samples: the discarded model, the benchmark run and the
    # exposure-measurement seeds are not part of the experimental programme.
    d = d[~d.config.str.startswith("e3mistral7_")]
    d = d[~((d.config.str.startswith("e3l_")) & (d.seed == 99))]
    d = d[d.seed < 900]
    d["is_qwen"] = d.config.str.contains(r"^e\dq_|__qwen", regex=True)
    d["sample_label"] = _sample_label(d)
    return d


# ---------------------------------------------------------------- Figure 1
def fig_crossover(d):
    panels = [("e3_", "Llama-3.1-8B"), ("e3q_", "Qwen2.5-14B")]
    have = [(p, l) for p, l in panels if not d[d.config.str.startswith(p)].empty]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have) * 2, figsize=(min(6.69, 3.35 * len(have)), 2.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    k = 0
    for prefix, label in have:
        e = d[d.config.str.startswith(prefix)]
        for metric, ylab in [("pol_var", "Opinion variance"), ("crosscut_rate", "Cross-cutting rate")]:
            ax = axes[k]; k += 1
            g = e.groupby(["cond__mu_agreeableness", "cond__mu_openness"])[metric]
            m, s = g.mean().unstack(), g.std().unstack()
            for j, A in enumerate(sorted(m.index)):
                ax.errorbar(m.columns, m.loc[A], yerr=s.loc[A], color=OKABE[j],
                            marker="osD"[j % 3], ms=4, lw=1.4, capsize=2,
                            label=f"$\\mu_A$={A:g}")
            ax.set_xlabel("Openness $\\mu$")
            ax.set_ylabel(ylab)
            ax.set_title(label, fontsize=8, pad=3)
            if k == 1:
                ax.legend(frameon=False, loc="best")
    save(fig, "fig1_crossover.pdf")


# ---------------------------------------------------------------- Figure 2
def fig_heterogeneity(d):
    rt = RES / "realized_traits.csv"
    if not rt.exists():
        print("skip fig2: realized_traits.csv missing (run diagnose_confounds.py)")
        return
    real = pd.read_csv(rt)
    m = d.merge(real[["config", "seed", "realized_sd_mean"]], on=["config", "seed"], how="left")
    e2 = m[m.config.isin(["e2_homog", "e2_mid", "e2_diverse"])].dropna(subset=["realized_sd_mean"])
    if e2.empty:
        return
    specs = [("pol_var", "Opinion variance", "dispersion"),
             ("pol_extremity", "Extremity", "dispersion"),
             ("pol_ei", "Echo-chamber closure", "segregation"),
             ("crosscut_rate", "Cross-cutting rate", "segregation")]
    fig, axes = plt.subplots(1, 4, figsize=(6.69, 1.95), constrained_layout=True)
    for ax, (metric, lab, kind) in zip(axes, specs):
        s = e2[["realized_sd_mean", metric]].dropna()
        if s.empty:
            continue
        col = OKABE[0] if kind == "dispersion" else OKABE[3]
        ax.scatter(s.realized_sd_mean, s[metric], s=14, color=col, alpha=0.75,
                   edgecolor="none")
        if len(s) > 2:
            b, a = np.polyfit(s.realized_sd_mean, s[metric], 1)
            xs = np.linspace(s.realized_sd_mean.min(), s.realized_sd_mean.max(), 50)
            ax.plot(xs, a + b * xs, color="k", lw=1.0, ls="--")
            r, p = sps.pearsonr(s.realized_sd_mean, s[metric])
            ax.annotate(f"$r={r:+.2f}$", xy=(0.05, 0.90), xycoords="axes fraction", fontsize=7)
        ax.set_xlabel("Realised trait SD")
        ax.set_ylabel(lab)
    save(fig, "fig2_heterogeneity.pdf")


def fig_heterogeneity_models(d):
    """Does the heterogeneity effect hold in the replication models?

    The dose-response above is the primary model. The same three heterogeneity conditions
    were run on two further models, and the ordering that defines the effect --- homogeneous
    below baseline, diverse above --- holds in one and fails in the other. That failure is
    the behavioural counterpart of the missing interaction reported in the cross-model
    section, and it is what makes that model's exception structured rather than arbitrary.
    """
    MODELS = [("e2_", "e1_norm_baseline", "Llama-3.1-8B"),
              ("e2q_", "e1q_norm_baseline", "Qwen2.5-14B"),
              ("e2qwen7_", "e1qwen7_norm_baseline", "Qwen2.5-7B")]
    CONDS = [("homog", "homogeneous"), ("mid", "moderate"), ("diverse", "diverse")]
    rows = []
    for pre, base, name in MODELS:
        b = d[d.config == base]
        if b.empty:
            continue
        for key, lab in CONDS:
            x = d[d.config == f"{pre}{key}"][["seed", "pol_var"]].merge(
                b[["seed", "pol_var"]], on="seed", suffixes=("", "_b")).dropna()
            if len(x):
                diff = x.pol_var - x.pol_var_b
                rows.append((name, lab, float(diff.mean()),
                             float(diff.std(ddof=1) / np.sqrt(len(diff)))))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.3), constrained_layout=True)
    names = [n for _, _, n in MODELS if any(r[0] == n for r in rows)]
    width = 0.26
    for k, (key, lab) in enumerate(CONDS):
        xs, ys, es = [], [], []
        for i, n in enumerate(names):
            hit = [r for r in rows if r[0] == n and r[1] == lab]
            if hit:
                xs.append(i + (k - 1) * width); ys.append(hit[0][2]); es.append(hit[0][3])
        ax.bar(xs, ys, width=width, yerr=es, capsize=2,
               color=[OKABE[0], "#BBBBBB", OKABE[3]][k], label=lab, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(names)), names, fontsize=6.5)
    ax.set_ylabel("Opinion variance,\ndifference from own baseline")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower left")
    save(fig, "fig2b_hetero_models.pdf")


# ---------------------------------------------------------------- Figure 3
def fig_crossmodel(d):
    pairs, metrics = [], ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]
    lla_base = d[d.config == "e1_norm_baseline"]
    qwn_base = d[d.config == "e1q_norm_baseline"]
    if lla_base.empty or qwn_base.empty:
        print("skip fig3: need both baselines")
        return
    conds = [c for c in d.config.unique()
             if c.startswith("e1_") and c != "e1_norm_baseline"]
    for c in sorted(conds):
        q = c.replace("e1_", "e1q_", 1)
        if d[d.config == q].empty:
            continue
        for met in metrics:
            pairs.append({"cond": c.replace("e1_", ""), "metric": met,
                          "llama": d.loc[d.config == c, met].mean() - lla_base[met].mean(),
                          "qwen": d.loc[d.config == q, met].mean() - qwn_base[met].mean()})
    if not pairs:
        return
    P = pd.DataFrame(pairs)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.69, 2.05), constrained_layout=True)
    lab = {"pol_var": "Opinion variance", "pol_extremity": "Extremity",
           "crosscut_rate": "Cross-cutting", "pol_ei": "Echo closure"}
    for ax, met in zip(axes, metrics):
        s = P[P.metric == met]
        agree = np.sign(s.llama) == np.sign(s.qwen)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        ax.scatter(s.llama[agree], s.qwen[agree], s=22, color=OKABE[2],
                   label="sign agrees", edgecolor="none")
        ax.scatter(s.llama[~agree], s.qwen[~agree], s=22, color=OKABE[3],
                   marker="X", label="sign flips")
        lim = max(abs(np.r_[s.llama, s.qwen]).max() * 1.15, 1e-3)
        ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.7, ls=":")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Effect (Llama)"); ax.set_ylabel("Effect (Qwen)")
        ax.set_title(lab[met], fontsize=8, pad=3)
        n_ok = int(agree.sum())
        ax.annotate(f"{n_ok}/{len(s)} agree", xy=(0.05, 0.90),
                    xycoords="axes fraction", fontsize=7)
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "fig3_crossmodel.pdf")


# ---------------------------------------------------------------- Figure 4
def fig_controls(d):
    rt = RES / "realized_traits.csv"
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.25), constrained_layout=True)

    ax = axes[0]
    # The text reports this control on the primary model's published runs; pooling in
    # six-topic or ablation runs would change the sample without changing the label,
    # which is how the figure and the text came to disagree.
    for j, (fam, lab) in enumerate([("primary", "Llama-3.1-8B"),
                                    ("replication", "replication models")]):
        s = d[d.sample_label == fam][["filler_variance", "pol_extremity"]].dropna()
        if len(s) < 5:
            continue
        r, p = sps.pearsonr(s.filler_variance, s.pol_extremity)
        ax.scatter(s.filler_variance, s.pol_extremity, s=10, color=OKABE[j], alpha=0.6,
                   edgecolor="none", label=f"{lab}: $r={r:+.2f}$, $p={p:.3f}$")
    ax.set_xlabel("Neutral filler-topic variance")
    ax.set_ylabel("Contested-topic extremity")
    ax.legend(frameon=False, fontsize=6.5, loc="best")

    ax = axes[1]
    if rt.exists():
        real = pd.read_csv(rt)
        m = d.merge(real[["config", "seed", "realized_sd_mean"]], on=["config", "seed"], how="left")
        s = m[m.sample_label == "primary"][["realized_sd_mean", "pol_var"]].dropna()
        if len(s) > 5:
            ax.scatter(s.realized_sd_mean, s.pol_var, s=10, color=OKABE[0], alpha=0.6,
                       edgecolor="none")
            b, a = np.polyfit(s.realized_sd_mean, s.pol_var, 1)
            xs = np.linspace(s.realized_sd_mean.min(), s.realized_sd_mean.max(), 50)
            ax.plot(xs, a + b * xs, color="k", lw=1.0, ls="--")
            r, p = sps.pearsonr(s.realized_sd_mean, s.pol_var)
            ax.annotate(f"$r={r:+.2f}$, $p={p:.1e}$", xy=(0.05, 0.92),
                        xycoords="axes fraction", fontsize=7)
    ax.set_xlabel("Realised trait SD")
    ax.set_ylabel("Opinion variance")
    save(fig, "fig4_controls.pdf")


def main():
    d = load()
    fig_crossover(d)
    fig_heterogeneity(d)
    fig_heterogeneity_models(d)
    fig_crossmodel(d)
    fig_controls(d)


if __name__ == "__main__":
    main()
