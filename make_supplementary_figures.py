#!/usr/bin/env python3
"""
make_supplementary_figures.py — Figures for the Additional file.

S1  Realised trait distributions per condition. The manipulation check: shows that the
    composition actually differed between conditions, and by how much, including the
    truncation-induced narrowing at extreme means that the main text quantifies.

S2  Collective-intelligence item screening. Every candidate item plotted by the model's
    solo error against the dispersion of its answers, with the two pre-specified
    thresholds drawn, so a reader can see exactly which items were admitted and why.

S3  Full response surfaces. All four polarization measures across the Openness by
    Agreeableness grid, in both model families: the complete version of the main-text
    figure, which shows only two measures.

S4  The hidden-profile floor. Why the hidden-profile measure carries no information in the
    replication model, and why several statistics in Table S4 are undefined.

Usage:  python make_supplementary_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "supplementary" / "figures"
BLUE, ORANGE, GREEN, VERM, PURPLE, GREY = ("#0072B2", "#E69F00", "#009E73",
                                           "#D55E00", "#CC79A7", "#666666")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def s1_realised_traits():
    p = RES / "realized_traits.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    e1 = r[r.config.str.match(r"^e1_")]
    if e1.empty:
        return
    order = ["e1_norm_baseline"] + [f"e1_{t}_{lv}" for t in TRAITS for lv in ("high", "low")]
    order = [c for c in order if c in set(e1.config)]
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.6), constrained_layout=True)

    ax = axes[0]
    for k, t in enumerate(TRAITS):
        mus = [e1[e1.config == c][f"real_mu_{t}"].mean() for c in order]
        ax.plot(range(len(order)), mus, marker="o", ms=3.4, lw=1.0,
                color=[BLUE, ORANGE, GREEN, VERM, PURPLE][k], label=t[:5].capitalize() + ".")
    ax.set_xticks(range(len(order)),
                  [c.replace("e1_", "").replace("_", " ") for c in order],
                  rotation=45, ha="right")
    ax.set_ylabel("realised trait mean")
    ax.legend(frameon=False, ncol=2, fontsize=5.8)

    ax = axes[1]
    sds = [e1[e1.config == c].realized_sd_mean.mean() for c in order]
    base = e1[e1.config == "e1_norm_baseline"].realized_sd_mean.mean()
    ax.bar(range(len(order)), sds, color=[GREY if c == "e1_norm_baseline" else BLUE
                                          for c in order], width=0.66)
    ax.axhline(base, color=VERM, lw=1.0, ls="--", label="baseline")
    ax.set_xticks(range(len(order)),
                  [c.replace("e1_", "").replace("_", " ") for c in order],
                  rotation=45, ha="right")
    ax.set_ylabel("realised trait SD (mean over traits)")
    ax.set_ylim(min(sds) * 0.97, max(sds) * 1.02)
    ax.legend(frameon=False, fontsize=6)
    save(fig, "figS1_realised_traits.pdf")


def s2_item_screen():
    p = RES / "ci_item_screen.csv"
    if not p.exists():
        print("skip S2: ci_item_screen.csv missing")
        return
    d = pd.read_csv(p)
    kc = "kept" if "kept" in d.columns else d.columns[-1]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for keep, col, lab, mk in [(True, GREEN, "retained", "o"), (False, VERM, "rejected", "X")]:
        s = d[d[kc].astype(str).str.lower().isin(["true", "1"]) == keep]
        if s.empty:
            continue
        ax.scatter(s.median_rel_error, s.log_spread, s=42, color=col, marker=mk,
                   edgecolor="white", linewidth=0.5, label=lab, zorder=4)
        for _, r in s.iterrows():
            ax.annotate(str(r.item_id).replace("wb_", ""), (r.median_rel_error, r.log_spread),
                        fontsize=5.2, xytext=(3, 3), textcoords="offset points", color=GREY)
    ax.axvline(0.15, color="k", lw=0.9, ls="--")
    ax.axhline(0.02, color="k", lw=0.9, ls="--")
    ax.annotate("headroom threshold", xy=(0.155, ax.get_ylim()[1] * 0.55), fontsize=5.8,
                rotation=90, color=GREY)
    ax.annotate("dispersion threshold", xy=(ax.get_xlim()[1] * 0.42, 0.028), fontsize=5.8,
                color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("solo median relative error")
    ax.set_ylabel("SD of $\\log_{10}$ answers")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "figS2_item_screen.pdf")


def s3_full_surfaces():
    p = RES / "derived_results.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    metrics = [("pol_var", "Opinion variance"), ("pol_extremity", "Extremity"),
               ("crosscut_rate", "Cross-cutting"), ("pol_ei", "Echo closure")]
    fams = [("e3_", "Llama-3.1-8B"), ("e3q_", "Qwen2.5-14B")]
    fig, axes = plt.subplots(2, 4, figsize=(6.69, 3.6), constrained_layout=True)
    for i, (pre, fam) in enumerate(fams):
        e = d[d.config.str.startswith(pre)]
        for j, (m, lab) in enumerate(metrics):
            ax = axes[i, j]
            if e.empty or m not in e:
                ax.axis("off"); continue
            piv = e.groupby(["cond__mu_agreeableness", "cond__mu_openness"])[m].mean().unstack()
            vals = piv.values.astype(float)
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            im = ax.imshow(vals, cmap="RdBu_r", origin="lower", aspect="auto",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns], fontsize=6)
            ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index], fontsize=6)
            for a in range(piv.shape[0]):
                for b in range(piv.shape[1]):
                    v = piv.values[a, b]
                    if np.isfinite(v):
                        # white text on saturated cells, black on pale ones
                        rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        tc = "white" if (rel < 0.18 or rel > 0.82) else "black"
                        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                                color=tc, fontweight="bold")
            if i == 0:
                ax.set_title(lab, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{fam}\n$\\mu_A$", fontsize=6.5)
            if i == 1:
                ax.set_xlabel("$\\mu_O$", fontsize=6.5)
            fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02).ax.tick_params(labelsize=5.2)
    save(fig, "figS3_full_surfaces.pdf")


def s4_hidden_profile_floor():
    p = RES / "derived_results.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).copy()
    d["fam"] = np.where(d.config.str.contains(r"^e\dq_|__qwen", regex=True),
                        "Qwen2.5-14B", "Llama-3.1-8B")
    s = d[["fam", "CI_hidden_profile_rate"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), constrained_layout=True)

    ax = axes[0]
    for k, fam in enumerate(["Llama-3.1-8B", "Qwen2.5-14B"]):
        v = s[s.fam == fam].CI_hidden_profile_rate.values
        ax.scatter(np.full(len(v), k) + np.random.default_rng(0).normal(0, 0.045, len(v)),
                   v, s=9, color=[BLUE, ORANGE][k], alpha=0.45, edgecolor="none")
        ax.plot([k - 0.2, k + 0.2], [v.mean()] * 2, color="k", lw=1.4, zorder=5)
    ax.set_xticks([0, 1], ["Llama-3.1-8B", "Qwen2.5-14B"])
    ax.set_ylabel("hidden-profile correct rate")
    ax.set_title("Every run", fontsize=7.5)

    ax = axes[1]
    for k, fam in enumerate(["Llama-3.1-8B", "Qwen2.5-14B"]):
        v = s[s.fam == fam].CI_hidden_profile_rate.values
        ax.hist(v, bins=np.linspace(0, 0.25, 26), alpha=0.62, color=[BLUE, ORANGE][k],
                label=f"{fam} (mean {v.mean():.3f})")
    ax.set_xlabel("hidden-profile correct rate")
    ax.set_ylabel("runs")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("Distribution", fontsize=7.5)
    save(fig, "figS4_hidden_profile_floor.pdf")


if __name__ == "__main__":
    s1_realised_traits()
    s2_item_screen()
    s3_full_surfaces()
    s4_hidden_profile_floor()
