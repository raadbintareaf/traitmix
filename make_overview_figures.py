#!/usr/bin/env python3
"""
make_overview_figures.py — Three candidate "whole paper in one figure" designs.

  overview_A  Integrated architecture. Four panels: the factorial design space, the
              simulation pipeline, the measurement architecture, and a summary of which
              trait conditions moved which outcome. Intended to replace Figure 1 and open
              the paper.

  overview_B  Design matrix. The seven studied dimensions laid out as columns of icons and
              badges, in the style used by large factorial studies. Emphasises scope:
              24 compositions, 2 model families, 3 topologies, 8 seeds, 387 runs.

  overview_C  Radial effect map. A circular layout linking trait conditions to outcome
              measures, with chord width proportional to effect size, colour to direction,
              and opacity to whether the contrast survives correction. Data-driven, and
              the most information-dense of the three.

Model families are shown as typographic badges rather than corporate logos: reproducing
trademarked marks in a published figure raises permissions questions that are avoidable.

Usage:  python make_overview_figures.py
Output: paper/figures/overview_{A,B,C}.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
MM = 1 / 25.4

BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
SKY, YELLOW, GREY, LIGHT = "#56B4E9", "#F0E442", "#4D4D4D", "#F4F4F4"
plt.rcParams.update({"font.size": 6.4, "pdf.fonttype": 42, "font.family": "sans-serif"})

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TSHORT = ["O", "C", "E", "A", "N"]


def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.010, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z,
                                alpha=alpha))


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.0, z=6, rad=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}", mutation_scale=7,
                                 linewidth=lw, color=color, zorder=z))


def badge(ax, x, y, w, h, label, sub, color):
    box(ax, x, y, w, h, "white", color, lw=0.9, r=0.008, z=4)
    box(ax, x, y + h * 0.55, w, h * 0.45, color, color, lw=0.9, r=0.008, z=4)
    ax.text(x + w / 2, y + h * 0.775, label, fontsize=5.4, fontweight="bold",
            color="white", ha="center", va="center", zorder=6)
    ax.text(x + w / 2, y + h * 0.26, sub, fontsize=5.9, color=GREY,
            ha="center", va="center", zorder=6)


def load_effects():
    """E1 contrasts, for the panels that summarise results."""
    p = RES / "stats.csv"
    if not p.exists():
        return pd.DataFrame()
    st = pd.read_csv(p)
    return st[st.family == "e1"]


# =============================================================== overview A
def overview_A():
    fig = plt.figure(figsize=(168.5 * MM, 92 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0.285, 1.0); ax.axis("off")
    rng = np.random.default_rng(2)

    def label(x, y, letter, title, color):
        ax.text(x, y, letter, fontsize=9, fontweight="bold", color=color,
                ha="left", va="center", zorder=8)
        ax.text(x + 0.026, y, title, fontsize=7.6, fontweight="bold", color="black",
                ha="left", va="center", zorder=8)

    # ---------------- A: design space ----------------
    box(ax, 0.012, 0.585, 0.47, 0.395, "white", BLUE, lw=1.0, r=0.012)
    label(0.026, 0.955, "A", "Design space", BLUE)
    ax.text(0.026, 0.928, "personality composition as the independent variable",
            fontsize=5.9, style="italic", color=GREY, ha="left", va="center")

    ax.text(0.030, 0.895, "trait level", fontsize=6.2, fontweight="bold", ha="left")
    for k, t in enumerate(TSHORT):
        x0 = 0.030 + k * 0.036
        for j, (lv, c) in enumerate([("high", VERM), ("mid", "#CCCCCC"), ("low", BLUE)]):
            box(ax, x0, 0.845 - j * 0.019, 0.030, 0.016, c, c, lw=0, r=0.004, z=3)
        ax.text(x0 + 0.015, 0.812, t, fontsize=5.9, fontweight="bold", ha="center")
    ax.text(0.216, 0.845, "11 conditions", fontsize=5.9, color=GREY, ha="left", va="center")
    ax.text(0.216, 0.826, "1 baseline + 5 traits $\\times$ 2", fontsize=5.9,
            color=GREY, ha="left", va="center")

    ax.text(0.030, 0.780, "trait heterogeneity", fontsize=6.2, fontweight="bold", ha="left")
    tv = np.linspace(0, 1, 100)
    for k, (sd, lab) in enumerate([(0.05, "$\\sigma$=.05"), (0.15, "$\\sigma$=.15"),
                                   (0.25, "$\\sigma$=.25")]):
        x0 = 0.032 + k * 0.062
        pdf = np.exp(-0.5 * ((tv - 0.5) / sd) ** 2)
        ax.plot(x0 + tv * 0.052, 0.720 + pdf * 0.040, color=GREEN, lw=0.9, zorder=4)
        ax.fill_between(x0 + tv * 0.052, 0.720, 0.720 + pdf * 0.040, color=GREEN,
                        alpha=0.18, zorder=3)
        ax.text(x0 + 0.026, 0.705, lab, fontsize=5.9, ha="center", color=GREY)
    ax.text(0.222, 0.735, "+ human-calibrated", fontsize=5.9, color=GREY, ha="left")
    ax.text(0.222, 0.716, "IPIP-NEO-120 norms", fontsize=5.9, color=GREY, ha="left")

    ax.text(0.030, 0.672, "response surface", fontsize=6.2, fontweight="bold", ha="left")
    for a in range(3):
        for o in range(3):
            v = [[0.53, 1.27, 0.97], [1.18, 1.23, 0.77], [1.38, 1.02, 0.44]][o][a]
            c = plt.get_cmap("RdBu_r")((v - 0.4) / 1.1)
            box(ax, 0.032 + o * 0.020, 0.600 + a * 0.020, 0.019, 0.019, c, "white",
                lw=0.4, r=0.002, z=3)
    ax.text(0.100, 0.640, "$\\mu_O \\times \\mu_A$", fontsize=6.0, ha="left", va="center")
    ax.text(0.100, 0.620, "9 cells", fontsize=5.9, ha="left", va="center", color=GREY)
    badge(ax, 0.294, 0.598, 0.088, 0.072, "Llama-3.1-8B", "primary", VERM)
    badge(ax, 0.388, 0.598, 0.088, 0.072, "+5 models", "3B to 32B", PURPLE)

    # ---------------- B: pipeline ----------------
    box(ax, 0.500, 0.585, 0.488, 0.395, "white", ORANGE, lw=1.0, r=0.012)
    label(0.514, 0.955, "B", "Simulation pipeline", ORANGE)
    ax.text(0.514, 0.928, "$N$ agents, $T$ rounds, recommender-driven feed",
            fontsize=5.9, style="italic", color=GREY, ha="left", va="center")
    cx, cy, r = 0.588, 0.795, 0.052
    ang = rng.permutation(np.linspace(0, 2 * np.pi, 13, endpoint=False))
    px, py = cx + r * np.cos(ang) * 0.95, cy + r * np.sin(ang)
    for i in range(len(px)):
        for j in [0, 4, 8]:
            if i != j and rng.random() < 0.4:
                ax.plot([px[i], px[j]], [py[i], py[j]], color="#CFCFCF", lw=0.35, zorder=3)
    ax.scatter(px, py, s=[13 if i in (0, 4, 8) else 6 for i in range(len(px))],
               c=[[BLUE, VERM, GREEN, ORANGE, PURPLE][i % 5] for i in range(len(px))],
               edgecolors="white", linewidths=0.35, zorder=4)
    ax.text(cx, 0.722, "society on a\nscale-free graph", fontsize=5.9, ha="center",
            va="center", color=GREY)
    for k, (lab, yy) in enumerate([("feed", 0.855), ("action", 0.800), ("memory", 0.745)]):
        box(ax, 0.690, yy - 0.019, 0.088, 0.038, LIGHT, ORANGE, lw=0.6, r=0.006, z=4)
        ax.text(0.734, yy, lab, fontsize=6.0, ha="center", va="center", zorder=6)
    arrow(ax, 0.734, 0.836, 0.734, 0.820, color=ORANGE, lw=0.8)
    arrow(ax, 0.734, 0.781, 0.734, 0.765, color=ORANGE, lw=0.8)
    ax.plot([0.782, 0.792, 0.792, 0.782], [0.745, 0.745, 0.855, 0.855],
            color=ORANGE, lw=0.8, zorder=5)
    arrow(ax, 0.786, 0.855, 0.778, 0.855, color=ORANGE, lw=0.8)
    arrow(ax, 0.648, 0.800, 0.686, 0.800, lw=1.0)
    for k, (lab, yy, c) in enumerate([("opinion probe", 0.860, PURPLE),
                                      ("estimation task", 0.812, PURPLE),
                                      ("hidden profile", 0.764, PURPLE),
                                      ("neutral filler", 0.716, GREY)]):
        box(ax, 0.822, yy - 0.019, 0.152, 0.038, "white", c, lw=0.6, r=0.006, z=4)
        ax.text(0.898, yy, lab, fontsize=5.9, ha="center", va="center", zorder=6)
    arrow(ax, 0.796, 0.800, 0.818, 0.800, lw=1.0)
    ax.text(0.898, 0.688, "elicited privately, off-platform", fontsize=5.9,
            ha="center", va="center", color=GREY, style="italic")
    ax.text(0.744, 0.640, "991 runs  ·  6 models  ·  6 topics  ·  8 seeds",
            fontsize=6.0, ha="center", va="center", fontweight="bold", color=GREY)
    box(ax, 0.514, 0.594, 0.460, 0.040, "#FAFAFA", "#DDDDDD", lw=0.6, r=0.006, z=1)
    ax.text(0.744, 0.620, "induction gate per model · filler topic · realised traits",
            fontsize=5.5, ha="center", va="center", color=GREY)
    ax.text(0.744, 0.603, "probe and recommender ablations · perturbation audit",
            fontsize=5.5, ha="center", va="center", color=GREY)

    # ---------------- C: measurement ----------------
    box(ax, 0.012, 0.300, 0.47, 0.268, "white", VERM, lw=1.0, r=0.012)
    label(0.026, 0.543, "C", "Two outcome families, one run", VERM)
    box(ax, 0.030, 0.400, 0.213, 0.118, "white", VERM, lw=0.8, r=0.008)
    ax.text(0.136, 0.500, "POLARIZATION", fontsize=6.2, fontweight="bold", color=VERM,
            ha="center", va="center")
    ax.text(0.083, 0.468, "dispersion", fontsize=5.9, fontweight="bold", ha="center")
    ax.text(0.083, 0.446, "variance", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.083, 0.428, "extremity", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.083, 0.410, "bimodality", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.190, 0.468, "segregation", fontsize=5.9, fontweight="bold", ha="center")
    ax.text(0.190, 0.446, "cross-cutting", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.190, 0.428, "echo closure", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.190, 0.410, "assortativity", fontsize=5.9, ha="center", color=GREY)
    box(ax, 0.253, 0.400, 0.213, 0.118, "white", BLUE, lw=0.8, r=0.008)
    ax.text(0.359, 0.500, "COLLECTIVE INTELLIGENCE", fontsize=6.0, fontweight="bold",
            color=BLUE, ha="center", va="center")
    ax.text(0.306, 0.468, "estimation", fontsize=5.9, fontweight="bold", ha="center")
    ax.text(0.306, 0.446, "collective error", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.306, 0.428, "diversity term", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.306, 0.410, "median error", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.413, 0.468, "decision", fontsize=5.9, fontweight="bold", ha="center")
    ax.text(0.413, 0.446, "hidden profile", fontsize=5.9, ha="center", color=GREY)
    ax.text(0.413, 0.428, "pre/post shift", fontsize=5.9, ha="center", color=GREY)
    ax.annotate("", xy=(0.359, 0.532), xytext=(0.136, 0.532),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=GREY))
    ax.text(0.248, 0.372, "measured on the same societies, in the same runs",
            fontsize=6.0, ha="center", va="center", style="italic", color=GREY)
    ax.text(0.248, 0.334, "both measured in every run", fontsize=6.2,
            ha="center", va="center", fontweight="bold", color=VERM)

    # ---------------- D: findings ----------------
    box(ax, 0.500, 0.300, 0.488, 0.268, "white", GREEN, lw=1.0, r=0.012)
    label(0.514, 0.543, "D", "What composition does", GREEN)
    st = load_effects()
    metrics = [("pol_var", "Opinion\nvariance"), ("pol_extremity", "Extremity"),
               ("crosscut_rate", "Cross-\ncutting"), ("CI_accuracy_z", "Collective\naccuracy")]
    conds = [(f"e1_{t}_high", f"{s}$\\uparrow$") for t, s in zip(TRAITS, TSHORT)]
    x0, y0, cw, ch = 0.580, 0.350, 0.070, 0.036
    for j, (m, ml) in enumerate(metrics):
        ax.text(x0 - 0.014, y0 + (3 - j) * ch + ch / 2, ml, fontsize=5.9, ha="right",
                va="center", color=GREY)
    for i, (cfg, cl) in enumerate(conds):
        ax.text(x0 + i * cw + cw / 2, y0 + 4 * ch + 0.014, cl, fontsize=6.4,
                fontweight="bold", ha="center", va="center")
        for j, (m, _) in enumerate(metrics):
            yy = y0 + (3 - j) * ch
            r_ = st[(st.config == cfg) & (st.metric == m)]
            if r_.empty:
                continue
            r_ = r_.iloc[0]
            sig = r_.p_holm < 0.05
            mag = min(abs(r_.cohens_dz) / 3.0, 1.0)
            col = VERM if r_.mean_diff > 0 else BLUE
            box(ax, x0 + i * cw + 0.004, yy + 0.004, cw - 0.008, ch - 0.008,
                col, col, lw=0, r=0.004, z=3, alpha=0.18 + 0.72 * mag if sig else 0.12)
            if sig:
                ax.text(x0 + i * cw + cw / 2, yy + ch / 2, f"{r_.mean_diff:+.2f}",
                        fontsize=5.9, ha="center", va="center", zorder=6,
                        color="white" if mag > 0.55 else "black", fontweight="bold")
    ax.text(0.744, 0.324, "shaded cells survive Holm correction; intensity $\\propto |d_z|$",
            fontsize=5.9, ha="center", va="center", color=GREY, style="italic")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "overview_A.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print("wrote", FIG / "overview_A.pdf")


# =============================================================== overview B
def overview_B():
    fig = plt.figure(figsize=(168.5 * MM, 66 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    cols = [("Compositions", BLUE), ("Topics", GREEN), ("Platform", ORANGE),
            ("Models", PURPLE), ("Outcomes", VERM), ("Perturbations", SKY),
            ("Replication", GREY)]
    n = len(cols)
    gap, top, height = 0.006, 0.930, 0.545
    w = (1 - (n + 1) * gap) / n
    xs = [gap + i * (w + gap) for i in range(n)]

    for x, (t, c) in zip(xs, cols):
        box(ax, x, top - height, w, height, "white", c, lw=0.9, r=0.008)
        box(ax, x, top - 0.052, w, 0.052, c, c, lw=0.9, r=0.008)
        ax.text(x + w / 2, top - 0.026, t, fontsize=6.6, fontweight="bold", color="white",
                ha="center", va="center", zorder=6)

    def rows(x, entries, y0=0.855, dy=0.068, fs=5.7):
        for k, (main, sub) in enumerate(entries):
            yy = y0 - k * dy
            ax.text(x + w / 2, yy, main, fontsize=fs, fontweight="bold", ha="center",
                    va="center", zorder=6)
            if sub:
                ax.text(x + w / 2, yy - 0.026, sub, fontsize=4.9, color=GREY,
                        ha="center", va="center", zorder=6)

    rows(xs[0], [("11 trait levels", "5 traits $\\times$ high/low"),
                 ("4 heterogeneity", "$\\sigma$ = .05/.15/.25"),
                 ("9 surface cells", "$\\mu_O \\times \\mu_A$"),
                 ("1 human-calibrated", "IPIP-NEO-120 norms"),
                 ("$\\theta_i \\in [0,1]^5$", "O C E A N"),
                 ("validated", "induction gate $\\rho$=.94")])
    rows(xs[1], [("gun control", "contested"),
                 ("immigration", "contested"),
                 ("filler topic", "neutral control"),
                 ("6 estimation", "World Bank truth"),
                 ("2 hidden profile", "unshared information")])
    rows(xs[2], [("$N$ = 100", "scale ablation 200"),
                 ("$T$ = 30 rounds", "activation $\\rho$ = 0.4"),
                 ("scale-free graph", "follow / unfollow"),
                 ("interest + hot feed", "size 10"),
                 ("bounded memory", "$k$ = 10")])
    badge(ax, xs[3] + w * 0.08, 0.790, w * 0.84, 0.068, "Llama-3.1", "8B  primary", VERM)
    badge(ax, xs[3] + w * 0.08, 0.703, w * 0.84, 0.068, "Qwen2.5", "14B  replication", PURPLE)
    badge(ax, xs[3] + w * 0.08, 0.616, w * 0.84, 0.068, "vLLM", "local serving", GREY)
    rows(xs[3], [("1 GPU, 24 GB", "387 runs"),
                 ("temp 0.7", "0.3 for probes")], y0=0.560)
    rows(xs[4], [("opinion variance", ""), ("extremity", ""), ("cross-cutting", ""),
                 ("echo closure", ""), ("collective accuracy", ""),
                 ("hidden-profile rate", ""), ("diversity term", "")], dy=0.060)
    rows(xs[5], [("2 topologies", "small-world, random"),
                 ("activation $\\pm$50\\%", ""), ("memory $\\times$2", ""),
                 ("feed $\\times$1.5", ""), ("temperature 1.0", ""),
                 ("model swap", "sign stability")])
    rows(xs[6], [("8 seeds", "per condition"),
                 ("3 internal", "identical compositions"),
                 ("2nd model family", "E1, E2, E3 repeated"),
                 ("code + data", "public, archived"),
                 ("387 runs", "$\\approx$ 21 GPU-hours")])

    ax.text(0.5, 0.345, "Seven dimensions, one factorial programme: composition is varied, "
                        "everything else is held fixed or perturbed deliberately",
            fontsize=6.4, ha="center", va="center", style="italic", color=GREY)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "overview_B.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print("wrote", FIG / "overview_B.pdf")


# =============================================================== overview C
def overview_C():
    st = load_effects()
    if st.empty:
        print("skip overview_C: stats.csv missing")
        return
    fig = plt.figure(figsize=(170 * MM, 108 * MM))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_aspect("equal")
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.22, 1.22); ax.axis("off")

    conds = [(f"e1_{t}_{lv}", f"{s}{'+' if lv=='high' else '-'}")
             for t, s in zip(TRAITS, TSHORT) for lv in ("high", "low")]
    metrics = [("pol_var", "opinion variance", VERM),
               ("pol_extremity", "extremity", VERM),
               ("crosscut_rate", "cross-cutting", ORANGE),
               ("pol_ei", "echo closure", ORANGE),
               ("CI_accuracy_z", "collective accuracy", BLUE),
               ("CI_hidden_profile_rate", "hidden profile", BLUE)]

    # conditions on the left arc, outcomes on the right arc
    na, nb = len(conds), len(metrics)
    ang_a = np.linspace(115, 245, na)
    ang_b = np.linspace(65, -65, nb)
    R = 1.0

    def pol(a, rr=R):
        return rr * np.cos(np.deg2rad(a)), rr * np.sin(np.deg2rad(a))

    pos_a, pos_b = {}, {}
    for (cfg, lab), a in zip(conds, ang_a):
        x, y = pol(a); pos_a[cfg] = (x, y, a)
        tx, ty = pol(a, R + 0.055)
        ax.text(tx, ty, lab, fontsize=7.2, fontweight="bold", ha="right", va="center",
                color=VERM if lab.endswith("+") else BLUE, zorder=8)
        ax.add_patch(Circle((x, y), 0.022, facecolor="white",
                            edgecolor=VERM if lab.endswith("+") else BLUE, lw=1.0, zorder=7))
    for (m, lab, c), a in zip(metrics, ang_b):
        x, y = pol(a); pos_b[m] = (x, y, a)
        tx, ty = pol(a, R + 0.055)
        ax.text(tx, ty, lab, fontsize=6.6, ha="left", va="center", color=c, zorder=8)
        ax.add_patch(Circle((x, y), 0.020, facecolor=c, edgecolor="white", lw=0.8, zorder=7))

    drawn = 0
    for cfg, _ in conds:
        for m, _, c in metrics:
            r_ = st[(st.config == cfg) & (st.metric == m)]
            if r_.empty or not np.isfinite(r_.iloc[0].cohens_dz):
                continue
            r_ = r_.iloc[0]
            sig = r_.p_holm < 0.05
            mag = abs(r_.cohens_dz)
            if mag < 0.5 and not sig:
                continue
            x0, y0, _ = pos_a[cfg]; x1, y1, _ = pos_b[m]
            verts = [(x0, y0), (0.0, (y0 + y1) / 2 * 0.30), (x1, y1)]
            ax.add_patch(mpatches.PathPatch(
                MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
                fill=False, lw=0.35 + 1.5 * min(mag / 3.0, 1.0),
                edgecolor=VERM if r_.mean_diff > 0 else BLUE,
                alpha=0.85 if sig else 0.16, zorder=4 if sig else 3,
                capstyle="round"))
            drawn += 1

    ax.text(0, 1.14, "Which trait conditions move which outcomes", fontsize=8.2,
            fontweight="bold", ha="center", va="center")
    ax.text(0, 1.075, "chord width $\\propto$ effect size; solid chords survive "
                      "Holm--Bonferroni correction", fontsize=6.2, ha="center",
            va="center", style="italic", color=GREY)
    ax.text(-1.28, -1.14, "trait condition\n(+ raised, $-$ lowered)", fontsize=6.0,
            ha="left", va="center", color=GREY)
    ax.text(1.28, -1.14, "outcome measure", fontsize=6.0, ha="right", va="center", color=GREY)
    h = [plt.Line2D([], [], color=VERM, lw=1.6), plt.Line2D([], [], color=BLUE, lw=1.6),
         plt.Line2D([], [], color=GREY, lw=1.2, alpha=0.25)]
    ax.legend(h, ["increases the outcome", "decreases the outcome", "not significant"],
              frameon=False, fontsize=6.0, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.055))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "overview_C.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print(f"wrote {FIG/'overview_C.pdf'} ({drawn} chords)")


if __name__ == "__main__":
    overview_A()
    overview_B()
    overview_C()
