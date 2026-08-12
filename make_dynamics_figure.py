#!/usr/bin/env python3
"""
make_dynamics_figure.py — Figure: opinion dynamics, and how they differ by topic.

Rebuilt on the six-topic runs. The earlier two-topic version placed gun control beside
immigration, which we now know is the one topic composition barely moves; showing that pair
alone presented the exception as though it were the rule. This version shows a
representative topic, the outlier explicitly labelled, and every topic's dispersion
trajectory beneath, so a reader can see five topics tracking together and one apart.

Requires results/timeseries/<run_id>.csv for the six-topic conditions.
Usage:  python make_dynamics_figure.py
Output: paper/figures/figC_dynamics.pdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
RES, FIG, TS = ROOT / "results", ROOT / "paper" / "figures", ROOT / "results" / "timeseries"

VERM, GREY = "#D55E00", "#666666"
DENSITY = LinearSegmentedColormap.from_list("d", ["#FFFFFF", "#BFD9EA", "#4F91C4", "#12456B"])
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                     "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

PANELS = [("e1t_agreeableness_high", "High Agreeableness", "insular consensus"),
          ("e1t_norm_baseline", "Baseline", "reference composition"),
          ("e1t_neuroticism_high", "High Neuroticism", "divergent polarization")]
SHOWN = ["T_guncontrol", "T_immigration"]          # representative, then the outlier
NICE = {"T_guncontrol": "gun control", "T_immigration": "immigration",
        "T_carbontax": "carbon tax", "T_nuclear": "nuclear", "T_ubi": "basic income",
        "T_socialmedia": "social media"}
TCOL = {"T_guncontrol": "#0072B2", "T_immigration": "#D55E00", "T_carbontax": "#009E73",
        "T_nuclear": "#CC79A7", "T_ubi": "#56B4E9", "T_socialmedia": "#8C8C00"}


def load_runs(config, max_seeds=8):
    der = pd.read_csv(RES / "derived_results.csv", usecols=["run_id", "config", "seed"])
    ids = der[der.config == config].sort_values("seed").run_id.tolist()
    frames = []
    for k, i in enumerate([i for i in ids if (TS / f"{i}.csv").exists()][:max_seeds]):
        d = pd.read_csv(TS / f"{i}.csv")
        d = d[~d.topic.astype(str).str.contains("filler", case=False, na=False)]
        d["seed_idx"] = k
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    data = {c: load_runs(c) for c, _, _ in PANELS}
    if any(v.empty for v in data.values()):
        sys.exit("six-topic per-round data not found for one or more conditions")
    topics = [t for t in TCOL if t in set(data[PANELS[0][0]].topic)]

    fig, axes = plt.subplots(3, 3, figsize=(6.69, 5.8), sharex=True,
                             constrained_layout=True)
    edges = np.arange(-3.5, 4.5, 1.0)

    for j, (cfg, title, subtitle) in enumerate(PANELS):
        d = data[cfg]
        rounds = np.sort(d["round"].unique())
        rep_seed = d.seed_idx.min()

        for ti, tp in enumerate(SHOWN):
            ax = axes[ti, j]
            rep = d[(d.seed_idx == rep_seed) & (d.topic == tp)]
            H = np.zeros((len(edges) - 1, len(rounds)))
            for ri, r in enumerate(rounds):
                h, _ = np.histogram(rep[rep["round"] == r].opinion.values, bins=edges)
                H[:, ri] = h / max(h.sum(), 1)
            im = ax.imshow(H, aspect="auto", origin="lower", cmap=DENSITY,
                           extent=[rounds.min(), rounds.max(), -3.5, 3.5], vmin=0, vmax=0.6)
            m = rep.groupby("round").opinion.mean()
            ax.plot(m.index, m.values, color=VERM, lw=1.6, zorder=5)
            ax.axhline(0, color=GREY, lw=0.6, ls=":", zorder=4)
            ax.set_ylim(-3.5, 3.5); ax.set_yticks([-3, 0, 3])
            ax.tick_params(labelbottom=False)
            if ti == 0:
                ax.set_title(title, fontsize=7.8, pad=13)
                ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=6.2, style="italic", color=GREY)
            if j == 0:
                note = "representative" if ti == 0 else "least responsive"
                ax.set_ylabel(f"{NICE[tp]}\n({note})", fontsize=6.6)

        ax = axes[2, j]
        for tp in topics:
            cur = []
            for si in sorted(d.seed_idx.unique()):
                v = d[(d.seed_idx == si) & (d.topic == tp)].groupby("round").opinion.var()
                cur.append(v.reindex(rounds))
            if not cur:
                continue
            mean = pd.concat(cur, axis=1).mean(axis=1)
            is_imm = tp == "T_immigration"
            ax.plot(rounds, mean.values, color=TCOL[tp], lw=2.0 if is_imm else 1.2,
                    ls="-" if not is_imm else (0, (4, 1.4)), zorder=6 if is_imm else 4,
                    label=NICE[tp])
        ax.set_ylim(0, 3.2)
        ax.set_xlabel("round")
        if j == 0:
            ax.set_ylabel("opinion variance")
            ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left")

    cb = fig.colorbar(im, ax=axes[:2, :], shrink=0.55, pad=0.012)
    cb.set_label("share of agents", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "figC_dynamics.pdf")
    plt.close(fig)
    print("wrote", FIG / "figC_dynamics.pdf")


if __name__ == "__main__":
    main()
