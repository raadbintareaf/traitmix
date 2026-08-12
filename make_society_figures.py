#!/usr/bin/env python3
"""
make_society_figures.py — Three candidate figures showing the 100-agent society itself.

  society_1  "The society over time". The reconstructed follower network at four rounds,
             with each agent coloured by its privately held opinion. Node size scales with
             in-degree, so the hubs that dominate the feed are visible. Two conditions are
             shown one above the other, which makes the difference between convergence and
             a stable split legible as spatial structure rather than as a statistic.

  society_2  "Opinion flow". A stream of the whole population across rounds, each band an
             opinion level, with the population split into the two camps above and below
             the neutral line. Shows the formation, or the failure, of consensus.

  society_3  "Every agent, every round". A 100 by 31 fingerprint: one row per agent, one
             column per round, sorted by final position, with the network neighbourhood
             composition beside it. The most information-dense of the three, and the only
             one in which individual agents remain traceable.

The follower network is reconstructed exactly from the condition and seed, replaying the
same random draws the simulation made. Edges are therefore the network as initialised;
agents also follow and unfollow during a run, so the rendered ties are the starting
structure rather than the final one, which the captions state.

Usage:  python make_society_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RES, TS, FIG = ROOT / "results", ROOT / "results" / "timeseries", ROOT / "paper" / "figures"

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
OPINION = LinearSegmentedColormap.from_list("op", ["#08519C", "#6BAED6", "#F0F0F0",
                                                  "#FC9272", "#A63603"])
NORM = Normalize(-3, 3)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.4,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "pdf.fonttype": 42,
                     "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

PAIR = [("e1_agreeableness_high", "High Agreeableness"),
        ("e1_neuroticism_high", "High Neuroticism")]

# the full-OCEAN variant: baseline plus each trait raised, one row each
OCEAN_LOW = [("e1_norm_baseline", "Baseline"),
             ("e1_openness_low", "Low Openness"),
             ("e1_conscientiousness_low", "Low Conscientiousness"),
             ("e1_extraversion_low", "Low Extraversion"),
             ("e1_agreeableness_low", "Low Agreeableness"),
             ("e1_neuroticism_low", "Low Neuroticism")]

OCEAN = [("e1_norm_baseline", "Baseline"),
         ("e1_openness_high", "High Openness"),
         ("e1_conscientiousness_high", "High Conscientiousness"),
         ("e1_extraversion_high", "High Extraversion"),
         ("e1_agreeableness_high", "High Agreeableness"),
         ("e1_neuroticism_high", "High Neuroticism")]
TOPIC = "T_guncontrol"


def rebuild_graph(config_name, seed):
    """Replay the simulation's random draws to recover the network it started from."""
    from traitmix.utils import load_config
    from traitmix import personality as pers
    from traitmix.engine import FIRST, OCC
    from traitmix.network import build
    hits = list((ROOT / "configs").rglob(f"{config_name}.yaml"))
    if not hits:
        return None
    cfg = load_config(hits[0])
    n = cfg["society"]["n_agents"]
    rng = np.random.default_rng(seed)
    pers.sample_society(cfg["composition"], n, rng)
    [{"name": f"{rng.choice(FIRST)}_{i}", "age": int(rng.integers(19, 66)),
      "occ": str(rng.choice(OCC))} for i in range(n)]
    return build(cfg["society"]["topology"], n, rng, **cfg["society"].get("topology_kw", {}))


def one_run(config_name):
    """The per-round opinions of the first analysed run of a condition, one topic."""
    der = pd.read_csv(RES / "derived_results.csv", usecols=["run_id", "config", "seed"])
    rows = der[der.config == config_name].sort_values("seed")
    for _, r in rows.iterrows():
        f = TS / f"{r.run_id}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d.topic == TOPIC] if TOPIC in set(d.topic) else d[d.topic == sorted(d.topic.unique())[0]]
            return d, int(r.seed)
    return None, None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# ------------------------------------------------------------------ candidate 1
def society_1(panels=PAIR, name="society_1_network.pdf", height=3.5, nodescale=1.0):
    rounds_to_show = [0, 10, 20, 30]
    fig, axes = plt.subplots(len(panels), len(rounds_to_show) + 1,
                             figsize=(6.69, height),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.62]},
                             constrained_layout=True)
    for row, (cfg, title) in enumerate(panels):
        d, seed = one_run(cfg)
        g = rebuild_graph(cfg, seed) if d is not None else None
        if d is None or g is None:
            for ax in axes[row]:
                ax.axis("off")
            continue
        pos = nx.spring_layout(g, seed=7, k=0.42, iterations=220)
        deg = dict(g.degree())
        avail = sorted(d["round"].unique())
        for col, rq in enumerate(rounds_to_show):
            ax = axes[row, col]
            r = min(avail, key=lambda a: abs(a - rq))
            ops = d[d["round"] == r].set_index("agent").opinion
            ax.axis("off")
            for u, v in g.edges():
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color="#D8D8D8", lw=0.18, zorder=1, alpha=0.7)
            xs = [pos[i][0] for i in g.nodes()]
            ys = [pos[i][1] for i in g.nodes()]
            cs = [OPINION(NORM(ops.get(i, 0))) for i in g.nodes()]
            ss = [(7 + 1.5 * deg.get(i, 1)) * nodescale for i in g.nodes()]
            ax.scatter(xs, ys, s=ss, c=cs, edgecolors="white", linewidths=0.25, zorder=3)
            if row == 0:
                ax.set_title(f"round {r}", fontsize=7, pad=2)
            if col == 0:
                ax.text(-0.06, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center",
                        fontsize=7.2 if len(panels) < 4 else 5.9, fontweight="bold")
        ax = axes[row, len(rounds_to_show)]
        last = max(avail)
        v = d[d["round"] == last].opinion.values
        cnt, edges = np.histogram(v, bins=np.arange(-3.5, 4.5))
        centres = (edges[:-1] + edges[1:]) / 2
        ax.barh(centres, cnt, height=0.86,
                color=[OPINION(NORM(c)) for c in centres], edgecolor="white", lw=0.4)
        ax.set_ylim(-3.6, 3.6); ax.set_yticks([-3, 0, 3])
        ax.set_xlabel("agents", fontsize=6.4)
        ax.set_title("final", fontsize=7, pad=2) if row == 0 else None
        ax.tick_params(labelsize=6)
    sm = ScalarMappable(norm=NORM, cmap=OPINION); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:, :].ravel().tolist(), shrink=0.55, pad=0.008,
                      ticks=[-3, 0, 3])
    cb.set_label("opinion", fontsize=6.4); cb.ax.tick_params(labelsize=6)
    if len(panels) > 3:
        # constrained_layout only fixes positions at draw time, so settle it first
        fig.draw_without_rendering()
        p0 = axes[0, 0].get_position()
        p1 = axes[1, 0].get_position()
        pr = axes[0, -1].get_position()
        y = (p0.y0 + p1.y1) / 2
        fig.add_artist(plt.Line2D([p0.x0 - 0.012, pr.x1], [y, y], color="#9A9A9A",
                                  lw=0.7, transform=fig.transFigure, zorder=10))
    save(fig, name)


# ------------------------------------------------------------------ candidate 2
def society_2():
    fig, axes = plt.subplots(1, len(PAIR), figsize=(6.69, 2.5), sharey=True,
                             constrained_layout=True)
    for j, (cfg, title) in enumerate(PAIR):
        ax = axes[j]
        d, _ = one_run(cfg)
        if d is None:
            ax.axis("off"); continue
        rounds = sorted(d["round"].unique())
        levels = list(range(-3, 4))
        counts = np.array([[((d[d["round"] == r].opinion == L).sum()) for r in rounds]
                           for L in levels], dtype=float)
        base_up = np.zeros(len(rounds))
        for L in [0, 1, 2, 3]:
            idx = levels.index(L)
            ax.fill_between(rounds, base_up, base_up + counts[idx],
                            color=OPINION(NORM(L)), lw=0.3, edgecolor="white", zorder=3)
            base_up = base_up + counts[idx]
        base_dn = np.zeros(len(rounds))
        for L in [-1, -2, -3]:
            idx = levels.index(L)
            ax.fill_between(rounds, base_dn, base_dn - counts[idx],
                            color=OPINION(NORM(L)), lw=0.3, edgecolor="white", zorder=3)
            base_dn = base_dn - counts[idx]
        ax.axhline(0, color="black", lw=0.8, zorder=5)
        ax.set_xlabel("round")
        ax.set_title(title, fontsize=7.6, pad=3)
        if j == 0:
            ax.set_ylabel("agents  (support $\\uparrow$ / oppose $\\downarrow$)")
        ax.set_xlim(min(rounds), max(rounds))
        ax.text(0.98, 0.94, f"final split {int(counts[3:].sum(axis=0)[-1])}:"
                            f"{int(counts[:3].sum(axis=0)[-1])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=GREY)
    sm = ScalarMappable(norm=NORM, cmap=OPINION); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01, ticks=[-3, 0, 3])
    cb.set_label("opinion", fontsize=6.4); cb.ax.tick_params(labelsize=6)
    save(fig, "society_2_flow.pdf")


# ------------------------------------------------------------------ candidate 3
def society_3():
    fig, axes = plt.subplots(1, 2 * len(PAIR), figsize=(6.69, 3.1),
                             gridspec_kw={"width_ratios": [1, 0.22] * len(PAIR)},
                             constrained_layout=True)
    for j, (cfg, title) in enumerate(PAIR):
        axm, axn = axes[2 * j], axes[2 * j + 1]
        d, seed = one_run(cfg)
        g = rebuild_graph(cfg, seed) if d is not None else None
        if d is None:
            axm.axis("off"); axn.axis("off"); continue
        piv = d.pivot_table(index="agent", columns="round", values="opinion")
        order = piv[piv.columns[-1]].sort_values(kind="mergesort").index
        piv = piv.loc[order]
        im = axm.imshow(piv.values, aspect="auto", cmap=OPINION, norm=NORM,
                        interpolation="nearest",
                        extent=[piv.columns.min(), piv.columns.max(), 0, len(piv)])
        axm.set_xlabel("round")
        axm.set_title(title, fontsize=7.6, pad=3)
        if j == 0:
            axm.set_ylabel("agents, ordered by final opinion")
        axm.set_yticks([])
        # neighbourhood composition: share of each agent's followees ending on the same side
        finals = piv[piv.columns[-1]]
        same = []
        for a in piv.index:
            nb = [v for u, v in g.edges() if u == a] + [u for u, v in g.edges() if v == a]
            nb = [b for b in nb if b in finals.index]
            if not nb or finals[a] == 0:
                same.append(np.nan); continue
            same.append(np.mean([np.sign(finals[b]) == np.sign(finals[a]) for b in nb]))
        y = np.arange(len(piv)) + 0.5
        axn.barh(y, same, height=1.0, color=[OPINION(NORM(v)) for v in finals],
                 edgecolor="none")
        axn.axvline(0.5, color="black", lw=0.7, ls=":")
        axn.set_ylim(0, len(piv)); axn.set_xlim(0, 1)
        axn.set_yticks([]); axn.set_xticks([0, 0.5, 1])
        axn.set_xlabel("same-side\nneighbours", fontsize=6.2)
        axn.tick_params(labelsize=5.8)
    sm = ScalarMappable(norm=NORM, cmap=OPINION); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.01, ticks=[-3, 0, 3])
    cb.set_label("opinion", fontsize=6.4); cb.ax.tick_params(labelsize=6)
    save(fig, "society_3_fingerprint.pdf")


if __name__ == "__main__":
    if not TS.exists() or not any(TS.glob("*.csv")):
        sys.exit("needs results/timeseries/*.csv")
    society_1()
    society_1(OCEAN, "society_1b_ocean.pdf", height=7.05, nodescale=0.52)
    society_1(OCEAN_LOW, "figS5_society_low.pdf", height=7.05, nodescale=0.52)
    society_2()
    society_3()
