#!/usr/bin/env python3
"""
make_segregation_figures.py — Three candidate designs for the segregation finding.

The article measures polarization along two axes and argues they are distinct. The
dispersion axis is visualised: agents are coloured by opinion in the society figure, and
opinion densities are plotted over rounds. The segregation axis is not. Cross-cutting
interaction and echo-chamber closure appear only as numbers in tables, even though they
carry half of the central claim and are the measures the recommender ablation was run to
defend. These three designs each visualise that axis, in different ways.

  design A  Structural opportunity against realised contact.
            The reconstructed follower network with ties drawn according to whether the two
            agents ended on opposing sides. Cross-camp ties are abundant in every condition;
            what changes is whether agents use them. Pairs the structure with the reply
            counts recovered for the denominator analysis, so the figure states both the
            opportunity and the behaviour.

  design B  The two axes come apart.
            Every condition placed in the plane of dispersion against segregation. If the
            two were one quantity the conditions would fall on a line; the spread away from
            that line is the claim. Model families are distinguished, so the reader can see
            that the decoupling is not specific to one model.

  design C  Camp formation as flow.
            Agents assigned to a camp at each probe round and drawn as a flow between camps
            over time. Shows whether a society's final division is reached by agents
            crossing sides or by uncommitted agents settling, which the aggregate density
            in the dynamics figure cannot distinguish.

Usage:  python make_segregation_figures.py
Output: paper/figures/seg_design_{A,B,C}.pdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RES, FIG, TS = ROOT / "results", ROOT / "paper" / "figures", ROOT / "results" / "timeseries"

BLUE, ORANGE, GREEN, VERM, PURPLE, GREY = ("#0072B2", "#E69F00", "#009E73",
                                           "#D55E00", "#CC79A7", "#666666")
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "pdf.fonttype": 42,
                     "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.20,
                     "axes.spines.top": False, "axes.spines.right": False})
FULL = 6.69
TOPIC = "T_guncontrol"

PANELS = [("e1_agreeableness_high", "High Agreeableness", 25, 598),
          ("e1_norm_baseline", "Baseline", 72, 435),
          ("e1_neuroticism_high", "High Neuroticism", 119, 583)]


def load(cfg, max_seeds=8):
    der = pd.read_csv(RES / "derived_results.csv", usecols=["run_id", "config", "seed"])
    ids = der[der.config == cfg].sort_values("seed").run_id.tolist()
    out = []
    for i in ids[:max_seeds]:
        f = TS / f"{i}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d = d[d.topic == TOPIC] if TOPIC in set(d.topic) else \
                d[d.topic == sorted(d.topic.unique())[0]]
            out.append((i, d))
    return out


def rebuild_graph(cfg_name, seed):
    """Replay the simulation's draws to recover the network it started from."""
    from traitmix.utils import load_config
    from traitmix import personality as pers
    from traitmix.engine import FIRST, OCC
    from traitmix.network import build
    hits = list((ROOT / "configs").rglob(f"{cfg_name}.yaml"))
    if not hits:
        return None
    cfg = load_config(hits[0])
    n = cfg["society"]["n_agents"]
    rng = np.random.default_rng(seed)
    pers.sample_society(cfg["composition"], n, rng)
    [{"name": f"{rng.choice(FIRST)}_{i}", "age": int(rng.integers(19, 66)),
      "occ": str(rng.choice(OCC))} for i in range(n)]
    return build(cfg["society"]["topology"], n, rng, **cfg["society"].get("topology_kw", {}))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# ----------------------------------------------------------------- design A
def design_a():
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.3),
                             gridspec_kw={"height_ratios": [3, 1]},
                             constrained_layout=True)
    for j, (cfg, title, cross, total) in enumerate(PANELS):
        runs = load(cfg, 1)
        if not runs:
            axes[0, j].axis("off"); axes[1, j].axis("off"); continue
        rid, d = runs[0]
        seed = int(rid.rsplit("_s", 1)[1])
        g = rebuild_graph(cfg, seed)
        last = d["round"].max()
        ops = d[d["round"] == last].set_index("agent").opinion.to_dict()

        ax = axes[0, j]
        ax.set_axis_off(); ax.grid(False)
        pos = nx.spring_layout(g, seed=7, k=0.42, iterations=200)
        same = [(u, v) for u, v in g.edges()
                if np.sign(ops.get(u, 0)) == np.sign(ops.get(v, 0)) != 0]
        cross_e = [(u, v) for u, v in g.edges()
                   if np.sign(ops.get(u, 0)) * np.sign(ops.get(v, 0)) < 0]
        for u, v in same:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#DDDDDD", lw=0.25, zorder=1)
        for u, v in cross_e:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=VERM, lw=0.45, alpha=0.55, zorder=2)
        xs = [pos[i][0] for i in g.nodes()]; ys = [pos[i][1] for i in g.nodes()]
        cs = [BLUE if ops.get(i, 0) < 0 else (ORANGE if ops.get(i, 0) > 0 else "#CCCCCC")
              for i in g.nodes()]
        ax.scatter(xs, ys, s=13, c=cs, edgecolors="white", linewidths=0.3, zorder=4)
        ax.set_title(title, fontsize=7.8, pad=4)
        ax.text(0.5, -0.02, f"{len(cross_e)} cross-camp ties of {g.number_of_edges()}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=GREY)

        ax = axes[1, j]
        ax.barh([0], [total - cross], color="#CFCFCF", height=0.55, label="within camp")
        ax.barh([0], [cross], left=[total - cross], color=VERM, height=0.55,
                label="across camps")
        ax.set_xlim(0, 760); ax.set_yticks([])
        ax.set_xlabel("replies exchanged", fontsize=6.8)
        ax.text(total + 12, 0, f"{cross}/{total}", va="center", fontsize=6.4,
                color=VERM, fontweight="bold")
        ax.grid(axis="y", visible=False)
        if j == 0:
            ax.legend(frameon=False, fontsize=5.9, loc="lower left",
                      bbox_to_anchor=(0, -1.05), ncol=2)
    save(fig, "seg_design_A.pdf")


# ----------------------------------------------------------------- design B
def design_b():
    d = pd.read_csv(RES / "derived_results.csv")
    d = d[~d.config.str.startswith("e3mistral7_")]
    d = d[d.seed < 900]
    d = d[~d.config.str.startswith(("abpr_", "abwi_", "abex_"))]

    def fam(c):
        if c.startswith(("e1qwen7_", "e2qwen7_", "e3qwen7_")): return "Qwen2.5-7B"
        if c.startswith(("e1q_", "e2q_", "e3q_")) or "__qwen" in c: return "Qwen2.5-14B"
        if c.startswith("e3l_"): return "Qwen2.5-32B"
        if c.startswith(("e3qwen3b_", "e3llama3b_")): return "excluded"
        return "Llama-3.1-8B"
    d["family"] = d.config.map(fam)
    g = d.groupby(["config", "family"], as_index=False).agg(
        var=("pol_var", "mean"), cross=("crosscut_rate", "mean"))
    g = g[g.family != "excluded"]

    fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    COL = {"Llama-3.1-8B": BLUE, "Qwen2.5-7B": GREEN,
           "Qwen2.5-14B": ORANGE, "Qwen2.5-32B": PURPLE}
    for f, sub in g.groupby("family"):
        ax.scatter(sub["var"], sub["cross"], s=26, color=COL[f], alpha=0.75,
                   edgecolor="white", linewidth=0.4, label=f, zorder=4)
    prim = g[g.family == "Llama-3.1-8B"]
    b, a = np.polyfit(prim["var"], prim["cross"], 1)
    xs = np.linspace(g["var"].min(), g["var"].max(), 50)
    ax.plot(xs, a + b * xs, color=GREY, lw=1.0, ls="--", zorder=3)
    ax.annotate("if the two axes were one quantity,\nconditions would lie on this line",
                xy=(xs[-8], a + b * xs[-8]), xytext=(0.42, 0.10),
                textcoords="axes fraction", fontsize=5.9, color=GREY,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
    for lab, cfg in [("homogeneous", "e2_homog"), ("diverse", "e2_diverse"),
                     ("high Agreeableness", "e1_agreeableness_high"),
                     ("high Neuroticism", "e1_neuroticism_high")]:
        r = g[g.config == cfg]
        if len(r):
            ax.annotate(lab, xy=(r["var"].iloc[0], r["cross"].iloc[0]),
                        xytext=(4, 5), textcoords="offset points", fontsize=5.9)
    ax.set_xlabel("Opinion variance  (dispersion)")
    ax.set_ylabel("Cross-cutting rate  (segregation)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "seg_design_B.pdf")


# ----------------------------------------------------------------- design C
def design_c():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6), sharey=True,
                             constrained_layout=True)
    CAMPS = [(-3, -1, "oppose", BLUE), (0, 0, "neutral", "#BBBBBB"), (1, 3, "support", ORANGE)]
    for j, (cfg, title, _, _) in enumerate(PANELS):
        runs = load(cfg, 1)
        ax = axes[j]
        if not runs:
            ax.axis("off"); continue
        _, d = runs[0]
        rounds = np.sort(d["round"].unique())
        piv = d.pivot_table(index="agent", columns="round", values="opinion")

        def camp(v):
            for k, (lo, hi, _, _) in enumerate(CAMPS):
                if lo <= v <= hi:
                    return k
            return 1
        C = piv.map(camp)
        base = np.zeros(len(rounds))
        for k, (_, _, lab, col) in enumerate(CAMPS):
            share = (C == k).sum(axis=0).reindex(rounds).values / C.shape[0]
            ax.fill_between(rounds, base, base + share, color=col, alpha=0.85,
                            lw=0.4, edgecolor="white", label=lab if j == 0 else None)
            base = base + share
        # how many agents actually changed camp between consecutive probes
        sw = (C.values[:, 1:] != C.values[:, :-1]).sum(axis=0)
        ax2 = ax.twinx()
        ax2.plot(rounds[1:], sw, color=VERM, lw=1.4, marker="o", ms=3, zorder=6)
        ax2.set_ylim(0, max(sw.max() * 1.6, 10)); ax2.grid(False)
        ax2.tick_params(labelsize=5.8, colors=VERM)
        if j == 2:
            ax2.set_ylabel("agents changing camp", fontsize=6.4, color=VERM)
        else:
            ax2.set_yticklabels([])
        ax.set_xlim(rounds.min(), rounds.max()); ax.set_ylim(0, 1)
        ax.set_xlabel("round"); ax.set_title(title, fontsize=7.6, pad=3)
        if j == 0:
            ax.set_ylabel("share of society")
            ax.legend(frameon=False, fontsize=6, loc="center left")
    save(fig, "seg_design_C.pdf")


if __name__ == "__main__":
    design_a()
    design_b()
    design_c()
