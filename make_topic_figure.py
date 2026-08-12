#!/usr/bin/env python3
"""make_topic_figure.py — The six-topic replication, as a figure."""
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats as sps
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
BLUE, VERM, GREY, GREEN = "#0072B2", "#D55E00", "#666666", "#009E73"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "legend.fontsize": 6.4,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "pdf.fonttype": 42,
                     "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv(RES / "derived_results.csv")
six = d[d.config.str.startswith(("e1t_", "e2t_"))]
base = six[six.config == "e1t_norm_baseline"]
tcols = [c for c in six.columns if c.startswith("T_") and c.endswith("__var")
         and "filler" not in c]
names = [c.replace("T_", "").replace("__var", "") for c in tcols]
NICE = {"guncontrol": "gun control", "immigration": "immigration", "carbontax": "carbon tax",
        "nuclear": "nuclear", "ubi": "basic income", "socialmedia": "social media"}

eff = {}
for c, n in zip(tcols, names):
    vals = {}
    for cfg in sorted(six.config.unique()):
        a = six[six.config == cfg][["seed", c]].rename(columns={c: "a"})
        b = base[["seed", c]].rename(columns={c: "b"})
        m = a.merge(b, on="seed").dropna()
        if len(m):
            vals[cfg] = float((m.a - m.b).mean())
    eff[n] = vals
P = pd.DataFrame(eff).dropna()

fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.5), constrained_layout=True)

# (a) two-topic vs six-topic effect sizes
S = pd.read_csv(RES / "topic_extension_summary.csv")
ax = axes[0]
for m, col, lab in [("pol_var", BLUE, "opinion variance"),
                    ("crosscut_rate", GREEN, "cross-cutting"),
                    ("pol_extremity", VERM, "extremity")]:
    s = S[S.metric == m]
    ax.scatter(s.two, s.six, s=16, color=col, alpha=0.75, edgecolor="none", label=lab)
lim = 1.05 * max(S.two.abs().max(), S.six.abs().max())
ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8, ls=":")
ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
ax.set_xlabel("effect, two topics"); ax.set_ylabel("effect, six topics")
ax.set_title("(a) effects are preserved", fontsize=7.5)
ax.legend(frameon=False, fontsize=5.8, loc="upper left")

# (b) agreement matrix
ax = axes[1]
M = np.eye(len(P.columns))
for i, a in enumerate(P.columns):
    for j, b in enumerate(P.columns):
        if i != j:
            M[i, j] = sps.spearmanr(P[a], P[b]).statistic
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
lbl = [NICE.get(n, n) for n in P.columns]
ax.set_xticks(range(len(lbl)), lbl, rotation=45, ha="right", fontsize=5.8)
ax.set_yticks(range(len(lbl)), lbl, fontsize=5.8)
for i in range(len(lbl)):
    for j in range(len(lbl)):
        if i != j:
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=5.0,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
ax.set_title("(b) agreement between topics", fontsize=7.5)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=5.4)

# (c) how far composition moves each topic
ax = axes[2]
rows = []
for c, n in zip(tcols, names):
    g = six.groupby("config")[c]
    rows.append((NICE.get(n, n), g.mean().std() / g.std().mean()))
R = pd.DataFrame(rows, columns=["topic", "ratio"]).sort_values("ratio")
cols = [VERM if t == "immigration" else BLUE for t in R.topic]
ax.barh(range(len(R)), R.ratio, color=cols, height=0.68)
ax.set_yticks(range(len(R)), R.topic, fontsize=6)
ax.set_xlabel("between-condition spread /\nwithin-condition noise")
ax.set_title("(c) purchase of composition", fontsize=7.5)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "figD_topics.pdf")
print("wrote", FIG / "figD_topics.pdf")
