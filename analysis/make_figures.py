"""raw/summary -> paper/figures/*.pdf. Journal-sized vector PDFs, Okabe–Ito palette,
error bands from seed std, no in-figure titles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
SINGLE, DOUBLE = (3.3, 2.4), (6.9, 2.6)

def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print("wrote", FIG / name)

def fig_timeseries(df):
    ts_files = list((RES / "timeseries").glob("*.csv"))
    if not ts_files: return
    id2cfg = dict(zip(df.run_id, df.config))
    frames = []
    for f in ts_files:
        cfg = id2cfg.get(f.stem)
        if cfg is None: continue
        t = pd.read_csv(f); t = t[~t.topic.str.contains("filler")]
        v = t.groupby(["round"]).opinion.var().rename("var").reset_index(); v["config"] = cfg
        frames.append(v)
    if not frames: return
    allv = pd.concat(frames)
    fig, ax = plt.subplots(figsize=SINGLE)
    for k, (cfg, g) in enumerate(sorted(allv.groupby("config"))[:8]):
        m = g.groupby("round")["var"].agg(["mean", "std"])
        ax.plot(m.index, m["mean"], color=OKABE[k % 8], label=cfg.replace("e1_", ""),
                marker="osv^D<>*"[k % 8], ms=3, lw=1.2)
        ax.fill_between(m.index, m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0),
                        color=OKABE[k % 8], alpha=0.15, lw=0)
    ax.set_xlabel("Round"); ax.set_ylabel("Opinion variance")
    ax.legend(ncol=2, frameon=False)
    _save(fig, "fig_timeseries_variance.pdf")

def fig_e1_forest():
    p = RES / "stats.csv"
    if not p.exists(): return
    st = pd.read_csv(p)
    st = st[st.config.str.startswith("e1_") & ~st.config.str.startswith("e1q")]
    if st.empty: return
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, metric, lab in [(axes[0], "pol_var", "Hedges g: opinion variance"),
                            (axes[1], "CI_mean_gain_vs_pre", "Hedges g: CI gain")]:
        sub = st[st.metric == metric].sort_values("config")
        if sub.empty: continue
        y = np.arange(len(sub))
        colors = [OKABE[3] if p_ < .05 else OKABE[0] for p_ in sub.p_holm.fillna(1)]
        ax.barh(y, sub.hedges_g, color=colors, height=0.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, [c.replace("e1_", "") for c in sub.config])
        ax.set_xlabel(lab)
    _save(fig, "fig_e1_forest.pdf")

def fig_e3_surface(df):
    e3 = df[df.config.str.startswith("e3_")]
    if e3.empty: return
    e3 = e3.copy()
    e3["O"] = e3["cond__mu_openness"]; e3["A"] = e3["cond__mu_agreeableness"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))
    for ax, col, lab in [(axes[0], "POL_composite", "Polarization composite"),
                         (axes[1], "CI_composite", "CI composite")]:
        piv = e3.groupby(["A", "O"])[col].mean().unstack()
        if piv.isna().all().all(): continue
        im = ax.contourf(piv.columns, piv.index, piv.values, levels=12, cmap="RdBu_r")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xlabel("Openness $\\mu$"); ax.set_ylabel("Agreeableness $\\mu$"); ax.set_title(lab, fontsize=8)
    g = e3.groupby("config")[["POL_composite", "CI_composite"]].mean().dropna()
    if not g.empty:
        ax = axes[2]
        ax.scatter(g.POL_composite, g.CI_composite, c=OKABE[0], s=18)
        pareto = []
        for c, r in g.iterrows():
            if not ((g.POL_composite < r.POL_composite) & (g.CI_composite > r.CI_composite)).any():
                pareto.append(c)
        gp = g.loc[pareto].sort_values("POL_composite")
        ax.plot(gp.POL_composite, gp.CI_composite, color=OKABE[3], lw=1.4, marker="o", ms=4,
                label="Pareto frontier")
        ax.set_xlabel("Polarization composite ($\\downarrow$ better)")
        ax.set_ylabel("CI composite ($\\uparrow$ better)"); ax.legend(frameon=False)
    _save(fig, "fig_e3_surface_frontier.pdf")

def fig_ci_decomposition(df):
    cols_ind = [c for c in df.columns if c.endswith("__post_avg_individual_sqerr")]
    if not cols_ind: return
    rows = []
    for _, r in df.iterrows():
        for c in cols_ind:
            item = c.split("__")[0]
            rows.append(dict(config=r.config, item=item,
                             individual=r[c], collective=r.get(f"{item}__post_collective_sqerr"),
                             diversity=r.get(f"{item}__post_diversity")))
    d = pd.DataFrame(rows).groupby("config")[["individual", "collective", "diversity"]].mean().dropna()
    if d.empty: return
    d = d.iloc[:10]
    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.individual, 0.4, color=OKABE[0], label="avg individual err$^2$")
    ax.bar(x + 0.2, d.collective, 0.4, color=OKABE[1], label="collective err$^2$")
    ax.plot(x, d.diversity, color=OKABE[2], marker="D", ms=4, lw=1.2, label="diversity")
    ax.set_xticks(x, [c[:14] for c in d.index], rotation=45, ha="right")
    ax.set_ylabel("log$_{10}$ error$^2$"); ax.legend(frameon=False)
    _save(fig, "fig_ci_decomposition.pdf")

def fig_audit_stability(df):
    e5 = df[df.config.str.startswith("e5_")]
    if e5.empty: return
    e5 = e5.copy()
    e5[["anchor", "pert"]] = e5.config.str.replace("e5_", "", 1).str.split("__", expand=True)
    piv = e5.groupby(["anchor", "pert"]).pol_var.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    fig.colorbar(im, ax=ax, shrink=0.8, label="opinion var")
    _save(fig, "fig_audit_heatmap.pdf")

def fig_validation():
    p = RES / "e0_measured_vs_target.csv"
    if not p.exists(): return
    v = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=SINGLE)
    for k, (tr, g) in enumerate(v.groupby("trait")):
        ax.scatter(g.target, g.measured, s=12, color=OKABE[k % 8], label=tr[:5])
    ax.plot([0, 1], [1, 5], color="k", lw=0.8, ls="--")
    ax.set_xlabel("Targeted trait level"); ax.set_ylabel("Measured (1–5)")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "fig_e0_validation.pdf")

def main():
    p = RES / "derived_results.csv"
    if not p.exists():
        print("run aggregate_results.py first"); return
    df = pd.read_csv(p)
    fig_timeseries(df); fig_e1_forest(); fig_e3_surface(df)
    fig_ci_decomposition(df); fig_audit_stability(df); fig_validation()

if __name__ == "__main__":
    main()
