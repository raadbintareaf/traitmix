#!/usr/bin/env python3
"""
reviewer_analyses.py — The analyses requested in the first round of review.

R1  Is the accuracy--cross-cutting association a common-cause artifact?
    The diversity-prediction decomposition states that collective error equals mean
    individual error minus prediction diversity. If composition raises diversity, it lowers
    collective error mechanically, and any correlation between a polarization measure and
    collective accuracy could be common-cause rather than substantive. We therefore partial
    the association on the diversity term, on mean individual error, and on realised trait
    dispersion, singly and jointly, and report what survives.

R2  Response surface at a defensible unit of analysis. Six parameters fitted to nine cell
    means, with p values computed on run-level n, over-states the evidence. We refit on the
    nine cell means with a reduced model, and as a mixed model with cell as the grouping
    factor, and report the interaction with honest degrees of freedom.

R5  Are the polarization measures mechanically dependent? Cross-cutting rate and extremity
    are both constrained by consensus. We report partial correlations of each segregation
    measure with composition holding opinion variance fixed, rather than arguing from the
    pattern of signs.

Usage:  python reviewer_analyses.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ITEMS = ["wb_npl_gdp", "wb_per_forest", "wb_gha_health", "wb_tun_unemp",
         "wb_lka_exports", "wb_pry_internet"]


def zneg(df, suffix):
    """Composite of per-item columns, z-scored then sign-flipped so higher = better."""
    cols = [f"{i}{suffix}" for i in ITEMS if f"{i}{suffix}" in df.columns]
    zs = [(df[c] - df[c].mean()) / df[c].std() for c in cols if df[c].std(skipna=True) > 0]
    return -pd.concat(zs, axis=1).mean(axis=1) if zs else pd.Series(np.nan, index=df.index)


def zpos(df, suffix):
    cols = [f"{i}{suffix}" for i in ITEMS if f"{i}{suffix}" in df.columns]
    zs = [(df[c] - df[c].mean()) / df[c].std() for c in cols if df[c].std(skipna=True) > 0]
    return pd.concat(zs, axis=1).mean(axis=1) if zs else pd.Series(np.nan, index=df.index)


def partial_corr(d, x, y, covars):
    """Correlation of x and y with covars removed from both, by OLS residualisation."""
    s = d[[x, y] + list(covars)].dropna()
    if len(s) < len(covars) + 12:
        return np.nan, np.nan, len(s)
    A = np.column_stack([np.ones(len(s))] + [s[c].values for c in covars])
    rx = s[x].values - A @ np.linalg.lstsq(A, s[x].values, rcond=None)[0]
    ry = s[y].values - A @ np.linalg.lstsq(A, s[y].values, rcond=None)[0]
    r, _ = sps.pearsonr(rx, ry)
    dof = len(s) - len(covars) - 2
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, 2 * sps.t.sf(abs(t), dof), len(s)


def main():
    d = pd.read_csv(RES / "derived_results.csv").copy()
    real = pd.read_csv(RES / "realized_traits.csv")
    d = d.merge(real[["config", "seed", "realized_sd_mean"]], on=["config", "seed"], how="left")
    d["is_q"] = d.config.str.contains(r"^e\dq_|__qwen", regex=True)
    L = d[~d.is_q].copy()

    # components of the diversity-prediction decomposition
    L["acc"] = zneg(L, "__post_collective_sqerr")        # higher = more accurate crowd
    L["indiv"] = zneg(L, "__post_avg_individual_sqerr")  # higher = better individuals
    L["divers"] = zpos(L, "__post_diversity")            # higher = more varied estimates

    print("=" * 78)
    print("R1  IS THE ALIGNMENT A COMMON-CAUSE ARTIFACT?")
    print("=" * 78)
    print("\nThe decomposition, verified in these data:")
    chk = L[["acc", "indiv", "divers"]].dropna()
    r_id, _ = sps.pearsonr(chk.indiv, chk.acc)
    r_dv, _ = sps.pearsonr(chk.divers, chk.acc)
    print(f"  individual skill  vs collective accuracy : r = {r_id:+.3f}")
    print(f"  estimate diversity vs collective accuracy: r = {r_dv:+.3f}")
    print(f"  diversity vs individual skill            : r = "
          f"{sps.pearsonr(chk.divers, chk.indiv)[0]:+.3f}")

    print("\nCross-cutting rate vs collective accuracy, progressively controlled:")
    tests = [
        ("raw (as reported in the manuscript)", []),
        ("| estimate diversity", ["divers"]),
        ("| mean individual skill", ["indiv"]),
        ("| realised trait SD", ["realized_sd_mean"]),
        ("| diversity + individual skill", ["divers", "indiv"]),
        ("| diversity + individual skill + trait SD", ["divers", "indiv", "realized_sd_mean"]),
    ]
    for lab, cov in tests:
        r, p, n = partial_corr(L, "crosscut_rate", "acc", cov)
        flag = "" if not np.isfinite(p) else ("  SURVIVES" if p < 0.05 else "  n.s.")
        print(f"  {lab:44s} r = {r:+.3f}  p = {p:.4f}  n = {n}{flag}")

    print("\nSame test for the other polarization measures:")
    for m in ["pol_var", "pol_ei", "pol_extremity"]:
        r0, p0, _ = partial_corr(L, m, "acc", [])
        r1, p1, n1 = partial_corr(L, m, "acc", ["divers", "indiv", "realized_sd_mean"])
        print(f"  {m:16s} raw r = {r0:+.3f} (p={p0:.4f})  ->  "
              f"controlled r = {r1:+.3f} (p={p1:.4f})")

    print("\nDoes cross-cutting predict accuracy WITHIN conditions?")
    print("  (composition held constant, so a common cause cannot drive it)")
    rs, ns = [], 0
    for cfg, g in L.groupby("config"):
        s = g[["crosscut_rate", "acc"]].dropna()
        if len(s) >= 6:
            rs.append(sps.pearsonr(s.crosscut_rate, s.acc)[0]); ns += 1
    if rs:
        rs = np.array(rs)
        t, p = sps.ttest_1samp(rs, 0)
        print(f"  mean within-condition r = {rs.mean():+.3f} over {ns} conditions, "
              f"t = {t:.2f}, p = {p:.4f}")
        print(f"  conditions with positive r: {int((rs > 0).sum())}/{ns}")

    print()
    print("=" * 78)
    print("R2  RESPONSE SURFACE AT A DEFENSIBLE UNIT")
    print("=" * 78)
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels unavailable"); return
    for pre, fam in [("e3_", "Llama-3.1-8B"), ("e3q_", "Qwen2.5-14B")]:
        e = d[d.config.str.startswith(pre)].copy()
        if e.empty:
            continue
        e["O"] = e["cond__mu_openness"]; e["A"] = e["cond__mu_agreeableness"]
        print(f"\n{fam}")
        m_run = smf.ols("pol_var ~ O + A + O:A + I(O**2) + I(A**2)", data=e).fit()
        print(f"  as reported (6 params, run-level n={int(m_run.nobs)}): "
              f"b(O:A) = {m_run.params['O:A']:+.3f}, p = {m_run.pvalues['O:A']:.2e}")
        cells = e.groupby(["O", "A"], as_index=False).pol_var.mean()
        m_cell = smf.ols("pol_var ~ O + A + O:A", data=cells).fit()
        print(f"  refit on {len(cells)} cell means, reduced model (4 params): "
              f"b(O:A) = {m_cell.params['O:A']:+.3f}, p = {m_cell.pvalues['O:A']:.4f}, "
              f"df = {int(m_cell.df_resid)}, R2 = {m_cell.rsquared:.3f}")
        try:
            e["cell"] = e.O.astype(str) + "_" + e.A.astype(str)
            md = smf.mixedlm("pol_var ~ O + A + O:A", e, groups=e["cell"]).fit(reml=True)
            print(f"  mixed model, cell as grouping factor: "
                  f"b(O:A) = {md.params['O:A']:+.3f}, p = {md.pvalues['O:A']:.4f}")
        except Exception as exc:
            print(f"  mixed model did not converge: {exc}")

    print()
    print("=" * 78)
    print("R5  ARE THE POLARIZATION MEASURES MECHANICALLY DEPENDENT?")
    print("=" * 78)
    print("\nPairwise correlations among the measures (Llama runs):")
    ms = ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]
    for a, b in combinations(ms, 2):
        s = L[[a, b]].dropna()
        print(f"  {a:15s} vs {b:15s} r = {sps.pearsonr(s[a], s[b])[0]:+.3f}")
    print("\nEach segregation measure vs condition, holding opinion variance fixed:")
    cond_dummies = pd.get_dummies(L.config, prefix="c", drop_first=True).astype(float)
    LL = pd.concat([L.reset_index(drop=True), cond_dummies.reset_index(drop=True)], axis=1)
    for m in ["crosscut_rate", "pol_ei", "pol_extremity"]:
        s = LL[[m, "pol_var"] + list(cond_dummies.columns)].dropna()
        y = s[m].values
        X0 = np.column_stack([np.ones(len(s)), s["pol_var"].values])
        X1 = np.column_stack([X0, s[list(cond_dummies.columns)].values])
        r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
        r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        ss0, ss1 = (r0 ** 2).sum(), (r1 ** 2).sum()
        k = X1.shape[1] - X0.shape[1]
        F = ((ss0 - ss1) / k) / (ss1 / (len(s) - X1.shape[1]))
        p = sps.f.sf(F, k, len(s) - X1.shape[1])
        print(f"  {m:16s} composition adds R2 = {(ss0-ss1)/ss0:.3f} beyond opinion variance, "
              f"F({k},{len(s)-X1.shape[1]}) = {F:.2f}, p = {p:.2e}")


if __name__ == "__main__":
    main()
