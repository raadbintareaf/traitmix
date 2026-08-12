#!/usr/bin/env python3
"""
analyse_topics.py — Does the finding survive a wider set of contested topics?

The reviewer's objection is that averaging two anti-correlated topics makes the reported
effect sizes a property of that topic pair rather than of composition. The six-topic
replication of E1 and E2 lets us test that directly. This script answers four questions.

  Q1  Do the trait-level and heterogeneity effects hold when six topics are averaged
      instead of two? Effects are re-estimated on the six-topic runs and placed beside the
      published two-topic estimates for the same conditions.

  Q2  How much do effects vary between topics? For each condition and each measure we
      compute the effect separately per topic and report its spread, so the reader can see
      whether the average conceals disagreement.

  Q3  Is the anti-correlation reported in the two-topic study a general property or a
      feature of that particular pair? With six topics we can compute agreement across all
      fifteen pairs rather than a single correlation.

  Q4  How is the variance in effect size divided between composition and topic? A two-way
      decomposition puts a number on which matters more.

Usage:  python analyse_topics.py
Writes results/topic_extension_summary.csv and prints the four analyses.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
POL = ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]
LABEL = {"pol_var": "opinion variance", "pol_extremity": "extremity",
         "crosscut_rate": "cross-cutting", "pol_ei": "echo closure"}


def topic_cols(d, metric):
    """Per-topic columns for a metric, e.g. T_guncontrol__var for pol_var."""
    suffix = {"pol_var": "__var", "pol_extremity": "__extremity"}.get(metric)
    if suffix is None:
        return []
    return [c for c in d.columns
            if c.startswith("T_") and c.endswith(suffix) and "filler" not in c]


def paired(d, cfg, base, metric):
    """Mean paired difference and a t test, matched on seed."""
    a = d[d.config == cfg][["seed", metric]].rename(columns={metric: "a"})
    b = d[d.config == base][["seed", metric]].rename(columns={metric: "b"})
    m = a.merge(b, on="seed").dropna()
    if len(m) < 3:
        return np.nan, np.nan, 0
    diff = m.a - m.b
    if diff.std(ddof=1) == 0:
        return float(diff.mean()), np.nan, len(m)
    return float(diff.mean()), float(sps.ttest_rel(m.a, m.b).pvalue), len(m)


def main() -> None:
    p = RES / "derived_results.csv"
    if not p.exists():
        sys.exit("run analysis/aggregate_results.py first")
    d = pd.read_csv(p)
    six = d[d.config.str.startswith(("e1t_", "e2t_"))]
    if six.empty:
        sys.exit("no six-topic runs found (expected configs named e1t_* / e2t_*)")

    tcols = topic_cols(six, "pol_var")
    print("=" * 76)
    print(f"SIX-TOPIC REPLICATION: {len(six)} runs, {six.config.nunique()} conditions, "
          f"{len(tcols)} contested topics")
    print("=" * 76)
    print("  topics:", ", ".join(c.replace("T_", "").replace("__var", "") for c in tcols))
    print("  seeds per condition:",
          sorted(six.groupby("config").seed.nunique().unique()))

    # ---------------- Q1: do the effects hold on six topics? ----------------
    print("\n" + "=" * 76)
    print("Q1  EFFECTS ON SIX TOPICS, BESIDE THE PUBLISHED TWO-TOPIC ESTIMATES")
    print("=" * 76)
    rows = []
    for cfg6 in sorted(c for c in six.config.unique() if not c.endswith("norm_baseline")):
        cfg2 = cfg6.replace("e1t_", "e1_").replace("e2t_", "e2_")
        if d[d.config == cfg2].empty:
            continue
        for m in POL:
            d6, p6, n6 = paired(d, cfg6, "e1t_norm_baseline", m)
            d2, p2, n2 = paired(d, cfg2, "e1_norm_baseline", m)
            if not np.isfinite(d6) or not np.isfinite(d2):
                continue
            rows.append(dict(condition=cfg6.replace("e1t_", "").replace("e2t_", ""),
                             metric=m, six=d6, six_p=p6, two=d2, two_p=p2,
                             same_sign=np.sign(d6) == np.sign(d2)))
    R = pd.DataFrame(rows)
    if R.empty:
        sys.exit("could not match six-topic conditions to their two-topic counterparts")
    R.to_csv(RES / "topic_extension_summary.csv", index=False)
    for m in POL:
        s = R[R.metric == m]
        if s.empty:
            continue
        agree = int(s.same_sign.sum())
        r, pr = sps.pearsonr(s.two, s.six) if len(s) > 3 else (np.nan, np.nan)
        print(f"\n  {LABEL[m]}")
        print(f"    sign agreement {agree}/{len(s)} conditions | "
              f"correlation of effect sizes r = {r:+.3f} (p = {pr:.4f})")
        big = s.reindex(s.two.abs().sort_values(ascending=False).index).head(4)
        for _, x in big.iterrows():
            star6 = "*" if np.isfinite(x.six_p) and x.six_p < .05 else " "
            star2 = "*" if np.isfinite(x.two_p) and x.two_p < .05 else " "
            print(f"      {x.condition:26s} two-topic {x.two:+.3f}{star2}   "
                  f"six-topic {x.six:+.3f}{star6}")

    # ---------------- Q2: how much do effects vary by topic? ----------------
    print("\n" + "=" * 76)
    print("Q2  SPREAD OF THE SAME EFFECT ACROSS TOPICS")
    print("=" * 76)
    base = six[six.config == "e1t_norm_baseline"]
    for cfg in sorted(c for c in six.config.unique() if not c.endswith("norm_baseline")):
        eff = []
        for c in tcols:
            a = six[six.config == cfg][["seed", c]].rename(columns={c: "a"})
            b = base[["seed", c]].rename(columns={c: "b"})
            m = a.merge(b, on="seed").dropna()
            if len(m):
                eff.append(float((m.a - m.b).mean()))
        if len(eff) >= 4:
            eff = np.array(eff)
            flips = int((np.sign(eff) != np.sign(eff.mean())).sum())
            print(f"  {cfg.replace('e1t_','').replace('e2t_',''):26s} "
                  f"mean {eff.mean():+.3f}  range [{eff.min():+.3f}, {eff.max():+.3f}]  "
                  f"topics against the mean sign: {flips}/{len(eff)}")

    # ---------------- Q3: agreement across all topic pairs ----------------
    print("\n" + "=" * 76)
    print("Q3  DO TOPICS AGREE WITH ONE ANOTHER?")
    print("=" * 76)
    per_topic = {}
    for c in tcols:
        vals = {}
        for cfg in sorted(six.config.unique()):
            a = six[six.config == cfg][["seed", c]].rename(columns={c: "a"})
            b = base[["seed", c]].rename(columns={c: "b"})
            m = a.merge(b, on="seed").dropna()
            if len(m):
                vals[cfg] = float((m.a - m.b).mean())
        per_topic[c.replace("T_", "").replace("__var", "")] = vals
    P = pd.DataFrame(per_topic).dropna()
    if len(P) >= 4 and P.shape[1] >= 3:
        rs = []
        print(f"  rank agreement between topic pairs, over {len(P)} conditions:")
        for a, b in combinations(P.columns, 2):
            rho = sps.spearmanr(P[a], P[b]).statistic
            rs.append(rho)
            print(f"    {a:14s} vs {b:14s} rho = {rho:+.3f}")
        rs = np.array(rs)
        print(f"\n  mean pairwise rank agreement = {rs.mean():+.3f} "
              f"({int((rs > 0).sum())}/{len(rs)} pairs positive)")
        print("  INTERPRETATION: a mean near zero means composition acts on each topic")
        print("    differently, so topic-averaged effects describe the topic set as much")
        print("    as the composition. Clearly positive means the two-topic result was a")
        print("    property of that pair and the effects do generalise.")

    # ---------------- Q4: variance decomposition ----------------
    print("\n" + "=" * 76)
    print("Q4  COMPOSITION OR TOPIC: WHICH EXPLAINS MORE?")
    print("=" * 76)
    long = []
    for c in tcols:
        for cfg in sorted(six.config.unique()):
            for _, r in six[six.config == cfg].iterrows():
                if np.isfinite(r.get(c, np.nan)):
                    long.append(dict(topic=c, config=cfg, seed=r.seed, value=r[c]))
    L = pd.DataFrame(long)
    if not L.empty:
        try:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            m = smf.ols("value ~ C(config) + C(topic)", data=L).fit()
            aov = sm.stats.anova_lm(m, typ=2)
            tot = aov["sum_sq"].sum()
            for term in ["C(config)", "C(topic)"]:
                if term in aov.index:
                    lab = "composition" if "config" in term else "topic"
                    print(f"  {lab:12s} explains {aov.loc[term,'sum_sq']/tot:6.1%} of "
                          f"variance in opinion variance  (F = {aov.loc[term,'F']:.1f}, "
                          f"p = {aov.loc[term,'PR(>F)']:.2e})")
            print(f"  {'residual':12s} {aov.loc['Residual','sum_sq']/tot:6.1%}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (decomposition unavailable: {exc})")

    print(f"\nwrote {RES/'topic_extension_summary.csv'}")


if __name__ == "__main__":
    main()
