"""raw_results.jsonl -> summary.csv, stats.csv, mixed_models.txt, derived_results.csv

INFERENCE HIERARCHY (as pre-registered in the design document):
  PRIMARY      linear mixed-effects model on run x topic (polarization) / run x item (CI)
               observations, seed as random intercept -> contrasts vs baseline.
  PARAMETRIC   paired t-test on run-level values (seeds are matched across conditions).
  CONFIRMATORY Wilcoxon signed-rank.

WILCOXON FLOOR: the signed-rank test with n paired observations cannot produce a
two-sided p below 2/2^n (n=5 -> 0.0625; n=8 -> 0.0078). With 5 seeds it is therefore
mathematically incapable of reaching significance after multiplicity correction, and a
"nothing is significant" verdict from it is an artifact of the test, not evidence of
absence. It is reported alongside, never as the primary test.

MULTIPLICITY: Holm-Bonferroni WITHIN each (experiment family x metric); the E1 trait
contrasts form one family and E3 cells are corrected separately.

Effect sizes (Cohen's dz for paired data, Hedges g, Cliff's delta) and bootstrap CIs on
the mean difference are reported for every contrast regardless of significance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASELINE = "e1_norm_baseline"
# Each experiment family must be contrasted against a baseline run with the SAME model,
# otherwise a Qwen condition differenced against the Llama baseline measures the model,
# not the composition. E5 is excluded from paired contrasts entirely: with 3 seeds its
# t-tests are meaningless (they produce |dz| > 9 artifacts) and its inferential role is
# sign stability, handled by audit_sign_stability.py.
AGGREGATOR_VERSION = "2026-08-regime-standardisation"

FAMILY_BASELINE = {"e1": "e1_norm_baseline", "e2": "e1_norm_baseline", "e3": "e1_norm_baseline",
                   "scale": "e1_norm_baseline",
                   "e1q": "e1q_norm_baseline", "e2q": "e1q_norm_baseline",
                   "e3q": "e1q_norm_baseline",
                   # six-topic replication: contrast against the six-topic baseline, not
                   # the two-topic one, or the topic set becomes part of the difference
                   "e1t": "e1t_norm_baseline", "e2t": "e1t_norm_baseline",
                   # larger-model surface: the centre cell is its own reference,
                   # since no separate baseline condition is run for it
                   "e3l": "e3l_O5_A5"}
SKIP_FAMILIES = {"e5"}
PRIMARY = ["pol_var", "pol_extremity", "pol_assort", "pol_ei", "crosscut_rate",
           "CI_accuracy_z", "CI_hidden_profile_rate", "CI_medrelerr_z", "CI_diversity",
           "CI_mean_gain_vs_pre", "trait_drift_mean"]
N_BOOT = 10000


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def topic_mean(suffix):
        cols = [c for c in df.columns
                if c.endswith(suffix) and c.startswith("T_") and "filler" not in c]
        return df[cols].mean(axis=1) if cols else np.nan

    df["pol_var"] = topic_mean("__var")
    df["pol_extremity"] = topic_mean("__extremity")
    df["pol_assort"] = topic_mean("__assort")
    df["pol_ei"] = -topic_mean("__ei_index")
    df["pol_bimodality"] = topic_mean("__sarle_bc")
    # --- collective-intelligence measures, derived retroactively from per-item columns ---
    # CI_mean_gain_vs_pre (change in accuracy) proved undiscriminating: private estimates
    # barely move pre->post, so the gain is a difference of near-identical numbers and is
    # swamped by noise. Post-discussion accuracy LEVEL discriminates strongly, so it is the
    # primary CI measure; the gain is retained and reported as a null.
    def _model_of(cfg: str) -> str:
        """Which model produced a run, inferred from its condition name.

        Standardisation must be relative to runs from the same model. A z-score taken
        against the whole dataset changes when unrelated models are added, so the same
        runs yield different values as the dataset grows, and the published number stops
        being reproducible from the released data.
        """
        c = str(cfg)
        if "__qwen" in c:
            return "qwen14"
        for pre, tag in [("e3qwen3b_", "qwen3b"), ("e3llama3b_", "llama3b"),
                         ("e1qwen7_", "qwen7"), ("e2qwen7_", "qwen7"), ("e3qwen7_", "qwen7"),
                         ("e1q_", "qwen14"), ("e2q_", "qwen14"), ("e3q_", "qwen14"),
                         ("e3l_", "qwen32")]:
            if c.startswith(pre):
                return tag
        return "primary"

    def _regime(row) -> str:
        """Runs are comparable only if measured the same way on the same model.

        An ablation changes the measurement instrument, so its runs must not enter the
        reference population used to standardise runs measured with the published
        instrument. Grouping on model and measurement regime makes a standardised value
        depend only on runs it is actually comparable with, so adding an experiment
        elsewhere in the programme cannot change a number already reported.
        """
        cfg = str(row.get("config", ""))
        pa = row.get("probe_anchors", True)
        wi = row.get("w_interest", 1.0)
        ie = row.get("interest_on_expressed", False)
        if cfg.startswith("abpr_"):
            pa = False
        elif cfg.startswith("abwi_"):
            wi = 0.0
        elif cfg.startswith("abex_"):
            ie = True
        pa = True if pd.isna(pa) else bool(pa)
        wi = 1.0 if pd.isna(wi) else float(wi)
        ie = False if pd.isna(ie) else bool(ie)
        return f"{_model_of(cfg)}|pa{int(pa)}|wi{wi:g}|ie{int(ie)}"

    df["_model"] = df.apply(_regime, axis=1)
    print(f"[aggregate_results {AGGREGATOR_VERSION}] standardising within "
          f"{df['_model'].nunique()} model-by-regime groups")

    def _neg_z(suffix):
        """Composite of per-item scores, z-scored within each model, then sign-flipped."""
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            return np.nan
        parts = []
        for c in cols:
            g = df.groupby("_model")[c]
            mu, sd = g.transform("mean"), g.transform("std")
            z = (df[c] - mu) / sd.where(sd > 0)
            if z.notna().any():
                parts.append(z)
        return -pd.concat(parts, axis=1).mean(axis=1) if parts else np.nan

    df["CI_accuracy_z"] = _neg_z("__post_collective_sqerr")   # higher = more accurate
    df["CI_medrelerr_z"] = _neg_z("__post_median_relerr")     # higher = more accurate
    div = [c for c in df.columns if c.endswith("__post_diversity")]
    df["CI_diversity"] = df[div].mean(axis=1) if div else np.nan

    for comp, cols in [("POL_composite", ["pol_var", "pol_extremity", "pol_assort", "pol_ei"]),
                       ("CI_composite", ["CI_accuracy_z", "CI_hidden_profile_rate"])]:
        zs = []
        for c in cols:
            if c not in df:
                continue
            g = df.groupby("_model")[c]
            mu, sd = g.transform("mean"), g.transform("std")
            z = (df[c] - mu) / sd.where(sd > 0)
            if z.notna().any():
                zs.append(z)
        df[comp] = pd.concat(zs, axis=1).mean(axis=1) if zs else np.nan
    df = df.copy()          # defragment after column additions
    df["family"] = df.config.str.split("_").str[0]
    return df


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def cohens_dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def boot_ci(d, rng, n_boot=N_BOOT, alpha=0.05):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    order = np.argsort(np.where(ok, p, np.inf))
    m, running = int(ok.sum()), 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def wilcoxon_floor(n):
    return 2.0 / (2 ** n) if n and n > 0 else np.nan



def baseline_for(family: str) -> str:
    """Baseline condition for a family.

    Response-surface families run on an additional model have no separate baseline
    condition: the centre cell of the grid is their reference. Falling back to that rule
    lets a new model be added without editing this file, and prevents a contrast from
    silently being taken against a different model's baseline.
    """
    if family in FAMILY_BASELINE:
        return FAMILY_BASELINE[family]
    if family.startswith("e3") and family != "e3":
        return f"{family}_O5_A5"
    if family.startswith(("e1", "e2")) and family not in ("e1", "e2"):
        return f"e1{family[2:]}_norm_baseline"
    # ablation families are contrasted against the ablated baseline, not the published
    # one, so that the switch under test is the only difference within a contrast
    for tag in ("abpr", "abwi", "abex"):
        if family == tag:
            return f"{tag}_e1_norm_baseline"
    return BASELINE

def paired_contrasts(df):
    rng = np.random.default_rng(0)
    rows = []
    for cfg in sorted(df.config.unique()):
        fam = cfg.split("_")[0]
        if fam in SKIP_FAMILIES:
            continue
        base_cfg = baseline_for(fam)
        if cfg == base_cfg:
            continue
        base = df[df.config == base_cfg]
        if base.empty:
            continue
        merged = df[df.config == cfg].merge(base, on="seed", suffixes=("", "_base"))
        for m in PRIMARY:
            if m not in df.columns:
                continue
            a, b = merged[m].astype(float), merged[f"{m}_base"].astype(float)
            ok = a.notna() & b.notna()
            a, b = a[ok].values, b[ok].values
            if len(a) < 2:
                continue
            d = a - b
            t_p = float(sps.ttest_rel(a, b).pvalue) if d.std(ddof=1) > 0 else np.nan
            try:
                w_p = float(sps.wilcoxon(a, b).pvalue)
            except ValueError:
                w_p = 1.0
            lo, hi = boot_ci(d, rng)
            rows.append(dict(family=fam, config=cfg, baseline=base_cfg, metric=m, n_pairs=len(a),
                             mean_baseline=float(b.mean()), mean_condition=float(a.mean()),
                             mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi,
                             cohens_dz=cohens_dz(d), hedges_g=hedges_g(a, b),
                             cliffs_delta=cliffs_delta(a, b), p_ttest=t_p, p_wilcoxon=w_p,
                             wilcoxon_min_possible_p=wilcoxon_floor(len(a))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for _, g in out.groupby(["family", "metric"]):
        out.loc[g.index, "p_holm"] = holm(g["p_ttest"].values)
    return out.sort_values(["family", "metric", "p_holm"])


def mixed_models(df, path):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        path.write_text("statsmodels not installed; see stats.csv (paired t-tests).\n")
        return
    lines = []
    specs = [("pol_var", [c for c in df.columns
                          if c.endswith("__var") and c.startswith("T_") and "filler" not in c]),
             ("CI_gain", [c for c in df.columns if c.endswith("__gain_vs_pre")])]
    for label, cols in specs:
        if not cols:
            continue
        for fam in sorted(df.family.dropna().unique()):
            if fam in SKIP_FAMILIES:
                continue
            base_cfg = baseline_for(fam)
            sub = df[(df.family == fam) | (df.config == base_cfg)]
            long = sub.melt(id_vars=["config", "seed"], value_vars=cols,
                            var_name="unit", value_name="value").dropna()
            if long.config.nunique() < 2 or len(long) < 20:
                continue
            try:
                md = smf.mixedlm(f"value ~ C(config, Treatment('{base_cfg}'))", long,
                                 groups=long["seed"]).fit(reml=True)
                lines.append(f"===== {label} | family {fam} | n_obs={len(long)} =====\n{md.summary()}")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"===== {label} | family {fam} =====\nmixedlm failed: {exc}")
    path.write_text("\n\n".join(lines) or "insufficient data for mixed models\n")


def main(raw=None):
    sys.path.insert(0, str(ROOT / "src"))
    from traitmix.utils import read_rows
    df = read_rows(raw)
    if df.empty:
        sys.exit("no results found - run experiments first.")
    df = derive(df.drop_duplicates(subset=["run_id"], keep="last"))

    metrics = [m for m in PRIMARY + ["POL_composite", "CI_composite", "pol_bimodality",
                                     "filler_variance", "distinct2", "self_bleu3", "runtime_s"]
               if m in df.columns]
    summary = df.groupby("config")[metrics].agg(["mean", "std", "count"])
    summary.columns = ["__".join(c) for c in summary.columns]
    summary.reset_index().to_csv(RES / "summary.csv", index=False)

    stats = paired_contrasts(df)
    stats.to_csv(RES / "stats.csv", index=False)
    mixed_models(df, RES / "mixed_models.txt")
    df.to_csv(RES / "derived_results.csv", index=False)

    n_seeds = int(df.groupby("config").size().min())
    print(f"configs={df.config.nunique()} runs={len(df)} min_seeds={n_seeds}")
    print(f"Wilcoxon floor at n={n_seeds}: smallest attainable two-sided p = "
          f"{wilcoxon_floor(n_seeds):.4f}; primary inference is the paired t-test / mixed model.")
    if stats.empty:
        return df

    sig = stats[stats.p_holm < 0.05]
    print(f"\nSIGNIFICANT after Holm within family x metric: {len(sig)} of {len(stats)}")
    if len(sig):
        print(sig[["family", "config", "metric", "mean_diff", "ci_lo", "ci_hi",
                   "cohens_dz", "p_ttest", "p_holm"]].round(4).to_string(index=False))

    band = stats[(stats.p_holm >= 0.05) & (stats.p_holm < 0.15)]
    if len(band):
        print("\nSEED-EXTENSION RULE (pre-registered): extend these contrasts to 8 seeds, once:")
        print(band[["family", "config", "metric", "mean_diff", "cohens_dz", "p_holm"]]
              .round(4).to_string(index=False))
        print("\nConfigs to re-run with seeds 6,7,8:")
        print("  " + " ".join(sorted(band.config.unique())))

    big = stats[(stats.cohens_dz.abs() > 0.8) & (stats.p_holm >= 0.05)]
    if len(big):
        print(f"\nLARGE effects not reaching corrected significance ({len(big)}) - "
              "power-limited; report with effect sizes and CIs:")
        print(big[["family", "config", "metric", "mean_diff", "cohens_dz", "p_holm"]]
              .round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
