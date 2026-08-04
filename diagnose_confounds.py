#!/usr/bin/env python3
"""
diagnose_confounds.py — Stress-test the E1/E3 findings against two alternative explanations.
Runs entirely offline on completed results; no GPU, no re-simulation.

CONFOUND 1 - RESPONSE-STYLE PRIMING.
    pol_extremity rose in almost every non-baseline condition, regardless of which trait
    was manipulated. A trait-specific effect should not behave that way. If conditions
    whose prompts contain "extremely/very high|low" also produce more extreme answers on
    the NEUTRAL FILLER topic ("pineapple on pizza"), the effect is generic response-style
    priming rather than opinion polarization. The filler topic is a pre-built control.

CONFOUND 2 - TRUNCATION OF THE TRAIT DISTRIBUTION.
    Traits are drawn from a truncated normal on [0,1]. At mu=0.5 nothing is clipped
    (realized sd ~ 0.149), but at mu=0.2/0.8 one tail is clipped (realized sd ~ 0.128).
    Extreme-mean conditions are therefore automatically LESS heterogeneous, so an apparent
    effect of trait LEVEL may really be an effect of trait SPREAD - the variable E2 is
    meant to isolate. Trait sampling is deterministic given (config, seed), so the
    realized distributions are reconstructed exactly here without re-running anything.

Usage (from repo root):  python diagnose_confounds.py
Writes results/realized_traits.csv and prints the two diagnostics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traitmix import personality as pers  # noqa: E402
from traitmix.utils import load_config, read_rows  # noqa: E402


def realized_traits(config_path: Path, seed: int, n_agents: int) -> dict:
    """Reproduce the trait matrix a run actually used.

    Mirrors Simulation.fresh_state: rng = default_rng(seed), then sample_society is the
    first draw from it, so the reconstruction is exact.
    """
    cfg = load_config(config_path)
    rng = np.random.default_rng(seed)
    theta = pers.sample_society(cfg["composition"], n_agents, rng)
    out = {"realized_sd_mean": float(theta.std(axis=0).mean())}
    for k, t in enumerate(pers.TRAITS):
        out[f"real_mu_{t}"] = float(theta[:, k].mean())
        out[f"real_sd_{t}"] = float(theta[:, k].std())
    return out


def find_config(name: str) -> Path | None:
    hits = list((ROOT / "configs").rglob(f"{name}.yaml"))
    return hits[0] if hits else None


def main() -> None:
    df = read_rows()
    if df.empty:
        sys.exit("no results found.")
    df = df.drop_duplicates(subset=["run_id"], keep="last")

    # ---- reconstruct realized trait distributions ------------------------------
    rows = []
    for (cfg_name, seed, n_agents), _ in df.groupby(["config", "seed", "n_agents"]):
        p = find_config(cfg_name)
        if p is None:
            print(f"  (skipping {cfg_name}: config file not found)")
            continue
        rows.append({"config": cfg_name, "seed": seed,
                     **realized_traits(p, int(seed), int(n_agents))})
    real = pd.DataFrame(rows)
    if real.empty:
        sys.exit("could not reconstruct trait distributions.")
    real.to_csv(ROOT / "results" / "realized_traits.csv", index=False)

    d = df.merge(real, on=["config", "seed"], how="inner")
    tcols = [c for c in d.columns if c.endswith("__var") and c.startswith("T_")
             and "filler" not in c]
    if tcols:
        d["pol_var"] = d[tcols].mean(axis=1)
    ecols = [c for c in d.columns if c.endswith("__extremity") and c.startswith("T_")
             and "filler" not in c]
    if ecols:
        d["pol_extremity"] = d[ecols].mean(axis=1)

    print("=" * 78)
    print("CONFOUND 2 - TRUNCATION: does realized trait SPREAD explain opinion variance?")
    print("=" * 78)
    per_cfg = (d.groupby("config")[["realized_sd_mean", "pol_var", "pol_extremity"]]
               .mean().round(4).sort_values("realized_sd_mean"))
    print(per_cfg.to_string())
    ok = d[["realized_sd_mean", "pol_var"]].dropna()
    if len(ok) > 5:
        r, p = sps.pearsonr(ok.realized_sd_mean, ok.pol_var)
        rs, ps = sps.spearmanr(ok.realized_sd_mean, ok.pol_var)
        print(f"\nrealized trait SD vs opinion variance: Pearson r={r:.3f} (p={p:.4f}), "
              f"Spearman rho={rs:.3f} (p={ps:.4f}), n={len(ok)} runs")
        print("INTERPRETATION: a strong positive correlation means conditions with extreme"
              "\n  trait MEANS look less polarized mainly because truncation made them less"
              "\n  heterogeneous - report realized SD as a covariate, or reframe as a spread"
              "\n  effect. A weak correlation clears the trait-level interpretation.")
        try:
            import statsmodels.formula.api as smf
            m = smf.ols("pol_var ~ realized_sd_mean + C(config)", data=d).fit()
            print(f"\nControlling for realized SD, config still explains variance: "
                  f"model R2={m.rsquared:.3f}")
            print(f"  coefficient on realized_sd_mean = {m.params.get('realized_sd_mean', float('nan')):.3f} "
                  f"(p={m.pvalues.get('realized_sd_mean', float('nan')):.4f})")
        except Exception as exc:  # noqa: BLE001
            print(f"  (OLS covariate check unavailable: {exc})")

    print()
    print("=" * 78)
    print("CONFOUND 1 - PRIMING: does the NEUTRAL FILLER topic move with the conditions?")
    print("=" * 78)
    if "filler_variance" not in d.columns or d.filler_variance.isna().all():
        print("filler_variance not logged - cannot run this check.")
        return
    fill = d.groupby("config")[["filler_variance", "pol_extremity", "pol_var"]].mean().round(4)
    print(fill.sort_values("pol_extremity", ascending=False).to_string())
    base = "e1_norm_baseline"
    if base in fill.index:
        print(f"\nbaseline filler_variance = {fill.loc[base, 'filler_variance']:.4f}")
    ok2 = d[["filler_variance", "pol_extremity"]].dropna()
    if len(ok2) > 5:
        r2, p2 = sps.pearsonr(ok2.filler_variance, ok2.pol_extremity)
        print(f"\nfiller variance vs topic extremity: Pearson r={r2:.3f} (p={p2:.4f}), n={len(ok2)}")
        print("INTERPRETATION: if the filler topic tracks topic extremity, agents in"
              "\n  trait-extreme conditions are answering EVERY scale more extremely -"
              "\n  a response-style artifact. If the filler stays flat while contested"
              "\n  topics move, the polarization effect is topic-specific and real.")


if __name__ == "__main__":
    main()
