#!/usr/bin/env python3
"""
apply_ipip_norms.py — Unblock the human-calibrated E2 conditions with CITED norms.

SOURCE (verified, open access CC-BY):
  Kajonius, P. J., & Johnson, J. A. (2019). Assessing the structure of the Five Factor
  Model of Personality (IPIP-NEO-120) in the public domain. Europe's Journal of
  Psychology, 15(2), 260-275. https://doi.org/10.5964/ejop.v15i2.1671
  Table A1, Sample Descriptive Statistics, N = 320,128 US respondents.

  Domain scales run 4-20 (six facets, four items each, 1-5 per item), so trait values
  are rescaled to the model's [0,1] space as (M - 4) / 16 and SD / 16.

      Neuroticism        M = 11.10  SD = 2.66   ->  mu = 0.4438  sigma = 0.1663
      Extraversion       M = 13.69  SD = 2.36   ->  mu = 0.6056  sigma = 0.1475
      Openness           M = 13.71  SD = 2.06   ->  mu = 0.6069  sigma = 0.1288
      Agreeableness      M = 14.87  SD = 2.01   ->  mu = 0.6794  sigma = 0.1256
      Conscientiousness  M = 14.95  SD = 2.34   ->  mu = 0.6844  sigma = 0.1463

LIMITATIONS TO REPORT IN THE PAPER (stated by the source authors):
  * The sample is large but self-selected online volunteers, NOT nationally
    representative; the authors note some facets (Emotionality, Intellect, Altruism)
    are likely unrepresentatively high because such traits characterise people
    interested in psychology.
  * Mean age 28.1 (SD 10.1), 40% male / 60% female, US only.
  Because of this the condition is labelled "human-calibrated", not "human-representative".

This script fills e2_human_norm (means + SDs, independent traits). The e2_human_corr
condition additionally needs a 5x5 domain correlation matrix, which the source paper
reports only as a facet-level heatmap - use compute_ipip_corr.py on Johnson's freely
available raw data (https://osf.io/tbmh5/) to derive it, or drop that condition.

Run from the repo root:  python apply_ipip_norms.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

CITATION = ("Kajonius & Johnson (2019), Europe's Journal of Psychology 15(2), 260-275, "
            "doi:10.5964/ejop.v15i2.1671, Table A1 (N=320,128). Domain scales 4-20 "
            "rescaled to [0,1] as (M-4)/16.")

MU = {"openness": 0.6069, "conscientiousness": 0.6844, "extraversion": 0.6056,
      "agreeableness": 0.6794, "neuroticism": 0.4438}
SIGMA = {"openness": 0.1288, "conscientiousness": 0.1463, "extraversion": 0.1475,
         "agreeableness": 0.1256, "neuroticism": 0.1663}


def main() -> None:
    cfg_dir = ROOT / "configs"
    if not cfg_dir.exists():
        sys.exit("Run from the repo root (where configs/ lives).")

    norms = {"citation": CITATION, "mu": MU, "sigma": SIGMA,
             "sample": "N=320,128 US online volunteers, mean age 28.1 (SD 10.1), 40% male",
             "caveat": "Large but self-selected sample; not nationally representative."}
    (cfg_dir / "norms.yaml").write_text(yaml.safe_dump(norms, sort_keys=False))
    print("wrote configs/norms.yaml")

    target = cfg_dir / "e2" / "e2_human_norm.yaml"
    if not target.exists():
        sys.exit(f"{target} not found - run configs/generate_grids.py first.")
    cfg = yaml.safe_load(target.read_text())
    cfg["composition"] = {"mu": MU, "sigma": SIGMA, "corr": "independent"}
    cfg["_norms_source"] = CITATION
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"patched {target.relative_to(ROOT)} (placeholder guard cleared)")

    corr_cfg = cfg_dir / "e2" / "e2_human_corr.yaml"
    if corr_cfg.exists():
        c = yaml.safe_load(corr_cfg.read_text())
        still_blocked = "_VERIFY" in (c.get("composition") or {})
        print(f"\ne2_human_corr: {'STILL BLOCKED' if still_blocked else 'check manually'} "
              "- needs a 5x5 domain correlation matrix.")
        print("  Options: (a) run compute_ipip_corr.py on Johnson's raw data "
              "(https://osf.io/tbmh5/), or (b) drop this condition and report E2 with "
              "four conditions.")

    print("\nVerify nothing is blocked:")
    print("  grep -rl _VERIFY configs/ || echo 'no placeholder norms remain'")


if __name__ == "__main__":
    main()
