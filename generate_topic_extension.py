#!/usr/bin/env python3
"""
generate_topic_extension.py — Build the six-topic replication requested in review.

The reviewer's point is that averaging two anti-correlated topics makes the reported effect
sizes a property of that topic pair rather than of composition. This script regenerates the
E1 (trait level) and E2 (heterogeneity) conditions with six contested topics instead of two,
so the topic-averaged effects can be re-estimated on a set wide enough to generalise over,
and so a topic-by-composition analysis becomes possible.

Design decisions worth recording:

  * The two original topics are retained, so the new runs are a superset and the earlier
    results remain directly comparable rather than being replaced.
  * Four new statements were chosen to be genuinely contested, to span different policy
    domains rather than clustering in one national debate, and to avoid content likely to
    trigger model refusals, which would confound a refusal with a null opinion.
  * Topics are written into each configuration file rather than into the shared topic
    file. Configurations therefore remain self-contained, the configuration hash changes,
    new run identifiers are created, and nothing already in results/ is overwritten.
  * The conditions are renamed e1t_* and e2t_*, so both topic sets can be analysed side by
    side and the comparison between them is itself reportable.

Run from the repository root:  python generate_topic_extension.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

ORIGINAL = [
    {"id": "T_guncontrol",
     "statement": "Stricter national gun-control laws would make society safer overall."},
    {"id": "T_immigration",
     "statement": "Current levels of immigration benefit the country more than they cost it."},
]

ADDED = [
    {"id": "T_carbontax",
     "statement": "A national carbon tax should be introduced to reduce emissions, "
                  "even if it raises household energy costs."},
    {"id": "T_nuclear",
     "statement": "Nuclear power should be expanded as a main source of the country's "
                  "electricity."},
    {"id": "T_ubi",
     "statement": "The government should provide an unconditional basic income to every "
                  "adult citizen."},
    {"id": "T_socialmedia",
     "statement": "Social media platforms should be legally required to verify the age of "
                  "their users."},
]

FILLER = {"id": "T_neutral_filler", "statement": "Pineapple belongs on pizza.",
          "role": "filler"}

TOPICS = ORIGINAL + ADDED + [FILLER]


def build(src_dir: str, dst_dir: str, old_prefix: str, new_prefix: str) -> list[str]:
    src = CFG / src_dir
    if not src.exists():
        sys.exit(f"{src} not found.")
    dst = CFG / dst_dir
    dst.mkdir(exist_ok=True)
    made = []
    for f in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        name = cfg.get("name", f.stem)
        if "human_corr" in name:          # still blocked on a correlation matrix
            continue
        new_name = name.replace(old_prefix, new_prefix, 1)
        cfg["name"] = new_name
        cfg["inherit"] = "../base.yaml"
        cfg["topics"] = {"source": "custom", "items": TOPICS}
        (dst / f"{new_name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        made.append(f"{dst_dir}/{new_name}.yaml")
    return made


def main() -> None:
    if not CFG.exists():
        sys.exit("Run from the repository root, where configs/ lives.")
    e1 = build("e1", "e1_t6", "e1_", "e1t_")
    e2 = build("e2", "e2_t6", "e2_", "e2t_")

    man_path = CFG / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["E1_T6"] = {"configs": e1, "seeds": SEEDS}
    man["E2_T6"] = {"configs": e2, "seeds": SEEDS}
    man_path.write_text(json.dumps(man, indent=1))

    n = (len(e1) + len(e2)) * len(SEEDS)
    print(f"E1_T6: {len(e1)} configs x {len(SEEDS)} seeds = {len(e1)*len(SEEDS)} runs")
    print(f"E2_T6: {len(e2)} configs x {len(SEEDS)} seeds = {len(e2)*len(SEEDS)} runs")
    print(f"total: {n} runs")
    print(f"\ntopics per run: {len(TOPICS)-1} contested + 1 neutral filler")
    for t in TOPICS:
        role = "  (filler)" if t.get("role") == "filler" else ""
        print(f"    {t['id']:18s} {t['statement'][:64]}{role}")

    print("\nsanity check: compositions must be unchanged from the originals")
    for a, b in [("e1/e1_agreeableness_high.yaml", "e1_t6/e1t_agreeableness_high.yaml"),
                 ("e2/e2_diverse.yaml", "e2_t6/e2t_diverse.yaml")]:
        pa, pb = CFG / a, CFG / b
        if pa.exists() and pb.exists():
            ca, cb = yaml.safe_load(pa.read_text()), yaml.safe_load(pb.read_text())
            print(f"  {b:34s} composition identical = "
                  f"{ca.get('composition') == cb.get('composition')}, "
                  f"topics = {len(cb['topics']['items'])}")


if __name__ == "__main__":
    main()
