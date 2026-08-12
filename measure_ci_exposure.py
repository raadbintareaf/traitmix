#!/usr/bin/env python3
"""
measure_ci_exposure.py — Did the estimation items ever enter the conversation?

Reviewer point R3 argues that if the estimation items were never surfaced in agents'
feeds, then the finding that private estimates barely move is a property of the design
rather than a result about deliberation, and the collective-intelligence construct would
have to be renamed.

The engine does surface them: at each item's pre-round an announcement post is appended
to the feed, carrying two likes so that it competes in the recommender ranking. This
script measures the consequence rather than asserting it. It runs a small number of
baseline simulations with checkpoints retained, then counts, from the actual post record:

  announcements   posts introducing an estimation or decision item
  replies         agent replies whose parent is one of those posts
  mentions        other agent posts referring to an item without replying to it
  agents engaged  distinct agents contributing any of the above

It also prints example post texts, so that a reader can see what the agents actually
wrote rather than trusting a count.

Usage:
    python measure_ci_exposure.py --runs 3
    python measure_ci_exposure.py --runs 3 --config configs/e1/e1_norm_baseline.yaml
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

MARK = re.compile(r"#estimate|#decision", re.I)
SOFT = re.compile(r"\bestimate|\bguess|\bmy take|billion|per capita|GDP|square kilom", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--config", default="configs/e1/e1_norm_baseline.yaml")
    ap.add_argument("--start-seed", type=int, default=901,
                    help="seeds outside the experimental design, so nothing is contaminated")
    args = ap.parse_args()

    from traitmix.utils import load_config
    from traitmix.runner import run_one
    from traitmix.classifier import make_scorer
    from traitmix import checkpointing as ck

    cfg = load_config(Path(args.config))
    scorer = make_scorer()
    rows, examples = [], []

    for k in range(args.runs):
        seed = args.start_seed + k
        print(f"run {k+1}/{args.runs} (seed {seed}) ...", flush=True)
        run_one(cfg, seed=seed, force=True, trait_scorer=scorer, keep_ckpt=True)
        rid = None
        reg = json.loads((RES / "registry.json").read_text()) if (RES / "registry.json").exists() else {}
        for key in reg:
            if key.endswith(f"_s{seed}") and cfg["name"] in key:
                rid = key
        st = ck.latest(rid) if rid else None
        if st is None:
            print("   could not recover the checkpoint; skipping")
            continue
        posts = st.get("posts", [])
        ann = {p["id"] for p in posts if MARK.search(str(p.get("text", "")))
               and p.get("author") == len(st.get("personas", [])) - 1}
        replies = [p for p in posts if p.get("reply_to") in ann]
        mentions = [p for p in posts if p["id"] not in ann and p.get("reply_to") not in ann
                    and (MARK.search(str(p.get("text", ""))) or SOFT.search(str(p.get("text", ""))))]
        engaged = {p["author"] for p in replies + mentions}
        rows.append(dict(seed=seed, posts=len(posts), announcements=len(ann),
                         replies=len(replies), mentions=len(mentions),
                         agents_engaged=len(engaged),
                         pct_agents=100 * len(engaged) / max(len(st.get("personas", [])), 1)))
        for p in (replies + mentions)[:4]:
            examples.append(str(p.get("text", ""))[:150])

    if not rows:
        sys.exit("no runs produced a recoverable checkpoint")

    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_csv(RES / "ci_exposure.csv", index=False)
    print("\n=== EXPOSURE TO THE COLLECTIVE-INTELLIGENCE ITEMS ===")
    print(D.to_string(index=False))
    print(f"\n  mean announcements per run : {D.announcements.mean():.1f}")
    print(f"  mean agent replies to them : {D.replies.mean():.1f}")
    print(f"  mean other mentions        : {D.mentions.mean():.1f}")
    print(f"  agents engaging per run    : {D.agents_engaged.mean():.1f} "
          f"({D.pct_agents.mean():.0f}% of the society)")
    print("\n  example agent posts referring to an item:")
    for e in examples[:6]:
        print(f"    {e}")
    print(f"\nwrote {RES / 'ci_exposure.csv'}")
    print("\nThese runs use seeds outside the experimental design and are excluded from")
    print("the analysis by seed; remove them with scan_dead_runs.py if you prefer.")


if __name__ == "__main__":
    main()
