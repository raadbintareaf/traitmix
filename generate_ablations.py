#!/usr/bin/env python3
"""
generate_ablations.py — Configurations for the measurement-circularity ablations.

Three ablations, on the ten conditions the reviewer specifies. Each copies a published
condition unchanged apart from the switch under test, so any difference in outcome is
attributable to that switch alone.

  ab_probe   probe_anchors = False
             The private opinion probe no longer states the agent's previous answer or the
             feed average. Reviewer point R1: those clauses perform social influence inside
             the measurement instrument, so an effect on opinion variance could arise from
             differential anchoring on a stated group position rather than from any social
             process on the network. Agreeableness is the trait most likely to modulate such
             anchoring, and it carries the strongest convergence result, so the ablation
             bears directly on a headline finding.

  ab_wint    w_interest = 0
             The recommender ranks on popularity and recency only. Reviewer point R2: the
             proximity term ranks posts by the distance between two agents' latent opinions,
             which is the variable the polarization measures are computed on, so the feed is
             built to bring like-minded agents together on precisely the quantity in which
             echo-chamber closure and cross-cutting are then scored.

  ab_expr    interest_on_expressed = True
             The proximity term uses the author's most recently posted stance instead of its
             latent opinion. This is the recommender a real platform could build, since none
             observes unexpressed belief, and it is the variant that supports the
             platform-design discussion.

Usage:
    python generate_ablations.py                    # all three
    python generate_ablations.py --which probe      # one at a time
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"

# the ten conditions named in the review, plus their source directories
CONDITIONS = [
    ("e1", "e1_norm_baseline"), ("e1", "e1_agreeableness_high"),
    ("e1", "e1_agreeableness_low"), ("e1", "e1_neuroticism_high"),
    ("e1", "e1_openness_high"),
    ("e2", "e2_homog"), ("e2", "e2_mid"), ("e2", "e2_diverse"),
    ("e3", "e3_O2_A2"), ("e3", "e3_O8_A8"),
]
# the expressed-stance variant is requested on a smaller set
EXPRESSED_ONLY = {"e1_norm_baseline", "e1_agreeableness_high", "e1_agreeableness_low"}

ABLATIONS = {
    "probe": ("abpr", {"probe_anchors": False}, CONDITIONS),
    "wint":  ("abwi", {"w_interest": 0.0}, CONDITIONS),
    "expr":  ("abex", {"interest_on_expressed": True},
              [c for c in CONDITIONS if c[1] in EXPRESSED_ONLY]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "probe", "wint", "expr"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if not CFG.exists():
        sys.exit("configs/ not found. Run from the repository root.")
    man_path = CFG / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    todo = ABLATIONS if args.which == "all" else {args.which: ABLATIONS[args.which]}

    for key, (tag, switches, conds) in todo.items():
        dst = CFG / f"ab_{key}"
        dst.mkdir(exist_ok=True)
        made = []
        for src_dir, name in conds:
            f = CFG / src_dir / f"{name}.yaml"
            if not f.exists():
                print(f"  missing {f}; skipped"); continue
            cfg = yaml.safe_load(f.read_text())
            new_name = f"{tag}_{name}"
            cfg["name"] = new_name
            cfg["inherit"] = "../base.yaml"
            cfg.setdefault("society", {}).update(switches)
            cfg["llm"] = {**cfg.get("llm", {}), "max_workers": args.workers}
            (dst / f"{new_name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            made.append(f"ab_{key}/{new_name}.yaml")
        man[f"AB_{key.upper()}"] = {"configs": made, "seeds": list(range(1, args.seeds + 1))}
        print(f"AB_{key.upper()}: {len(made)} conditions x {args.seeds} seeds = "
              f"{len(made) * args.seeds} runs   switches={switches}")

        # every ablation condition must match its source apart from the switch
        ok = True
        for rel in made:
            b = yaml.safe_load((CFG / rel).read_text())
            orig_name = b["name"].replace(f"{tag}_", "", 1)
            src = next((CFG / d / f"{orig_name}.yaml" for d, n in conds if n == orig_name), None)
            a = yaml.safe_load(src.read_text())
            if a.get("composition") != b.get("composition"):
                ok = False; print(f"    COMPOSITION DIFFERS: {b['name']}")
            for k, v in b.get("society", {}).items():
                if k in switches:
                    continue
                if a.get("society", {}).get(k, v) != v:
                    ok = False; print(f"    SOCIETY DIFFERS on {k}: {b['name']}")
        print(f"    identical to source apart from the switch: {ok}")

    man_path.write_text(json.dumps(man, indent=1))
    total = sum(len(man[f"AB_{k.upper()}"]["configs"]) * args.seeds for k in todo)
    print(f"\ntotal: {total} runs")


if __name__ == "__main__":
    main()
