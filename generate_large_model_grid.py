#!/usr/bin/env python3
"""
generate_large_model_grid.py — Response-surface replication on a larger model.

Reviewer point 3 was that both models used are small (8B and 14B), so claims of generality
should be limited to models of that class. This adds a third size on the same hardware.

The conditions are the nine cells of the Openness x Agreeableness grid, copied unchanged
except for the model. Crucially the original two-topic set is retained rather than the
six-topic set, so the new runs are directly comparable with the published E3 and with the
Qwen-14B replication; changing two things at once would make the comparison uninterpretable.

Conditions are named e3l_* ("large") so they cannot collide with e3_*, e3q_*, or the e3x_*
names used by the separate third-family replication.

Usage (from the repository root):
    python generate_large_model_grid.py
    python generate_large_model_grid.py --model Qwen/Qwen2.5-32B-Instruct-AWQ --seeds 5
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--workers", type=int, default=8,
                    help="keep this low: the KV cache is small once a 32B model is loaded")
    ap.add_argument("--seeds", type=int, default=5,
                    help="5 matches the Qwen-14B replication, keeping the two comparable")
    args = ap.parse_args()

    src = CFG / "e3"
    if not src.exists():
        sys.exit("configs/e3 not found. Run from the repository root.")
    dst = CFG / "e3_large"
    dst.mkdir(exist_ok=True)

    made = []
    for f in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        name = cfg.get("name", f.stem)
        new_name = name.replace("e3_", "e3l_", 1)
        cfg["name"] = new_name
        cfg["inherit"] = "../base.yaml"
        cfg["llm"] = {**cfg.get("llm", {}), "backend": "vllm", "model": args.model,
                      "base_url": args.base_url, "max_workers": args.workers}
        # topics deliberately left at the default two-topic set, matching E3 and E3_QWEN
        (dst / f"{new_name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        made.append(f"e3_large/{new_name}.yaml")

    man_path = CFG / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["E3_LARGE"] = {"configs": made, "seeds": list(range(1, args.seeds + 1))}
    man_path.write_text(json.dumps(man, indent=1))

    print(f"wrote {len(made)} configurations to configs/e3_large/")
    print(f"  model       : {args.model}")
    print(f"  concurrency : {args.workers}")
    print(f"  planned     : {len(made)} cells x {args.seeds} seeds = {len(made)*args.seeds} runs")
    a = yaml.safe_load((src / "e3_O8_A8.yaml").read_text())
    b = yaml.safe_load((dst / "e3l_O8_A8.yaml").read_text())
    print(f"\n  composition identical to the published grid : "
          f"{a.get('composition') == b.get('composition')}")
    print(f"  topic set left at the default (two topics)  : {'topics' not in b}")


if __name__ == "__main__":
    main()
