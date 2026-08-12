#!/usr/bin/env python3
"""
patch_runner_exposure.py — Record collective-intelligence exposure in the results row.

The engine counts how many posts announce a collective-intelligence item and how many
replies those posts attract, but the runner was not writing those counts into the results
row, so they were computed and discarded. This patch adds them, along with the three
ablation switches actually in force for a run, so that a released results file states its
own provenance rather than requiring the reader to infer it from the condition name.

Reviewer point R3 asks whether the estimation items ever entered the conversation. The
answer is in the engine: an announcement post is appended to the feed for every item at
its pre-round. These counts make that measurable rather than a matter of reading the
source.

Usage:  python patch_runner_exposure.py
"""
import shutil
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "src" / "traitmix" / "runner.py"


def main() -> None:
    if not RUNNER.exists():
        sys.exit(f"{RUNNER} not found. Run from the repository root.")
    src = RUNNER.read_text()
    if "ci_posts" in src:
        sys.exit("runner.py already patched; nothing to do.")
    anchor = '    row["runtime_s"] = round(time.time() - t0, 1)'
    if anchor not in src:
        sys.exit("anchor line not found; runner.py differs from the expected version.")
    shutil.copy(RUNNER, RUNNER.with_suffix(".py.pre_exposure"))
    src = src.replace(anchor, '''    # exposure to the collective-intelligence items, so that it is reported rather than
    # assumed, and the ablation switches in force, so the file states its own provenance
    for _k in ("ci_posts", "ci_replies", "ci_mentions"):
        row[_k] = state.get(_k)
    _soc = cfg.get("society", {})
    row["probe_anchors"] = _soc.get("probe_anchors", True)
    row["w_interest"] = _soc.get("w_interest", 1.0)
    row["interest_on_expressed"] = _soc.get("interest_on_expressed", False)
''' + anchor, 1)
    RUNNER.write_text(src)
    print(f"patched {RUNNER}")
    print(f"backup  {RUNNER.with_suffix('.py.pre_exposure')}")
    print("\nNext: measure exposure on a small number of runs with")
    print("  python measure_ci_exposure.py --runs 3")


if __name__ == "__main__":
    main()
