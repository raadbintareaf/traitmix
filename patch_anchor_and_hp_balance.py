#!/usr/bin/env python3
"""
patch_anchor_and_hp_balance.py — Two fixes from the third pilot audit.

FIX 1 (CRITICAL - prompt/screen parity).
    The in-simulation estimate prompt ended with "(for example: 12500000)". That example
    anchored every answer to 10^6-10^7 regardless of the question: Paraguay internet
    penetration (truth 76.3 percent) came back as 7,050,000. The screening script used
    NO example and the same items scored 0.19-0.23 median error, so screening and
    deployment were measuring different things - which invalidates the screen. The
    example is removed; the simulation prompt now matches the screening prompt exactly.

FIX 2 (task calibration - hidden profile).
    All nine unshared facts favour candidate A, and each agent held two of them, so
    82.5 percent already chose correctly BEFORE discussion: the hidden-profile paradigm
    had inverted into an easy task with a ceiling. Rebalanced so the information every
    agent shares points clearly to the wrong candidate (six substantive pro-B facts plus
    two anti-A facts) while each agent holds only ONE unshared pro-A fact. Pre-discussion
    accuracy should now be low, leaving headroom for discussion - and for personality
    composition - to move it.

Run from the repo root:  python patch_anchor_and_hp_balance.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "src" / "traitmix" / "engine.py"
DATA = ROOT / "src" / "traitmix" / "data.py"

OLD_EST = '''f"no units and no words (for example: 12500000).")'''
NEW_EST = '''f"no units and no words.")'''

NEW_HP = '''def make_hidden_profile(rng: np.random.Generator, n_agents: int, task_id="hp_0",
                       clues_per_agent=1):
    """Stasser-Titus hidden profile, scaled to large societies.

    Everyone shares the same eight facts, which point clearly to candidate B (six
    substantive pro-B facts plus two anti-A facts). Candidate A is objectively better,
    but the facts establishing that are UNSHARED: each agent holds only
    `clues_per_agent` of them. No individual can solve the task alone; the group
    collectively holds every fact, so only discussion that surfaces unshared
    information can recover the correct answer (candidate A)."""
    a_pros = [f"Candidate A {s}" for s in
              ["cut departmental costs by 12% in a previous role",
               "has led teams of more than 40 people",
               "holds the professional certification the role legally requires",
               "shipped two major products ahead of schedule",
               "speaks the main client's language fluently",
               "has a clean compliance record over eleven years"]]
    b_cons_unshared = [f"Candidate B {s}" for s in
                       ["was formally cited for a compliance violation",
                        "missed two product deadlines last year",
                        "has never managed anyone before"]]
    shared_pro_b = ["Candidate B interviewed with exceptional confidence",
                    "Candidate B holds a degree from a prestigious university",
                    "Candidate B is widely liked by the people who met them",
                    "Candidate B gave the strongest presentation of any applicant",
                    "Candidate B has eight years of experience in this industry",
                    "Candidate B asked thoughtful questions about the team's strategy"]
    shared_anti_a = ["Candidate A missed a quarterly target two years ago",
                     "Candidate A has changed jobs three times in six years"]
    shared = shared_pro_b + shared_anti_a
    unshared = a_pros + b_cons_unshared
    k = int(min(max(1, clues_per_agent), len(unshared) - 1))
    clues, private = {}, {}
    for i in range(n_agents):
        picks = [unshared[j] for j in rng.choice(len(unshared), size=k, replace=False)]
        private[i] = picks
        clues[i] = list(shared) + picks
    return {"id": task_id,
            "prompt": "Your team must choose the better job candidate: A) Candidate A  B) Candidate B.",
            "correct": "A", "clues": clues, "private": private}'''


def main() -> None:
    if not ENGINE.exists() or not DATA.exists():
        sys.exit("Run this from the repo root (where src/traitmix/ lives).")
    print("Patching anchor bug + hidden-profile balance:")

    e = ENGINE.read_text()
    if OLD_EST in e:
        ENGINE.write_text(e.replace(OLD_EST, NEW_EST, 1))
        print("  patched engine.py (removed anchoring example)")
    elif "for example: 12500000" in e:
        sys.exit("PATCH FAILED: found the example but not the expected line; inspect engine.py")
    else:
        print("  already patched: engine.py")

    d = DATA.read_text()
    if "shared_pro_b" in d:
        print("  already patched: data.py")
    else:
        start = d.index("def make_hidden_profile(")
        end = d.index("def load_ipip(", start)
        DATA.write_text(d[:start] + NEW_HP + "\n\n\n" + d[end:])
        print("  patched data.py (rebalanced hidden profile)")

    print("\nSelf-test:")
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    import numpy as np
    import traitmix.data as D
    importlib.reload(D)
    t = D.make_hidden_profile(np.random.default_rng(0), 100)
    sizes = {len(v) for v in t["clues"].values()}
    priv = {len(v) for v in t["private"].values()}
    reach = len({x for v in t["private"].values() for x in v})
    print(f"  clues per agent: {sizes} | private per agent: {priv} | unshared facts reaching society: {reach}/9")
    ok = sizes == {9} and priv == {1} and reach == 9
    import traitmix.engine as E
    importlib.reload(E)
    src = Path(E.__file__).read_text()
    ok &= "for example: 12500000" not in src
    print("  estimate prompt free of anchoring example:", "for example: 12500000" not in src)
    print("\nAll checks passed." if ok else "\nCHECKS FAILED - inspect before running.")
    print("\nNext: rm -rf checkpoints/* results/raw_results.jsonl results/registry.json results/timeseries/*")
    print("      then re-run the pilot.")


if __name__ == "__main__":
    main()
