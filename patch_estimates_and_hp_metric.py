#!/usr/bin/env python3
"""
patch_estimates_and_hp_metric.py — Two fixes found in the first real-LLM pilot.

FIX 1 (data integrity, important): magnitude-aware estimate parsing.
    The old regex read "about 55 million" as 55 against a ground truth of 55,339,003.
    Because agents write more conversationally AFTER discussion, this systematically
    inflated post-discussion error and would have manufactured a spurious
    "discussion destroys collective intelligence" result. The new parser understands
    thousand/million/billion/trillion/lakh/crore, k/m/bn/t suffixes, comma and space
    grouping, scientific notation, and leading words. Unit-tested (16 cases) before
    shipping. Raw model replies are now also stored per item so parsing can be audited
    after the fact instead of trusted blindly.

FIX 2 (statistical power): continuous hidden-profile DV.
    CI_hidden_profile_rate thresholded at "majority correct > 0.5". Real correct rates
    sit near 0.10, so the composite read exactly 0.0 in every condition - no variance,
    no ability to detect composition effects. It now reports the mean post-discussion
    correct rate (continuous), plus CI_hp_gain for the pre->post change. The binary
    solved-majority flag is retained per item for reporting.

Run from the repo root:  python patch_estimates_and_hp_metric.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "src" / "traitmix" / "engine.py"
METRICS = ROOT / "src" / "traitmix" / "metrics_ci.py"

PARSER = '''
# --- magnitude-aware numeric parsing for CI estimation answers -------------------
SCALE_WORDS = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6,
               "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12,
               "t": 1e12, "lakh": 1e5, "crore": 1e7}
SCI_RE = re.compile(r"(-?\\d+\\.?\\d*)[eE]([+-]?\\d+)")
NUM_RE = re.compile(r"(-?\\d[\\d,\\s]*\\.?\\d*)\\s*([a-zA-Z]+)?")


def parse_number(text):
    """'about 55 million' -> 55000000.0 ; '4.34e11' -> 4.34e11 ; 'no idea' -> None."""
    if not text:
        return None
    t = str(text).replace("\\u00a0", " ").strip()
    t = re.sub(r"^[^\\d\\-+.]*", "", t)          # strip leading words ("about", "roughly")
    m = SCI_RE.search(t)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    m = NUM_RE.search(t)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower().strip(".")
    if suffix in SCALE_WORDS:
        val *= SCALE_WORDS[suffix]
    return val
'''

OLD_ANCHOR = 'ACTION_RE = re.compile(r"\\b(POST|REPLY|LIKE|FOLLOW|UNFOLLOW|PASS)\\b\\s*(p?\\d+|\\w+)?\\s*(.*)", re.S)'
NEW_ANCHOR = OLD_ANCHOR + "\n" + PARSER

OLD_PROMPT = '''                u = (f"[ESTIMATE] Privately estimate: {item['question']}{hidden} "
                     f"Reply with a single number only (no units, no words).")'''
NEW_PROMPT = '''                u = (f"[ESTIMATE] Privately estimate: {item['question']}{hidden} "
                     f"Reply with a single plain number in full digits, no units and no words "
                     f"(for example: 12500000).")'''

OLD_PARSE = '''            else:
                m = re.search(r"-?\\d[\\d,]*\\.?\\d*(?:[eE][+-]?\\d+)?", (o or "").replace(",", ""))
                st["ci"][tid][phase][i] = float(m.group()) if m else None'''
NEW_PARSE = '''            else:
                st["ci"][tid][phase][i] = parse_number(o)
        st["ci"][tid].setdefault("raw", {})[phase] = {i: (o or "")[:80] for i, o in enumerate(outs)}'''

OLD_COMPOSITE = '''    gains = [v for k, v in out.items() if k.endswith("__gain_vs_pre")]
    solved = [v for k, v in out.items() if k.endswith("__solved_majority") and np.isfinite(v)]
    if gains: out["CI_mean_gain_vs_pre"] = float(np.mean(gains))
    if solved: out["CI_hidden_profile_rate"] = float(np.mean(solved))
    return out'''

NEW_COMPOSITE = '''    gains = [v for k, v in out.items() if k.endswith("__gain_vs_pre")]
    if gains:
        out["CI_mean_gain_vs_pre"] = float(np.mean(gains))
    # Continuous hidden-profile DV: mean post-discussion correct rate (the binary
    # "majority solved" flag has no variance when correct rates sit near 0.1).
    post = [v for k, v in out.items() if k.endswith("__post_correct_rate") and np.isfinite(v)]
    pre = [v for k, v in out.items() if k.endswith("__pre_correct_rate") and np.isfinite(v)]
    if post:
        out["CI_hidden_profile_rate"] = float(np.mean(post))
    if post and pre:
        out["CI_hp_gain"] = float(np.mean(post) - np.mean(pre))
    solved = [v for k, v in out.items() if k.endswith("__solved_majority") and np.isfinite(v)]
    if solved:
        out["CI_hp_solved_majority"] = float(np.mean(solved))
    return out'''


def patch(path: Path, pairs, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"  already patched: {path.name}")
        return
    for old, new in pairs:
        if old not in text:
            sys.exit(f"PATCH FAILED in {path.name}: anchor not found:\n{old[:140]}...")
        text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name}")


def main() -> None:
    if not ENGINE.exists() or not METRICS.exists():
        sys.exit("Run this from the repo root (where src/traitmix/ lives).")
    print("Patching estimate parsing + hidden-profile metric:")
    patch(ENGINE, [(OLD_ANCHOR, NEW_ANCHOR), (OLD_PROMPT, NEW_PROMPT), (OLD_PARSE, NEW_PARSE)],
          marker="def parse_number")
    patch(METRICS, [(OLD_COMPOSITE, NEW_COMPOSITE)], marker="CI_hp_gain")
    print("\nSelf-test:")
    sys.path.insert(0, str(ROOT / "src"))
    from traitmix.engine import parse_number
    cases = [("55 million", 55e6), ("55,339,003", 55339003), ("about 3.2 billion", 3.2e9),
             ("80.6", 80.6), ("4.34e11", 4.34e11), ("7.5k", 7500), ("no idea", None)]
    ok = True
    for s, exp in cases:
        got = parse_number(s)
        good = (got is None and exp is None) or (got is not None and exp is not None
                                                 and abs(got - exp) / max(abs(exp), 1) < 1e-9)
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {s!r:22} -> {got}")
    print("\nAll parser tests passed." if ok else "\nPARSER TESTS FAILED - do not run experiments.")
    print("Remember to discard any pre-patch results before re-running.")


if __name__ == "__main__":
    main()
