#!/usr/bin/env python3
"""
screen_ci_items.py — Select the final collective-intelligence estimation items.

WHY THIS IS REQUIRED (not optional):
A crowd cannot show wisdom on a question the model already knows exactly. In the pilot,
every one of 100 agents answered Kenya's population with the identical string
"52,573,000" - zero dispersion, so Page's diversity term is 0 and collective error
equals individual error by construction. Items must therefore satisfy BOTH:

  (a) HEADROOM     median relative error of a solo agent >= --min-error (default 0.15)
  (b) DISPERSION   sd of log10(answers) >= --min-spread (default 0.02), i.e. agents
                   genuinely disagree, leaving room for aggregation to help or hurt

Items passing both are written to data/ci/wb_items.json and used by every experiment.
Ground truths are fetched from the free World Bank API and cached in wb_truths.json.

Usage (vLLM server must be running):
    python screen_ci_items.py                 # screen with defaults, 20 samples/item
    python screen_ci_items.py --n 30 --keep 6 # more samples, keep at most 6 items
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traitmix.data import DEFAULT_WB_ITEMS, fetch_wb_truth  # noqa: E402
from traitmix.engine import parse_number  # noqa: E402
from traitmix.llm import VLLMClient  # noqa: E402

SYS = ("You are a 35-year-old analyst taking part in a private research survey. "
       "Answer factual questions to the best of your knowledge.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=20, help="samples per item")
    ap.add_argument("--min-error", type=float, default=0.15)
    ap.add_argument("--min-spread", type=float, default=0.02)
    ap.add_argument("--keep", type=int, default=6, help="max items to keep")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    truths_p = ROOT / "data" / "ci" / "wb_truths.json"
    truths = json.loads(truths_p.read_text()) if truths_p.exists() else {}

    llm = VLLMClient(model=args.model, max_workers=32)
    rows = []
    print(f"screening {len(DEFAULT_WB_ITEMS)} candidate items, {args.n} samples each\n")

    for it in DEFAULT_WB_ITEMS:
        if truths.get(it["id"]) is None:
            try:
                truths[it["id"]] = fetch_wb_truth(it)
            except Exception as e:
                print(f"  SKIP {it['id']}: ground truth unavailable ({str(e).splitlines()[0][:70]})")
                continue
        truth = truths[it["id"]]
        prompt = (f"[ESTIMATE] Privately estimate: {it['question']} "
                  "This is a private research survey, not a social media post. "
                  "Do not role-play, do not use asterisks or stage directions, do not explain. "
                  "Reply with a single plain number in full digits, no units and no words.")
        outs = llm.generate_batch([(SYS, prompt)] * args.n, max_tokens=16,
                                  temperature=args.temperature)
        vals = [v for v in (parse_number(o) for o in outs) if v is not None and v > 0]
        if len(vals) < max(5, args.n // 3):
            print(f"  SKIP {it['id']}: only {len(vals)}/{args.n} parseable answers")
            continue
        arr = np.array(vals, float)
        med_err = float(np.median(np.abs(arr - truth) / abs(truth)))
        spread = float(np.std(np.log10(arr)))
        ok = med_err >= args.min_error and spread >= args.min_spread
        rows.append({"item": it, "median_rel_error": med_err, "log_spread": spread,
                     "n_parsed": len(vals), "truth": truth, "keep": ok})
        flag = "KEEP" if ok else ("no headroom" if med_err < args.min_error else "no spread")
        print(f"  {flag:11s} {it['id']:18s} truth={truth:<16.4g} "
              f"median_rel_err={med_err:5.2f}  log_spread={spread:5.3f}  parsed={len(vals)}/{args.n}")

    truths_p.parent.mkdir(parents=True, exist_ok=True)
    truths_p.write_text(json.dumps({k: v for k, v in truths.items() if v is not None}, indent=1))

    kept = [r for r in rows if r["keep"]]
    kept.sort(key=lambda r: -r["median_rel_error"])
    kept = kept[: args.keep]
    if len(kept) < 3:
        print(f"\nONLY {len(kept)} ITEMS PASSED - do not start the campaign.")
        print("Loosen --min-error slightly, raise --n, or add harder candidate items to")
        print("DEFAULT_WB_ITEMS in src/traitmix/data.py, then re-run.")
        sys.exit(1)

    out = ROOT / "data" / "ci" / "wb_items.json"
    out.write_text(json.dumps([r["item"] for r in kept], indent=1))
    (ROOT / "results" / "ci_item_screen.csv").parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(ROOT / "results" / "ci_item_screen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "truth", "median_rel_error", "log_spread", "n_parsed", "kept"])
        for r in rows:
            w.writerow([r["item"]["id"], r["truth"], round(r["median_rel_error"], 4),
                        round(r["log_spread"], 4), r["n_parsed"], r["keep"]])

    print(f"\nKEPT {len(kept)} items -> {out}")
    for r in kept:
        print(f"   {r['item']['id']:18s} rel_err={r['median_rel_error']:.2f} spread={r['log_spread']:.3f}")
    print("\nScreening record saved to results/ci_item_screen.csv (report this in the paper).")
    print("Set ci.n_estimation in configs/base.yaml to at most", len(kept))


if __name__ == "__main__":
    main()
