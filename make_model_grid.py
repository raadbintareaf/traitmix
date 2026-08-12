#!/usr/bin/env python3
"""
make_model_grid.py — Replicate any experiment family on an additional model.

Copies the conditions of an experiment (e1 trait levels, e2 heterogeneity, e3 response
surface) unchanged except for the model, so the only difference from the published grid is
the model itself. Conditions are renamed with a per-model tag, which keeps them separate in
results/ and lets the analysis treat each model as its own family.

Usage (from the repository root):
    python make_model_grid.py --exp e1 --tag qwen7 --model Qwen/Qwen2.5-7B-Instruct-AWQ
    python make_model_grid.py --exp e2 --tag qwen7 --model Qwen/Qwen2.5-7B-Instruct-AWQ --workers 8
"""
import argparse, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["e1", "e2", "e3"])
    ap.add_argument("--tag", required=True, help="short model tag, e.g. qwen7")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    src = CFG / args.exp
    if not src.exists():
        sys.exit(f"{src} not found. Run from the repository root.")
    dst = CFG / f"{args.exp}_{args.tag}"
    dst.mkdir(exist_ok=True)

    made, skipped = [], []
    for f in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        name = cfg.get("name", f.stem)
        if "human_corr" in name:
            skipped.append(name); continue          # still blocked on a correlation matrix
        cfg["name"] = name.replace(f"{args.exp}_", f"{args.exp}{args.tag}_", 1)
        cfg["inherit"] = "../base.yaml"
        cfg["llm"] = {**cfg.get("llm", {}), "backend": "vllm", "model": args.model,
                      "base_url": args.base_url, "max_workers": args.workers}
        (dst / f"{cfg['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        made.append(f"{args.exp}_{args.tag}/{cfg['name']}.yaml")

    man_path = CFG / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    key = f"{args.exp.upper()}_{args.tag.upper()}"
    man[key] = {"configs": made, "seeds": list(range(1, args.seeds + 1))}
    man_path.write_text(json.dumps(man, indent=1))

    print(f"{key}: {len(made)} configs x {args.seeds} seeds = {len(made)*args.seeds} runs")
    print(f"  model       : {args.model}")
    print(f"  concurrency : {args.workers}")
    if skipped:
        print(f"  skipped     : {skipped}")
    # verify every condition against its published counterpart, matched by name
    same = True
    for rel in made:
        b = yaml.safe_load((CFG / rel).read_text())
        orig = src / (b["name"].replace(f"{args.exp}{args.tag}_", f"{args.exp}_", 1) + ".yaml")
        if not orig.exists():
            same = False; print(f"  missing counterpart for {b['name']}"); continue
        a = yaml.safe_load(orig.read_text())
        if a.get("composition") != b.get("composition"):
            same = False; print(f"  COMPOSITION DIFFERS: {b['name']}")
        if "topics" in b:
            same = False; print(f"  topic set overridden in {b['name']}")
    print(f"  all {len(made)} compositions identical to the published grid, "
          f"topics left at default: {same}")


if __name__ == "__main__":
    main()
