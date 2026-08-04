#!/usr/bin/env python3
"""
generate_qwen_grids.py — Create Qwen replications of E2 (heterogeneity) and E3 (O x A
surface), mirroring how E1_QWEN was generated.

Each source config is copied verbatim except that llm.model is set to the Qwen checkpoint,
so composition, topology, rounds, CI schedule and seeds are identical to the Llama runs and
the two model families are directly comparable.

Writes configs/e2_qwen/, configs/e3_qwen/ and adds E2_QWEN / E3_QWEN entries to
configs/MANIFEST.json (existing entries are preserved).

Run from the repo root:  python generate_qwen_grids.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs"
QWEN = "Qwen/Qwen2.5-14B-Instruct-AWQ"
SEEDS = [1, 2, 3, 4, 5]


def build(src_dir: str, dst_dir: str, prefix_old: str, prefix_new: str) -> list[str]:
    src = CFG / src_dir
    if not src.exists():
        sys.exit(f"{src} not found - run configs/generate_grids.py first.")
    dst = CFG / dst_dir
    dst.mkdir(exist_ok=True)
    made = []
    for f in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        name = cfg.get("name", f.stem)
        new_name = name.replace(prefix_old, prefix_new, 1)
        cfg["name"] = new_name
        cfg["inherit"] = "../base.yaml"
        cfg["llm"] = {**cfg.get("llm", {}), "model": QWEN}
        (dst / f"{new_name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        made.append(f"{dst_dir}/{new_name}.yaml")
    return made


def main() -> None:
    if not CFG.exists():
        sys.exit("Run from the repo root (where configs/ lives).")
    e2 = build("e2", "e2_qwen", "e2_", "e2q_")
    e3 = build("e3", "e3_qwen", "e3_", "e3q_")

    man_path = CFG / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["E2_QWEN"] = {"configs": e2, "seeds": SEEDS}
    man["E3_QWEN"] = {"configs": e3, "seeds": SEEDS}
    man_path.write_text(json.dumps(man, indent=1))

    print(f"E2_QWEN: {len(e2)} configs x {len(SEEDS)} seeds = {len(e2)*len(SEEDS)} runs")
    print(f"E3_QWEN: {len(e3)} configs x {len(SEEDS)} seeds = {len(e3)*len(SEEDS)} runs")
    print(f"total added: {(len(e2)+len(e3))*len(SEEDS)} runs")
    print("\nSanity check - compositions must match the Llama originals:")
    for a, b in [("e2/e2_diverse.yaml", "e2_qwen/e2q_diverse.yaml"),
                 ("e3/e3_O8_A8.yaml", "e3_qwen/e3q_O8_A8.yaml")]:
        pa, pb = CFG / a, CFG / b
        if pa.exists() and pb.exists():
            ca, cb = yaml.safe_load(pa.read_text()), yaml.safe_load(pb.read_text())
            same = ca.get("composition") == cb.get("composition")
            print(f"  {a} vs {b}: composition identical = {same} | "
                  f"model = {cb.get('llm', {}).get('model')}")


if __name__ == "__main__":
    main()
