#!/usr/bin/env python3
"""
prepare_release.py — Build a clean, public, citable copy of the TraitMix repository.

Produces ../traitmix-release/ containing the code, configurations, results and paper
artifacts needed to reproduce every number in the manuscript, plus LICENSE, README,
CITATION.cff and a frozen environment.

IMPORTANT: this script also scans for credentials before you publish anything. It refuses
to finish if it finds something that looks like a token or key. Read that output carefully;
a leaked token in git history is very hard to undo.

Usage (from the repo root):
    python prepare_release.py
    python prepare_release.py --out ../traitmix-release
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INCLUDE_DIRS = ["src", "configs", "analysis", "notebooks", "data", "paper"]
INCLUDE_FILES = [
    "requirements.txt", "README.md",
    "audit_sign_stability.py", "diagnose_confounds.py", "scan_dead_runs.py",
    "screen_ci_items.py", "make_paper_figures.py", "make_paper_tables.py",
    "apply_ipip_norms.py", "generate_qwen_grids.py", "prepare_release.py",
    "patch_hidden_profile.py", "patch_estimates_and_hp_metric.py",
    "patch_probe_quality.py", "patch_anchor_and_hp_balance.py",
]
RESULT_FILES = [
    "raw_results.jsonl", "derived_results.csv", "summary.csv", "stats.csv",
    "mixed_models.txt", "realized_traits.csv", "audit_sign_stability.csv",
    "e0_validation.csv", "e0_measured_vs_target.csv", "ci_item_screen.csv",
]
EXCLUDE_PATTERNS = [".ipynb_checkpoints", "__pycache__", ".pyc", ".ckpt",
                    "checkpoints/", ".git/", ".DS_Store", ".bak"]

# Patterns that must never reach a public repository.
SECRET_PATTERNS = [
    (r"gh[pousr]_[A-Za-z0-9]{16,}", "GitHub token"),
    (r"hf_[A-Za-z0-9]{20,}", "Hugging Face token"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API key"),
    (r"s2k-[A-Za-z0-9]{20,}", "Semantic Scholar API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?i)\b(api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"]{12,}['\"]", "credential assignment"),
]

LICENSE_MIT = """MIT License

Copyright (c) 2026 Raad Bin Tareaf, Samia Loucif, Murad Al-Rajab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CITATION_CFF = """cff-version: 1.2.0
message: "If you use this software or data, please cite the accompanying article."
title: "TraitMix: personality composition, polarization and collective intelligence in LLM-based social simulations"
authors:
  - family-names: "Bin Tareaf"
    given-names: "Raad"
    affiliation: "German University of Digital Science"
    email: "raad.bintareaf@german-uds.de"
  - family-names: "Loucif"
    given-names: "Samia"
    affiliation: "Zayed University"
  - family-names: "Al-Rajab"
    given-names: "Murad"
    affiliation: "Abu Dhabi University"
license: MIT
repository-code: "https://github.com/REPLACE/traitmix"
preferred-citation:
  type: article
  title: "Diverse Minds, Divided Networks? Personality Composition, Polarization, and Collective Intelligence in LLM-Based Social Simulations"
  authors:
    - family-names: "Bin Tareaf"
      given-names: "Raad"
    - family-names: "Loucif"
      given-names: "Samia"
    - family-names: "Al-Rajab"
      given-names: "Murad"
  year: 2026
  journal: "TO BE COMPLETED ON ACCEPTANCE"
"""

GITIGNORE = """__pycache__/
*.pyc
.ipynb_checkpoints/
checkpoints/
results/timeseries/
*.ckpt
*.bak
.env
.DS_Store
"""


def excluded(path: Path) -> bool:
    s = str(path)
    return any(p in s for p in EXCLUDE_PATTERNS)


def scan_secrets(root: Path):
    hits = []
    for f in root.rglob("*"):
        if not f.is_file() or excluded(f) or f.suffix in {".pdf", ".png", ".xlsx", ".zip"}:
            continue
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for pattern, label in SECRET_PATTERNS:
            for m in re.finditer(pattern, text):
                hits.append((str(f.relative_to(root)), label, m.group(0)[:12] + "..."))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../traitmix-release")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n = 0
    for d in INCLUDE_DIRS:
        src = ROOT / d
        if not src.exists():
            continue
        for f in src.rglob("*"):
            if f.is_file() and not excluded(f):
                dst = out / f.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    for name in INCLUDE_FILES:
        f = ROOT / name
        if f.exists():
            shutil.copy2(f, out / name)
            n += 1
    res_out = out / "results"
    res_out.mkdir(exist_ok=True)
    for name in RESULT_FILES:
        f = ROOT / "results" / name
        if f.exists():
            shutil.copy2(f, res_out / name)
            n += 1
        else:
            print(f"  WARNING: results/{name} not found - the release will be incomplete")

    (out / "LICENSE").write_text(LICENSE_MIT)
    (out / "CITATION.cff").write_text(CITATION_CFF)
    (out / ".gitignore").write_text(GITIGNORE)

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
        (out / "environment-freeze.txt").write_text(
            "# Exact environment used to produce the published results.\n"
            "# Serving stack: vLLM 0.8.5.post1, PyTorch 2.6.0+cu124, CUDA 12.4,\n"
            "# NVIDIA driver 550.163.01, single RTX 4090 (24 GB).\n" + freeze)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not freeze environment: {e})")

    print(f"\ncopied {n} files -> {out}")

    print("\nSECRET SCAN")
    hits = scan_secrets(out)
    if hits:
        print(f"  {len(hits)} POSSIBLE CREDENTIAL(S) FOUND - DO NOT PUBLISH UNTIL RESOLVED:")
        for path, label, frag in hits:
            print(f"    {path}: {label} ({frag})")
        print("\n  Remove them, then re-run. If any real credential was ever committed,")
        print("  rotate it: deleting the file does not remove it from git history.")
        sys.exit(1)
    print("  clean - no credential patterns detected")

    print("\nSIZE")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  {total/1e6:.1f} MB")
    if total > 100e6:
        print("  WARNING: over 100 MB. GitHub warns above 50 MB per file and 1 GB per repo;")
        print("  consider putting the large result files on Zenodo only.")

    print("\nNEXT: follow the GitHub and Zenodo steps in the release instructions.")


if __name__ == "__main__":
    main()
