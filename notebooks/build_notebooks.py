"""Builds the 8 orchestration notebooks (.ipynb). Rerun any time: python notebooks/build_notebooks.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent

def nb(cells):
    out = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
           "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
           "nbformat": 4, "nbformat_minor": 5}
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update({"outputs": [], "execution_count": None})
        out["cells"].append(c)
    return out

HDR = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
print("project root:", ROOT)"""

N = {}

N["00_setup_and_smoke_test.ipynb"] = [
("markdown", """# 00 — Setup & smoke test
Verifies the environment, runs the **demo-scale** pipeline end-to-end with the MockLLM
(integration test only — **never paper numbers**), and demonstrates checkpoint/resume.

**Checkpoint levels:** (1) round-level `.ckpt` files in `checkpoints/<run_id>/`,
(2) run-level `results/registry.json` (completed runs are skipped),
(3) HF `checkpoint-*` dirs for QLoRA/classifier training."""),
("code", HDR),
("code", """# 1) Install core requirements (safe to re-run)
# !pip install -r ../requirements.txt"""),
("markdown", """## vLLM server (on your 24 GB machine — run in a separate terminal)
```bash
pip install vllm==0.5.4
# Primary model (E1-E3, E5 non-swap):
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.90
# After notebook 03 (T-arm):
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --lora-modules big5=checkpoints/qlora_big5chat/final
# Model swaps for E1_QWEN / E5: restart with Qwen/Qwen2.5-14B-Instruct-AWQ or google/gemma-2-9b-it
```"""),
("code", """# 2) Smoke run (MockLLM, N=8, T=4) — DEMO-SCALE ONLY
from traitmix.utils import load_config, Registry
from traitmix.runner import run_one
cfg = load_config(ROOT / "configs" / "smoke.yaml")
row = run_one(cfg, seed=1, force=True, keep_ckpt=True)
{k: v for k, v in row.items() if k in ("run_id", "pol_var", "CI_mean_gain_vs_pre", "trait_drift_mean", "runtime_s")}"""),
("code", """# 3) Checkpoint/resume demonstration: wipe the registry entry, keep round ckpts, rerun.
# The engine resumes from the last saved round instead of starting over.
from traitmix.utils import run_id, Registry
from traitmix import checkpointing as ck
rid = run_id(cfg, 1); reg = Registry(); reg.mark(rid, status="interrupted")
print("latest checkpoint round:", (ck.latest(rid) or {}).get("t_done"))
row2 = run_one(cfg, seed=1, resume=True, force=True)
print("resumed & completed:", row2["run_id"], "runtime_s:", row2["runtime_s"])"""),
("code", """# 4) Sanity: raw_results has rows; registry marks them done
from traitmix.utils import read_rows
df = read_rows()
df.tail(3)[[c for c in ["run_id", "config", "seed", "backend", "runtime_s"] if c in df]]"""),
("markdown", """**Gate to proceed:** both cells above completed and `backend == mock` rows appear.
Real experiments (notebooks 02+) require the vLLM server and will refuse mock-only shortcuts."""),
]

N["01_data_preparation.ipynb"] = [
("markdown", """# 01 — Data preparation (all free sources)
Builds: contested-topic file, World-Bank ground truths + **headroom screen**, IPIP check,
and the **cited BFI-2 norms** for the human-calibrated condition (required before full runs)."""),
("code", HDR),
("code", '''# 1) Topics: starter set + (recommended) Chuang et al. statements + ANES-derived items.
import json
topics = [
 {"id": "T_guncontrol", "statement": "Stricter national gun-control laws would make society safer overall."},
 {"id": "T_immigration", "statement": "Current levels of immigration benefit the country more than they cost it."},
 {"id": "T_neutral_filler", "statement": "Pineapple belongs on pizza.", "role": "filler"},
]
# ANES 2024 (free registration): https://electionstudies.org/data-center/2024-time-series-study/
# -> Optionally rephrase 2-4 policy items as statements and append here, citing variable IDs.
# Chuang et al. materials: https://github.com/yunshiuan/llm-agent-opinion-dynamics
(ROOT / "data" / "topics").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "topics" / "topics.json").write_text(json.dumps(topics, indent=1))
print("topics.json written:", [t["id"] for t in topics])'''),
("code", """# 2) World Bank ground truths (free API; graceful manual fallback)
from traitmix.data import load_ci_estimation
items, truths = load_ci_estimation(require_truths=True)
truths"""),
("code", """# 3) HEADROOM SCREEN (needs vLLM up): keep items where the solo model's median relative
# error >= 15% -> writes the filtered wb_items.json used by all experiments.
RUN_SCREEN = False   # set True on your GPU machine
if RUN_SCREEN:
    import json, re, numpy as np
    from traitmix.llm import VLLMClient
    from traitmix.data import DEFAULT_WB_ITEMS
    llm = VLLMClient(model="meta-llama/Llama-3.1-8B-Instruct")
    keep = []
    for it in DEFAULT_WB_ITEMS:
        outs = llm.generate_batch([("You are a careful estimator.",
                f"[ESTIMATE] {it['question']} Reply with a single number only.")]*20, max_tokens=12, temperature=0.8)
        vals = [float(m.group()) for o in outs if (m := re.search(r"-?\\d[\\d,]*\\.?\\d*(?:[eE][+-]?\\d+)?", (o or "").replace(",", "")))]
        err = np.median([abs(v - truths[it["id"]]) / truths[it["id"]] for v in vals]) if vals else 1.0
        print(it["id"], "median rel err:", round(float(err), 3), "->", "KEEP" if err >= 0.15 else "DROP")
        if err >= 0.15: keep.append(it)
    (ROOT / "data" / "ci" / "wb_items.json").write_text(json.dumps(keep, indent=1))
    print("kept", len(keep), "items")"""),
("code", """# 4) IPIP-NEO-120: verify the real item file is in place (public domain: https://ipip.ori.org)
from traitmix.data import load_ipip
try:
    items, name = load_ipip(); print(name, len(items), "items OK")
except FileNotFoundError as e:
    print("ACTION NEEDED:\\n", e)"""),
("code", '''# 5) BFI-2 human norms (REQUIRED before full runs): paste published normative
# means/SDs (rescaled to [0,1]) and the 5x5 trait correlation matrix, WITH the citation.
NORMS = {
  "citation": "TODO e.g. Soto & John (2017), J. Pers. Soc. Psychol., Table X",
  "mu":    {"openness": None, "conscientiousness": None, "extraversion": None, "agreeableness": None, "neuroticism": None},
  "sigma": {"openness": None, "conscientiousness": None, "extraversion": None, "agreeableness": None, "neuroticism": None},
  "corr":  None,  # 5x5 nested list, trait order O,C,E,A,N
}
import yaml, glob
if all(v is not None for v in NORMS["mu"].values()) and NORMS["corr"] and "TODO" not in NORMS["citation"]:
    (ROOT / "configs" / "norms.yaml").write_text(yaml.safe_dump(NORMS))
    for f in glob.glob(str(ROOT / "configs" / "e2" / "e2_human*.yaml")):
        cfg = yaml.safe_load(open(f)); cfg["composition"] = {"mu": NORMS["mu"], "sigma": NORMS["sigma"], "corr": NORMS["corr"]}
        open(f, "w").write(yaml.safe_dump(cfg, sort_keys=False))
    print("human-calibrated configs patched with cited norms.")
else:
    print("Fill NORMS with published values + citation, then re-run this cell. Full runs are blocked until then.")'''),
]

N["02_validation_gate_E0.ipynb"] = [
("markdown", """# 02 — E0 validation gate (run BEFORE spending GPU-days)
Administers IPIP-NEO-120 to every E1 persona configuration (both arms if the LoRA is served).
**Gate: mean Spearman r >= 0.60** (reference: r=.80-.90 for much larger models, Serapio-García et al. 2025).
Pre-registered rule: if P-arm fails and T-arm passes, T-arm becomes primary."""),
("code", HDR),
("code", """import numpy as np, pandas as pd, yaml, glob
from traitmix.data import load_ipip
from traitmix.llm import make_llm
from traitmix.questionnaire import administer, convergent_validity
from traitmix import personality as pers

SMOKE = False          # True = MockLLM + demo battery (pipeline test only)
llm = make_llm({"backend": "mock"} if SMOKE else
               {"backend": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct"})
items, battery = load_ipip(demo_ok=SMOKE)
print("battery:", battery, len(items), "items | backend:", llm.name)"""),
("code", """# target battery = the 11 E1 mu-vectors x 3 sampled agents each
cfgs = sorted(glob.glob(str(ROOT / "configs" / "e1" / "*.yaml")))
targets, measured, rows_long = [], [], []
rng = np.random.default_rng(0)
for f in cfgs:
    from traitmix.utils import load_config
    comp = load_config(f)["composition"]
    for rep in range(3):
        th = pers.sample_society(comp, 1, rng)[0]
        persona = {"name": f"Val_{len(targets)}", "age": 35, "occ": "analyst"}
        sc = administer(llm, th, persona, items, induction="prompt")
        targets.append(th); measured.append([sc[t] for t in pers.TRAITS])
        for k, t in enumerate(pers.TRAITS):
            rows_long.append({"trait": t, "target": th[k], "measured": sc[t]})
val = convergent_validity(np.array(targets), np.array(measured))
pd.DataFrame(rows_long).to_csv(ROOT / "results" / "e0_measured_vs_target.csv", index=False)
pd.DataFrame([{"arm": "prompt", **val}]).to_csv(ROOT / "results" / "e0_validation.csv", index=False)
val"""),
("code", """GATE = 0.60
print("PASS — proceed to full runs" if val["mean_r"] >= GATE else
      "FAIL — do NOT burn GPU-days: train/serve the T-arm (notebook 03), rerun with induction='tags';"
      " if both fail, fall back to binary high/low granularity and report honestly.")"""),
("markdown", """Drift is measured **continuously in-run** (`trait_drift_mean` in every results row) and can be
re-checked post-hoc; optional mid-run questionnaire re-administration can be added to pilot runs."""),
]

N["03_qlora_training_Tarm.ipynb"] = [
("markdown", """# 03 — T-arm training (QLoRA on BIG5-CHAT) + expressed-trait classifier
Both are **resumable** (HF `checkpoint-*`). Skip if the P-arm passed E0 and you accept
prompting-only induction (narrows contribution claim 2)."""),
("code", HDR),
("code", """# QLoRA adapter (~8-12 h on a 24 GB card, 4-bit). Resume: just re-run the cell.
RUN = False   # set True on the GPU machine
if RUN:
    from traitmix.qlora import train
    train(max_rows=None, epochs=1)"""),
("code", """# Expressed-trait classifier (~1 h). Used automatically by telemetry once trained.
RUN_CLF = False
if RUN_CLF:
    from traitmix.classifier import train
    train(max_rows=40000, epochs=1)"""),
("markdown", """Serve for E4: `vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --lora-modules big5=checkpoints/qlora_big5chat/final`"""),
]

N["04_run_experiments_E1_E3.ipynb"] = [
("markdown", """# 04 — Main experiments: E1 (trait levels), E1_QWEN (replication), E2 (heterogeneity), E3 (surface)
Fully **resumable**: completed runs are skipped via the registry; interrupted runs resume mid-round.
Re-running this notebook after a crash continues exactly where it stopped."""),
("code", HDR),
("code", """import json
from traitmix.runner import run_grid
from traitmix.classifier import make_scorer
MAN = json.loads((ROOT / "configs" / "MANIFEST.json").read_text())
scorer = make_scorer()   # classifier if trained, else lexical fallback
def go(exp, model_note=""):
    print(f"== {exp} {model_note} ==")
    paths = [ROOT / "configs" / c for c in MAN[exp]["configs"]]
    return run_grid(paths, MAN[exp]["seeds"], trait_scorer=scorer)"""),
("code", """rows = go("E1")            # Llama-3.1-8B served"""),
("code", """rows = go("E2")"""),
("code", """rows = go("E3")"""),
("code", """# Restart vLLM with Qwen/Qwen2.5-14B-Instruct-AWQ first, then:
rows = go("E1_QWEN", "(restart vLLM with Qwen first)")"""),
("code", """# progress + interim aggregate preview (safe to run any time)
import subprocess, pandas as pd
subprocess.run([sys.executable, str(ROOT / "analysis" / "aggregate_results.py")])
pd.read_csv(ROOT / "results" / "summary.csv").head(12)"""),
]

N["05_run_experiments_E4_E5_scale.ipynb"] = [
("markdown", """# 05 — E4 (T-arm), E5 (robustness audit incl. model swaps), scale ablation
E5 groups by served model — restart vLLM between groups as printed."""),
("code", HDR),
("code", """import json, yaml
from traitmix.runner import run_grid
from traitmix.classifier import make_scorer
MAN = json.loads((ROOT / "configs" / "MANIFEST.json").read_text()); scorer = make_scorer()"""),
("code", """# E4 — requires the LoRA-enabled server (see notebook 03)
rows = run_grid([ROOT / "configs" / c for c in MAN["E4"]["configs"]], MAN["E4"]["seeds"], trait_scorer=scorer)"""),
("code", """# E5 — run in model groups; the registry makes re-runs after each server restart free.
by_model = {}
for c in MAN["E5"]["configs"]:
    m = yaml.safe_load(open(ROOT / "configs" / c)).get("llm", {}).get("model", "meta-llama/Llama-3.1-8B-Instruct")
    by_model.setdefault(m, []).append(c)
for model, cfgs in by_model.items():
    print(f"\\n### Serve this model, then continue: vllm serve {model}\\n({len(cfgs)} configs)")
    rows = run_grid([ROOT / "configs" / c for c in cfgs], MAN["E5"]["seeds"], trait_scorer=scorer)"""),
("code", """rows = run_grid([ROOT / "configs" / c for c in MAN["SCALE"]["configs"]], MAN["SCALE"]["seeds"], trait_scorer=scorer)"""),
]

N["06_analysis_stats.ipynb"] = [
("markdown", """# 06 — Aggregation & pre-registered statistics
mean±std, Wilcoxon vs. baseline, Holm–Bonferroni, Hedges g / Cliff's δ, mixed models,
and the **one-time seed-extension rule** report."""),
("code", HDR),
("code", """import importlib, sys
sys.path.insert(0, str(ROOT / "analysis"))
import aggregate_results; importlib.reload(aggregate_results)
df = aggregate_results.main()"""),
("code", """import pandas as pd
pd.read_csv(ROOT / "results" / "stats.csv").sort_values("p_holm").head(20)"""),
("code", """print(open(ROOT / "results" / "mixed_models.txt").read()[:3000])"""),
("code", """# classical reference curves (CPU): Deffuant + Friedkin-Johnsen on matched initial opinions
import numpy as np
from traitmix.classical import deffuant, friedkin_johnsen
x0 = np.random.default_rng(0).integers(-3, 4, 100).astype(float)
print("Deffuant final var (eps=1.0):", np.var(deffuant(x0, eps=1.0)).round(3),
      "| (eps=3.0):", np.var(deffuant(x0, eps=3.0)).round(3))"""),
]

N["07_figures_tables_supplementary.ipynb"] = [
("markdown", """# 07 — Figures, tables, supplementary
Emits `paper/figures/*.pdf`, `paper/tables/*.tex`, `paper/supplementary/supplementary.tex`.
The Phase-8 manuscript \\input{}s these — numbers are never typed by hand."""),
("code", HDR),
("code", """import subprocess
for script in ["aggregate_results.py", "make_tables.py", "make_figures.py", "make_supplementary.py"]:
    print("::", script); subprocess.run([sys.executable, str(ROOT / "analysis" / script)], check=True)"""),
("code", """for sub in ["figures", "tables", "supplementary"]:
    print(f"-- paper/{sub}:")
    for f in sorted((ROOT / "paper" / sub).glob("*")): print("  ", f.name)"""),
]

for name, cells in N.items():
    (HERE / name).write_text(json.dumps(nb(cells), indent=1))
    print("built", name)
