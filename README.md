# TraitMix — Diverse Minds, Divided Networks?
Reproducible codebase for: *Personality Composition, Polarization, and Collective Intelligence
in LLM-Based Social Simulations* (target: EPJ Data Science).

Runs entirely on **one 24 GB GPU + free data**. Every paper number comes from
`results/raw_results.jsonl` via `analysis/` — tables and figures are generated, never typed.

## Layout
```
configs/            base.yaml, smoke.yaml, generate_grids.py -> e1..e5/, MANIFEST.json (289 planned runs)
notebooks/          00-07 orchestration notebooks (the intended workflow)
src/traitmix/       engine, personality induction, metrics (single implementation), runner, qlora, classifier
analysis/           aggregate_results.py, make_tables.py, make_figures.py, make_supplementary.py
data/               topics/, ci/ (real World Bank truths cached), ipip/ (YOU supply the item file)
results/            raw_results.jsonl (append-only), registry.json, timeseries/, summary/stats
checkpoints/        <run_id>/round_*.ckpt (mid-run resume) + qlora_big5chat/ + trait_classifier/
paper/              figures/*.pdf, tables/*.tex, supplementary/supplementary.tex
```

## Setup (experiment machine)
```bash
pip install -r requirements.txt
pip install torch==2.3.1 vllm==0.5.4 transformers==4.43.3 peft==0.11.1 trl==0.9.6 \
            datasets==2.20.0 accelerate==0.33.0 bitsandbytes==0.43.1        # GPU stack
# terminal 2 — model server:
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.90
```

## Exact reproduction sequence
1. `notebooks/00_setup_and_smoke_test.ipynb` — MockLLM smoke + checkpoint/resume demo (**demo-scale only**).
2. `notebooks/01_data_preparation.ipynb` — topics.json; World-Bank truths; **headroom screen** (vLLM);
   place `data/ipip/ipip_neo_120.csv` (public domain, https://ipip.ori.org); paste **cited BFI-2 norms**
   (full runs are blocked until the norms cell is completed — by design).
3. `notebooks/02_validation_gate_E0.ipynb` — **gate: mean Spearman r ≥ 0.60** before any GPU-days.
4. `notebooks/03_qlora_training_Tarm.ipynb` — QLoRA on BIG5-CHAT (resumable) + trait classifier. Optional if E0 passes on prompting.
5. `notebooks/04_run_experiments_E1_E3.ipynb` — E1 (55), E2 (25), E3 (45); then restart vLLM with
   `Qwen/Qwen2.5-14B-Instruct-AWQ` and run E1_QWEN (55).
6. `notebooks/05_run_experiments_E4_E5_scale.ipynb` — E4 (LoRA server), E5 audit grouped by model
   (Llama / Qwen / `google/gemma-2-9b-it`), scale ablation.
7. `notebooks/06_analysis_stats.ipynb` — summary/stats/mixed models + **seed-extension report**.
8. `notebooks/07_figures_tables_supplementary.ipynb` — all PDFs/TeX for the manuscript.

CLI equivalents: `python analysis/aggregate_results.py && python analysis/make_tables.py &&
python analysis/make_figures.py && python analysis/make_supplementary.py`

## Checkpointing (three levels)
- **Round-level**: `checkpoints/<run_id>/round_XXXX.ckpt` — a killed run resumes mid-simulation
  (RNG state included; demonstrated in notebook 00).
- **Run-level**: `results/registry.json` — completed (config × seed) runs are skipped; re-running a
  notebook after a crash/server-restart continues exactly where it stopped. Force with `force=True`.
- **Training-level**: HF `checkpoint-*` for QLoRA and the classifier (`resume_from_checkpoint`).

## Integrity boundary
`llm.backend: mock` exists ONLY for integration tests; runs are labeled `backend=mock` in every row
and must never enter paper tables. All paper numbers: vLLM backends on your machine.
`configs/e2/e2_human*.yaml` refuse to run until notebook 01's norms cell is completed with a citation.

## Troubleshooting
- World Bank timeout -> rerun cell (retries built in) or add the value manually to `data/ci/wb_truths.json`
  (instructions printed by the error).
- Registry says done but you want a rerun -> `run_one(cfg, seed, force=True)` or delete the entry.
- vLLM model swap mid-E5 -> just restart the server with the printed model; the registry makes re-runs free.
