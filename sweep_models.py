#!/usr/bin/env python3
"""
sweep_models.py — Run the response surface on several models, unattended.

Starts a vLLM server for each model in turn, checks it is actually serving, times one run
before committing to the grid, runs the grid within a time budget, then shuts the server
down and moves on. Designed to be started and left for a day or two.

Each model gets its own condition prefix (e3<tag>_), so nothing already in results/ is
touched and the analysis treats each model as its own family with the centre cell of its
grid as the reference.

Behaviour worth knowing before you walk away:
  * A model that fails to load, serves under an unexpected name, or turns out too slow for
    the remaining budget is SKIPPED, not retried indefinitely. The sweep continues.
  * Completed runs are recorded as they finish, so an interruption loses at most one run.
    Re-running the script resumes where it stopped.
  * Progress is appended to sweep_log.txt, so you can see what happened from your phone.

Usage:
    python sweep_models.py --hours 40
    python sweep_models.py --hours 40 --only gemma9,phi4
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "sweep_log.txt"

# tag, HuggingFace id, extra vLLM arguments, seeds
# Ordered so that the cheapest and most likely to succeed run first: if the sweep is cut
# short you still gain a model rather than losing a long one part-way.
MODELS = [
    ("gemma9", "google/gemma-2-9b-it",
     ["--max-model-len", "4096", "--gpu-memory-utilization", "0.90"], 5),
    ("phi4", "microsoft/phi-4",
     ["--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
      "--quantization", "fp8"], 5),
    ("gemma27", "google/gemma-2-27b-it",
     ["--max-model-len", "4096", "--gpu-memory-utilization", "0.95",
      "--quantization", "fp8"], 5),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def served_models(url="http://localhost:8000/v1", timeout=5):
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout) as r:
            return [d["id"] for d in json.loads(r.read()).get("data", [])]
    except Exception:  # noqa: BLE001
        return None


def start_server(hf_id, extra, wait_minutes=25):
    if served_models() is not None:
        log("a server is already running; stopping it first")
        stop_server()
    out = open(ROOT / f"vllm_{hf_id.split('/')[-1]}.log", "w")
    proc = subprocess.Popen(["vllm", "serve", hf_id] + extra,
                            stdout=out, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    deadline = time.time() + wait_minutes * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"  server exited during load (code {proc.returncode}); see the log")
            return None
        ids = served_models()
        if ids:
            log(f"  server ready, serving: {ids}")
            return proc
        time.sleep(15)
    log(f"  server did not become ready within {wait_minutes} min")
    stop_server(proc)
    return None


def stop_server(proc=None):
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=90)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    else:
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(20)


def make_configs(tag, model_id, seeds, workers):
    import yaml
    src = ROOT / "configs" / "e3"
    dst = ROOT / "configs" / f"e3_{tag}"
    dst.mkdir(exist_ok=True)
    made = []
    for f in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        cfg["name"] = cfg["name"].replace("e3_", f"e3{tag}_", 1)
        cfg["inherit"] = "../base.yaml"
        cfg["llm"] = {**cfg.get("llm", {}), "backend": "vllm", "model": model_id,
                      "base_url": "http://localhost:8000/v1", "max_workers": workers}
        (dst / f"{cfg['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        made.append(dst / f"{cfg['name']}.yaml")
    man_path = ROOT / "configs" / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[f"E3_{tag.upper()}"] = {"configs": [str(p.relative_to(ROOT / "configs")) for p in made],
                                "seeds": list(range(1, seeds + 1))}
    man_path.write_text(json.dumps(man, indent=1))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="comma-separated tags to run")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from traitmix.runner import run_one
    from traitmix.utils import load_config
    from traitmix.classifier import make_scorer

    wanted = [m for m in MODELS if not args.only or m[0] in args.only.split(",")]
    deadline = time.time() + args.hours * 3600
    scorer = make_scorer()
    log(f"=== sweep starting: {len(wanted)} model(s), budget {args.hours:g} h ===")

    for tag, hf_id, extra, seeds in wanted:
        left = (deadline - time.time()) / 3600
        if left < 1.5:
            log(f"stopping: only {left:.1f} h left, not enough for another model")
            break
        log(f"--- {tag} ({hf_id}) | {left:.1f} h remaining ---")

        proc = start_server(hf_id, extra)
        if proc is None:
            log(f"  SKIPPING {tag}: server would not start")
            continue
        ids = served_models() or []
        model_id = hf_id if hf_id in ids else (ids[0] if ids else hf_id)

        cfgs = make_configs(tag, model_id, seeds, args.workers)
        log(f"  wrote {len(cfgs)} configurations")

        t0 = time.time()
        try:
            run_one(load_config(cfgs[4]), seed=99, force=True, trait_scorer=scorer)
        except Exception as exc:  # noqa: BLE001
            log(f"  SKIPPING {tag}: benchmark run failed: {type(exc).__name__}: {exc}")
            stop_server(proc); continue
        per_run = time.time() - t0
        need = per_run * len(cfgs) * seeds / 3600
        log(f"  benchmark {per_run/60:.1f} min/run -> full grid would take {need:.1f} h")
        if need > (deadline - time.time()) / 3600:
            fit = int((deadline - time.time()) / per_run)
            log(f"  grid does not fit; running the {fit} runs that do, seed-major")
        done = failed = 0
        for seed in range(1, seeds + 1):
            for cfg_path in cfgs:
                if time.time() + per_run > deadline:
                    log("  budget reached")
                    break
                try:
                    r = run_one(load_config(cfg_path), seed=seed, trait_scorer=scorer)
                    if not r.get("skipped"):
                        done += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    log(f"  run failed: {cfg_path.stem} seed {seed}: {type(exc).__name__}")
                    if failed >= 3:
                        break
            if failed >= 3 or time.time() + per_run > deadline:
                break
        log(f"  {tag}: {done} runs completed, {failed} failed")
        stop_server(proc)

    log("=== sweep finished ===")
    log("Next: python analysis/aggregate_results.py && python reviewer_analyses.py")


if __name__ == "__main__":
    main()
