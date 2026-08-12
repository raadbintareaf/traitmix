"""One (config, seed) run end-to-end -> one row in raw_results.csv. Resume-aware at
run level (registry) and round level (checkpoints)."""
import copy, time
import pandas as pd
from .utils import seed_everything, run_id, Registry, append_row, RESULTS
from .llm import make_llm
from .engine import Simulation
from .data import load_topics, load_ci_estimation
from .metrics_polarization import polarization_battery
from .metrics_ci import ci_battery
from . import telemetry, checkpointing as ck

def build_ci_schedule(cfg):
    sched = []
    for j, item in enumerate(cfg["ci"]["estimation_items"]):
        sched.append({**item, "type": "estimate",
                      "pre_round": cfg["ci"]["est_pre_rounds"][j % len(cfg["ci"]["est_pre_rounds"])],
                      "post_round": cfg["ci"]["est_post_rounds"][j % len(cfg["ci"]["est_post_rounds"])]})
    for j in range(cfg["ci"].get("n_hidden_profile", 0)):
        sched.append({"id": f"hp_{j}", "type": "hidden_profile",
                      "pre_round": cfg["ci"]["hp_pre_rounds"][j],
                      "post_round": cfg["ci"]["hp_post_rounds"][j]})
    return sched

def run_one(cfg: dict, seed: int, llm=None, registry: Registry | None = None,
            resume=True, force=False, keep_ckpt=False, trait_scorer=None):
    registry = registry or Registry()
    cfg = copy.deepcopy(cfg)          # never mutate the caller's config
    rid = run_id(cfg, seed)           # identity frozen BEFORE any runtime injection
    if registry.done(rid) and not force:
        return {"run_id": rid, "skipped": True}
    if force:
        ck.clear(rid)   # a completed round-checkpoint would otherwise short-circuit the rerun
    t0 = time.time()
    if "_VERIFY" in cfg.get("composition", {}) and cfg["llm"]["backend"] != "mock":
        raise RuntimeError("Human-calibrated composition still contains placeholder norms. "
                           "Run notebook 01 (BFI-2 norms cell, with citation) first.")
    seed_everything(seed)
    llm = llm or make_llm(cfg["llm"])
    topics = load_topics() if cfg["topics"].get("source", "default") == "default" else cfg["topics"]["items"]
    items, truths = load_ci_estimation(require_truths=cfg["ci"].get("require_truths", True))
    cfg.setdefault("ci", {})["estimation_items"] = items[: cfg["ci"]["n_estimation"]]
    schedule = build_ci_schedule(cfg)
    registry.mark(rid, status="running")
    sim = Simulation(cfg, seed, llm, rid)
    state = sim.run(topics, schedule, truths, resume=resume)

    topic_ids = [t["id"] for t in topics if t.get("role") != "filler"]
    filler = next((t["id"] for t in topics if t.get("role") == "filler"), None)
    row = {"run_id": rid, "config": cfg["name"], "seed": seed, "backend": llm.name,
           "n_agents": cfg["society"]["n_agents"], "rounds": cfg["society"]["rounds"],
           "induction": cfg["induction"]["arm"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    row.update({f"cond__{k}": v for k, v in flatten_condition(cfg).items()})
    row.update(polarization_battery(state, topic_ids))
    row.update(ci_battery(state, schedule, truths))
    _, drift = telemetry.expressed_traits(state, scorer=trait_scorer)
    row["trait_drift_mean"] = drift
    row.update(telemetry.homogenization(state))
    if filler:
        row["filler_variance"] = telemetry.filler_variance(state, filler)
    # exposure to the collective-intelligence items, so that it is reported rather than
    # assumed, and the ablation switches in force, so the file states its own provenance
    for _k in ("ci_posts", "ci_replies", "ci_mentions"):
        row[_k] = state.get(_k)
    _soc = cfg.get("society", {})
    row["probe_anchors"] = _soc.get("probe_anchors", True)
    row["w_interest"] = _soc.get("w_interest", 1.0)
    row["interest_on_expressed"] = _soc.get("interest_on_expressed", False)
    row["runtime_s"] = round(time.time() - t0, 1)
    append_row(row)
    pd.DataFrame(state["op_history"], columns=["round", "topic", "agent", "opinion"]) \
        .to_csv(RESULTS / "timeseries" / f"{rid}.csv", index=False)
    registry.mark(rid, status="done", runtime_s=row["runtime_s"])
    if not keep_ckpt:
        ck.clear(rid)
    return row

def flatten_condition(cfg):
    comp = cfg["composition"]
    out = {f"mu_{k}": v for k, v in comp["mu"].items()}
    sig = comp["sigma"]
    out.update({f"sigma_{k}": v for k, v in sig.items()} if isinstance(sig, dict) else {"sigma": sig})
    out["corr"] = "human" if isinstance(comp.get("corr"), list) else comp.get("corr", "independent")
    out["topology"] = cfg["society"]["topology"]
    out["model"] = cfg["llm"].get("model", cfg["llm"]["backend"])
    return out

def run_grid(config_paths, seeds, overrides=None, llm=None, **kw):
    from .utils import load_config
    from tqdm import tqdm
    registry = Registry(); rows = []
    jobs = [(p, s) for p in config_paths for s in seeds]
    for p, s in tqdm(jobs, desc="runs"):
        cfg = load_config(p, overrides)
        rows.append(run_one(cfg, s, llm=llm, registry=registry, **kw))
    return rows
