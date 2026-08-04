"""Collective-intelligence metric battery, incl. Page's diversity-prediction decomposition
(computed in log10 space for magnitude-estimation items; identity is exact)."""
import numpy as np

def _valid(d): return {k: v for k, v in d.items() if v is not None and np.isfinite(v) and v > 0}

def estimation_metrics(pre: dict, post: dict, truth: float, prefix: str):
    out = {}
    for phase, d in [("pre", pre), ("post", post)]:
        vals = np.array(list(_valid(d).values()), float)
        if len(vals) < 3 or not truth or truth <= 0:
            out[f"{prefix}__{phase}_n"] = len(vals); continue
        logs, lt = np.log10(vals), np.log10(truth)
        out[f"{prefix}__{phase}_n"] = len(vals)
        out[f"{prefix}__{phase}_collective_sqerr"] = float((logs.mean() - lt) ** 2)
        out[f"{prefix}__{phase}_median_relerr"] = float(abs(np.median(vals) - truth) / truth)
        out[f"{prefix}__{phase}_avg_individual_sqerr"] = float(((logs - lt) ** 2).mean())
        out[f"{prefix}__{phase}_diversity"] = float(logs.var())
    if f"{prefix}__post_collective_sqerr" in out and f"{prefix}__pre_collective_sqerr" in out:
        out[f"{prefix}__gain_vs_pre"] = out[f"{prefix}__pre_collective_sqerr"] - out[f"{prefix}__post_collective_sqerr"]
        out[f"{prefix}__gain_vs_individuals"] = (out[f"{prefix}__post_avg_individual_sqerr"]
                                                 - out[f"{prefix}__post_collective_sqerr"])
    return out

def hidden_profile_metrics(pre: dict, post: dict, correct: str, prefix: str):
    def rate(d):
        v = [x for x in d.values() if x in ("A", "B")]
        return float(np.mean([x == correct for x in v])) if v else float("nan")
    return {f"{prefix}__pre_correct_rate": rate(pre), f"{prefix}__post_correct_rate": rate(post),
            f"{prefix}__solved_majority": float(rate(post) > 0.5) if np.isfinite(rate(post)) else float("nan")}

def ci_battery(state, ci_schedule, truths):
    out = {}
    for item in ci_schedule:
        rec = state["ci"].get(item["id"], {"pre": {}, "post": {}})
        if item.get("type") == "hidden_profile":
            correct = state["hp_tasks"][item["id"]]["correct"]
            out.update(hidden_profile_metrics(rec["pre"], rec["post"], correct, item["id"]))
        else:
            out.update(estimation_metrics(rec["pre"], rec["post"], truths.get(item["id"]), item["id"]))
    # composites
    gains = [v for k, v in out.items() if k.endswith("__gain_vs_pre")]
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
    return out
