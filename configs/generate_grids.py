"""Generates every experiment config (E1-E5 + scale ablation) from base.yaml.
Run: python configs/generate_grids.py  -> writes configs/e*/ *.yaml + configs/MANIFEST.json"""
import json, copy, itertools
from pathlib import Path
import yaml

HERE = Path(__file__).parent
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
HI, LO, MID = 0.8, 0.2, 0.5
# Human-calibrated (mu, sigma, corr) -- placeholder values MUST be replaced with published
# BFI-2 normative statistics (with citation) in notebook 01 before full runs. Guard enforces this.
HUMAN_NORM = {"mu": {t: 0.5 for t in TRAITS}, "sigma": {t: 0.15 for t in TRAITS},
              "corr": "independent", "_VERIFY": "REPLACE with cited BFI-2 norms in notebook 01"}

def write(group, name, cfg):
    d = HERE / group; d.mkdir(exist_ok=True)
    cfg = dict(cfg, name=name)
    (d / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return f"{group}/{name}.yaml"

def main():
    manifest = {}
    # ---- E1: trait levels (11 conditions) ----
    e1 = [write("e1", "e1_norm_baseline", {"inherit": "../base.yaml"})]
    for t, lvl, v in [(t, l, v) for t in TRAITS for l, v in [("high", HI), ("low", LO)]]:
        mu = {x: MID for x in TRAITS}; mu[t] = v
        e1.append(write("e1", f"e1_{t}_{lvl}", {"inherit": "../base.yaml",
                                                "composition": {"mu": mu, "sigma": 0.15}}))
    manifest["E1"] = {"configs": e1, "seeds": [1, 2, 3, 4, 5]}
    # ---- E1 replication on Qwen (Option B) ----
    e1q = [c.replace("e1/", "e1_qwen/").replace("e1_", "e1q_") for c in e1]
    for src, dst in zip(e1, e1q):
        cfg = yaml.safe_load((HERE / src).read_text())
        cfg["inherit"] = "../base.yaml"
        cfg["llm"] = {"model": "Qwen/Qwen2.5-14B-Instruct-AWQ"}
        name = Path(dst).stem
        write("e1_qwen", name, cfg)
    manifest["E1_QWEN"] = {"configs": [f"e1_qwen/{Path(d).stem}.yaml" for d in e1q], "seeds": [1, 2, 3, 4, 5]}
    # ---- E2: heterogeneity (5 conditions) ----
    e2 = []
    for name, sig, corr in [("homog", 0.05, "independent"), ("mid", 0.15, "independent"),
                            ("diverse", 0.25, "independent"),
                            ("human_corr", "HUMAN", "human"), ("human_norm", "HNORM", "human")]:
        comp = copy.deepcopy(HUMAN_NORM) if corr == "human" else {"mu": {t: MID for t in TRAITS}, "sigma": sig}
        if sig == "HUMAN":
            comp["sigma"] = {t: 0.25 for t in TRAITS}
        e2.append(write("e2", f"e2_{name}", {"inherit": "../base.yaml", "composition": comp}))
    manifest["E2"] = {"configs": e2, "seeds": [1, 2, 3, 4, 5]}
    # ---- E3: O x A response surface (3x3) ----
    e3 = []
    for vo, va in itertools.product([LO, MID, HI], repeat=2):
        mu = {t: MID for t in TRAITS}; mu["openness"], mu["agreeableness"] = vo, va
        e3.append(write("e3", f"e3_O{int(vo*10)}_A{int(va*10)}",
                        {"inherit": "../base.yaml", "composition": {"mu": mu, "sigma": 0.15}}))
    manifest["E3"] = {"configs": e3, "seeds": [1, 2, 3, 4, 5]}
    # ---- E4: T-arm subset (5 conditions) ----
    e4 = []
    for src in ["e1/e1_norm_baseline", "e1/e1_openness_high", "e1/e1_openness_low",
                "e1/e1_agreeableness_high", "e1/e1_neuroticism_high"]:
        cfg = yaml.safe_load((HERE / f"{src}.yaml").read_text())
        cfg["induction"] = {"arm": "tags"}
        cfg["llm"] = {"model": "big5"}   # LoRA module name served by vLLM --enable-lora
        e4.append(write("e4", f"e4_{Path(src).stem[3:]}_tags", cfg))
    manifest["E4"] = {"configs": e4, "seeds": [1, 2, 3, 4, 5]}
    # ---- E5: robustness audit (3 anchors x 8 perturbations) ----
    anchors = ["e1/e1_norm_baseline", "e1/e1_openness_high", "e1/e1_neuroticism_high"]
    perts = {
        "ws":      {"society": {"topology": "watts_strogatz", "topology_kw": {"k": 6, "p": 0.1}}},
        "er":      {"society": {"topology": "erdos_renyi", "topology_kw": {"p_edge": 0.06}}},
        "rho_lo":  {"society": {"activation_prob": 0.2}},
        "rho_hi":  {"society": {"activation_prob": 0.6}},
        "mem":     {"society": {"memory_k": 20}},
        "feed":    {"society": {"feed_size": 15}},
        "temp":    {"llm": {"temperature": 1.0}},
        "qwen":    {"llm": {"model": "Qwen/Qwen2.5-14B-Instruct-AWQ"}},
        "gemma":   {"llm": {"model": "google/gemma-2-9b-it"}},   # 3rd family at anchors (Option B)
    }
    e5 = []
    for a in anchors:
        base_cfg = yaml.safe_load((HERE / f"{a}.yaml").read_text())
        for pname, patch in perts.items():
            cfg = copy.deepcopy(base_cfg)
            for k, v in patch.items():
                cfg.setdefault(k, {}); cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
            e5.append(write("e5", f"e5_{Path(a).stem[3:]}__{pname}", cfg))
    manifest["E5"] = {"configs": e5, "seeds": [1, 2, 3]}
    # ---- Scale ablation ----
    sa = [write("scale", "scale_n200_norm", {"inherit": "../base.yaml",
                                             "society": {"n_agents": 200}})]
    manifest["SCALE"] = {"configs": sa, "seeds": [1, 2, 3]}
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    total = sum(len(v["configs"]) * len(v["seeds"]) for v in manifest.values())
    print(f"wrote {sum(len(v['configs']) for v in manifest.values())} configs; total planned runs = {total}")

if __name__ == "__main__":
    main()
