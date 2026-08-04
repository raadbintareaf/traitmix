"""Administer a personality inventory to a persona configuration and score it."""
import re
import numpy as np
from scipy import stats
from . import personality as pers

def administer(llm, theta, persona, items, induction="prompt", reps=1):
    sys_p = pers.persona_system_prompt(persona["name"], persona["age"], persona["occ"], theta, induction)
    prompts = []
    for _ in range(reps):
        for it in items:
            u = (f"[QUESTIONNAIRE] trait={it['trait']} keyed={it['keyed']} "
                 f"Statement: \"{it['item_text']}\". How accurately does this describe you? "
                 f"Answer with a single number: 1=very inaccurate, 2, 3=neutral, 4, 5=very accurate.")
            prompts.append((sys_p, u))
    outs = llm.generate_batch(prompts, max_tokens=4, temperature=0.3)
    scores = {t: [] for t in pers.TRAITS}
    for it, o in zip(items * reps, outs):
        m = re.search(r"[1-5]", o or "")
        if not m:
            continue
        v = int(m.group())
        if it["keyed"].strip() == "-":
            v = 6 - v
        scores[it["trait"]].append(v)
    return {t: (float(np.mean(v)) if v else float("nan")) for t, v in scores.items()}

def convergent_validity(targets: np.ndarray, measured: np.ndarray):
    """Spearman r per trait across persona configurations."""
    out = {}
    for k, t in enumerate(pers.TRAITS):
        tt, mm = targets[:, k], measured[:, k]
        ok = np.isfinite(mm)
        out[t] = float(stats.spearmanr(tt[ok], mm[ok]).statistic) if ok.sum() > 2 else float("nan")
    out["mean_r"] = float(np.nanmean([out[t] for t in pers.TRAITS]))
    return out
