"""Manipulation & artifact checks: expressed-trait drift, homogenization, sycophancy hook."""
import numpy as np
from . import personality as pers

def expressed_traits(state, scorer=None):
    """scorer: callable(texts)->{trait:score in [0,1]}; defaults to lexical fallback.
    Returns per-agent expressed vector + mean absolute drift from target."""
    scorer = scorer or pers.lexical_trait_scores
    drift = []
    per_agent = {}
    for i in range(len(state["personas"])):
        texts = [p["text"] for p in state["posts"] if p["author"] == i]
        s = scorer(texts)
        per_agent[i] = s
        tgt = state["theta"][i]
        drift.append(np.mean([abs(s[t] - tgt[k]) for k, t in enumerate(pers.TRAITS)]))
    return per_agent, float(np.nanmean(drift))

def homogenization(state, sample=120):
    texts = [p["text"] for p in state["posts"]][-sample:]
    if len(texts) < 5:
        return {"distinct2": float("nan"), "self_bleu3": float("nan")}
    def ngrams(s, n):
        w = s.lower().split(); return set(zip(*[w[i:] for i in range(n)]))
    all2 = [g for t in texts for g in ngrams(t, 2)]
    distinct2 = len(set(all2)) / max(1, len(all2))
    sims = []
    for i in range(min(40, len(texts))):
        a = ngrams(texts[i], 3)
        others = set().union(*(ngrams(t, 3) for j, t in enumerate(texts[:60]) if j != i)) or {("_",)}
        sims.append(len(a & others) / max(1, len(a)))
    return {"distinct2": float(distinct2), "self_bleu3": float(np.mean(sims))}

def filler_variance(state, filler_id):
    if filler_id not in state["opinions"]:
        return float("nan")
    x = np.array(list(state["opinions"][filler_id].values()), float)
    return float(np.var(x))
