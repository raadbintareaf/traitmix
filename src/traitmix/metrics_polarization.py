"""Polarization metric battery. Implemented once; imported everywhere."""
import numpy as np
import networkx as nx
from scipy import stats
from .network import opinion_assortativity, ei_index

def sarle_bimodality(x):
    x = np.asarray(x, float); n = len(x)
    if n < 4 or np.std(x) == 0:
        return float("nan")
    g = stats.skew(x); k = stats.kurtosis(x)  # excess
    return float((g**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3))))

def ashman_d(x):
    """Separation of a 2-means split; D>2 ~ clear bimodality."""
    x = np.sort(np.asarray(x, float))
    best = np.nan
    for cut in range(2, len(x) - 2):
        a, b = x[:cut], x[cut:]
        s = np.sqrt((a.var() + b.var()))
        d = np.sqrt(2) * abs(a.mean() - b.mean()) / s if s > 0 else np.nan
        best = np.nanmax([best, d])
    return float(best)

def dip_stat(x):
    try:
        import diptest
        return float(diptest.dipstat(np.asarray(x, float)))
    except Exception:
        return float("nan")  # fallback: Sarle + Ashman carry bimodality reporting

def cross_cutting_rate(posts, opinions_by_topic):
    num = den = 0
    for p in posts:
        if p["reply_to"] is None or p.get("topic") not in opinions_by_topic:
            continue
        parent = posts[p["reply_to"]]
        ops = opinions_by_topic[p["topic"]]
        s1, s2 = np.sign(ops[p["author"]]), np.sign(ops[parent["author"]])
        if s1 == 0 or s2 == 0:
            continue
        den += 1; num += int(s1 != s2)
    return float(num / den) if den else float("nan")

def toxicity_cross_camp(posts, opinions_by_topic):
    try:
        from detoxify import Detoxify
    except Exception:
        return float("nan")
    texts = []
    for p in posts:
        if p["reply_to"] is None or p.get("topic") not in opinions_by_topic:
            continue
        ops = opinions_by_topic[p["topic"]]
        parent = posts[p["reply_to"]]
        if np.sign(ops[p["author"]]) * np.sign(ops[parent["author"]]) == -1:
            texts.append(p["text"])
    if not texts:
        return float("nan")
    scores = Detoxify("original").predict(texts[:200])["toxicity"]
    return float(np.mean(scores))

def polarization_battery(state, topic_ids):
    g = nx.DiGraph(); g.add_nodes_from(range(len(state["personas"]))); g.add_edges_from(state["edges"])
    out = {}
    for tid in topic_ids:
        x = np.array([state["opinions"][tid][i] for i in range(len(state["personas"]))], float)
        out[f"{tid}__var"] = float(np.var(x))
        out[f"{tid}__extremity"] = float(np.mean(np.abs(x)))
        out[f"{tid}__sarle_bc"] = sarle_bimodality(x)
        out[f"{tid}__ashman_d"] = ashman_d(x)
        out[f"{tid}__dip"] = dip_stat(x)
        out[f"{tid}__assort"] = opinion_assortativity(g.copy(), state["opinions"][tid])
        out[f"{tid}__ei_index"] = ei_index(g, state["opinions"][tid])
    ops = {tid: state["opinions"][tid] for tid in topic_ids}
    out["crosscut_rate"] = cross_cutting_rate(state["posts"], ops)
    out["toxicity_crosscamp"] = toxicity_cross_camp(state["posts"], ops)
    return out
