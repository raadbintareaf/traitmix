"""Network construction and structural metrics."""
import networkx as nx
import numpy as np

def build(topology: str, n: int, rng: np.random.Generator, **kw) -> nx.DiGraph:
    seed = int(rng.integers(0, 2**31 - 1))
    if topology == "barabasi_albert":
        g = nx.barabasi_albert_graph(n, kw.get("m", 3), seed=seed)
    elif topology == "watts_strogatz":
        g = nx.watts_strogatz_graph(n, kw.get("k", 6), kw.get("p", 0.1), seed=seed)
    elif topology == "erdos_renyi":
        g = nx.gnp_random_graph(n, kw.get("p_edge", 0.06), seed=seed)
    else:
        raise ValueError(topology)
    return nx.DiGraph(g)  # mutualize undirected -> both directions

def opinion_assortativity(g: nx.DiGraph, opinions: dict) -> float:
    nx.set_node_attributes(g, opinions, "op")
    try:
        return float(nx.numeric_assortativity_coefficient(g, "op"))
    except Exception:
        return float("nan")

def ei_index(g: nx.DiGraph, opinions: dict) -> float:
    """Krackhardt E-I on opinion sign (negative = echo-chamber-like closure). Neutral(0) excluded."""
    ext = intr = 0
    for u, v in g.edges():
        su, sv = np.sign(opinions.get(u, 0)), np.sign(opinions.get(v, 0))
        if su == 0 or sv == 0:
            continue
        if su == sv: intr += 1
        else: ext += 1
    return float((ext - intr) / (ext + intr)) if (ext + intr) else float("nan")
