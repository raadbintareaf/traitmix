"""Classical opinion-dynamics reference models (CPU, seconds)."""
import numpy as np

def deffuant(x0, eps=1.0, mu=0.3, steps=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.array(x0, float)
    for _ in range(steps):
        i, j = rng.integers(0, len(x), 2)
        if abs(x[i] - x[j]) < eps:
            d = mu * (x[j] - x[i]); x[i] += d; x[j] -= d
    return x

def friedkin_johnsen(W, s, lam=0.5, steps=200):
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    x = np.array(s, float)
    for _ in range(steps):
        x = lam * W @ x + (1 - lam) * s
    return x
