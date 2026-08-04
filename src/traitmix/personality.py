"""Big Five trait sampling, prompt-based induction (P-arm), trait tags (T-arm),
and lexical expressed-trait scoring used as a fallback when the classifier is absent."""
import numpy as np

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
SHORT = dict(zip(TRAITS, ["O", "C", "E", "A", "N"]))

# 9-level linguistic anchors (Serapio-Garcia-style graded qualifiers on IPIP-marker phrasing).
LEVELS = ["extremely low", "very low", "low", "slightly low", "average",
          "slightly high", "high", "very high", "extremely high"]
MARKERS = {  # positive-pole marker descriptions; negated automatically for low levels
    "openness": "curious about many different things, imaginative, and quick to explore new ideas",
    "conscientiousness": "organized, thorough, reliable, and someone who plans and follows through",
    "extraversion": "talkative, energetic, assertive, and someone who seeks out social interaction",
    "agreeableness": "considerate, cooperative, trusting, and someone who avoids conflict",
    "neuroticism": "easily stressed, prone to worry, and emotionally reactive",
}

def sample_society(cfg_comp: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """theta ~ TruncNormal5(mu, Sigma) on [0,1]. cfg_comp: {mu:{trait:val}, sigma:{trait:val}|float, corr:'independent'|matrix}"""
    mu = np.array([cfg_comp["mu"][t] for t in TRAITS])
    sig = cfg_comp["sigma"]
    sig = np.array([sig[t] for t in TRAITS]) if isinstance(sig, dict) else np.full(5, float(sig))
    corr = cfg_comp.get("corr", "independent")
    C = np.eye(5) if corr == "independent" else np.array(corr, dtype=float)
    cov = np.outer(sig, sig) * C
    theta = rng.multivariate_normal(mu, cov, size=n)
    for _ in range(20):  # resample-out-of-bounds truncation
        bad = (theta < 0) | (theta > 1)
        if not bad.any():
            break
        theta[bad.any(1)] = rng.multivariate_normal(mu, cov, size=int(bad.any(1).sum()))
    return np.clip(theta, 0, 1)

def level_index(v: float) -> int:
    return int(np.clip(round(v * 8), 0, 8))

def trait_sentence(trait: str, v: float) -> str:
    li = level_index(v); lvl = LEVELS[li]
    if li == 4:
        return f"You are about average in {trait}."
    pole = MARKERS[trait]
    if li > 4:
        return f"Your {trait} is {lvl}: you are {pole}."
    return f"Your {trait} is {lvl}: you are the opposite of someone {pole}."

def persona_system_prompt(name: str, age: int, occupation: str, theta: np.ndarray,
                          induction: str = "prompt") -> str:
    lines = [f"You are {name}, a {age}-year-old {occupation} using a social media platform.",
             "Stay in character at all times. Write casual, short social-media posts (max 60 words).",
             "Never mention that you are an AI, a language model, or part of a simulation."]
    if induction == "prompt":
        lines.append("Your personality:")
        lines += ["- " + trait_sentence(t, v) for t, v in zip(TRAITS, theta)]
    elif induction == "tags":  # T-arm: control tags the QLoRA adapter was trained on
        tags = " ".join(f"<{SHORT[t]}={LEVELS[level_index(v)].replace(' ', '_')}>" for t, v in zip(TRAITS, theta))
        lines.append(f"[personality control] {tags}")
    return "\n".join(lines)

# --- crude lexical expressed-trait fallback (classifier in classifier.py is preferred) ---
_LEX = {
    "extraversion": (["!", "party", "everyone", "let's", "we ", "friends", "hey"], ["alone", "quiet", "stay home"]),
    "agreeableness": (["thanks", "agree", "great point", "love this", "appreciate"], ["wrong", "stupid", "nonsense", "disagree"]),
    "neuroticism": (["worried", "anxious", "scared", "stressed", "afraid", "terrible"], ["calm", "relaxed", "no worries"]),
    "openness": (["curious", "interesting", "imagine", "what if", "new", "idea"], ["boring", "same old", "pointless"]),
    "conscientiousness": (["plan", "schedule", "carefully", "checked", "source", "verify"], ["whatever", "who cares", "later"]),
}

def lexical_trait_scores(texts: list[str]) -> dict:
    blob = " ".join(texts).lower() if texts else ""
    out = {}
    for t, (pos, neg) in _LEX.items():
        p = sum(blob.count(w) for w in pos); m = sum(blob.count(w) for w in neg)
        out[t] = 0.5 if (p + m) == 0 else p / (p + m)
    return out
