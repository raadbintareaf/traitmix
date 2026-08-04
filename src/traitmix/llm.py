"""LLM backends: MockLLM (deterministic, demo-scale only) and VLLMClient
(OpenAI-compatible local server). Batched generation with retries."""
import re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class BaseLLM:
    name = "base"
    def generate_batch(self, prompts: list[tuple[str, str]], max_tokens=120, temperature=0.7) -> list[str]:
        raise NotImplementedError

class MockLLM(BaseLLM):
    """DEMO-SCALE ONLY. Deterministic pseudo-agent: behavior driven by the trait vector and
    the numeric context embedded in the prompt. Exists to integration-test the pipeline;
    its outputs must never be treated as paper results."""
    name = "mock"
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def _theta(self, sys_prompt):
        vals = re.findall(r"(extremely low|very low|slightly low|slightly high|very high|extremely high|low|high|average)", sys_prompt)
        m = {"extremely low": .0, "very low": .12, "low": .25, "slightly low": .38, "average": .5,
             "slightly high": .62, "high": .75, "very high": .88, "extremely high": 1.}
        th = [m.get(v, .5) for v in vals[:5]]
        return (th + [.5] * 5)[:5]
    def generate_batch(self, prompts, max_tokens=120, temperature=0.7):
        outs = []
        for sys_p, user_p in prompts:
            O, C, E, A, N = self._theta(sys_p)
            if "[PROBE]" in user_p:
                cur = re.search(r"current opinion(?: is)? (-?\d)", user_p)
                cur = int(cur.group(1)) if cur else 0
                feed = re.search(r"feed average (-?\d+\.?\d*)", user_p)
                target = float(feed.group(1)) if feed else 0.0
                step = (target - cur) * (0.15 + 0.5 * A) + self.rng.normal(0, 0.4 + 0.8 * N)
                outs.append(str(int(np.clip(round(cur + step), -3, 3))))
            elif "[ESTIMATE]" in user_p:
                true = re.search(r"hidden_true=(\d+\.?\d*)", user_p)
                base = float(true.group(1)) if true else 100.0
                bias = self.rng.lognormal(0, 0.6 - 0.3 * C)
                outs.append(str(round(base * bias, 2)))
            elif "[QUESTIONNAIRE]" in user_p:
                key = 1 if "keyed=+" in user_p else -1
                tr = re.search(r"trait=(\w+)", user_p)
                idx = {"openness": O, "conscientiousness": C, "extraversion": E,
                       "agreeableness": A, "neuroticism": N}.get(tr.group(1) if tr else "", .5)
                mean = 3 + key * (idx - .5) * 3.6
                outs.append(str(int(np.clip(round(self.rng.normal(mean, 0.7)), 1, 5))))
            elif "[CHOOSE]" in user_p:
                opts = re.findall(r"\b([ABC])\)", user_p)
                outs.append(self.rng.choice(opts) if opts else "A")
            else:  # feed action
                stance = int(np.clip(self.rng.normal((O - N) * 2, 1.2), -3, 3))
                w = np.clip(np.array([.25 + .3 * E, .2 + .2 * A, .2, .35 - .3 * E - .2 * A]), 0.01, None)
                act = self.rng.choice(["POST", "REPLY", "LIKE", "PASS"], p=w / w.sum())
                if act == "POST":
                    tone = "love this, great point" if A > .6 else ("this is wrong" if A < .4 else "interesting")
                    worry = " worried about where this goes" if N > .6 else ""
                    outs.append(f"POST [stance:{stance}] {tone}{worry} #topic")
                elif act == "REPLY":
                    pid = re.search(r"\[p(\d+)\]", user_p)
                    agree = "agree, thanks" if A > .5 else "disagree, nonsense"
                    outs.append(f"REPLY p{pid.group(1) if pid else 0} {agree}")
                elif act == "LIKE":
                    pid = re.search(r"\[p(\d+)\]", user_p)
                    outs.append(f"LIKE p{pid.group(1) if pid else 0}")
                else:
                    outs.append("PASS")
        return outs

class VLLMClient(BaseLLM):
    """Talks to a local vLLM OpenAI-compatible server (started by the user, see notebook 00)."""
    def __init__(self, base_url="http://localhost:8000/v1", model="meta-llama/Llama-3.1-8B-Instruct",
                 api_key="EMPTY", max_workers=32, max_retries=4):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.name = model, model
        self.max_workers, self.max_retries = max_workers, max_retries
    def _one(self, sys_p, user_p, max_tokens, temperature):
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
                return r.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"PASS  (LLM_ERROR: {type(e).__name__})"
                time.sleep(2 ** attempt)
    def generate_batch(self, prompts, max_tokens=120, temperature=0.7):
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(self._one, s, u, max_tokens, temperature) for s, u in prompts]
            return [f.result() for f in futs]

def make_llm(cfg_llm: dict):
    if cfg_llm["backend"] == "mock":
        return MockLLM(seed=cfg_llm.get("mock_seed", 0))
    if cfg_llm["backend"] == "vllm":
        return VLLMClient(base_url=cfg_llm.get("base_url", "http://localhost:8000/v1"),
                          model=cfg_llm["model"], max_workers=cfg_llm.get("max_workers", 32))
    raise ValueError(f"unknown backend {cfg_llm['backend']}")
