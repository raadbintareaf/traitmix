"""OASIS-style social simulation engine, purpose-built for controlled factorial studies:
interest+hot-score recommender, follow/unfollow, bounded memory, private probe channels,
CI task scheduling, and round-level checkpoint/resume."""
import re
import numpy as np
from . import personality as pers
from . import checkpointing as ck
from .data import make_hidden_profile

FIRST = ["Alex","Sam","Jordan","Taylor","Morgan","Casey","Riley","Avery","Quinn","Rowan",
         "Maya","Leo","Nina","Omar","Ivy","Hugo","Zara","Eli","Ada","Noor"]
OCC = ["nurse","teacher","mechanic","designer","farmer","analyst","barista","engineer",
       "librarian","driver","chef","student","electrician","accountant","photographer"]

ACTION_RE = re.compile(r"\b(POST|REPLY|LIKE|FOLLOW|UNFOLLOW|PASS)\b\s*(p?\d+|\w+)?\s*(.*)", re.S)

# --- magnitude-aware numeric parsing for CI estimation answers -------------------
SCALE_WORDS = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6,
               "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12,
               "t": 1e12, "lakh": 1e5, "crore": 1e7}
SCI_RE = re.compile(r"(-?\d+\.?\d*)[eE]([+-]?\d+)")
NUM_RE = re.compile(r"(-?\d[\d,\s]*\.?\d*)\s*([a-zA-Z]+)?")


NO_ROLEPLAY = ("This is a private research survey, not a social media post. Do not role-play, do not use asterisks or stage directions, do not explain. ")


def parse_number(text):
    """'about 55 million' -> 55000000.0 ; '4.34e11' -> 4.34e11 ; 'no idea' -> None."""
    if not text:
        return None
    t = str(text).replace("\u00a0", " ").strip()
    t = re.sub(r"^[^\d\-+.]*", "", t)          # strip leading words ("about", "roughly")
    m = SCI_RE.search(t)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    m = NUM_RE.search(t)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower().strip(".")
    if suffix in SCALE_WORDS:
        val *= SCALE_WORDS[suffix]
    return val


class Simulation:
    def __init__(self, cfg: dict, seed: int, llm, rid: str):
        self.cfg, self.seed, self.llm, self.rid = cfg, seed, llm, rid
        self.n = cfg["society"]["n_agents"]
        self.T = cfg["society"]["rounds"]
        self.rho = cfg["society"]["activation_prob"]
        self.probe_every = cfg["society"].get("probe_every", 5)
        self.feed_size = cfg["society"].get("feed_size", 10)
        self.mem_k = cfg["society"].get("memory_k", 10)
        self.rng = np.random.default_rng(seed)

    # ---------- state ----------
    def fresh_state(self, topics):
        from .network import build
        theta = pers.sample_society(self.cfg["composition"], self.n, self.rng)
        personas = [{"name": f"{self.rng.choice(FIRST)}_{i}", "age": int(self.rng.integers(19, 66)),
                     "occ": str(self.rng.choice(OCC))} for i in range(self.n)]
        g = build(self.cfg["society"]["topology"], self.n, self.rng,
                  **self.cfg["society"].get("topology_kw", {}))
        init_op = {t["id"]: {i: int(self.rng.integers(-2, 3)) for i in range(self.n)} for t in topics}
        return dict(theta=theta, personas=personas,
                    edges=list(g.edges()), posts=[], memories={i: [] for i in range(self.n)},
                    opinions=init_op,                     # latest probed opinion per topic per agent
                    op_history=[],                        # rows: (round, topic, agent, opinion)
                    ci={}, action_log=[], t_done=0, hp_tasks={})

    # ---------- prompts ----------
    def sys_prompt(self, st, i):
        p = st["personas"][i]
        return pers.persona_system_prompt(p["name"], p["age"], p["occ"], st["theta"][i],
                                          induction=self.cfg["induction"]["arm"])

    def feed_for(self, st, i, topics):
        posts = st["posts"]
        if not posts:
            return []
        followees = {v for u, v in st["edges"] if u == i}
        my_ops = {t["id"]: st["opinions"][t["id"]][i] for t in topics}
        def score(p):
            hot = 1.0 + 0.3 * p["likes"] + max(0.0, 3.0 - (st["t_done"] - p["round"]))
            interest = 1.0
            top = p.get("topic")
            if top in my_ops and p["author"] != i:
                a_op = st["opinions"][top][p["author"]]
                interest = 1.0 - abs(my_ops[top] - a_op) / 6.0
            net = 1.5 if p["author"] in followees else 1.0
            return (self.cfg["society"].get("w_interest", 1.0) * interest
                    + self.cfg["society"].get("w_hot", 0.5) * hot) * net
        cand = [p for p in posts if p["author"] != i][-200:]
        return sorted(cand, key=score, reverse=True)[: self.feed_size]

    def action_prompt(self, st, i, feed, topics):
        lines = [f"Discussion topics on the platform right now:"]
        lines += [f"- {t['statement']}" for t in topics if t.get("role") != "filler"]
        if st["memories"][i]:
            lines.append("Things you remember: " + " | ".join(st["memories"][i][-3:]))
        lines.append("Your feed:")
        if feed:
            for p in feed:
                lines.append(f"[p{p['id']}] @{st['personas'][p['author']]['name']}: {p['text'][:180]}")
        else:
            lines.append("(empty)")
        lines.append("Choose ONE action and reply in EXACTLY one of these formats:\n"
                     "POST <your post>\nREPLY p<id> <your reply>\nLIKE p<id>\n"
                     "FOLLOW <username>\nUNFOLLOW <username>\nPASS")
        return "\n".join(lines)

    # ---------- action application ----------
    def apply_action(self, st, i, raw, topics, name2id):
        m = ACTION_RE.search(raw or "")
        verb, arg, rest = (m.group(1), (m.group(2) or "").strip(), (m.group(3) or "").strip()) if m else ("PASS", "", "")
        if verb == "POST" and (arg or rest):
            text = (arg + " " + rest).strip()[:400]
            topic = self._closest_topic(text, topics)
            st["posts"].append({"id": len(st["posts"]), "author": i, "round": st["t_done"],
                                "text": text, "reply_to": None, "likes": 0, "topic": topic})
        elif verb == "REPLY" and arg.startswith("p") and arg[1:].isdigit():
            pid = int(arg[1:])
            if pid < len(st["posts"]):
                st["posts"].append({"id": len(st["posts"]), "author": i, "round": st["t_done"],
                                    "text": rest[:400], "reply_to": pid, "likes": 0,
                                    "topic": st["posts"][pid]["topic"]})
        elif verb == "LIKE" and arg.startswith("p") and arg[1:].isdigit():
            pid = int(arg[1:])
            if pid < len(st["posts"]):
                st["posts"][pid]["likes"] += 1
        elif verb == "FOLLOW" and arg in name2id and name2id[arg] != i:
            e = (i, name2id[arg])
            if e not in st["edges"]: st["edges"].append(e)
        elif verb == "UNFOLLOW" and arg in name2id:
            e = (i, name2id[arg])
            if e in st["edges"]: st["edges"].remove(e)
        st["action_log"].append((st["t_done"], i, verb))
        st["memories"][i] = (st["memories"][i] + [f"round {st['t_done']}: I did {verb} {arg} {rest[:40]}"])[-self.mem_k:]

    def _closest_topic(self, text, topics):
        tl = text.lower()
        for t in topics:
            key = t["id"].split("_", 1)[-1][:6].lower()
            if key and key in tl:
                return t["id"]
        return topics[0]["id"] if topics else None

    # ---------- probes ----------
    def probe_opinions(self, st, topics):
        prompts, meta = [], []
        for t in topics:
            for i in range(self.n):
                cur = st["opinions"][t["id"]][i]
                seen = [p for p in st["posts"][-100:] if p.get("topic") == t["id"] and p["author"] != i]
                feed_avg = float(np.mean([st["opinions"][t["id"]][p["author"]] for p in seen])) if seen else 0.0
                u = (f"[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
                     f"to +3 (strongly agree), what is your CURRENT view on: \"{t['statement']}\"? "
                     f"For context, your previous answer / current opinion is {cur}, and recent posts you saw "
                     f"lean toward a feed average {feed_avg:.1f}. {NO_ROLEPLAY}"
                     f"Reply with a single integer between -3 and 3.")
                prompts.append((self.sys_prompt(st, i), u)); meta.append((t["id"], i))
        outs = self.llm.generate_batch(prompts, max_tokens=6, temperature=0.3)
        for (tid, i), o in zip(meta, outs):
            v = re.search(r"-?\d", o or "")
            if v:
                st["opinions"][tid][i] = int(np.clip(int(v.group()), -3, 3))
            st["op_history"].append((st["t_done"], tid, i, st["opinions"][tid][i]))

    # ---------- CI tasks ----------
    def ci_phase(self, st, item, phase, truth=None):
        tid = item["id"]; st["ci"].setdefault(tid, {"pre": {}, "post": {}, "type": item.get("type", "estimate")})
        prompts = []
        hidden = f" hidden_true={truth}" if (self.llm.name == "mock" and truth is not None) else ""
        for i in range(self.n):
            if item.get("type") == "hidden_profile":
                clues = " ".join(st["hp_tasks"][tid]["clues"][i][:12])
                u = (f"[CHOOSE] {st['hp_tasks'][tid]['prompt']} Facts you personally know: {clues}. "
                     f"{NO_ROLEPLAY}Reply with exactly one character: A or B.")
            else:
                u = (f"[ESTIMATE] Privately estimate: {item['question']}{hidden} "
                     f"{NO_ROLEPLAY}Reply with a single plain number in full digits, "
                     f"no units and no words.")
            prompts.append((self.sys_prompt(st, i), u))
        outs = self.llm.generate_batch(prompts, max_tokens=12, temperature=0.3)
        for i, o in enumerate(outs):
            if item.get("type") == "hidden_profile":
                st["ci"][tid][phase][i] = parse_choice(o)
            else:
                st["ci"][tid][phase][i] = parse_number(o)
        st["ci"][tid].setdefault("raw", {})[phase] = {i: (o or "")[:80] for i, o in enumerate(outs)}
        if phase == "pre":
            if item.get("type") == "hidden_profile":
                # Announce the decision on the platform AND put each agent's private facts
                # into memory, so unshared information can actually enter the discussion.
                task = st["hp_tasks"][tid]
                st["posts"].append({"id": len(st["posts"]), "author": self.n - 1, "round": st["t_done"],
                                    "text": f"Team decision: {task['prompt']} Share any facts you "
                                            f"personally know - others may know things you do not. #decision",
                                    "reply_to": None, "likes": 3, "topic": None})
                for i in range(self.n):
                    facts = task.get("private", {}).get(i, [])
                    if facts:
                        st["memories"][i] = (st["memories"][i] +
                                             ["Facts only I know: " + "; ".join(facts)])[-self.mem_k:]
            else:
                st["posts"].append({"id": len(st["posts"]), "author": self.n - 1, "round": st["t_done"],
                                    "text": f"Team question: {item['question']} What do people think? #estimate",
                                    "reply_to": None, "likes": 2, "topic": None})

    # ---------- main loop ----------
    def run(self, topics, ci_schedule, ci_truths, resume=True):
        st = ck.latest(self.rid) if resume else None
        if st is None:
            st = self.fresh_state(topics)
            for item in [c for c in ci_schedule if c.get("type") == "hidden_profile"]:
                st["hp_tasks"][item["id"]] = make_hidden_profile(self.rng, self.n, item["id"])
            self.probe_opinions(st, topics)  # t=0 baseline
        name2id = {p["name"]: i for i, p in enumerate(st["personas"])}
        start = st["t_done"] + 1
        for t in range(start, self.T + 1):
            st["t_done"] = t
            for item in ci_schedule:
                if item["pre_round"] == t:
                    if item.get("type") == "hidden_profile" and item["id"] not in st["hp_tasks"]:
                        st["hp_tasks"][item["id"]] = make_hidden_profile(self.rng, self.n, item["id"])
                    self.ci_phase(st, item, "pre", ci_truths.get(item["id"]))
            active = [i for i in range(self.n) if self.rng.random() < self.rho]
            prompts = [(self.sys_prompt(st, i), self.action_prompt(st, i, self.feed_for(st, i, topics), topics))
                       for i in active]
            outs = self.llm.generate_batch(prompts, max_tokens=90, temperature=self.cfg["llm"].get("temperature", 0.7))
            for i, o in zip(active, outs):
                self.apply_action(st, i, o, topics, name2id)
            if t % self.probe_every == 0 or t == self.T:
                self.probe_opinions(st, topics)
            for item in ci_schedule:
                if item["post_round"] == t:
                    self.ci_phase(st, item, "post", ci_truths.get(item["id"]))
            ck.save(self.rid, t, st)
        return st


def parse_choice(text):
    """Extract an A/B decision. Case-SENSITIVE so the article "a" is never read as
    choice A (a real bug found in pilot replies)."""
    if not text:
        return None
    t = str(text).strip()
    m = re.match(r"^\s*\(?\*{0,2}([AB])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:choose|choosing|chose|pick|picking|go with|going with|vote for|"
                  r"leaning towards?|prefer|select|selecting)\s+(?:candidate\s+)?([AB])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"candidate\s+([AB])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", t)
    return m.group(1) if m else None
