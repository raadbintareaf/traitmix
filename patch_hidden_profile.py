#!/usr/bin/env python3
"""
patch_hidden_profile.py — Fix the hidden-profile collective-intelligence task.

TWO BUGS FIXED
1. The clues were never surfaced into the feed: ci_phase seeded a discussion post only
   for estimation items, so hidden-profile pre/post answers were made on identical
   information and the task could not be solved. Now the task is announced on the
   platform AND each agent's private clues are written into their memory, so they can
   actually share what only they know during the discussion window.
2. Clue distribution did not scale to N=100: 9 unshared clues went to 9 random agents,
   leaving 91 agents with only the (misleading) shared information. Now every agent
   receives a random subset of the unshared clues, so no individual can solve it alone
   (the hidden-profile paradigm is preserved) but the group collectively holds every
   fact many times over and discussion can, in principle, aggregate them.

Run from the repo root:  python patch_hidden_profile.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "src" / "traitmix" / "data.py"
ENGINE = ROOT / "src" / "traitmix" / "engine.py"

OLD_HP = '''def make_hidden_profile(rng: np.random.Generator, n_agents: int, task_id="hp_0"):
    """Candidate A is objectively best (6 pros / 2 cons) but its pros are UNSHARED
    (each known to one agent); B looks better on shared info (3 shared pros)."""'''

NEW_HP = '''def make_hidden_profile(rng: np.random.Generator, n_agents: int, task_id="hp_0",
                       clues_per_agent=2):
    """Candidate A is objectively best (6 pros / 2 cons) but its pros are UNSHARED;
    B looks better on the information everyone holds (3 shared pros + A's 2 cons).

    Each agent receives all shared clues plus a random subset of the unshared clues
    (clues_per_agent), so no individual can solve the task alone while the group
    collectively holds every fact - the classic Stasser-Titus structure, scaled to
    large societies."""'''

OLD_TAIL = '''    shared = b_pros + a_cons
    unshared = a_pros + b_cons_unshared
    clues = {i: list(shared) for i in range(n_agents)}
    for k, clue in enumerate(unshared):
        clues[int(rng.integers(0, n_agents))].append(clue)
    return {"id": task_id, "prompt": "Your team must choose the better job candidate: A) Candidate A  B) Candidate B.",
            "correct": "A", "clues": clues}'''

NEW_TAIL = '''    shared = b_pros + a_cons
    unshared = a_pros + b_cons_unshared
    k = int(min(max(1, clues_per_agent), len(unshared) - 1))  # never give one agent everything
    clues, private = {}, {}
    for i in range(n_agents):
        picks = [unshared[j] for j in rng.choice(len(unshared), size=k, replace=False)]
        private[i] = picks
        clues[i] = list(shared) + picks
    return {"id": task_id,
            "prompt": "Your team must choose the better job candidate: A) Candidate A  B) Candidate B.",
            "correct": "A", "clues": clues, "private": private}'''

OLD_SEED = '''        if item.get("type") != "hidden_profile" and phase == "pre":
            # seed discussion with a pinned prompt post
            st["posts"].append({"id": len(st["posts"]), "author": self.n - 1, "round": st["t_done"],
                                "text": f"Team question: {item['question']} What do people think? #estimate",
                                "reply_to": None, "likes": 2, "topic": None})'''

NEW_SEED = '''        if phase == "pre":
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
                                    "reply_to": None, "likes": 2, "topic": None})'''


def patch(path: Path, pairs) -> None:
    text = path.read_text()
    for old, new in pairs:
        if new.split("\n")[0].strip() in text and old not in text:
            print(f"  already patched: {path.name}")
            return
        if old not in text:
            sys.exit(f"PATCH FAILED in {path.name}: anchor not found:\n{old[:120]}...")
        text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"  patched {path.name}")


def main() -> None:
    if not DATA.exists() or not ENGINE.exists():
        sys.exit("Run this from the repo root (where src/traitmix/ lives).")
    print("Patching hidden-profile task:")
    patch(DATA, [(OLD_HP, NEW_HP), (OLD_TAIL, NEW_TAIL)])
    patch(ENGINE, [(OLD_SEED, NEW_SEED)])
    print("\nDone. Because this changes task behaviour without changing config hashes,")
    print("clear the old results before re-running so the two versions never mix:")
    print("  mv results/raw_results.jsonl results/raw_results_PREPATCH.jsonl")
    print("  mv results/registry.json     results/registry_PREPATCH.json")


if __name__ == "__main__":
    main()
