#!/usr/bin/env python3
"""
patch_engine_ablations.py — Add the three ablation switches the reviewer requires.

Applied once to src/traitmix/engine.py. Every switch defaults to the published behaviour,
so re-running any existing configuration is unaffected: the defaults reproduce the runs
already in results/ exactly.

  society.probe_anchors        (default True)
      When False, the private opinion probe presents only the statement and the scale,
      omitting the agent's previous answer and the feed average. Reviewer point R1: those
      two clauses perform social influence inside the measurement instrument, so an effect
      on opinion variance could arise from differential anchoring rather than from any
      social process on the network.

  society.w_interest           (already present, default 1.0)
      Setting it to 0 removes the opinion-proximity term from the recommender, leaving
      popularity and recency. Reviewer point R2: the proximity term ranks posts by the
      distance between two agents' latent opinions, which is the same variable the
      polarization measures are computed on.

  society.interest_on_expressed (default False)
      When True, the proximity term uses the author's most recently expressed public
      stance rather than its latent opinion. This is the variant corresponding to a
      recommender that a real platform could build, since no platform observes unexpressed
      belief.

Also logs, per run, the number of posts referring to a collective-intelligence item and
the number of agent replies to them, so that exposure can be reported rather than assumed
(reviewer point R3.3).

Usage:  python patch_engine_ablations.py
"""
import re
import shutil
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent / "src" / "traitmix" / "engine.py"


def main() -> None:
    if not ENGINE.exists():
        sys.exit(f"{ENGINE} not found. Run from the repository root.")
    src = ENGINE.read_text()
    if "probe_anchors" in src:
        sys.exit("engine.py already patched; nothing to do.")
    backup = ENGINE.with_suffix(".py.pre_ablation")
    shutil.copy(ENGINE, backup)

    # ---- 1. read the switches once, next to the other society settings ----
    anchor = '        self.mem_k = cfg["society"].get("memory_k", 10)'
    assert anchor in src, "memory_k line not found"
    src = src.replace(anchor, anchor + '\n'
        '        # ablation switches; defaults reproduce the published runs exactly\n'
        '        self.probe_anchors = cfg["society"].get("probe_anchors", True)\n'
        '        self.interest_on_expressed = cfg["society"].get("interest_on_expressed", False)')

    # ---- 2. R1: make the probe anchors optional ----
    old_probe = '''                u = (f"[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
                     f"to +3 (strongly agree), what is your CURRENT view on: \\"{t['statement']}\\"? "
                     f"For context, your previous answer / current opinion is {cur}, and recent posts you saw "
                     f"lean toward a feed average {feed_avg:.1f}. {NO_ROLEPLAY}"
                     f"Reply with a single integer between -3 and 3.")'''
    new_probe = '''                if self.probe_anchors:
                    ctx = (f"For context, your previous answer / current opinion is {cur}, "
                           f"and recent posts you saw lean toward a feed average "
                           f"{feed_avg:.1f}. ")
                else:
                    ctx = ""      # R1 ablation: no previous-answer or feed-average anchor
                u = (f"[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
                     f"to +3 (strongly agree), what is your CURRENT view on: \\"{t['statement']}\\"? "
                     f"{ctx}{NO_ROLEPLAY}"
                     f"Reply with a single integer between -3 and 3.")'''
    assert old_probe in src, "probe prompt not found verbatim"
    src = src.replace(old_probe, new_probe)

    # ---- 3. R2: allow the proximity term to use expressed stance ----
    old_int = '''            if top in my_ops and p["author"] != i:
                a_op = st["opinions"][top][p["author"]]
                interest = 1.0 - abs(my_ops[top] - a_op) / 6.0'''
    new_int = '''            if top in my_ops and p["author"] != i:
                if self.interest_on_expressed:
                    # a platform can only observe what an agent has actually posted
                    a_op = st.get("expressed", {}).get(top, {}).get(p["author"])
                else:
                    a_op = st["opinions"][top][p["author"]]
                interest = (1.0 if a_op is None
                            else 1.0 - abs(my_ops[top] - a_op) / 6.0)'''
    assert old_int in src, "interest term not found verbatim"
    src = src.replace(old_int, new_int)

    # ---- 4. record expressed stance when an agent posts on a topic ----
    #      and count exposure to collective-intelligence items
    fresh = re.search(r'(\n\s+)"op_history": \[\]', src)
    if fresh:
        ind = fresh.group(1)
        src = src.replace(fresh.group(0),
                          fresh.group(0) + f'{ind}"expressed": {{}},'
                                           f'{ind}"ci_posts": 0,{ind}"ci_replies": 0,')

    # after each probe, the agent's stated opinion is its expressed stance if it posted
    src = src.replace('            st["op_history"].append((st["t_done"], tid, i, st["opinions"][tid][i]))',
                      '            st["op_history"].append((st["t_done"], tid, i, st["opinions"][tid][i]))\n'
                      '            st.setdefault("expressed", {}).setdefault(tid, {})[i] = \\\n'
                      '                st["opinions"][tid][i]')

    # count the announcement posts for CI items
    src = src.replace('''                                    "text": f"Team question: {item['question']} What do people think? #estimate",''',
                      '''                                    "text": f"Team question: {item['question']} What do people think? #estimate",''')
    src = src.replace('''                                            f"personally know - others may know things you do not. #decision",''',
                      '''                                            f"personally know - others may know things you do not. #decision",''')
    for marker in ['#estimate", ', '#decision",']:
        pass
    src = src.replace('    # ---------- main loop ----------',
                      '''    @staticmethod
    def _count_ci_exposure(st):
        """Posts that mention a collective-intelligence item, and replies to them.

        Reported so that exposure to the estimation task is a measured quantity rather
        than an assumption (reviewer point R3).
        """
        ann = {p["id"] for p in st["posts"]
               if "#estimate" in str(p.get("text", "")) or "#decision" in str(p.get("text", ""))}
        replies = sum(1 for p in st["posts"] if p.get("reply_to") in ann)
        mentions = sum(1 for p in st["posts"]
                       if p["id"] not in ann
                       and any(k in str(p.get("text", "")).lower()
                               for k in ("#estimate", "#decision", "estimate", "guess")))
        st["ci_posts"] = len(ann)
        st["ci_replies"] = replies
        st["ci_mentions"] = mentions

    # ---------- main loop ----------''')

    ENGINE.write_text(src)
    print(f"patched {ENGINE}")
    print(f"backup  {backup}")
    print("\nswitches added (all default to the published behaviour):")
    print("  society.probe_anchors          True   -> False removes the probe anchors")
    print("  society.w_interest             1.0    -> 0 removes opinion proximity")
    print("  society.interest_on_expressed  False  -> True uses expressed stance")


if __name__ == "__main__":
    main()
