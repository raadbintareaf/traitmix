#!/usr/bin/env python3
"""
patch_probe_quality.py — Three fixes found by auditing raw model replies in the pilot.

FIX 1  A/B choice parsing false-positive (data integrity).
    The parser uppercased the reply and matched \\b[AB]\\b, so the English article "a"
    scored as choice A - e.g. "It's a tough call but I pick B" was recorded as A.
    Because A is the correct answer, this inflated hidden-profile accuracy. The new
    parser is case-sensitive and choice-aware (leading letter -> explicit choice verb ->
    "Candidate X" -> standalone capital). Verified against the actual pilot replies.

FIX 2  Role-play contamination of private probes.
    Agents answered numeric probes with "*stressed sigh*" and "*posts a photo of a
    bustling Vietnamese market*" because the persona prompt tells them to write casual
    social posts. Post-discussion answers were therefore noisier than pre-discussion
    ones for reasons unrelated to discussion. Probes now explicitly state they are
    private surveys, not posts, and forbid role-play markup.

FIX 3  Estimation items with ambiguous units.
    "What percentage of India's population used the Internet in 2022?" (truth 55.9)
    was answered with counts of people (37,200,000); the Vietnam GDP item drew answers
    three orders of magnitude off. Items now state the unit and answer format
    explicitly, and the candidate pool is widened to 14 items across less
    heavily-memorised countries/indicators so the headroom screen has something to
    select from.

Run from the repo root:  python patch_probe_quality.py
Then run:                python screen_ci_items.py      (REQUIRED - selects final items)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "src" / "traitmix" / "engine.py"
DATA = ROOT / "src" / "traitmix" / "data.py"

CHOICE_PARSER = '''

def parse_choice(text):
    """Extract an A/B decision. Case-SENSITIVE so the article "a" is never read as
    choice A (a real bug found in pilot replies)."""
    if not text:
        return None
    t = str(text).strip()
    m = re.match(r"^\\s*\\(?\\*{0,2}([AB])\\b", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:choose|choosing|chose|pick|picking|go with|going with|vote for|"
                  r"leaning towards?|prefer|select|selecting)\\s+(?:candidate\\s+)?([AB])\\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"candidate\\s+([AB])\\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\b([AB])\\b", t)
    return m.group(1) if m else None
'''

NO_RP = ("This is a private research survey, not a social media post. "
         "Do not role-play, do not use asterisks or stage directions, do not explain. ")

OLD_CHOOSE = '''                u = (f"[CHOOSE] {st['hp_tasks'][tid]['prompt']} Facts you personally know: {clues}. "
                     f"Reply with a single letter: A or B.")'''
NEW_CHOOSE = '''                u = (f"[CHOOSE] {st['hp_tasks'][tid]['prompt']} Facts you personally know: {clues}. "
                     f"{NO_ROLEPLAY}Reply with exactly one character: A or B.")'''

OLD_EST = '''                u = (f"[ESTIMATE] Privately estimate: {item['question']}{hidden} "
                     f"Reply with a single plain number in full digits, no units and no words "
                     f"(for example: 12500000).")'''
NEW_EST = '''                u = (f"[ESTIMATE] Privately estimate: {item['question']}{hidden} "
                     f"{NO_ROLEPLAY}Reply with a single plain number in full digits, "
                     f"no units and no words (for example: 12500000).")'''

OLD_CHOICE_PARSE = '''            if item.get("type") == "hidden_profile":
                m = re.search(r"\\b([AB])\\b", (o or "").upper()); st["ci"][tid][phase][i] = m.group(1) if m else None'''
NEW_CHOICE_PARSE = '''            if item.get("type") == "hidden_profile":
                st["ci"][tid][phase][i] = parse_choice(o)'''

OLD_PROBE = '''                u = (f"[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
                     f"to +3 (strongly agree), what is your CURRENT view on: \\"{t['statement']}\\"? "
                     f"For context, your previous answer / current opinion is {cur}, and recent posts you saw "
                     f"lean toward a feed average {feed_avg:.1f}. Reply with a single integer between -3 and 3.")'''
NEW_PROBE = '''                u = (f"[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
                     f"to +3 (strongly agree), what is your CURRENT view on: \\"{t['statement']}\\"? "
                     f"For context, your previous answer / current opinion is {cur}, and recent posts you saw "
                     f"lean toward a feed average {feed_avg:.1f}. {NO_ROLEPLAY}"
                     f"Reply with a single integer between -3 and 3.")'''

NEW_ITEMS = '''DEFAULT_WB_ITEMS = [
    # Units and answer format are stated explicitly: the pilot showed agents answering a
    # percentage item with population counts. Countries/indicators are deliberately less
    # headline-famous so the headroom screen has candidates with genuine uncertainty.
    {"id": "wb_bgd_elec", "question": "What percentage of Bangladesh's population had access to electricity in 2022? Answer as a percentage between 0 and 100.",
     "country": "BGD", "indicator": "EG.ELC.ACCS.ZS", "year": 2022, "unit": "percent"},
    {"id": "wb_per_forest", "question": "What was the total forest area of Peru in 2021, measured in square kilometres? Answer as a number of square kilometres.",
     "country": "PER", "indicator": "AG.LND.FRST.K2", "year": 2021, "unit": "km2"},
    {"id": "wb_mar_gdppc", "question": "What was Morocco's GDP per capita in 2023, in current US dollars? Answer as a number of dollars.",
     "country": "MAR", "indicator": "NY.GDP.PCAP.CD", "year": 2023, "unit": "USD"},
    {"id": "wb_uzb_pop", "question": "What was the total population of Uzbekistan in 2023? Answer as a number of people.",
     "country": "UZB", "indicator": "SP.POP.TOTL", "year": 2023, "unit": "people"},
    {"id": "wb_zmb_fert", "question": "What was the fertility rate in Zambia in 2022, in births per woman? Answer as a number of births per woman.",
     "country": "ZMB", "indicator": "SP.DYN.TFRT.IN", "year": 2022, "unit": "births"},
    {"id": "wb_ecu_urban", "question": "What percentage of Ecuador's population lived in urban areas in 2023? Answer as a percentage between 0 and 100.",
     "country": "ECU", "indicator": "SP.URB.TOTL.IN.ZS", "year": 2023, "unit": "percent"},
    {"id": "wb_lka_exports", "question": "Exports of goods and services were what percentage of Sri Lanka's GDP in 2022? Answer as a percentage between 0 and 100.",
     "country": "LKA", "indicator": "NE.EXP.GNFS.ZS", "year": 2022, "unit": "percent"},
    {"id": "wb_tun_unemp", "question": "What was Tunisia's total unemployment rate in 2023, as a percentage of the labour force? Answer as a percentage between 0 and 100.",
     "country": "TUN", "indicator": "SL.UEM.TOTL.ZS", "year": 2023, "unit": "percent"},
    {"id": "wb_gha_health", "question": "What was Ghana's current health expenditure per capita in 2021, in current US dollars? Answer as a number of dollars.",
     "country": "GHA", "indicator": "SH.XPD.CHEX.PC.CD", "year": 2021, "unit": "USD"},
    {"id": "wb_bol_life", "question": "What was life expectancy at birth in Bolivia in 2022, in years? Answer as a number of years.",
     "country": "BOL", "indicator": "SP.DYN.LE00.IN", "year": 2022, "unit": "years"},
    {"id": "wb_npl_gdp", "question": "What was Nepal's total GDP in 2023, in current US dollars? Answer as a number of dollars.",
     "country": "NPL", "indicator": "NY.GDP.MKTP.CD", "year": 2023, "unit": "USD"},
    {"id": "wb_pry_internet", "question": "What percentage of Paraguay's population used the Internet in 2022? Answer as a percentage between 0 and 100.",
     "country": "PRY", "indicator": "IT.NET.USER.ZS", "year": 2022, "unit": "percent"},
    {"id": "wb_ken_pop", "question": "What was the total population of Kenya in 2023? Answer as a number of people.",
     "country": "KEN", "indicator": "SP.POP.TOTL", "year": 2023, "unit": "people"},
    {"id": "wb_deu_life", "question": "What was life expectancy at birth in Germany in 2022, in years? Answer as a number of years.",
     "country": "DEU", "indicator": "SP.DYN.LE00.IN", "year": 2022, "unit": "years"},
]'''


def replace_items(text: str) -> str:
    start = text.index("DEFAULT_WB_ITEMS = [")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:start] + NEW_ITEMS + text[i + 1:]


def main() -> None:
    if not ENGINE.exists() or not DATA.exists():
        sys.exit("Run this from the repo root (where src/traitmix/ lives).")
    print("Patching probe quality:")

    t = ENGINE.read_text()
    if "def parse_choice" in t:
        print("  already patched: engine.py")
    else:
        for old, new in [(OLD_CHOOSE, NEW_CHOOSE), (OLD_EST, NEW_EST),
                         (OLD_CHOICE_PARSE, NEW_CHOICE_PARSE), (OLD_PROBE, NEW_PROBE)]:
            if old not in t:
                sys.exit(f"PATCH FAILED (engine.py): anchor not found:\n{old[:140]}...")
            t = t.replace(old, new, 1)
        anchor = "def parse_number(text):"
        t = t.replace(anchor, f'NO_ROLEPLAY = ("{NO_RP}")\n\n\n{anchor}', 1)
        t = t.rstrip() + "\n" + CHOICE_PARSER
        ENGINE.write_text(t)
        print("  patched engine.py")

    d = DATA.read_text()
    if "wb_bgd_elec" in d:
        print("  already patched: data.py")
    else:
        DATA.write_text(replace_items(d))
        print("  patched data.py")

    print("\nSelf-test:")
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    import traitmix.engine as E
    importlib.reload(E)
    cases = [("I'm choosing A. Candidate A's past mistakes", "A"), ("B. I know it's a tough call", "B"),
             ("It's a tough call but I pick B", "B"), ("I'm torn between these two candidates.", None),
             ("*whispers* I'd go with Candidate B", "B"), ("A.", "A")]
    ok = True
    for s, exp in cases:
        got = E.parse_choice(s)
        good = got == exp
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {s[:44]!r:48} -> {got}")
    print("\nAll choice-parser tests passed." if ok else "\nTESTS FAILED - do not run experiments.")
    print("\nNEXT (required): python screen_ci_items.py")
    print("Then discard pre-patch results:  rm -f results/raw_results.jsonl results/registry.json")


if __name__ == "__main__":
    main()
