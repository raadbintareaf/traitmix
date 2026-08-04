"""Data assets: contested topics, World Bank ground truths (free API, graceful errors),
hidden-profile task generation, forecasting items, IPIP questionnaire loading."""
import csv, json
from pathlib import Path
import numpy as np
import requests
from .utils import ROOT

DATA = ROOT / "data"

# --- Contested topics (starter set; notebook 01 extends with Chuang-et-al. + ANES statements) ---
DEFAULT_TOPICS = [
    {"id": "T_guncontrol", "statement": "Stricter national gun-control laws would make society safer overall."},
    {"id": "T_immigration", "statement": "Current levels of immigration benefit the country more than they cost it."},
    {"id": "T_neutral_filler", "statement": "Pineapple belongs on pizza.", "role": "filler"},
]

def load_topics():
    p = DATA / "topics" / "topics.json"
    if p.exists():
        return json.loads(p.read_text())
    return DEFAULT_TOPICS

# --- World Bank ground truths for T1 estimation items ---
WB_API = "https://api.worldbank.org/v2/country/{c}/indicator/{i}?format=json&date={y}"
DEFAULT_WB_ITEMS = [
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
]

def fetch_wb_truth(item: dict, timeout=25, retries=3):
    url = WB_API.format(c=item["country"], i=item["indicator"], y=item["year"])
    last = None
    for a in range(retries):
        try:
            js = requests.get(url, timeout=timeout).json()
            val = js[1][0]["value"]
            if val is None:
                raise ValueError("null value (indicator/year may be unavailable)")
            return float(val)
        except Exception as e:
            last = e
            import time as _t; _t.sleep(2 * (a + 1))
    try:
        raise last
    except Exception as e:
        raise RuntimeError(
            f"World Bank fetch failed for {item['id']} ({url}): {e}\n"
            f"Manual fix: open https://data.worldbank.org, look up the value, and add it to "
            f"data/ci/wb_truths.json as {{'{item['id']}': <value>}}.") from e

def load_ci_estimation(require_truths=True):
    items_p = DATA / "ci" / "wb_items.json"
    items = json.loads(items_p.read_text()) if items_p.exists() else DEFAULT_WB_ITEMS
    truths_p = DATA / "ci" / "wb_truths.json"
    truths = json.loads(truths_p.read_text()) if truths_p.exists() else {}
    for it in items:
        if truths.get(it["id"]) is None:
            truths[it["id"]] = fetch_wb_truth(it) if require_truths else None
    truths_p.parent.mkdir(parents=True, exist_ok=True)
    truths_p.write_text(json.dumps({k: v for k, v in truths.items() if v is not None}, indent=1))
    return items, truths

# --- Hidden-profile tasks (synthetic Stasser-Titus structure) ---
def make_hidden_profile(rng: np.random.Generator, n_agents: int, task_id="hp_0",
                       clues_per_agent=2):
    """Stasser-Titus hidden profile, scaled to large societies.

    Everyone shares the same eight facts, which point clearly to candidate B (six
    substantive pro-B facts plus two anti-A facts). Candidate A is objectively better,
    but the facts establishing that are UNSHARED: each agent holds only
    `clues_per_agent` of them. No individual can solve the task alone; the group
    collectively holds every fact, so only discussion that surfaces unshared
    information can recover the correct answer (candidate A)."""
    a_pros = [f"Candidate A {s}" for s in
              ["cut departmental costs by 12% in a previous role",
               "has led teams of more than 40 people",
               "holds the professional certification the role legally requires",
               "shipped two major products ahead of schedule",
               "speaks the main client's language fluently",
               "has a clean compliance record over eleven years"]]
    b_cons_unshared = [f"Candidate B {s}" for s in
                       ["was formally cited for a compliance violation",
                        "missed two product deadlines last year",
                        "has never managed anyone before"]]
    shared_pro_b = ["Candidate B interviewed with exceptional confidence",
                    "Candidate B holds a degree from a prestigious university",
                    "Candidate B is widely liked by the people who met them",
                    "Candidate B gave the strongest presentation of any applicant",
                    "Candidate B has eight years of experience in this industry",
                    "Candidate B asked thoughtful questions about the team's strategy"]
    shared_anti_a = ["Candidate A missed a quarterly target two years ago",
                     "Candidate A has changed jobs three times in six years"]
    shared = shared_pro_b + shared_anti_a
    unshared = a_pros + b_cons_unshared
    k = int(min(max(1, clues_per_agent), len(unshared) - 1))
    clues, private = {}, {}
    for i in range(n_agents):
        picks = [unshared[j] for j in rng.choice(len(unshared), size=k, replace=False)]
        private[i] = picks
        clues[i] = list(shared) + picks
    return {"id": task_id,
            "prompt": "Your team must choose the better job candidate: A) Candidate A  B) Candidate B.",
            "correct": "A", "clues": clues, "private": private}


def load_ipip(path=None, demo_ok=False):
    """Load the real IPIP-NEO-120 item file the user places at data/ipip/ipip_neo_120.csv
    (columns: item_text,trait,keyed[+/-]). Items are public domain: https://ipip.ori.org.
    A tiny DEMO battery (NOT IPIP items; written for pipeline testing only) is used when demo_ok=True."""
    p = Path(path or DATA / "ipip" / "ipip_neo_120.csv")
    if p.exists():
        with open(p) as f:
            items = [r for r in csv.DictReader(f)]
        assert {"item_text", "trait", "keyed"} <= set(items[0]), "CSV must have item_text,trait,keyed columns"
        return items, "ipip_neo_120"
    if not demo_ok:
        raise FileNotFoundError(
            f"{p} not found. Download the public-domain IPIP-NEO-120 items from https://ipip.ori.org "
            "(or J.A. Johnson's IPIP-NEO materials), save as CSV with columns item_text,trait,keyed. "
            "Demo battery is only allowed for smoke tests (demo_ok=True).")
    demo = []
    for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        demo.append({"item_text": f"[DEMO ITEM] I show behaviors typical of high {t}.", "trait": t, "keyed": "+"})
        demo.append({"item_text": f"[DEMO ITEM] I show behaviors typical of low {t}.", "trait": t, "keyed": "-"})
    return demo, "DEMO_battery"
