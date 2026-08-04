"""Builds paper/supplementary/supplementary.tex: full prompts appendix, config manifest,
PIMMUR compliance table, complete results tables, homogenization metrics."""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, SUP = ROOT / "results", ROOT / "paper" / "supplementary"
sys.path.insert(0, str(ROOT / "src"))

PIMMUR = [
    ("Profile", "Trait vectors sampled per condition from TruncNormal$_5(\\mu,\\Sigma)$; validated via E0 gate (Table~1), not asserted."),
    ("Interaction", "Networked exchange via feeds with interest+hot recommender; follow/unfollow rewiring; no broadcast shortcuts."),
    ("Memory", "Per-agent bounded memory ($k{=}10$) plus running self-summary."),
    ("Minimal control", "Open-ended action prompts (App.~A); no outcome-leading instructions; full prompt text released."),
    ("Unawareness", "Agents never told they are in an experiment; system prompt forbids AI self-reference."),
    ("Realism", "Human-calibrated $(\\mu,\\Sigma)$ condition (cited BFI-2 norms); ANES-derived topics; classical-ABM reference curves."),
]

def tex_escape(s): return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")

def prompts_appendix():
    from traitmix import personality as pers
    import numpy as np
    theta = np.array([.8, .5, .5, .2, .5])
    example = pers.persona_system_prompt("Alex_0", 34, "nurse", theta)
    probe = ("[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) to +3 "
             "(strongly agree), what is your CURRENT view on: \"<statement>\"? ... Reply with a single integer.")
    action = ("Discussion topics ... Your feed: [p<id>] @<name>: <text> ... Choose ONE action and reply in "
              "EXACTLY one of these formats: POST <text> / REPLY p<id> <text> / LIKE p<id> / FOLLOW <name> / "
              "UNFOLLOW <name> / PASS")
    est = "[ESTIMATE] Privately estimate: <question> Reply with a single number only (no units, no words)."
    quest = ("[QUESTIONNAIRE] Statement: \"<IPIP item>\". How accurately does this describe you? "
             "1=very inaccurate ... 5=very accurate.")
    blocks = [("Persona system prompt (example, O=0.8, A=0.2)", example),
              ("Opinion probe", probe), ("Feed action prompt (schema)", action),
              ("Estimation probe", est), ("Questionnaire item", quest)]
    out = []
    for title, body in blocks:
        out += [f"\\subsection*{{{tex_escape(title)}}}", "\\begin{quote}\\small\\ttfamily",
                tex_escape(body).replace("\n", "\\\\ "), "\\end{quote}"]
    return "\n".join(out)

def full_table():
    p = RES / "summary.csv"
    if not p.exists(): return "% summary.csv not yet generated"
    s = pd.read_csv(p)
    cols = [c for c in s.columns if c.endswith("__mean")][:8]
    head = "Config & " + " & ".join(tex_escape(c[:-6]) for c in cols) + " \\\\"
    rows = [tex_escape(r.config) + " & " + " & ".join(
            ("--" if pd.isna(r[c]) else f"{r[c]:.3f}") for c in cols) + " \\\\" for _, r in s.iterrows()]
    return "\n".join(["\\begin{longtable}{l" + "c"*len(cols) + "}", "\\toprule", head, "\\midrule",
                      *rows, "\\bottomrule", "\\end{longtable}"])

def main():
    SUP.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "configs" / "MANIFEST.json").read_text()) \
        if (ROOT / "configs" / "MANIFEST.json").exists() else {}
    man_rows = [f"{k} & {len(v['configs'])} & {len(v['seeds'])} & {len(v['configs'])*len(v['seeds'])} \\\\"
                for k, v in manifest.items()]
    doc = f"""\\documentclass[11pt]{{article}}
\\usepackage{{booktabs,longtable,geometry,hyperref}}
\\geometry{{margin=1in}}
\\title{{Supplementary Material:\\\\Diverse Minds, Divided Networks?}}
\\begin{{document}}\\maketitle

\\section{{S1. Experiment manifest}}
\\begin{{tabular}}{{lccc}}\\toprule
Experiment & Configs & Seeds & Runs \\\\ \\midrule
{chr(10).join(man_rows)}
\\bottomrule\\end{{tabular}}

\\section{{S2. PIMMUR compliance}}
\\begin{{tabular}}{{p{{2.6cm}}p{{11cm}}}}\\toprule
Principle & Implementation \\\\ \\midrule
{chr(10).join(f"{p} & {d} \\\\" for p, d in PIMMUR)}
\\bottomrule\\end{{tabular}}

\\section{{S3. Full prompt texts}}
{prompts_appendix()}

\\section{{S4. Complete results (all configurations)}}
{full_table()}

\\section{{S5. Homogenization and manipulation checks}}
Distinct-2, self-BLEU-3, filler-topic variance and expressed-trait drift per configuration are
reported in \\texttt{{results/summary.csv}} (columns \\texttt{{distinct2, self\\_bleu3,
filler\\_variance, trait\\_drift\\_mean}}); the audit heat-map is Figure~\\texttt{{fig\\_audit\\_heatmap.pdf}}.

\\end{{document}}"""
    (SUP / "supplementary.tex").write_text(doc)
    print("wrote", SUP / "supplementary.tex")

if __name__ == "__main__":
    main()
