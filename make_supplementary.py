#!/usr/bin/env python3
"""
make_supplementary.py — Build the Additional file (supplementary material) as LaTeX.

EPJ Data Science permits Additional files, which are published alongside the accepted
article. This document carries the material that supports the manuscript but would crowd
it: the full prompt texts, the complete condition-by-metric results, the per-perturbation
audit values, the collective-intelligence item screening record, and the validity-principle
compliance table.

Everything here is generated from results/, so it cannot drift from the main text.

Usage:  python make_supplementary.py
Output: paper/supplementary/additional_file_1.tex  (compile standalone with pdfLaTeX)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, SUP = ROOT / "results", ROOT / "paper" / "supplementary"
sys.path.insert(0, str(ROOT / "src"))

PIMMUR = [
    ("Profile heterogeneity",
     "Trait vectors are sampled per condition from a truncated multivariate normal and "
     "validated through the induction gate (Table~S1), not asserted."),
    ("Interaction",
     "Agents exchange content through a recommender-driven feed with follow and unfollow "
     "actions; there is no broadcast shortcut."),
    ("Memory",
     "Each agent carries a bounded memory of its recent actions and of any private "
     "information it holds."),
    ("Minimal control",
     "Action prompts are open-ended and contain no instruction that presupposes an "
     "outcome. The full prompt text is reproduced in Section~S2."),
    ("Unawareness",
     "Agents are never told that they are in an experiment, and the persona prompt "
     "forbids self-reference as a language model."),
    ("Realism",
     "One condition is calibrated to published population norms for the same instrument; "
     "discussion topics derive from contested policy questions."),
]



def sec(num, title, body, sub=None):
    """Unnumbered heading that still appears in the table of contents with a page number."""
    head = f"S{num}\\quad {title}"
    out = [f"\\section*{{{head}}}",
           f"\\addcontentsline{{toc}}{{section}}{{S{num}\\quad {title}}}", body]
    return "\n".join(out)


def subsec(title, body):
    return "\n".join([f"\\subsection*{{{title}}}",
                       f"\\addcontentsline{{toc}}{{subsection}}{{{title}}}", body])


def landscape(body, n_cols=99, threshold=8):
    """Rotate only when the table is too wide for the portrait text block.

    A rotated page cannot share space with anything else, so rotating a table that would
    have fitted portrait costs most of a page for nothing.
    """
    if n_cols <= threshold:
        return body
    return "\\begin{landscape}\n" + body + "\n\\end{landscape}"


def figure(path, caption, label, width="\\linewidth"):
    return ("\\begin{figure}[H]\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption*{{\\small\\textbf{{{label}}}\\quad {caption}}}\n"
            "\\end{figure}")

def esc(x):
    """Escape text for LaTeX. Backslash first, or the other replacements corrupt it."""
    t = str(x)
    t = t.replace("\\", "\\textbackslash{}")
    for ch in "&%$#_{}":
        t = t.replace(ch, "\\" + ch)
    t = t.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    t = t.replace("<", "$<$").replace(">", "$>$")
    # a line beginning with "[" would be read as the optional argument of \\
    t = t.replace("[", "{[}")
    return t


SHORT = {"pol_var": "Opin.var", "pol_extremity": "Extrem.", "pol_assort": "Assort.",
         "pol_ei": "EchoClos", "crosscut_rate": "CrossCut", "CI_accuracy_z": "CollAcc",
         "CI_hidden_profile_rate": "HidProf", "CI_medrelerr_z": "MedErr",
         "CI_diversity": "Divers.", "CI_mean_gain_vs_pre": "CIgain",
         "trait_drift_mean": "Drift", "mean_diff": "diff", "cohens_dz": "$d_z$",
         "p_ttest": "$p$", "p_holm": "$p_{Holm}$", "n_pairs": "$n$",
         "ci_lo": "CI lo", "ci_hi": "CI hi", "median_rel_error": "med.rel.err",
         "log_spread": "log spread", "n_parsed": "parsed", "realized_sd_mean": "real.SD"}


def head_label(c):
    c = str(c).replace("__mean", "").replace("d_", "")
    return SHORT.get(c, c.replace("_", " "))


def tabular(df, cols=None, fmt="{:.3f}", maxrows=None, small="scriptsize"):
    cols = cols or list(df.columns)
    d = df[cols].head(maxrows) if maxrows else df[cols]
    out = [f"\\{small}", "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{longtable}{@{}l" + "r" * (len(cols) - 1) + "@{}}",
           "\\toprule",
           " & ".join(head_label(c) for c in cols) + " \\\\", "\\midrule", "\\endhead"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(v))
            elif isinstance(v, float):
                cells.append("--" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append(esc(v))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{longtable}", "\\normalsize"]
    return "\n".join(out)


def prompts_section():
    from traitmix import personality as pers
    theta = np.array([0.8, 0.5, 0.5, 0.2, 0.5])
    persona = pers.persona_system_prompt("Alex_0", 34, "nurse", theta)
    blocks = [
        ("Persona system prompt (example: Openness 0.8, Agreeableness 0.2)", persona),
        ("Feed action prompt",
         "Discussion topics on the platform right now:\n- <statement>\n"
         "Things you remember: <recent memory>\nYour feed:\n[p<id>] @<name>: <text>\n"
         "Choose ONE action and reply in EXACTLY one of these formats:\n"
         "POST <your post> / REPLY p<id> <your reply> / LIKE p<id> / FOLLOW <username> / "
         "UNFOLLOW <username> / PASS"),
        ("Private opinion probe",
         "[PROBE] Privately, not visible to anyone: on a scale from -3 (strongly disagree) "
         "to +3 (strongly agree), what is your CURRENT view on: \"<statement>\"? For "
         "context, your previous answer is <x>, and recent posts you saw lean toward a feed "
         "average <f>. This is a private research survey, not a social media post. Do not "
         "role-play, do not use asterisks or stage directions, do not explain. Reply with a "
         "single integer between -3 and 3."),
        ("Private estimation probe",
         "[ESTIMATE] Privately estimate: <question> This is a private research survey, not a "
         "social media post. Do not role-play, do not use asterisks or stage directions, do "
         "not explain. Reply with a single plain number in full digits, no units and no "
         "words."),
        ("Hidden-profile probe",
         "[CHOOSE] Your team must choose the better job candidate: A) Candidate A  "
         "B) Candidate B. Facts you personally know: <shared facts + this agent's private "
         "facts>. This is a private research survey, not a social media post. Do not "
         "role-play, do not use asterisks or stage directions, do not explain. Reply with "
         "exactly one character: A or B."),
        ("Questionnaire item (induction gate)",
         "[QUESTIONNAIRE] Statement: \"<IPIP-NEO-120 item>\". How accurately does this "
         "describe you? Answer with a single number: 1=very inaccurate, 2, 3=neutral, 4, "
         "5=very accurate."),
    ]
    out = []
    for title, body in blocks:
        out.append(f"\\subsection*{{{title}}}")
        out.append(f"\\addcontentsline{{toc}}{{subsection}}{{{title}}}")
        out.append("\\vspace{-2pt}\\begin{quote}\\footnotesize\\ttfamily\\sloppy"
                   "\\setlength{\\parskip}{0pt}\\noindent")
        out.append(esc(body).replace("\n", " \\\\\n"))
        out.append("\\end{quote}")
    return "\n".join(out)


def main():
    SUP.mkdir(parents=True, exist_ok=True)
    parts = []

    # ---- S1 induction validation ----
    q = RES / "e0_validation.csv"
    if q.exists():
        parts.append(sec(1, "Induction validation",
            "Spearman correlation between targeted and measured trait level. The "
            "IPIP-NEO-120 was administered to every persona configuration used in the main "
            "experiment before any experimental run; the pre-specified gate was a mean "
            "correlation of at least 0.60.\n" + tabular(pd.read_csv(q))))

    # ---- S2 realised trait distributions ----
    parts.append(sec(2, "Realised trait distributions",
        "Trait sampling is deterministic given a condition and a random seed, so the trait "
        "matrix actually used by every completed run can be reconstructed exactly without "
        "re-simulation. The left panel confirms that the intended trait was the one that "
        "moved in each condition. The right panel shows the narrowing of realised dispersion "
        "at extreme trait means caused by truncation of the sampling distribution, which the "
        "main text enters as a covariate and bounds at roughly one twelfth of the largest "
        "effect.\n"
        + figure("figures/figS1_realised_traits.pdf",
                 "Realised trait means (left) and realised trait dispersion (right) by "
                 "condition, averaged over seeds.", "Figure S1")))

    # ---- S3 the society under lowered traits ----
    parts.append(sec(3, "The society under lowered traits",
        "Figure~2 of the main article shows the society for the baseline and for each trait "
        "raised. The companion below shows the same conditions with each trait lowered. The "
        "contrast is the asymmetry reported in the main text: raising a trait produced ten "
        "corrected-significant contrasts, whereas lowering one produced none. Visually, the "
        "lowered conditions remain mixed rather than settling onto one side, and none "
        "reproduces either the convergence seen under raised Agreeableness or the stable "
        "split seen under raised Neuroticism.\n"
        + figure("figures/figS5_society_low.pdf",
                 "The simulated society over thirty rounds for the baseline and each "
                 "lowered trait, in the same format as Figure~2 of the main article.",
                 "Figure S5")))

    # ---- S4 the six contested topics, verbatim ----
    TOPICS = [
        ("T_guncontrol", "Stricter national gun-control laws would make society safer overall."),
        ("T_immigration", "Current levels of immigration benefit the country more than they cost it."),
        ("T_carbontax", "A national carbon tax should be introduced to reduce emissions, even if it raises household energy costs."),
        ("T_nuclear", "Nuclear power should be expanded as a main source of the country's electricity."),
        ("T_ubi", "The government should provide an unconditional basic income to every adult citizen."),
        ("T_socialmedia", "Social media platforms should be legally required to verify the age of their users."),
        ("T_neutral_filler", "Pineapple belongs on pizza."),
    ]
    rows = "\n".join(f"\\texttt{{{esc(i)}}} & {esc(t)} \\\\[3pt]" for i, t in TOPICS)
    parts.append(sec(4, "Discussion topics, verbatim",
        "Agents were asked to what extent they agreed with each statement, on a scale from "
        "$-3$ to $+3$. The first two were used throughout the study; the remaining four were "
        "added for the six-topic replication. The final entry is the neutral control, which "
        "is never included in any polarization measure and is used only to detect "
        "response-style artifacts.\n\n"
        "\\begin{longtable}{@{}p{3.1cm}p{10.6cm}@{}}\n\\toprule\n"
        "Identifier & Statement \\\\\n\\midrule\n\\endhead\n"
        + rows + "\n\\bottomrule\n\\end{longtable}"))

    # ---- S5 per-topic effects ----
    der = RES / "derived_results.csv"
    if der.exists():
        dd = pd.read_csv(der)
        six_ = dd[dd.config.str.startswith(("e1t_", "e2t_"))]
        if not six_.empty:
            base_ = six_[six_.config == "e1t_norm_baseline"]
            tcols = [c for c in six_.columns if c.startswith("T_")
                     and c.endswith("__var") and "filler" not in c]
            recs = []
            for cfg in sorted(six_.config.unique()):
                if cfg == "e1t_norm_baseline":
                    continue
                r = {"condition": cfg.replace("e1t_", "").replace("e2t_", "")}
                for c in tcols:
                    a = six_[six_.config == cfg][["seed", c]].rename(columns={c: "a"})
                    b = base_[["seed", c]].rename(columns={c: "b"})
                    m = a.merge(b, on="seed").dropna()
                    r[c.replace("T_", "").replace("__var", "")] = \
                        float((m.a - m.b).mean()) if len(m) else float("nan")
                recs.append(r)
            PT = pd.DataFrame(recs)
            parts.append(sec(5, "Effect on opinion variance, by topic",
                "The difference from the six-topic baseline, computed separately for each "
                "topic. The main article reports the average of these columns. Reading "
                "across a row shows how consistently a condition acts; reading down a column "
                "shows how responsive a topic is to composition.\n"
                + landscape(tabular(PT, list(PT.columns), small="scriptsize"), len(PT.columns))))

    # ---- S6 two-topic against six-topic ----
    tes = RES / "topic_extension_summary.csv"
    if tes.exists():
        T = pd.read_csv(tes)
        T = T[["condition", "metric", "two", "six", "same_sign"]]
        parts.append(sec(6, "Two-topic and six-topic estimates compared",
            "Every condition and measure for which both estimates exist. The correlation "
            "between the two columns is $+0.90$ for opinion variance, $+0.93$ for "
            "cross-cutting interaction, $+0.87$ for extremity and $+0.67$ for echo-chamber "
            "closure. Sign disagreement is concentrated where the two-topic effect was close "
            "to zero: agreement is 82\\% for contrasts exceeding $0.10$ in absolute value "
            "and 61\\% below it.\n"
            + landscape(tabular(T.sort_values(["metric", "condition"]),
                                list(T.columns), small="scriptsize"), len(T.columns))))

    # ---- S8 models excluded by induction validation ----
    parts.append(sec(7, "Models excluded by induction validation",
        "Two of the six models were excluded before their results were interpreted. "
        "Qwen2.5-3B fails the pre-specified rank criterion on Agreeableness "
        "($\\rho = 0.48$ against a threshold of $0.60$). Llama-3.2-3B passes on rank but "
        "fails on magnitude: the full manipulation moves its measured scores by $0.35$ and "
        "$0.71$ points on a five-point instrument, against $2.22$ and $3.06$ in the primary "
        "model, and the range of its condition means in opinion variance is $0.09$ against "
        "$0.45$ to $2.06$ in every admitted model. Their response surfaces are shown here so "
        "that the exclusions can be judged rather than taken on trust. We note in particular "
        "that excluding Llama-3.2-3B removes a model whose flat surface would otherwise "
        "appear to contradict the interaction reported in the main article.\n"
        + figure("figures/figE_sixmodel.pdf",
                 "Response surfaces for all six models. The two excluded models are the "
                 "faded panels; the vertical range of the first spans $0.2$ units against "
                 "$0.9$ for the primary model.", "Figure S6")))

    # ---- S9 per-model induction detail ----
    ind = RES / "induction_summary_all.csv"
    if ind.exists():
        I = pd.read_csv(ind)
        cols = [c for c in ["model", "size", "rho_O", "delta_O", "rho_A", "delta_A",
                            "gate", "span", "surface"] if c in I.columns]
        parts.append(sec(8, "Induction validation, by model",
            "The IPIP-NEO-120 administered to each of the nine response-surface "
            "configurations, separately for every model. $\\rho$ is the rank correlation "
            "between targeted and measured trait level; $\\Delta$ is the difference in mean "
            "measured score between the highest and lowest targeted level, on the "
            "instrument's 1--5 scale. Both criteria are applied to each manipulated trait. "
            "The per-configuration measurements are released with the code.\n"
            + tabular(I[cols], cols)))

    # ---- S10 the demoted cross-model figures ----
    parts.append(sec(9, "Further cross-model material",
        "Two figures were moved here from the main article. The first shows the "
        "Openness by Agreeableness surfaces for opinion variance and cross-cutting "
        "interaction in the primary and first replication models; it is superseded for "
        "opinion variance by the six-model figure in the main text, and the cross-cutting "
        "interaction reaches significance at cell level in only one model. The second shows "
        "sign agreement for trait-level main effects, which the main text reports as close "
        "to chance and does not claim as replication.\n"
        + figure("figures/fig1_crossover.pdf",
                 "Openness by Agreeableness surfaces for two measures in two models.",
                 "Figure S7")
        + "\n" + figure("figures/fig3_crossmodel.pdf",
                 "Sign agreement of trait-level effects between the primary and replication "
                 "models.", "Figure S8")))

    # ---- ablation comparison ----
    dd = pd.read_csv(RES / "derived_results.csv")
    dd = dd[~dd.config.str.startswith("e3mistral7_")]
    dd = dd[dd.seed < 900]
    POL = ["pol_var", "pol_extremity", "crosscut_rate", "pol_ei"]
    CONDS = ["e1_agreeableness_high", "e1_agreeableness_low", "e1_neuroticism_high",
             "e1_openness_high", "e2_homog", "e2_mid", "e2_diverse",
             "e3_O2_A2", "e3_O8_A8"]
    def _eff(pre, base, cfg, metric):
        b = dd[dd.config == base]
        x = dd[dd.config == pre + cfg][["seed", metric]].merge(
            b[["seed", metric]], on="seed", suffixes=("", "_b")).dropna()
        return float((x[metric] - x[f"{metric}_b"]).mean()) if len(x) else float("nan")
    rows = []
    for metric in POL:
        for c in CONDS:
            rows.append({"metric": metric, "condition": c,
                         "published": _eff("", "e1_norm_baseline", c, metric),
                         "no anchors": _eff("abpr_", "abpr_e1_norm_baseline", c, metric),
                         "no proximity": _eff("abwi_", "abwi_e1_norm_baseline", c, metric)})
    AB = pd.DataFrame(rows).dropna()
    parts.append(sec(10, "Circularity ablations, condition by condition",
        "Each condition effect computed against the baseline of its own variant, so that the "
        "switch under test is the only difference within a contrast. Removing the "
        "opinion-proximity term from the recommender leaves the ordering almost unchanged "
        "(rank agreement $\\rho = 0.975$, 33 of 36 signs preserved). Removing the probe "
        "anchors preserves the ordering less completely ($\\rho = 0.729$, 29 of 36) and "
        "raises extremity, which is consistent with the anchors having compressed opinions "
        "toward the stated feed average; the effects the article leads on are unaffected.\n"
        + landscape(tabular(AB, list(AB.columns), small="scriptsize"), len(AB.columns))))

    # ---- denominators ----
    q = RES / "denominators.csv"
    if q.exists():
        DN = pd.read_csv(q)
        keep = [c for c in ["config", "crosscut_rate", "reply_total", "reply_cross",
                            "pairs_opposing", "pairs_possible", "pct_pairs_opposing",
                            "frac_nonzero"] if c in DN.columns]
        parts.append(sec(11, "Counts behind the segregation rates",
            "Cross-cutting interaction and the E--I index are ratios, and a rate near zero "
            "means something different when few opposing pairs remain. These counts are "
            "recovered from the post and opinion records of runs at seeds outside the "
            "experimental design. The denominator does not collapse in the condition where "
            "the rate is lowest: under raised Agreeableness agents exchanged more replies "
            "than at baseline and over four thousand ordered pairs held opposing opinions.\n"
            + tabular(DN[keep], keep)))

    # ---- ledger and run accounting ----
    q = RES / "results_ledger.csv"
    if q.exists():
        LD = pd.read_csv(q)
        parts.append(sec(12, "Results ledger",
            "Every quantity quoted in the article, with the file, column and aggregation "
            "that produced it. Collective-intelligence composites are standardised within "
            "model and measurement regime, so a value does not depend on which other "
            "experiments are present in the released data.\n"
            + landscape(tabular(LD, list(LD.columns), small="tiny"), len(LD.columns))))
    q = RES / "run_accounting.csv"
    if q.exists():
        RA = pd.read_csv(q)
        parts.append(sec(13, "Run accounting",
            "Condition prefixes mapped to experiment, model and topic set. The run counts "
            "sum to the total reported in the article.\n" + tabular(RA, list(RA.columns))))

    # ---- S3 prompts ----
    parts.append(sec(14, "Complete prompt texts",
        "Every prompt used in the simulation is reproduced below. Angle brackets denote "
        "values substituted at run time.\n" + prompts_section()))

    # ---- S4 item screening ----
    q = RES / "ci_item_screen.csv"
    if q.exists():
        parts.append(sec(15, "Collective-intelligence item screening",
            "Candidate estimation items were admitted only where the model's solo median "
            "relative error was at least 0.15, so that a crowd had room to be right or "
            "wrong, and where the standard deviation of $\\log_{10}$ answers was at least "
            "0.02, so that agents genuinely disagreed and the diversity term of the "
            "decomposition was not degenerate. Six of fourteen candidates passed both. The "
            "rejected items are as informative as the retained ones: several were answered "
            "almost identically by every agent, leaving no aggregation for a crowd to "
            "perform.\n"
            + figure("figures/figS2_item_screen.pdf",
                     "Every candidate item by solo error and answer dispersion, with the "
                     "two pre-specified thresholds. Both axes are logarithmic.",
                     "Figure S2")
            + "\n" + tabular(pd.read_csv(q))))

    # ---- S5 complete results ----
    q = RES / "summary.csv"
    if q.exists():
        sm = pd.read_csv(q)
        keep = ["config"] + [c for c in sm.columns if c.endswith("__mean")][:8]
        parts.append(sec(16, "Complete results by condition",
            "Means over seeds for every condition and primary metric. The complete "
            "run-level file, including all metrics and all seeds, is released with the "
            "code.\n" + landscape(tabular(sm, keep, small="tiny"), len(keep))))

    # ---- S6 all contrasts, with the undefined-statistic explanation ----
    q = RES / "stats.csv"
    if q.exists():
        st = pd.read_csv(q)
        n_undef = int(st.p_ttest.isna().sum())
        cols = [c for c in ["family", "config", "metric", "n_pairs", "mean_diff", "ci_lo",
                            "ci_hi", "cohens_dz", "p_ttest", "p_holm"] if c in st.columns]
        note = ("Every contrast tested, significant or not, with effect sizes and bootstrap "
                "intervals. Contrasts are corrected within experiment family and metric.\n\n"
                "\\textbf{On the empty cells.} In " + str(n_undef) + " rows the effect size "
                "and $p$ values are empty. These are not failed computations. In each case "
                "the paired difference between the condition and its baseline was "
                "\\emph{exactly zero in every seed}, so the paired $t$ statistic is "
                "$0/0$ and undefined, and no effect size can be formed. Every such row is "
                "the hidden-profile measure in the replication-model families, where "
                "accuracy sat at or extremely near zero in all runs (Section~S9). We report "
                "the difference as $0.000$ and leave the test statistics empty rather than "
                "substituting an imputed value, since the quantity is undefined rather than "
                "unknown.\n")
        parts.append(sec(17, "All contrasts against baseline",
                         note + landscape(tabular(st.sort_values(["family", "metric", "p_holm"]),
                                                  cols, small="tiny"), len(cols))))

    # ---- S7 per-perturbation audit ----
    q = RES / "audit_sign_stability.csv"
    if q.exists():
        a = pd.read_csv(q)
        dcols = [c for c in a.columns if c.startswith("d_")]
        parts.append(sec(18, "Robustness audit, per perturbation",
            "Difference between the anchor condition and the baseline under each "
            "perturbation, each anchor being compared with the baseline subjected to the "
            "same perturbation. The main text reports the reference value, the range and "
            "the sign-stability fraction.\n"
            + landscape(tabular(a, ["anchor", "metric"] + dcols, small="tiny"), 2 + len(dcols))))

    # ---- S8 full surfaces ----
    parts.append(sec(19, "Full response surfaces",
        "The main text plots two measures across the Openness by Agreeableness grid. All "
        "four polarization measures are shown here, in both model families. The "
        "moderation is visible in every measure: the effect of Openness reverses or "
        "steepens according to the level of Agreeableness.\n"
        + figure("figures/figS3_full_surfaces.pdf",
                 "Response surfaces for all four polarization measures, in both model "
                 "families. Cell values are means over seeds.", "Figure S3")))

    # ---- S9 hidden-profile floor ----
    parts.append(sec(20, "The hidden-profile floor in the replication model",
        "The hidden-profile task discriminates weakly in the primary model and not at all "
        "in the replication model, where societies almost never recovered the unshared "
        "information. This is why several statistics in Section~S6 are undefined, and why "
        "the alignment result in the main text rests on the accuracy measure rather than "
        "on hidden-profile solving.\n"
        + figure("figures/figS4_hidden_profile_floor.pdf",
                 "Hidden-profile accuracy in every run, by model family. The replication "
                 "model sits at an absolute floor.", "Figure S4", width="0.86\\linewidth")))

    # ---- S10 validity principles ----
    rows = "\n".join(f"{esc(k)} & {v} \\\\[2pt]" for k, v in PIMMUR)
    parts.append(sec(21, "Validity-principle compliance",
        "\\begin{longtable}{@{}p{3.2cm}p{10.5cm}@{}}\n\\toprule\n"
        "Principle & Implementation \\\\\n\\midrule\n\\endhead\n"
        + rows + "\n\\bottomrule\n\\end{longtable}"))

    doc = ("\\documentclass[11pt]{article}\n"
           "\\usepackage[margin=1.9cm]{geometry}\n"
           "\\usepackage{booktabs,longtable,amsmath,amssymb,graphicx}\n"
           "\\usepackage{pdflscape,float}\n"
           "\\usepackage[colorlinks=true,allcolors=blue]{hyperref}\n"
           "\\setlength{\\parskip}{2pt}\\setlength{\\parindent}{0pt}\n"
           "\\setlength{\\LTpre}{3pt}\\setlength{\\LTpost}{6pt}\n"
           "\\usepackage{titlesec}\n"
           "\\titlespacing*{\\section}{0pt}{8pt}{3pt}\n"
           "\\titlespacing*{\\subsection}{0pt}{6pt}{2pt}\n"
           "\\setlength{\\textfloatsep}{6pt}\\setlength{\\intextsep}{6pt}\n"
           "\\title{Additional file 1: Supplementary material\\\\\n"
           "\\large Diverse Minds, Divided Networks? Personality Composition, Polarization,\n"
           "and Collective Intelligence in LLM-Based Social Simulations}\n"
           "\\author{}\\date{}\n\\begin{document}\\maketitle\n"
           "\\noindent This file contains supporting material referenced from the main "
           "article. All tables and figures are generated directly from the released "
           "results, available at \\url{https://doi.org/10.5281/zenodo.21792633} and "
           "\\url{https://github.com/raadbintareaf/traitmix}.\n\n"
           "\\vspace{6pt}\\hrule\\vspace{6pt}\n"
           "\\renewcommand{\\contentsname}{Contents}\n"
           "\\tableofcontents\n\\clearpage\n\n"
           + "\n\n\\medskip\n\n".join(parts) + "\n\n\\end{document}\n")
    out = SUP / "additional_file_1.tex"
    out.write_text(doc)
    print("wrote", out, f"({len(parts)} sections)")


if __name__ == "__main__":
    main()
