#!/usr/bin/env python3
"""
build_ipip_csv.py — Convert the public-domain IPIP-NEO-120 keys workbook into
TraitMix's questionnaire schema: item_text,trait,keyed[,item_no,facet]

SOURCE (public domain items; see ipip.ori.org — IPIP scales may be used by anyone,
for any purpose, without permission):
  Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model with a
  120-item public domain inventory: Development of the IPIP-NEO-120.
  Journal of Research in Personality, 51, 78-89.
  Workbook packaging: github.com/chfhhd/ipip-neo-120-paper-pencil (items public domain;
  packaging CC-BY-SA 4.0).

USAGE:
  python analysis/build_ipip_csv.py \
      --keys data/ipip/ipip-neo-120-paper-pencil/IPIP-NEO-120-Keys.xlsx \
      --out  data/ipip/ipip_neo_120.csv

The 'grouped' sheet lists items under facet headers like "N1: ANXIETY(Alpha = .78)";
the facet letter prefix (N/E/O/A/C) gives the Big Five domain. The 'sorted' sheet gives
items 1..120 in order with keying and verbatim text. We join on item number, so item
wording always comes from the source file — never retyped.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DOMAIN = {"N": "neuroticism", "E": "extraversion", "O": "openness",
          "A": "agreeableness", "C": "conscientiousness"}
FACET_HEADER = re.compile(r"^\s*([NEOAC])(\d)\s*:\s*([A-Za-z0-9\-\s/&.']+?)\s*(?:\(|$)")

# The source workbook uses typographic dashes for reverse-keyed items (e.g. "\u2013 keyed")
# and non-breaking spaces inside facet headers. Normalise both before parsing.
DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
          "\u2014": "-", "\u2015": "-", "\u2212": "-"}


def normalise(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    for bad, good in DASHES.items():
        text = text.replace(bad, good)
    return text


def parse_grouped(xl: pd.ExcelFile, sheet: str) -> dict[int, tuple[str, str]]:
    """item_no -> (trait, facet_label), read from facet headers in the grouped sheet."""
    df = xl.parse(sheet, header=None)
    mapping: dict[int, tuple[str, str]] = {}
    current: tuple[str, str] | None = None
    for _, row in df.iterrows():
        header_cell = normalise(row.get(1, "") or "")
        m = FACET_HEADER.match(header_cell)
        if m:
            letter, num, name = m.group(1), m.group(2), m.group(3).strip().title()
            current = (DOMAIN[letter], f"{letter}{num}: {name}")
            continue
        raw_no = row.get(0)
        if current and pd.notna(raw_no):
            try:
                item_no = int(float(raw_no))
            except (TypeError, ValueError):
                continue
            if 1 <= item_no <= 120:
                mapping[item_no] = current
    return mapping


def parse_sorted(xl: pd.ExcelFile, sheet: str) -> dict[int, tuple[str, str]]:
    """item_no -> (keyed, item_text) from the sorted sheet."""
    df = xl.parse(sheet, header=None)
    out: dict[int, tuple[str, str]] = {}
    for _, row in df.iterrows():
        raw_no, keyed_cell, text_cell = row.get(0), row.get(1), row.get(2)
        if pd.isna(raw_no) or pd.isna(text_cell):
            continue
        try:
            item_no = int(float(raw_no))
        except (TypeError, ValueError):
            continue
        if not 1 <= item_no <= 120:
            continue
        keyed_str = normalise(keyed_cell).strip().lower()
        keyed = "+" if keyed_str.startswith("+") else "-" if keyed_str.startswith("-") else None
        if keyed is None:
            continue
        text = " ".join(normalise(text_cell).split())
        out[item_no] = (keyed, text)
    return out


def pick_sheet(names: list[str], *needles: str) -> str:
    for n in names:
        low = n.lower()
        if all(x in low for x in needles):
            return n
    sys.exit(f"Could not find a sheet matching {needles} in {names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="path to IPIP-NEO-120-Keys.xlsx")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    xl = pd.ExcelFile(args.keys)
    grouped = parse_grouped(xl, pick_sheet(xl.sheet_names, "keys", "grouped"))
    items = parse_sorted(xl, pick_sheet(xl.sheet_names, "keys", "sorted"))

    rows = []
    missing_trait = []
    for item_no in sorted(items):
        keyed, text = items[item_no]
        if item_no not in grouped:
            missing_trait.append(item_no)
            continue
        trait, facet = grouped[item_no]
        rows.append({"item_text": text, "trait": trait, "keyed": keyed,
                     "item_no": item_no, "facet": facet})

    df = pd.DataFrame(rows)

    # ---- validation: fail loudly rather than silently shipping a broken instrument ----
    problems = []
    if len(df) != 120:
        problems.append(f"expected 120 items, parsed {len(df)}")
    if missing_trait:
        problems.append(f"items with no facet mapping: {missing_trait}")
    counts = df.trait.value_counts().to_dict()
    for trait in DOMAIN.values():
        if counts.get(trait, 0) != 24:
            problems.append(f"{trait}: expected 24 items, got {counts.get(trait, 0)}")
    if df.item_text.duplicated().any():
        dupes = df.loc[df.item_text.duplicated(keep=False), "item_no"].tolist()
        problems.append(f"duplicate item text at items {dupes}")
    if not set(df.keyed) <= {"+", "-"}:
        problems.append(f"unexpected keying values: {set(df.keyed)}")
    if (df.keyed == "-").sum() == 0:
        problems.append("no reverse-keyed items parsed - the workbook likely uses an "
                        "unrecognised dash character; inspect col 1 with repr() and extend DASHES")

    print(f"parsed {len(df)} items | per-trait counts: {counts}")
    print(f"keying: {df.keyed.value_counts().to_dict()} | facets: {df.facet.nunique()}")
    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nOK -> {out}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
