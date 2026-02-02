#!/usr/bin/env python3
"""
Compute Eton Fives rankings from TournamentResults.txt.

This is a faithful port of the provided doRanking.R logic:
- parses tournament sections (headers starting with '20..')
- extracts round outcomes (NQ/L16/QF/SF/F/W)
- builds a complete tournament x player grid
- applies a 7-year window with decay
- computes RPA/POSS/PC and ranks
- outputs:
  - rankings_latest.csv (matches the structure produced by the original R script)
  - rankings_latest.json (for the website)

Usage:
  python scripts/calc_rankings.py --results data/TournamentResults.txt --outdir docs/data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import re
import pandas as pd

def ensure_initial_surname(df: pd.DataFrame) -> pd.DataFrame:
    # If Initial/Surname are present in the index (e.g. after groupby/pivot), pull them into columns.
    if getattr(df.index, "names", None):
        idx_names = [n for n in df.index.names if n]
        if "Initial" in idx_names or "Surname" in idx_names:
            df = df.reset_index()

    # Normalise common column-name variants.
    rename_map: dict[str, str] = {}
    cols = list(df.columns)
    if "Initial" not in cols:
        for c in cols:
            lc = str(c).lower()
            if lc == "initial" or lc.endswith("initial") or lc.startswith("initial"):
                rename_map[c] = "Initial"
                break
    if "Surname" not in cols:
        for c in cols:
            lc = str(c).lower()
            if lc == "surname" or lc.endswith("surname") or lc.startswith("surname"):
                rename_map[c] = "Surname"
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Initial" in df.columns and "Surname" in df.columns:
        return df

    # Find a name column to derive from (but **do not** use internal numeric IDs like 'p').
    # Prefer columns with alphabetic content.
    candidates = [c for c in ["Name", "Player", "Pair", "player", "name"] if c in df.columns]
    chosen = None
    for c in candidates:
        series = df[c].astype(str)
        # choose if any value contains a letter (e.g. 'R.Houlden')
        if series.str.contains(r"[A-Za-z]", regex=True, na=False).any():
            chosen = c
            break

    if chosen is None:
        raise KeyError(
            "Missing player identifiers. Expected columns 'Initial'/'Surname' or a name column like 'Name'/'Player'. "
            f"Available columns: {list(df.columns)}"
        )

    def split_name(x: str):
        s = str(x).strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            return ("", "")
        # Names are usually like 'R.Houlden' or 'R Houlden' or 'R. Houlden'
        s2 = s.replace(".", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        parts = s2.split(" ")
        surname = parts[-1].upper() if parts else ""
        first = re.sub(r"[^A-Za-z]", "", parts[0]) if parts else ""
        initial = first[:1].upper() if first else ""
        return (initial, surname)

    vals = df[chosen].apply(split_name)
    df = df.copy()
    df["Initial"] = vals.apply(lambda t: t[0])
    df["Surname"] = vals.apply(lambda t: t[1])
    return df





import datetime


ROUND_HEADER_RE = re.compile(
    r"^(non\s*-?\s*qualifiers?|last\s*16|quarter\s*-?\s*finals?|semi\s*-?\s*finals?|final)$",
    re.I
)

def parse_non_qualifiers_clean(z: List[str]) -> List[str]:
    nq_parts: List[str] = []
    for i, raw in enumerate(z):
        s = _norm_dash(raw).strip()
        if re.match(r"^non\s*-?\s*qualifiers?", s, flags=re.I):
            inline = re.sub(r"^non\s*-?\s*qualifiers?\s*:?", "", s, flags=re.I).strip()
            if inline:
                nq_parts.append(inline)
            j = i + 1
            while j < len(z):
                sj = _norm_dash(z[j]).strip()
                if not sj:
                    j += 1
                    continue
                if ROUND_HEADER_RE.match(sj):
                    break
                nq_parts.append(sj)
                j += 1
            break
    if not nq_parts:
        return []
    blob = ", ".join([p for p in nq_parts if p])
    people: List[str] = []
    for part in [p.strip() for p in blob.split(",") if p.strip()]:
        for name in [n.strip() for n in part.split(" & ") if n.strip()]:
            people.append(name)
    return people

def parse_non_qualifiers_pairs_clean(z: List[str]) -> List[Tuple[str, Optional[str], str]]:
    """Return NQ teams as (player1, player2, team_display). player2 may be None."""
    nq_parts: List[str] = []
    for i, raw in enumerate(z):
        s = _norm_dash(raw).strip()
        if re.match(r"^non\s*-?\s*qualifiers?", s, flags=re.I):
            inline = re.sub(r"^non\s*-?\s*qualifiers?\s*:?", "", s, flags=re.I).strip()
            if inline:
                nq_parts.append(inline)
            j = i + 1
            while j < len(z):
                sj = _norm_dash(z[j]).strip()
                if not sj:
                    j += 1
                    continue
                if ROUND_HEADER_RE.match(sj):
                    break
                nq_parts.append(sj)
                j += 1
            break
    if not nq_parts:
        return []

    blob = ", ".join([p for p in nq_parts if p])
    teams: List[Tuple[str, Optional[str], str]] = []
    for part in [p.strip() for p in blob.split(",") if p.strip()]:
        # Prefer '&' pairs, but tolerate single names
        if " & " in part:
            a, b = [x.strip() for x in part.split(" & ", 1)]
            teams.append((a, b, f"{a} & {b}"))
        else:
            teams.append((part, None, part))
    return teams

def _name_to_key(name: str) -> str:
    s = str(name).strip().upper()
    s = s.replace("DE ", "DE").replace("VAN ", "VAN").replace("SOUZA GIRAO", "SOUZAGIRAO")
    s = s.replace(" ", ".")
    if "." in s:
        initial = re.sub(r"[^A-Z]", "", s.split(".")[0])[:1]
        surname = re.sub(r"[^A-Z\-\']", "", s.split(".")[-1])
    else:
        parts = re.sub(r"\s+", " ", s).split(" ")
        initial = re.sub(r"[^A-Z]", "", parts[0])[:1] if parts else ""
        surname = re.sub(r"[^A-Z\-\']", "", parts[-1]) if parts else ""
    return f"{initial}|{surname}"

# Match line parsing
# We accept a variety of name formats:
#   R.Houlden & J.Ho beat N.Caplin & H.Young 3-0 (12-4, 12-7, 12-0)
#   T Dunbar & S Cooley beat J Ho & R Houlden 3-0 (12-1, 12-4, 14-12)
#   ... and tolerate extra spaces / unicode dashes.
BEAT_SPLIT_RE = re.compile(r"\s+(?:bt|beat)\s+", re.I)
SCORE_START_RE = re.compile(r"\s+(?:\d+\s*-\s*\d+|\()")


def parse_match_line(line: str) -> dict | None:
    s = _norm_dash(line).strip()
    s = re.sub(r"\s+", " ", s)

    # Must contain a beat token and a pair separator
    if "&" not in s or not re.search(r"\b(?:bt|beat)\b", s, flags=re.I):
        return None

    parts = BEAT_SPLIT_RE.split(s, maxsplit=1)
    if len(parts) != 2:
        return None

    wteam = parts[0].strip()
    right = parts[1].strip()

    # Split loser team from score.
    # We support:
    #  - normal results like "3-0 (12-1, 12-1, 12-4)"
    #  - lines that only have set scores like "(12-6, 12-0, 12-0)"
    #  - walkovers like "w/o" at the end
    score = ""
    lteam = right

    # Walkover / no-play markers at end (must be treated as score, not a name)
    mwo = re.search(r"\b(w/o|wo|walkover)\b\.?$", right, flags=re.I)
    if mwo:
        lteam = right[: mwo.start()].strip()
        score = mwo.group(1).lower().replace("walkover", "w/o")
    else:
        # Split at the first "3-0" / "2-3" etc.
        ms = SCORE_START_RE.search(right)
        if ms:
            lteam = right[: ms.start()].strip()
            score = right[ms.start() :].strip()
        else:
            # If no game-score token, split at first "(" to capture set scores
            paren = right.find("(")
            if paren != -1:
                lteam = right[:paren].strip()
                score = right[paren:].strip()

    # Normalise spacing around ampersand in display strings
    wteam = re.sub(r"\s*&\s*", " & ", wteam)
    lteam = re.sub(r"\s*&\s*", " & ", lteam)

    # Basic validation: both teams should look like pairs
    if " & " not in wteam or " & " not in lteam:
        return None

    winners = [_name_to_key(x.strip()) for x in wteam.split("&")]
    losers = [_name_to_key(x.strip()) for x in lteam.split("&")]

    return {
        "winnerTeam": wteam.strip(),
        "loserTeam": lteam.strip(),
        "score": score,
        "winners": winners,
        "losers": losers,
        "raw": s,
    }

def parse_match_details(tournaments: Dict[str, List[str]]) -> dict:
    out = {"schema": 1, "generatedAt": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z", "tournaments": {}}
    for tname, block in tournaments.items():
        if not block:
            continue
        if block[0].strip() == "Not held":
            continue
        try:
            year_str, comp = tname.split(" ", 1)
            year = int(year_str)
        except Exception:
            continue
        comp_letter = comp.strip()[0].upper()
        short = f"{year % 100:02d} {comp_letter}"
        z = clean_block(block)
        nq_teams = []
        for a, b, team in parse_non_qualifiers_pairs_clean(z):
            players = [_name_to_key(a)]
            if b:
                players.append(_name_to_key(b))
            nq_teams.append({"team": team, "players": players})
        tinfo = {"tournament": tname, "year": year, "comp": comp, "key": short,
                 "nq": sorted({p for t in nq_teams for p in t.get("players", [])}),
                 "nqTeams": nq_teams,
                 "matches": []}
        for line in segment_between(z, "Last 16", r"Quarter[- ]Final", 1):
            pm = parse_match_line(line)
            if pm:
                pm["round"]="L16"; pm["roundName"]="Last 16"; tinfo["matches"].append(pm)
        for line in segment_between(z, r"Quarter[- ]Finals", r"Semi-Finals", 1):
            pm = parse_match_line(line)
            if pm:
                pm["round"]="QF"; pm["roundName"]="Quarter-Finals"; tinfo["matches"].append(pm)
        for line in segment_between(z, r"Semi-Finals", r"Final$", 1):
            pm = parse_match_line(line)
            if pm:
                pm["round"]="SF"; pm["roundName"]="Semi-Finals"; tinfo["matches"].append(pm)
        final_line=None
        for i, s in enumerate(z):
            if re.search(r"Final$", _norm_dash(s), flags=re.I):
                if i+1 < len(z):
                    final_line=z[i+1]
                break
        if final_line:
            pm=parse_match_line(final_line)
            if pm:
                pm["round"]="F"; pm["roundName"]="Final"; tinfo["matches"].append(pm)
        out["tournaments"][short]=tinfo
    return out
LOSER_RE = re.compile(r".* (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*", flags=re.I)
WINNER_RE = re.compile(r"([A-z.\' -]* & [A-z.\' -]*) (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*", flags=re.I)

# Points tables (as per doRanking.R)
K1 = {"W": 15, "F": 13.5, "SF": 9.6, "QF": 5.6, "L16": 2, "NQ": 0, "DNS": 0, "NA": 0}
K2 = {"W": 15, "F": 15, "SF": 12, "QF": 8, "L16": 4, "NQ": 2, "DNS": 1, "NA": 0}

L1 = {"W": 10, "F": 8, "SF": 5.2, "QF": 3, "NQ": 0, "DNS": 0, "NA": 0}
L2 = {"W": 10, "F": 10, "SF": 8, "QF": 6, "NQ": 2, "DNS": 1, "NA": 0}

N1 = {"W": 10, "F": 8, "SF": 5.2, "QF": 3, "NQ": 0, "DNS": 0, "NA": 0}
N2 = {"W": 10, "F": 10, "SF": 8, "QF": 6, "NQ": 2, "DNS": 1, "NA": 0}

MAPS_RPA = {"Kinnaird": K1, "London": L1, "Northern": N1}
MAPS_POSS = {"Kinnaird": K2, "London": L2, "Northern": N2}

DECAY = [1, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1] + [0] * 993


def read_tournaments(results_file: Path) -> Dict[str, List[str]]:
    lines = results_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"^20..", l)]
    tournaments: Dict[str, List[str]] = {}
    for idx, start in enumerate(starts):
        end = (starts[idx + 1] - 1) if idx + 1 < len(starts) else (len(lines) - 1)
        tournaments[lines[start].strip()] = lines[start + 1 : end + 1]
    return tournaments


def clean_block(block: List[str]) -> List[str]:
    z = [s.strip() for s in block if s.strip() != ""]
    z = [s.replace(":", "") for s in z]
    # Normalize unicode dashes/hyphens to '-' (helps with issue-submitted text)
    z = [s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
           .replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-") for s in z]
    # Insert a space after comma only when comma is between two word characters (matches doRanking.R)
    z = [re.sub(r"(\w),(\w)", r"\1, \2", s) for s in z]
    z = [re.sub(r"\([1-8]\)", "", s) for s in z]
    return [s.strip() for s in z]


def _norm_dash(s: str) -> str:
    return (s or '').replace('\u2010','-').replace('\u2011','-').replace('\u2012','-') \
        .replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')


def segment_between(z: List[str], start_pat: str, end_pat: str | None, start_offset: int = 1) -> List[str]:
    s = None
    for i, line in enumerate(z):
        if re.search(start_pat, _norm_dash(line), flags=re.I):
            s = i + start_offset
            break
    if s is None:
        return []
    e = len(z)
    if end_pat:
        for j in range(s, len(z)):
            if re.search(end_pat, _norm_dash(z[j]), flags=re.I):
                e = j
                break
    return z[s:e]


def apply_loser_gsub(line: str) -> str:
    # If pattern matches, returns losing pair; otherwise returns original line (as in R's gsub).
    return LOSER_RE.sub(r"\2", line)


def apply_winner_gsub(line: str) -> str:
    return WINNER_RE.sub(r"\1", line)


def parse_results(tournaments: Dict[str, List[str]]) -> pd.DataFrame:
    rows: List[Tuple[str, str, str]] = []

    for tname, block in tournaments.items():
        if not block:
            continue
        if block[0].strip() == "Not held":
            continue

        z = clean_block(block)

                # Non-qualifiers
        for name in parse_non_qualifiers_clean(z):
            rows.append((tname, "NQ", name.strip()))

# Last 16
        for line in segment_between(z, "Last 16", r"Quarter[- ]Final", 1):
            txt = apply_loser_gsub(line)
            for name in txt.split(" & "):
                rows.append((tname, "L16", name.strip()))

        # Quarter-Finals
        for line in segment_between(z, r"Quarter[- ]Finals", r"Semi-Finals", 1):
            txt = apply_loser_gsub(line)
            for name in txt.split(" & "):
                rows.append((tname, "QF", name.strip()))

        # Semi-Finals
        for line in segment_between(z, r"Semi-Finals", r"Final$", 1):
            txt = apply_loser_gsub(line)
            for name in txt.split(" & "):
                rows.append((tname, "SF", name.strip()))

        # Final line (line immediately after "Final")
        final_line = None
        for i, s in enumerate(z):
            if re.search(r"Final$", s, flags=re.I):
                if i + 1 < len(z):
                    final_line = z[i + 1]
                break

        if final_line:
            # Finalists (losers)
            txt = apply_loser_gsub(final_line)
            for name in txt.split(" & "):
                rows.append((tname, "F", name.strip()))
            # Winners
            txtw = apply_winner_gsub(final_line)
            for name in txtw.split(" & "):
                rows.append((tname, "W", name.strip()))

    df = pd.DataFrame(rows, columns=["Tournament", "Round", "Name"])
    df["Name"] = df["Name"].astype(str).str.strip()

    # Mirror cleanup steps from doRanking.R
    df.loc[df["Name"].str.contains("injury", case=False, na=False), "Name"] = df.loc[
        df["Name"].str.contains("injury", case=False, na=False), "Name"
    ].str.replace(r"(.*) (quali|reach).*", r"\1", regex=True, flags=re.I)

    df["Name"] = df["Name"].str.replace(r"\(.*", "", regex=True).str.strip()
    df["Name"] = df["Name"].str.replace("’", "'")
    df = df[df["Name"].str.strip() != ""]

    return df


def add_name_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Name2"] = (
        out["Name"]
        .str.upper()
        .str.replace("DE ", "DE", regex=False)
        .str.replace("SOUZA GIRAO", "SOUZAGIRAO", regex=False)
        .str.replace("VAN ", "VAN", regex=False)
        .str.replace(" ", ".", regex=False)
    )
    out["Initial"] = np.where(out["Name2"].str.contains(r"\.", regex=True), out["Name2"].str.replace(r"([^.]*)\..*", r"\1", regex=True), "")
    out["Surname"] = np.where(out["Name2"].str.contains(r"\.", regex=True), out["Name2"].str.replace(r"^.*\.([^.]*)$", r"\1", regex=True), out["Name2"])
    out[["Year", "Comp"]] = out["Tournament"].str.split(" ", n=1, expand=True)
    out["Year"] = out["Year"].astype(int)
    return out


def adjust_leading_dns_to_na(m: pd.DataFrame) -> pd.DataFrame:
    # Equivalent to:
    # dtres2[order(Year, Comp),
    #   Round := ifelse(1:.N < which.min(Round %in% c('DNS','NA') | (LastHeld-Year)>7),
    #                  ifelse(Round == 'DNS','NA',Round), Round),
    #   by = .(Initial, Surname)]
    def _adj(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["Year", "Comp"])
        valid = (~g["Round"].isin(["DNS", "NA"])) & ((g["LastHeld"] - g["Year"]) <= 7)
        if valid.any():
            pos = int(np.where(valid.values)[0][0])
            idxs = g.index[:pos]
            g.loc[idxs, "Round"] = g.loc[idxs, "Round"].replace({"DNS": "NA"})
        return g

    return m.groupby(["Initial", "Surname"], group_keys=False).apply(_adj)


def compute_rankings(m: pd.DataFrame) -> pd.DataFrame:
    # Some upstream transforms can leave player identifiers in the index or with
    # variant column names. Normalise so we always have Initial/Surname columns.
    df = ensure_initial_surname(m).copy()
    age = (df["LastHeld"] - df["Year"]).astype(int)
    df["decay"] = age.apply(lambda a: DECAY[a] if 0 <= a < len(DECAY) else 0)

    df["rpa"] = df.apply(lambda r: MAPS_RPA[str(r["Comp"])].get(r["Round"], 0), axis=1)
    df["poss"] = df.apply(lambda r: MAPS_POSS[str(r["Comp"])].get(r["Round"], 0), axis=1)

    df["rpa_w"] = df["decay"] * df["rpa"]
    df["poss_w"] = df["decay"] * df["poss"]

    # Order for "missed last" etc is descending Year, then descending Comp (factor order reversed)
    df_sorted = df.sort_values(["Initial", "Surname", "Year", "Comp"], ascending=[True, True, False, False])

    rows = []
    for (ini, sur), g in df_sorted.groupby(["Initial", "Surname"], sort=False):
        rpa_sum = float(g["rpa_w"].sum())
        poss_sum = float(g["poss_w"].sum())

        played_mask = ((g["LastHeld"] - g["Year"]) < 7) & (~g["Round"].isin(["DNS", "NA"]))
        comps_played = int(played_mask.sum())

        rounds = list(g["Round"])
        consec_missed = 0
        if rounds and rounds[0] == "DNS":
            for r in rounds:
                if r == "DNS":
                    consec_missed += 1
                else:
                    break

        missed_last6 = sum(1 for r in rounds[:6] if r == "DNS")
        missed_last9 = sum(1 for r in rounds[:9] if r == "DNS")

        rows.append((ini, sur, rpa_sum, poss_sum, comps_played, consec_missed, missed_last6, missed_last9))

    out = pd.DataFrame(
        rows,
        columns=[
            "Initial",
            "Surname",
            "RPA",
            "POSS",
            "CompsPlayed",
            "ConsecutiveCompsMissed",
            "CompsMissed_Last6",
            "CompsMissed_Last9",
        ],
    )

    out["PC"] = (out["RPA"] / out["POSS"]) * 100
    penal = (out["ConsecutiveCompsMissed"] >= 3) | (out["CompsPlayed"] < 3)
    out["PC2"] = out["PC"] / np.where(penal, 1000, 1)

    out["RANK"] = out["PC"].rank(ascending=False, method="min")
    out["RANK2"] = out["PC2"].rank(ascending=False, method="min")

    # frank(.SD, -PC2, -POSS, ties.method='first') equivalent:
    out_sorted = out.sort_values(["PC2", "POSS"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    out_sorted["RANK3"] = np.arange(1, len(out_sorted) + 1)
    out = out.merge(out_sorted[["Initial", "Surname", "RANK3"]], on=["Initial", "Surname"], how="left")

    return out


def build_output_table(dt: pd.DataFrame) -> pd.DataFrame:
    dt1 = dt[dt["Year"] > 2012].copy()
    dt1["Comp"] = pd.Categorical(dt1["Comp"], categories=["Northern", "Kinnaird", "London"], ordered=True)
    dt1["LastHeld"] = dt1.groupby("Comp")["Year"].transform("max")

    players = dt1[["Initial", "Surname"]].drop_duplicates().reset_index(drop=True)
    tourns = dt1[["Tournament", "Year", "Comp", "LastHeld"]].drop_duplicates()

    # Cartesian product Tournament x Player (like CJ2)
    all_idx = pd.MultiIndex.from_product([tourns["Tournament"].unique(), range(len(players))], names=["Tournament", "p"])
    all_df = pd.DataFrame(index=all_idx).reset_index()
    all_df = all_df.merge(tourns[["Tournament", "Year", "Comp", "LastHeld"]], on="Tournament", how="left")
    all_df = all_df.merge(players.reset_index().rename(columns={"index": "p"}), on="p", how="left")

    m = all_df.merge(dt1[["Tournament", "Initial", "Surname", "Round"]], on=["Tournament", "Initial", "Surname"], how="left")
    # Ensure Initial/Surname are present (never derive from numeric 'p')
    players_idx = players.reset_index().rename(columns={"index": "p"})
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left")
    else:
        # In case merges created suffixed columns, prefer the player lookup values
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left", suffixes=("", "_pl"))
        if "Initial_pl" in m.columns:
            m["Initial"] = m["Initial"].fillna(m["Initial_pl"])
            m["Surname"] = m["Surname"].fillna(m["Surname_pl"])
            m = m.drop(columns=["Initial_pl", "Surname_pl"])
    # If still missing (unexpected), try deriving from a true name column
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = ensure_initial_surname(m)
    m["Round"] = m["Round"].fillna("DNS")

    m["Comp"] = pd.Categorical(m["Comp"], categories=["Northern", "Kinnaird", "London"], ordered=True)
    m["LastHeld"] = m.groupby("Comp")["Year"].transform("max")

    # Outside the 7-year window: DNS becomes NA
    m.loc[((m["LastHeld"] - m["Year"]) > 7) & (m["Round"] == "DNS"), "Round"] = "NA"

    # Leading DNS before first appearance become NA
    m = adjust_leading_dns_to_na(m)

    # Ensure player identifiers exist before ranking computation
    # Ensure Initial/Surname are present (never derive from numeric 'p')
    players_idx = players.reset_index().rename(columns={"index": "p"})
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left")
    else:
        # In case merges created suffixed columns, prefer the player lookup values
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left", suffixes=("", "_pl"))
        if "Initial_pl" in m.columns:
            m["Initial"] = m["Initial"].fillna(m["Initial_pl"])
            m["Surname"] = m["Surname"].fillna(m["Surname_pl"])
            m = m.drop(columns=["Initial_pl", "Surname_pl"])
    # If still missing (unexpected), try deriving from a true name column
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = ensure_initial_surname(m)
    # Fallback: if Initial/Surname were lost, re-attach from player index mapping
    if 'Initial' not in m.columns or 'Surname' not in m.columns:
        _players_map = players.reset_index().rename(columns={'index': 'p'})[['p','Initial','Surname']]
        m = m.merge(_players_map, on='p', how='left')

    rankings = compute_rankings(m).rename(
        columns={"CompsPlayed": "Played", "ConsecutiveCompsMissed": "MissedLast"}
    )
    rankings = rankings[
        ["Initial", "Surname", "RPA", "POSS", "Played", "MissedLast", "PC", "PC2", "RANK", "RANK2", "RANK3"]
    ]

    # YC columns: ordered by Year then Comp
    yc_levels = m[["Year", "Comp"]].drop_duplicates().sort_values(["Year", "Comp"])
    level_order = (yc_levels["Year"].astype(str) + " " + yc_levels["Comp"].astype(str)).tolist()
    short_labels = [s[2:6] for s in level_order]  # substr(,3,6)

    m["YC"] = pd.Categorical(m["Year"].astype(str) + " " + m["Comp"].astype(str), categories=level_order, ordered=True)
    m["RoundDisp"] = m["Round"].replace({"NA": "", "DNS": "P"})

    # Ensure Initial/Surname are present (never derive from numeric 'p')
    players_idx = players.reset_index().rename(columns={"index": "p"})
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left")
    else:
        # In case merges created suffixed columns, prefer the player lookup values
        m = m.merge(players_idx[["p", "Initial", "Surname"]], on="p", how="left", suffixes=("", "_pl"))
        if "Initial_pl" in m.columns:
            m["Initial"] = m["Initial"].fillna(m["Initial_pl"])
            m["Surname"] = m["Surname"].fillna(m["Surname_pl"])
            m = m.drop(columns=["Initial_pl", "Surname_pl"])
    # If still missing (unexpected), try deriving from a true name column
    if "Initial" not in m.columns or "Surname" not in m.columns:
        m = ensure_initial_surname(m)


    pivot = m.pivot_table(index=["Initial", "Surname"], columns="YC", values="RoundDisp", aggfunc="first", fill_value="")
    pivot.columns = [short_labels[level_order.index(str(c))] for c in pivot.columns]
    pivot = pivot.reset_index()

    out = pivot.merge(rankings, on=["Initial", "Surname"], how="left")
    out = out.sort_values("RANK3")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--date", type=str, default=None, help="Optional YYYYMMDD for filenames; defaults to today")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    tournaments = read_tournaments(args.results)
    parsed = parse_results(tournaments)
    parsed = add_name_fields(parsed)

    out = build_output_table(parsed)

    # Save CSV (latest)
    csv_path = args.outdir / "rankings_latest.csv"
    out.to_csv(csv_path, index=False)

    # Save JSON (records) for the website. Add a Rank column at the front (RANK3).
    records_df = out.copy()
    records_df.insert(0, "Rank", records_df["RANK3"].astype(int))
    # Keep floats to 3dp-ish like Excel display (still numeric)
    json_path = args.outdir / "rankings_latest.json"
    json_path.write_text(records_df.to_json(orient="records"), encoding="utf-8")

    # Save match details for click-through UI
    matches_path = args.outdir / "matches_latest.json"
    matches = parse_match_details(tournaments)
    matches_path.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")


    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {matches_path}")


if __name__ == "__main__":
    main()
