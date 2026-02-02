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
    """Convert a player name into a stable player key used across rankings + match data.

    Rules:
      - Dot or space separated initial+surname should map to the same key:
          "R.Houlden" == "R Houlden" == "R. Houlden" -> "R|HOULDEN"
      - If the first token is a full (unambiguous) first name, preserve it:
          "SAHIL SHAH" -> "SAHIL|SHAH"
          "SAAJAN SHAH" -> "SAAJAN|SHAH"
      - Common surname particles are normalised (DE, VAN) and a known compound surname (SOUZA GIRAO).
    """
    s = str(name).strip().upper()
    s = s.replace("SOUZA GIRAO", "SOUZAGIRAO")
    # Normalise common particles so they stay attached to surname tokenisation.
    s = re.sub(r"\bDE\s+", "DE", s)
    s = re.sub(r"\bVAN\s+", "VAN", s)
    # Normalise separators: convert dots to spaces, collapse whitespace
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "|"

    parts = s.split(" ")
    first = parts[0]
    last = parts[-1]

    # If first token is more than 1 letter, treat it as full first name token (unambiguous)
    first_alpha = re.sub(r"[^A-Z]", "", first)
    if first_alpha and len(first_alpha) > 1:
        surname = re.sub(r"[^A-Z\-\']", "", last)
        return f"{first_alpha}|{surname}"

    # Otherwise treat as initial
    initial = re.sub(r"[^A-Z]", "", first)[:1]
    surname = re.sub(r"[^A-Z\-\']", "", last)
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
    """Vectorised equivalent of the original doRanking.R leading-DNS -> NA adjustment.

    For each player, looking in chronological order (Year, Comp), any initial runs of DNS
    before the player's first "valid" appearance become NA.
    """
    if m.empty:
        return m

    gcols = ["Initial", "Surname"]
    out = m.copy()

    # Ensure deterministic chronological order
    out = out.sort_values(gcols + ["Year", "Comp"], ascending=[True, True, True, True]).reset_index(drop=True)

    valid = (~out["Round"].isin(["DNS", "NA"])) & ((out["LastHeld"] - out["Year"]) <= 7)
    pos = out.groupby(gcols, sort=False).cumcount()

    # First valid position per player (NaN if none)
    first_valid_pos = pos.where(valid).groupby([out[c] for c in gcols], sort=False).transform("min")

    mask = (out["Round"] == "DNS") & first_valid_pos.notna() & (pos < first_valid_pos)
    out.loc[mask, "Round"] = "NA"

    return out


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




# ------------------------
# V2 Elo-style ratings (experimental)
# ------------------------
ELO_MU = 1500.0
ELO_K_BASE = 12.0
ELO_SIGMA_NEW = 350.0
ELO_SIGMA_EST = 120.0
ELO_SIGMA_MAX = 400.0

ROUND_WT = {"L16": 1.00, "QF": 1.05, "SF": 1.10, "F": 1.15}
COMP_ORDER = {"Northern": 0, "Kinnaird": 1, "London": 2}


def _elo_expected(r_you: float, r_opp: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_opp - r_you) / 400.0))


def _apply_inactivity(state: dict, current_year: int) -> None:
    last = state.get("last_year")
    if last is None:
        state["last_year"] = int(current_year)
        return
    gap = int(current_year) - int(last)
    if gap <= 0:
        state["last_year"] = int(current_year)
        return
    r = float(state["r"])
    s = float(state["sigma"])
    for _ in range(gap):
        # Very gentle mean reversion
        r = ELO_MU + (r - ELO_MU) * 0.98
        s = min(ELO_SIGMA_MAX, s + 50.0)
    state["r"] = r
    state["sigma"] = s
    state["last_year"] = int(current_year)


def _sigma_after_match(sigma: float) -> float:
    return max(ELO_SIGMA_EST, float(sigma) * 0.85)


def _k_eff(sigma: float) -> float:
    return ELO_K_BASE * (1.0 + float(sigma) / 300.0)


def _retirement_coeff(score: str) -> float:
    """Coefficient for retirements based on match state.

    Maps progress towards a best-of-5 (first to 3 sets) win to [0.25, 1.0].
    If parsing fails, return 0.50.
    """
    if not score or "ret" not in score.lower():
        return 1.0

    # Try game score first (e.g. 3-1)
    m = re.search(r"\b(\d+)\s*-\s*(\d+)\b", score)
    sets_won = None
    sets_lost = None
    if m:
        try:
            a = int(m.group(1)); b = int(m.group(2))
            if 0 <= a <= 5 and 0 <= b <= 5 and (a + b) <= 7:
                sets_won, sets_lost = a, b
        except Exception:
            sets_won = sets_lost = None

    if sets_won is None:
        # Fall back to counting set scores
        pairs = re.findall(r"(\d+)\s*-\s*(\d+)", score)
        if not pairs:
            return 0.50
        sw = 0
        sl = 0
        for a, b in pairs:
            try:
                ai = int(a); bi = int(b)
            except Exception:
                continue
            if ai > bi:
                sw += 1
            elif bi > ai:
                sl += 1
        sets_won, sets_lost = sw, sl

    target = 3
    if sets_won >= target:
        return 1.0
    progress = max(0.0, min(1.0, sets_won / float(target)))
    return float(max(0.25, min(1.0, 0.25 + 0.75 * progress)))


def compute_elo_v2(matches: dict) -> tuple[pd.DataFrame, dict]:
    """Compute Elo V2 ratings and per-tournament snapshots.

    Returns:
      elo_df: Initial, Surname, Elo, EloSigma
      snapshots: dict keyed by tournament key (e.g. '19 N') with rows incl EloRank
    """
    players: dict[str, dict] = {}

    def get_state(pkey: str, year: int):
        if pkey not in players:
            ini, sur = pkey.split("|", 1)
            players[pkey] = {"r": ELO_MU, "sigma": ELO_SIGMA_NEW, "last_year": year, "Initial": ini, "Surname": sur}
        st = players[pkey]
        _apply_inactivity(st, year)
        return st

    def update_pair(win_keys, lose_keys, score: str, round_code: str, year: int):
        coeff = 1.0
        sl = (score or "").lower()
        if re.search(r"\b(w/o|wo|walkover)\b", sl):
            coeff = 0.25
        elif "ret" in sl:
            coeff = _retirement_coeff(score)

        wt = ROUND_WT.get(round_code, 1.0)

        st_w = [get_state(k, year) for k in win_keys]
        st_l = [get_state(k, year) for k in lose_keys]

        r_w = st_w[0]["r"] + st_w[1]["r"]
        r_l = st_l[0]["r"] + st_l[1]["r"]
        sigma_pair = (st_w[0]["sigma"] + st_w[1]["sigma"] + st_l[0]["sigma"] + st_l[1]["sigma"]) / 4.0
        k_pair = _k_eff(sigma_pair) * wt * coeff

        e = _elo_expected(r_w, r_l)
        delta = k_pair * (1.0 - e)

        # Passenger counter split
        # Win: 60% to lower-rated partner
        rw1, rw2 = st_w[0]["r"], st_w[1]["r"]
        low_i, high_i = (0, 1) if rw1 <= rw2 else (1, 0)
        st_w[low_i]["r"] += delta * 0.60
        st_w[high_i]["r"] += delta * 0.40

        # Loss: 60% penalty to higher-rated partner
        rl1, rl2 = st_l[0]["r"], st_l[1]["r"]
        high_i_l, low_i_l = (0, 1) if rl1 >= rl2 else (1, 0)
        st_l[high_i_l]["r"] -= delta * 0.60
        st_l[low_i_l]["r"] -= delta * 0.40

        for st in st_w + st_l:
            st["sigma"] = _sigma_after_match(st["sigma"])
            st["last_year"] = year

    # chronological tournaments
    tourn_list = []
    for tkey, tinfo in matches.get("tournaments", {}).items():
        try:
            y = int(tinfo.get("year"))
        except Exception:
            continue
        comp = str(tinfo.get("comp", ""))
        tourn_list.append((y, COMP_ORDER.get(comp, 99), tkey, tinfo))
    tourn_list.sort()

    snapshots: dict[str, dict] = {}
    round_rank = {"L16": 0, "QF": 1, "SF": 2, "F": 3}

    for year, _, tkey, tinfo in tourn_list:
        ms = list(tinfo.get("matches", []))
        ms.sort(key=lambda m: round_rank.get(m.get("round"), 99))
        for m in ms:
            w = m.get("winners", [])
            l = m.get("losers", [])
            if len(w) == 2 and len(l) == 2:
                update_pair(w, l, m.get("score", ""), m.get("round", ""), year)

        rows = []
        for pkey, st in players.items():
            rows.append({
                "Initial": st.get("Initial", pkey.split("|", 1)[0]),
                "Surname": st.get("Surname", pkey.split("|", 1)[1]),
                "Elo": round(float(st["r"])),
                "EloSigma": float(st["sigma"]),
            })
        snap_df = pd.DataFrame(rows)
        if not snap_df.empty:
            snap_df = snap_df.sort_values(["Elo", "Surname", "Initial"], ascending=[False, True, True]).reset_index(drop=True)
            snap_df.insert(0, "EloRank", snap_df.index + 1)
        snapshots[tkey] = {
            "key": tkey,
            "label": str(tinfo.get("tournament", tkey)),
            "year": int(year),
            "comp": tinfo.get("comp"),
            "rows": snap_df.to_dict(orient="records"),
        }

    # Final df
    final_rows = []
    for pkey, st in players.items():
        final_rows.append({
            "Initial": st.get("Initial", pkey.split("|", 1)[0]),
            "Surname": st.get("Surname", pkey.split("|", 1)[1]),
            "Elo": round(float(st["r"])),
            "EloSigma": float(st["sigma"]),
        })
    elo_df = pd.DataFrame(final_rows)
    if not elo_df.empty:
        elo_df = elo_df.sort_values(["Elo", "Surname", "Initial"], ascending=[False, True, True]).reset_index(drop=True)
    return elo_df, snapshots



def build_official_snapshots(parsed: pd.DataFrame, start_key: str = "19 N") -> List[dict]:
    """Generate official ranking-table snapshots efficiently.

    Snapshots are generated after each tournament starting at `start_key` (e.g. '19 N').
    This mirrors the official table structure (Rank, Initial, Surname, tournament columns, metrics).

    NOTE: This function intentionally reuses a single pre-built Tournament x Player matrix
    to avoid re-building the Cartesian product for every snapshot.
    """

    # Build the master Tournament x Player matrix once (as per build_output_table)
    dt1 = parsed[parsed["Year"] > 2012].copy()
    dt1["Comp"] = pd.Categorical(dt1["Comp"], categories=["Northern", "Kinnaird", "London"], ordered=True)

    players = dt1[["Initial", "Surname"]].drop_duplicates().reset_index(drop=True)
    tourns = dt1[["Tournament", "Year", "Comp"]].drop_duplicates()

    all_idx = pd.MultiIndex.from_product([tourns["Tournament"].unique(), range(len(players))], names=["Tournament", "p"])
    all_df = pd.DataFrame(index=all_idx).reset_index()
    all_df = all_df.merge(tourns, on="Tournament", how="left")
    all_df = all_df.merge(players.reset_index().rename(columns={"index": "p"}), on="p", how="left")

    m_base = all_df.merge(
        dt1[["Tournament", "Initial", "Surname", "Round"]],
        on=["Tournament", "Initial", "Surname"],
        how="left",
    )
    m_base["Round"] = m_base["Round"].fillna("DNS")
    m_base["Comp"] = pd.Categorical(m_base["Comp"], categories=["Northern", "Kinnaird", "London"], ordered=True)

    # Order tournaments chronologically in the same way as build_output_table
    yc_levels = m_base[["Year", "Comp"]].drop_duplicates().sort_values(["Year", "Comp"])
    level_order = (yc_levels["Year"].astype(str) + " " + yc_levels["Comp"].astype(str)).tolist()
    short_labels = [s[2:6] for s in level_order]

    if start_key not in short_labels:
        return []
    start_idx = short_labels.index(start_key)

    snaps: List[dict] = []
    # Iterate over snapshots by expanding set of included tournaments
    for i in range(start_idx, len(short_labels)):
        allowed = set(short_labels[: i + 1])

        m = m_base.copy()
        m["YCShort"] = (m["Year"].astype(int) % 100).map(lambda x: f"{x:02d}") + " " + m["Comp"].astype(str).str[0].str.upper()
        m = m[m["YCShort"].isin(allowed)].copy()

        if m.empty:
            continue

        # Recompute LastHeld for this snapshot (critical for decay + NA cutoff)
        m["LastHeld"] = m.groupby("Comp")["Year"].transform("max")

        # Outside 7-year window: DNS becomes NA
        m.loc[((m["LastHeld"] - m["Year"]) > 7) & (m["Round"] == "DNS"), "Round"] = "NA"

        # Leading DNS before first appearance become NA (within this snapshot)
        m = adjust_leading_dns_to_na(m)

        rankings = compute_rankings(m).rename(columns={"CompsPlayed": "Played", "ConsecutiveCompsMissed": "MissedLast"})
        rankings = rankings[["Initial", "Surname", "RPA", "POSS", "Played", "MissedLast", "PC", "PC2", "RANK", "RANK2", "RANK3"]]

        # Pivot tournament results columns
        m["YC"] = pd.Categorical(m["Year"].astype(str) + " " + m["Comp"].astype(str), categories=level_order[: i + 1], ordered=True)
        m["RoundDisp"] = m["Round"].replace({"NA": "", "DNS": "P"})

        pivot = m.pivot_table(index=["Initial", "Surname"], columns="YC", values="RoundDisp", aggfunc="first", fill_value="")
        # Rename columns to short form (e.g. '19 N')
        # categories in YC are level_order[:i+1], so map directly
        pivot.columns = short_labels[: i + 1]
        pivot = pivot.reset_index()

        out = pivot.merge(rankings, on=["Initial", "Surname"], how="left").sort_values("RANK3")
        out2 = out.copy()
        out2.insert(0, "Rank", out2["RANK3"].astype(int))

        snaps.append({
            "key": short_labels[i],
            "label": level_order[i],
            "records": out2.to_dict(orient="records"),
        })

    return snaps

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

    # Match details for click-through UI (and Elo)
    matches_path = args.outdir / "matches_latest.json"
    matches = parse_match_details(tournaments)
    matches_path.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Experimental Elo V2 outputs (also merged into latest table for the website) ---
    elo_df = None
    elo_snaps = None
    try:
        elo_df, elo_snaps = compute_elo_v2(matches)
        if elo_df is not None and not elo_df.empty:
            elo_df = elo_df.copy()
            # Add EloRank based on current Elo ordering
            elo_df = elo_df.sort_values(["Elo", "Surname", "Initial"], ascending=[False, True, True]).reset_index(drop=True)
            elo_df.insert(0, "EloRank", elo_df.index + 1)
            # Merge Elo into the latest official table
            out = out.merge(elo_df[["Initial", "Surname", "Elo", "EloSigma", "EloRank"]], on=["Initial", "Surname"], how="left")

        # Write Elo files
        if elo_df is not None:
            elo_csv = args.outdir / "elo_latest.csv"
            elo_json = args.outdir / "elo_latest.json"
            elo_df.to_csv(elo_csv, index=False)
            elo_json.write_text(elo_df.to_json(orient="records"), encoding="utf-8")
    except Exception as e:
        print("WARNING: Elo V2 generation failed:", e)



    # Reorder columns for CSV to mirror website ordering (excluding the published Rank column)
    try:
        tour_cols = [c for c in out.columns if re.match(r"^\d{2} [NKL]$", str(c))]
        def _tour_sort_key(c):
            yy = int(str(c).split(' ')[0])
            comp = str(c).split(' ')[1]
            order = {'N':0,'K':1,'L':2}.get(comp, 9)
            return (yy, order)
        tour_cols = sorted(tour_cols, key=_tour_sort_key)
        meta_cols = [c for c in ["POSS","RPA","PC","PC2","Played","MissedLast"] if c in out.columns]
        elo_cols = [c for c in ["EloRank","Elo","EloSigma"] if c in out.columns]
        keep = elo_cols + ["Initial","Surname"] + meta_cols + tour_cols
        # Keep RANK3 as internal numeric rank in CSV (some scripts may rely on it)
        if "RANK3" in out.columns:
            keep = ["RANK3"] + keep
        out = out[keep]
    except Exception:
        pass
    # Save CSV (latest)
    csv_path = args.outdir / "rankings_latest.csv"
    out.to_csv(csv_path, index=False)

    # Save JSON (records) for the website.
    # We publish a clean column order so the UI can keep Rank/Elo sticky and the rest scrollable.
    records_df = out.copy()
    records_df.insert(0, "Rank", records_df["RANK3"].astype(int))

    # Choose tournament columns in chronological order
    tour_cols = [c for c in records_df.columns if re.match(r"^\d{2} [NKL]$", str(c))]
    def _tour_sort_key(c):
        yy = int(str(c).split(' ')[0])
        comp = str(c).split(' ')[1]
        order = {'N':0,'K':1,'L':2}.get(comp, 9)
        return (yy, order)
    tour_cols = sorted(tour_cols, key=_tour_sort_key)

    meta_cols = [c for c in ["POSS","RPA","PC","PC2","Played","MissedLast"] if c in records_df.columns]
    elo_cols = [c for c in ["EloRank","Elo","EloSigma"] if c in records_df.columns]

    # Final column order
    keep = ["Rank"] + elo_cols + ["Initial","Surname"] + meta_cols + tour_cols

    # Ensure we keep only those columns (drop internal RANK/RANK2/RANK3 etc)
    records_df = records_df[keep]

    json_path = args.outdir / "rankings_latest.json"
    json_path.write_text(records_df.to_json(orient="records"), encoding="utf-8")

    # Build official table snapshots from 2019 Northern onwards, and attach Elo fields if available
    try:
        official_snaps = build_official_snapshots(parsed, start_key="19 N")
        merged = []
        for snap in official_snaps:
            key = snap["key"]
            records = snap["records"]
            elo_rows = {}
            if elo_snaps is not None:
                elo_rows = {f"{r['Initial']}|{r['Surname']}": r for r in (elo_snaps.get(key, {}).get("rows", []) or [])}
            for r in records:
                pk = f"{r.get('Initial','')}|{r.get('Surname','')}"
                er = elo_rows.get(pk)
                if er:
                    r["Elo"] = er.get("Elo")
                    r["EloSigma"] = er.get("EloSigma")
                    r["EloRank"] = er.get("EloRank")
                else:
                    r["Elo"] = None
                    r["EloSigma"] = None
                    r["EloRank"] = None
            merged.append({"key": key, "label": snap["label"], "records": records})

        snaps_path = args.outdir / "rankings_snapshots.json"
        snaps_path.write_text(json.dumps({"schema": 1, "snapshots": merged}, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {snaps_path}")
    except Exception as e:
        print("WARNING: Snapshot generation failed:", e)


    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {matches_path}")


if __name__ == "__main__":
    main()
