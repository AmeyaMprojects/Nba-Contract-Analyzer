"""
parse_raw.py
============
Parses the raw data pulled from basketball-reference (contracts, per-game
stats, advanced stats) into clean pandas DataFrames and merges them into a
single player table. This stands in for what the production scraper
(scrapers/scrape_contracts.py, scrapers/scrape_stats.py) would output when
run locally - here we're parsing a real snapshot fetched directly from
basketball-reference.com/contracts/players.html and the 2025-26 stats pages.
"""
import re
import pandas as pd
import numpy as np
import unicodedata

RAW_DIR = "/home/claude/nba-contract-analyzer/data/raw"
OUT_DIR = "/home/claude/nba-contract-analyzer/data/processed"


def normalize_name(name: str) -> str:
    """Strip accents/punctuation so names match across sources that render
    them slightly differently (e.g. 'Nikola Jokić' vs 'Nikola Jokic')."""
    if not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = re.sub(r"[^a-zA-Z0-9 ]", "", ascii_name)
    return ascii_name.strip().lower()


def money_to_float(s):
    if pd.isna(s) or s is None:
        return np.nan
    s = str(s).replace("$", "").replace(",", "").strip()
    if s == "" or s == "—":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_contracts():
    rows = []
    with open(f"{RAW_DIR}/contracts_raw.md") as f:
        lines = [l.strip() for l in f if l.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    for line in lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        rows.append(row)
    df = pd.DataFrame(rows)
    year_cols = ["2025-26", "2026-27", "2027-28", "2028-29", "2029-30", "2030-31"]
    for c in year_cols:
        df[c] = df[c].apply(money_to_float)
    df["Guaranteed"] = df["Guaranteed"].apply(money_to_float)
    df = df.rename(columns={"Player": "player", "Tm": "team"})
    df["name_key"] = df["player"].apply(normalize_name)
    # contract length = number of non-null year columns from 2025-26 onward
    df["contract_years_remaining"] = df[year_cols].notna().sum(axis=1)
    df["salary_2025_26"] = df["2025-26"]
    df["salary_2026_27"] = df["2026-27"]
    df["total_remaining_value"] = df[year_cols].sum(axis=1, skipna=True)
    keep = ["player", "name_key", "team", "salary_2025_26", "salary_2026_27",
            "contract_years_remaining", "total_remaining_value", "Guaranteed"] + year_cols
    df = df.rename(columns={"Guaranteed": "guaranteed_total"})
    return df[["player", "name_key", "team", "salary_2025_26", "salary_2026_27",
               "contract_years_remaining", "total_remaining_value", "guaranteed_total"] + year_cols]


def parse_per_game():
    df = pd.read_csv(f"{RAW_DIR}/per_game_raw.psv", sep="|")
    df = df.rename(columns={"Player": "player", "Team": "team", "Pos": "pos",
                             "Age": "age", "G": "gp", "GS": "gs", "MP": "mpg",
                             "PTS": "ppg", "TRB": "rpg", "AST": "apg",
                             "STL": "spg", "BLK": "bpg", "TOV": "topg",
                             "FGpct": "fg_pct", "3Ppct": "fg3_pct", "FTpct": "ft_pct",
                             "eFGpct": "efg_pct"})
    df["name_key"] = df["player"].apply(normalize_name)
    # de-duplicate players traded mid-season: keep the TOT row only
    df["is_tot"] = df["team"] == "TOT"
    df = df.sort_values(["name_key", "is_tot"], ascending=[True, False])
    df = df.drop_duplicates(subset="name_key", keep="first")
    num_cols = ["age", "gp", "gs", "mpg", "ppg", "rpg", "apg", "spg", "bpg",
                "topg", "fg_pct", "fg3_pct", "ft_pct", "efg_pct"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["player", "name_key", "team", "pos", "age", "gp", "gs", "mpg",
               "ppg", "rpg", "apg", "spg", "bpg", "topg",
               "fg_pct", "fg3_pct", "ft_pct", "efg_pct"]]


def parse_advanced():
    df = pd.read_csv(f"{RAW_DIR}/advanced_raw.psv", sep="|")
    df = df.rename(columns={"Player": "player", "Team": "team", "TSpct": "ts_pct",
                             "USGpct": "usg_pct", "WS48": "ws_per_48"})
    df["name_key"] = df["player"].apply(normalize_name)
    df["is_tot"] = df["team"] == "TOT"
    df = df.sort_values(["name_key", "is_tot"], ascending=[True, False])
    df = df.drop_duplicates(subset="name_key", keep="first")
    num_cols = ["PER", "ts_pct", "usg_pct", "OWS", "DWS", "WS", "ws_per_48",
                "OBPM", "DBPM", "BPM", "VORP"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["name_key", "PER", "ts_pct", "usg_pct", "OWS", "DWS", "WS",
               "ws_per_48", "OBPM", "DBPM", "BPM", "VORP"]]


def build_merged_dataset():
    contracts = parse_contracts()
    per_game = parse_per_game()
    advanced = parse_advanced()

    # inner-join stats sources on name, then left-join contracts so we keep
    # every player we have *stats* for and attach contract info where it exists
    stats = per_game.merge(advanced, on="name_key", how="left")
    merged = stats.merge(
        contracts.drop(columns=["player", "team"]), on="name_key", how="left"
    )
    merged = merged.sort_values("ppg", ascending=False).reset_index(drop=True)
    merged.to_csv(f"{OUT_DIR}/players_merged.csv", index=False)
    return merged


if __name__ == "__main__":
    df = build_merged_dataset()
    print(f"Merged dataset: {len(df)} players, {df['salary_2025_26'].notna().sum()} with contract data")
    print(df.columns.tolist())
    print(df[["player", "team", "ppg", "salary_2025_26"]].head(15))
