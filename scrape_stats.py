"""
scrape_stats.py
================
Pulls full-league per-game and advanced stats tables for a given season from
basketball-reference.com. Same network caveat as scrape_contracts.py - run
this locally, not inside a restricted sandbox.

Improvements vs. the 2022 version:
  - Targets the current season by default (was hardcoded to 2021-22).
  - Pulls BOTH per-game and advanced stats in one pass and merges them,
    instead of treating "stats" as a single undifferentiated blob.
  - Properly drops the duplicate "TOT" rows basketball-reference inserts for
    players traded mid-season while preserving the season-total line.
  - Designed to also pull *multiple historical seasons* in one run (see
    --seasons), which the value model needs to train on more than ~90 rows -
    a known limitation of the v1 snapshot shipped in this repo.

Usage:
    pip install requests beautifulsoup4 pandas --break-system-packages
    python scrape_stats.py --seasons 2024 2025 2026 --out-dir ../data/raw
"""
import argparse
import time
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}


def fetch(url: str, retries: int = 3, delay: float = 2.0) -> str:
    for attempt in range(retries):
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.text
        print(f"  ...status {resp.status_code}, retrying ({attempt+1}/{retries})", file=sys.stderr)
        time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def scrape_table(url: str, table_id: str) -> pd.DataFrame:
    html = fetch(url)
    if f'id="{table_id}"' not in html and "<!--" in html:
        import re
        html = re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": table_id})
    if table is None:
        raise RuntimeError(f"Table id={table_id} not found at {url} - markup may have changed.")
    df = pd.read_html(str(table))[0]
    df = df[df["Rk"] != "Rk"]  # drop repeated header rows
    return df


def scrape_season(year: int) -> pd.DataFrame:
    pg_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    adv_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"

    print(f"Season {year-1}-{str(year)[2:]}: fetching per-game stats...")
    per_game = scrape_table(pg_url, "per_game_stats")
    time.sleep(1.5)
    print(f"Season {year-1}-{str(year)[2:]}: fetching advanced stats...")
    advanced = scrape_table(adv_url, "advanced")

    # Drop duplicate rows for traded players, keeping the multi-team total
    for df in (per_game, advanced):
        if "Team" in df.columns:
            df["is_total_row"] = df["Team"].isin(["TOT", "2TM", "3TM", "4TM"])
            df.sort_values(["Player", "is_total_row"], ascending=[True, False], inplace=True)
            df.drop_duplicates(subset="Player", keep="first", inplace=True)
            df.drop(columns="is_total_row", inplace=True)

    merged = per_game.merge(
        advanced[["Player", "PER", "TS%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP"]],
        on="Player", how="left", suffixes=("", "_adv")
    )
    merged["season"] = f"{year-1}-{str(year)[2:]}"
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2026],
                     help="End year of each season, e.g. 2026 for 2025-26")
    ap.add_argument("--out-dir", default="../data/raw")
    args = ap.parse_args()

    all_seasons = []
    for yr in args.seasons:
        df = scrape_season(yr)
        all_seasons.append(df)
        time.sleep(2)

    combined = pd.concat(all_seasons, ignore_index=True)
    out_path = f"{args.out_dir}/stats_full.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} player-season rows to {out_path}")
