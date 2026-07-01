"""
scrape_contracts.py
====================
Pulls the full NBA contracts table from basketball-reference.com/contracts/players.html.

WHY THIS FILE CAN'T BE RUN IN THIS SANDBOX:
This was developed and tested inside an environment whose outbound network is
restricted to package registries (pypi, npm, github) - it cannot reach
basketball-reference.com directly. The dashboard in this repo was instead
built from a live snapshot fetched through a separate browsing tool and
hand-converted into data/raw/*.{md,psv}. Running THIS script on your own
machine (where basketball-reference is reachable) is how you refresh the
dataset to the full ~530-contract league-wide table instead of the ~100-player
subset shipped here.

Improvements vs. the 2022 version of this project:
  - Single page for the whole league (contracts/players.html) instead of
    looping over 30 team pages - faster, fewer requests, less brittle.
  - Parses the *raw HTML* (not pandas.read_html on rendered text) so it can
    recover player option (PO) / team option (TO) flags from the cell CSS
    classes basketball-reference uses, which a plain text/markdown scrape
    loses. Update the SELECTOR constants below if the site's class names
    change - check by inspecting the page in a browser dev console first.
  - Pulls "Guaranteed" totals per player directly instead of approximating.

Usage:
    pip install requests beautifulsoup4 pandas --break-system-packages
    python scrape_contracts.py --out ../data/raw/contracts_full.csv
"""
import argparse
import time
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.basketball-reference.com/contracts/players.html"
HEADERS = {
    # a real browser UA reduces (but doesn't eliminate) bot-detection blocks;
    # if you get 403s, add a delay between requests and consider a session
    # with cookies from a real browser visit.
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


def parse_option_flag(cell) -> str:
    """basketball-reference flags player/team options with a CSS class on the
    <td> (historically something like 'salary-pl' / 'salary-tm', or a title
    attribute) rather than inline text. INSPECT THE LIVE PAGE before relying
    on this - site markup changes periodically. This function tries a few
    likely signals and falls back to 'unknown' rather than guessing wrong."""
    classes = cell.get("class") or []
    title = (cell.get("title") or "").lower()
    text = cell.get_text(strip=True)
    if any("option" in c.lower() and "player" in c.lower() for c in classes) or "player option" in title:
        return "player_option"
    if any("option" in c.lower() and "team" in c.lower() for c in classes) or "team option" in title:
        return "team_option"
    if "(p" in text.lower() or text.lower().endswith("po"):
        return "player_option"
    if "(t" in text.lower() or text.lower().endswith("to"):
        return "team_option"
    return None


def scrape_contracts() -> pd.DataFrame:
    print(f"Fetching {BASE_URL} ...")
    html = fetch(BASE_URL)
    # basketball-reference often ships the main data table inside an HTML
    # comment to deter naive scraping - check for that and unwrap if needed
    if '<table id="player-contracts"' not in html and "<!--" in html:
        import re
        html = re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "player-contracts"}) or soup.find("table")
    if table is None:
        raise RuntimeError("Could not find the contracts table - basketball-reference's "
                            "markup may have changed. Inspect the page and update the "
                            "table id/selector above.")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue  # repeated header rows inside long tables
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = {}
        for cell in cells:
            stat = cell.get("data-stat", cell.name)
            row[stat] = cell.get_text(strip=True)
            if stat not in ("player", "team_id", "ranker"):
                opt = parse_option_flag(cell)
                if opt:
                    row[f"{stat}_option_type"] = opt
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} contract rows.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/raw/contracts_full.csv")
    args = ap.parse_args()
    df = scrape_contracts()
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")
