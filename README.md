# NBA Contract Analyzer

An end-to-end pipeline + interactive dashboard for evaluating NBA contracts
against current performance, built with actual 2023 CBA mechanics (rookie
scale, max tiers, supermax, Bird rights, the apron system) instead of
treating "salary" as an undifferentiated number.

**Open `app/index.html` in a browser to use it.** Everything is static - no
server needed. It's built from a real, live snapshot of 2025-26 contracts and
stats pulled from basketball-reference.com.

## What changed from the 2022 version

The original project (`1-scrape.ipynb` → `4-prepare.ipynb`) scraped 5 years
of free-agent stats and salaries, normalized salary to the cap, and trained a
`RandomForestClassifier` to predict which $5M salary bucket a player belonged
in. "Surplus value" was the gap between predicted bucket and actual salary.

Two things were outdated and one thing was missing entirely:

- **Outdated data**: 2021-22 season, salary cap table stopping at ~$112M.
  The cap is $165M for 2026-27 - a >45% jump - and supermax deals now run
  $300M+ over five years. The old $5M-bucket classifier could not represent
  this even if re-run on fresh data.
- **Outdated model shape**: bucketed classification into coarse $5M bands
  is a poor fit for a market where rookie minimums sit around $1.2M and
  supermax AAVs sit around $70M - a 60x range collapsed into a handful of
  buckets loses almost all signal at the top of the distribution.
- **Missing entirely**: actual contract structure. The old pipeline never
  modeled years remaining, options, guarantees, rookie-scale status, or any
  CBA mechanic - it only ever compared "stats" to "this season's salary
  number." That's a performance-value model, not a contract analyzer.

This version adds a real CBA rules layer (`pipeline/cba_rules.py`) and
contract-structure fields (years remaining, guaranteed total, contract type
classification) alongside a regression-based value model, and surfaces all
of it in an interactive dashboard rather than static notebook output.

## Architecture

```
scrapers/            Production scrapers (run locally - see note below)
  scrape_contracts.py    Full-league contracts table, single page, with
                          option-flag parsing from cell CSS classes
  scrape_stats.py         Per-game + advanced stats, multi-season capable

pipeline/
  cba_rules.py          Salary cap/tax/apron history, max contract tiers,
                         supermax rules, rookie scale, Bird rights, MLE
                         variants, apron-status classifier
  parse_raw.py           Parses raw scraped data into clean DataFrames,
                          merges contracts + per-game + advanced stats
  build_dataset.py        Feature engineering, Ridge regression value
                          model, contract grading, team payroll/apron
                          aggregation, exports dashboard_data.json

data/
  raw/                  Raw scraped snapshots
  processed/             Cleaned CSVs + final dashboard_data.json

app/
  template.html          Dashboard source (data injected at build time)
  index.html              Final standalone dashboard (open this one)
```

### Why the scrapers can't be run from this environment

This was built inside a sandboxed environment whose network is restricted to
package registries (pypi/npm/github) - it cannot reach basketball-reference.com
directly. The dataset actually shipped in `data/raw/` and baked into
`app/index.html` is a **real, live snapshot** (101 players, 30 teams) pulled
through a separate browsing tool during development, not synthetic data -
but it's a partial slice of the league (the highest-minute, highest-salary
players), not the full ~530-contract / ~450-player league.

To get full coverage: run `scrapers/scrape_contracts.py` and
`scrapers/scrape_stats.py` on your own machine (where basketball-reference is
reachable), then re-run `pipeline/parse_raw.py` and `pipeline/build_dataset.py`
pointed at the new raw files, then regenerate `app/index.html` with the
injection step at the bottom of this README.

## The value model, honestly

`build_dataset.py` trains a Ridge regression predicting a player's
performance-implied "market value" as a percentage of the salary cap, from
this season's per-game and advanced stats. Rookie-scale contracts are
excluded from training (their pay is fixed by draft slot, not market
performance) but still get predictions, which is why they show up as
"locked-in value" rather than bargains or overpays.

This is trained on roughly 90 veteran contracts from a single season
snapshot. That's enough to be directionally useful and a legitimate
demonstration of the pipeline end-to-end, but it is **not** enough data to
treat the dollar outputs as precise valuations - real production versions of
this kind of model (what Spotrac, DunksAndThrees, and similar sites do)
train on several seasons of player-seasons, thousands of rows. The
multi-season capability is already built into `scrape_stats.py --seasons`;
running it across 3-4 past seasons and re-training is the highest-leverage
next step if this becomes a longer-running project.

Players with fewer than 40 games played this season are flagged with a
small-sample caveat in the dashboard - several stars (Tatum, Embiid, Giannis)
are having injury-shortened seasons, which skews their box-score-derived
"predicted value" downward relative to their true level.

## Known simplifications (documented honestly, not hidden)

- **Rookie-scale detection is heuristic** (age ≤ 22 and salary ≤ $16M), not
  a real draft-year lookup. The production scraper should pull draft year
  directly from each player's basketball-reference page.
- **Player/team option flags are not in the v1 dataset** - the markdown/text
  extraction used to build the snapshot loses basketball-reference's CSS-based
  option markers. `scrape_contracts.py` is written to parse these from raw
  HTML; the live snapshot in `app/` doesn't have them yet.
- **Team payroll totals are partial** - they sum only the ~100 players in
  this dataset, not full 15-man rosters, so apron status will read as more
  cap-friendly than reality for most teams. Flagged in the dashboard's Teams
  tab.
- **Cap holds are not modeled** - true team cap room during free agency
  depends on free-agent cap holds, not just signed contracts. Noted as a
  next step in the dashboard's glossary.
- **Bird rights, individual max eligibility, and supermax qualification are
  not computed per-player** - the CBA rules module has the formulas, but
  applying them correctly requires years-of-service and award-history data
  this snapshot doesn't carry yet.

## Regenerating the dashboard after a data refresh

```bash
cd pipeline
python3 parse_raw.py        # rebuilds data/processed/players_merged.csv
python3 build_dataset.py    # rebuilds data/processed/dashboard_data.json
cd ..
python3 -c "
import json
data = json.load(open('data/processed/dashboard_data.json'))
template = open('app/template.html').read()
open('app/index.html', 'w').write(template.replace('__DATA_JSON__', json.dumps(data)))
"
```
