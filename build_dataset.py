"""
build_dataset.py
=================
Feature engineering + value model + contract grading. This replaces the old
project's RandomForestClassifier-into-$5M-buckets approach with:

  1. A regression model (not a coarse classifier) predicting a player's
     market value as a % of the salary cap, from current performance.
  2. Training data that EXCLUDES rookie-scale contracts - those prices are
     set by draft slot, not by the market, so including them would teach the
     model that great performance correlates with cheap pay. We still
     generate predictions for rookie-scale players (to show how big a bargain
     they are), we just don't train on their salaries.
  3. A CBA-aware "contract grade" that combines the predicted-vs-actual
     surplus with the player's contract structure (years remaining, total
     guaranteed) and CBA contract-type bucket from cba_rules.py.

MODEL NOTE: this is trained on a single-season snapshot of ~80-90 veteran
contracts pulled live from basketball-reference. That's enough to be
directionally useful and a legitimate portfolio demonstration of the
pipeline, but it is NOT enough data to treat the dollar predictions as
precise valuations - real production versions of this (Spotrac/DunksAndThrees
style models) train on several seasons of player-seasons (thousands of rows).
The scrapers/ folder is built to pull multiple historical seasons so this can
be extended that way; see README.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict

import cba_rules as cba

DATA_DIR = "/home/claude/nba-contract-analyzer/data/processed"
CAP_2025_26 = cba.CAP_HISTORY["2025-26"]["cap"]

# Smaller, more complete feature set than an initial draft that used 15
# features on ~60-90 rows (overfit risk, noisy extrapolation at the tails).
# Ridge + standardization generalizes far better than a tree ensemble here -
# linear models extrapolate more sensibly to the handful of supermax players
# that sit at the edge of the observed distribution.
FEATURES = ["ppg", "rpg", "apg", "spg", "bpg", "ts_pct", "usg_pct", "WS", "VORP", "age"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cap_pct"] = df["salary_2025_26"] / CAP_2025_26
    df["per36_pts"] = df["ppg"] / df["mpg"] * 36
    df["per36_reb"] = df["rpg"] / df["mpg"] * 36
    df["per36_ast"] = df["apg"] / df["mpg"] * 36
    # heuristic flag for "likely still on a rookie-scale contract": young AND
    # cheap relative to what their stats would otherwise command. Without a
    # real draft-year field this is an approximation - the production
    # scraper (scrapers/scrape_contracts.py) should pull actual draft year
    # from basketball-reference player pages to replace this.
    df["likely_rookie_scale"] = (df["age"] <= 22) & (df["salary_2025_26"].fillna(0) <= 16_000_000)
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def train_value_model(df: pd.DataFrame):
    train_df = df[
        df["salary_2025_26"].notna()
        & ~df["likely_rookie_scale"]
    ].copy()

    X = train_df[FEATURES].values
    y = train_df["cap_pct"].values

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=3.0),
    )

    n = len(train_df)
    n_splits = min(5, max(2, n // 12))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(model, X, y, cv=kf)
    train_df["predicted_cap_pct"] = np.clip(cv_preds, 0.01, 0.42)

    model.fit(X, y)

    full_mask = df[FEATURES].notna().sum(axis=1) >= len(FEATURES) - 2  # tolerate 1-2 missing
    df.loc[full_mask, "predicted_cap_pct_full"] = np.clip(
        model.predict(df.loc[full_mask, FEATURES].values), 0.01, 0.42
    )

    df = df.merge(
        train_df[["name_key", "predicted_cap_pct"]], on="name_key", how="left"
    )
    # use CV prediction where available (trained vets), else the full-fit
    # prediction (rookies / players we didn't train on)
    df["predicted_cap_pct"] = df["predicted_cap_pct"].fillna(df["predicted_cap_pct_full"])
    df = df.drop(columns=["predicted_cap_pct_full"])

    ridge = model.named_steps["ridge"]
    importances = dict(zip(FEATURES, np.round(ridge.coef_, 3)))
    return df, importances, len(train_df)


def grade_contract(row) -> str:
    if pd.isna(row.get("salary_2025_26")) or pd.isna(row.get("predicted_cap_pct")):
        return "Unknown"
    surplus_pct = row["predicted_cap_pct"] - row["cap_pct"]
    if row.get("likely_rookie_scale"):
        return "Rookie scale (locked-in value)"
    if surplus_pct >= 0.06:
        return "Significant bargain"
    if surplus_pct >= 0.02:
        return "Good value"
    if surplus_pct >= -0.02:
        return "Fair value"
    if surplus_pct >= -0.06:
        return "Slight overpay"
    return "Significant overpay"


def build_team_summaries(df: pd.DataFrame):
    """Aggregate known player salaries by team and classify each team's
    apron status. NOTE: because our dataset only covers ~100 of the league's
    ~450 rostered players (the highest-earners and highest-minute players),
    these team totals UNDERSTATE actual payroll - they're a partial cap
    sheet, not the full one. The production scraper pulls full team payroll
    pages (scrapers/scrape_contracts.py) to fix this; flagged clearly in the
    dashboard."""
    team_totals = (
        df[df["salary_2025_26"].notna() & (df["team"] != "TOT")]
        .groupby("team")["salary_2025_26"]
        .sum()
        .sort_values(ascending=False)
    )
    teams = []
    for team, total in team_totals.items():
        if not isinstance(team, str) or len(team) > 4:
            continue
        status = cba.ApronStatus(team_salary=total, season="2025-26")
        teams.append({
            "team": team,
            "known_payroll": round(total),
            "apron_status": status.label,
            "available_exceptions": status.available_exceptions,
        })
    return teams


def build():
    df = pd.read_csv(f"{DATA_DIR}/players_merged.csv")
    df = engineer_features(df)
    df, importances, n_train = train_value_model(df)

    df["predicted_aav"] = df["predicted_cap_pct"] * CAP_2025_26
    df["surplus_value"] = df["predicted_aav"] - df["salary_2025_26"]
    df["contract_type"] = df.apply(
        lambda r: cba.classify_contract_type({
            "years_of_service": None,
            "salary_current": r.get("salary_2025_26"),
            "cap_pct": r.get("cap_pct"),
        }), axis=1
    )
    df["contract_grade"] = df.apply(grade_contract, axis=1)
    df["small_sample_caveat"] = df["gp"] < 40

    df.to_csv(f"{DATA_DIR}/players_final.csv", index=False)

    # ---- build a compact JSON for the dashboard ----
    records = []
    for _, r in df.iterrows():
        if pd.isna(r.get("salary_2025_26")):
            continue  # dashboard focuses on players with known contracts
        records.append({
            "player": r["player"],
            "team": r["team"],
            "pos": r["pos"],
            "age": int(r["age"]) if pd.notna(r["age"]) else None,
            "stats": {
                "gp": int(r["gp"]) if pd.notna(r["gp"]) else None,
                "ppg": round(r["ppg"], 1) if pd.notna(r["ppg"]) else None,
                "rpg": round(r["rpg"], 1) if pd.notna(r["rpg"]) else None,
                "apg": round(r["apg"], 1) if pd.notna(r["apg"]) else None,
                "ts_pct": round(r["ts_pct"], 3) if pd.notna(r["ts_pct"]) else None,
                "per": round(r["PER"], 1) if pd.notna(r["PER"]) else None,
                "ws": round(r["WS"], 1) if pd.notna(r["WS"]) else None,
                "vorp": round(r["VORP"], 1) if pd.notna(r["VORP"]) else None,
                "bpm": round(r["BPM"], 1) if pd.notna(r["BPM"]) else None,
            },
            "contract": {
                "salary_2025_26": r["salary_2025_26"],
                "salary_2026_27": r["salary_2026_27"] if pd.notna(r["salary_2026_27"]) else None,
                "years_remaining": int(r["contract_years_remaining"]) if pd.notna(r["contract_years_remaining"]) else None,
                "total_remaining_value": r["total_remaining_value"] if pd.notna(r["total_remaining_value"]) else None,
                "guaranteed_total": r["guaranteed_total"] if pd.notna(r["guaranteed_total"]) else None,
                "cap_pct": round(r["cap_pct"], 4) if pd.notna(r["cap_pct"]) else None,
                "type": r["contract_type"],
            },
            "valuation": {
                "predicted_cap_pct": round(r["predicted_cap_pct"], 4) if pd.notna(r["predicted_cap_pct"]) else None,
                "predicted_aav": round(r["predicted_aav"]) if pd.notna(r["predicted_aav"]) else None,
                "surplus_value": round(r["surplus_value"]) if pd.notna(r["surplus_value"]) else None,
                "grade": r["contract_grade"],
                "small_sample_caveat": bool(r["small_sample_caveat"]),
            },
        })

    meta = {
        "season": cba.CURRENT_SEASON,
        "cap": CAP_2025_26,
        "tax": cba.CAP_HISTORY["2025-26"]["tax"],
        "apron1": cba.CAP_HISTORY["2025-26"]["apron1"],
        "apron2": cba.CAP_HISTORY["2025-26"]["apron2"],
        "next_season": "2026-27",
        "next_cap": cba.CAP_HISTORY["2026-27"]["cap"],
        "n_players": len(records),
        "n_training_players": n_train,
        "feature_importances": importances,
        "data_source": "basketball-reference.com (contracts/players.html, 2025-26 per-game & advanced stats)",
        "generated_note": "Snapshot pulled live; full-league refresh requires running scrapers/ locally.",
    }

    with open(f"{DATA_DIR}/dashboard_data.json", "w") as f:
        json.dump({"meta": meta, "players": records, "teams": build_team_summaries(df)}, f, indent=2)

    print(f"Trained on {n_train} non-rookie-scale veteran contracts")
    print(f"Feature importances: {importances}")
    print(f"Final dataset: {len(records)} players written to dashboard_data.json")
    print(df[["player", "salary_2025_26", "predicted_aav", "surplus_value", "contract_grade"]]
          .sort_values("surplus_value", ascending=False).head(10))
    print(df[["player", "salary_2025_26", "predicted_aav", "surplus_value", "contract_grade"]]
          .sort_values("surplus_value", ascending=True).head(10))


if __name__ == "__main__":
    build()
