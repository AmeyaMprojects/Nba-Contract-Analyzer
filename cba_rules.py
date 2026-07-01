"""
cba_rules.py
============
Models the NBA's Collective Bargaining Agreement (2023 CBA) salary mechanics.
This is the piece that didn't exist at all in the 2022 version of this project -
the old pipeline only ever compared "stats" to "a single salary number." This
module encodes the actual rules that determine what a player CAN be paid, which
is necessary context for judging whether what they ARE paid is reasonable.

Sources: basketball-reference salary cap history page, NBA CBA summaries
(Larry Coon's salary cap FAQ conventions), current-season cap figures pulled
directly from basketball-reference.com/contracts as of June 2026.

NOTE ON ACCURACY: this models the *mechanics that matter for contract analysis*
(max tiers, supermax, rookie scale, apron lines, cap/tax history). It is not a
substitute for the full CBA text - things like Over-38 rule edge cases, offer
sheet matching nuances, and stretch provision math are simplified or omitted.
Where simplified, it's noted in the docstring.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Salary cap / tax / apron history, by season.
# 2025-26 and 2026-27 figures are the real, current numbers (basketball-reference
# salary cap history page + NBA cap announcements). Earlier years are the
# actual historical NBA cap figures. All in dollars.
# ---------------------------------------------------------------------------
CAP_HISTORY = {
    "2018-19": {"cap": 101_869_000, "tax": 123_733_000},
    "2019-20": {"cap": 109_140_000, "tax": 132_627_000},
    "2020-21": {"cap": 109_140_000, "tax": 132_627_000},  # frozen, COVID
    "2021-22": {"cap": 112_414_000, "tax": 136_606_000},
    "2022-23": {"cap": 123_655_000, "tax": 150_267_000},
    "2023-24": {"cap": 136_021_000, "tax": 165_294_000},
    "2024-25": {"cap": 140_588_000, "tax": 170_814_000, "apron1": 178_132_000, "apron2": 188_931_000},
    "2025-26": {"cap": 154_647_000, "tax": 187_895_000, "apron1": 195_945_000, "apron2": 207_824_000},
    "2026-27": {"cap": 165_000_000, "tax": 201_000_000, "apron1": 209_000_000, "apron2": 222_000_000,
                "floor": 149_000_000},
}

CURRENT_SEASON = "2025-26"  # the season the live stats/contract snapshot represents

# ---------------------------------------------------------------------------
# Max contract tiers (2023 CBA, Article II).
# Max salary as a % of the cap depends on Years of Service (YOS) at signing:
#   0-6 YOS   -> 25% of cap ("first apron max" / 25% max)
#   7-9 YOS   -> 30% of cap
#   10+ YOS   -> 35% of cap
# Supermax ("Designated Veteran Player" extension, the "Rose Rule") lets a
# player with 7-9 YOS sign for 35% if, in the prior season, they won
# MVP, DPOY, or made an All-NBA team - AND they re-sign with the team that
# drafted them (or that traded for them as a rookie-scale player, with some
# exceptions). This is why guys like Tatum/SGA can have 35%-of-cap deals
# despite having fewer than 10 years in the league.
# ---------------------------------------------------------------------------
MAX_PCT_BY_YOS = [
    (0, 6, 0.25),
    (7, 9, 0.30),
    (10, 99, 0.35),
]
SUPERMAX_PCT = 0.35
SUPERMAX_MIN_YOS, SUPERMAX_MAX_YOS = 7, 9

# First-year max raise rules: extensions/re-signings with Bird rights can raise
# up to 8% per year (5% for non-Bird signings with a new team), but the
# practical ceiling is the max % of cap above.
RAISE_PCT_BIRD = 0.08
RAISE_PCT_NON_BIRD = 0.05


def max_salary_pct(years_of_service: int, supermax_eligible: bool = False) -> float:
    """Return the max first-year salary as a fraction of the cap for a player
    with the given years of service. supermax_eligible should be True only if
    the player actually met the performance criteria AND re-signed with the
    qualifying team - this script can't verify that automatically, so it's
    left as a manual override flag per player where known."""
    if supermax_eligible and SUPERMAX_MIN_YOS <= years_of_service <= SUPERMAX_MAX_YOS:
        return SUPERMAX_PCT
    for lo, hi, pct in MAX_PCT_BY_YOS:
        if lo <= years_of_service <= hi:
            return pct
    return 0.25


# ---------------------------------------------------------------------------
# Rookie scale (first-round picks). 2024 rookie scale shown as representative;
# the scale increases each year with the cap. Year 1 amount by pick number,
# approximate for 2025-26 rookie scale. Years 3-4 are team options.
# Simplified to 5 pick-bands rather than all 30 individual slots.
# ---------------------------------------------------------------------------
ROOKIE_SCALE_BANDS_2025 = [
    (1, 1, 13_140_000),
    (2, 5, 10_500_000),
    (6, 10, 8_300_000),
    (11, 20, 5_900_000),
    (21, 30, 3_900_000),
]


def rookie_scale_estimate(pick: int) -> int:
    for lo, hi, amt in ROOKIE_SCALE_BANDS_2025:
        if lo <= pick <= hi:
            return amt
    return 2_300_000  # second-round / undrafted minimum-adjacent estimate


# ---------------------------------------------------------------------------
# Bird rights categories - govern how far over the cap a team can go to
# re-sign its own free agent.
# ---------------------------------------------------------------------------
BIRD_RIGHTS = {
    "full_bird": "3+ consecutive years with same team (allowing for trades/qualifying exceptions); "
                  "can be re-signed up to the player's individual max, unrestricted by team cap room.",
    "early_bird": "2 consecutive years with same team; can be re-signed for the greater of 175% of "
                   "prior salary or the league average salary.",
    "non_bird": "Less than 2 years with team (or has Bird rights renounced); can be re-signed for "
                 "120% of prior salary or 120% of the minimum, whichever is greater.",
}


# ---------------------------------------------------------------------------
# Mid-level / other exceptions, 2025-26 amounts (approximate, indexed to cap).
# Availability is gated by team apron status:
#   - Non-taxpayer (full) MLE: teams below the first apron only
#   - Taxpayer MLE: teams between the tax line and the second apron
#   - Teams above the SECOND apron: no MLE access at all, no bi-annual,
#     cannot aggregate salaries in trade, cannot send cash, cannot take back
#     more salary than they send out, first-round pick frozen 7 years out
#     if repeatedly over the second apron.
# ---------------------------------------------------------------------------
EXCEPTIONS_2025_26 = {
    "non_taxpayer_mle": 14_100_000,
    "taxpayer_mle": 5_700_000,
    "room_mle": 8_400_000,
    "bi_annual": 4_700_000,
}


@dataclass
class ApronStatus:
    team_salary: float
    season: str

    def _thresholds(self):
        return CAP_HISTORY.get(self.season, CAP_HISTORY[CURRENT_SEASON])

    @property
    def over_cap(self) -> bool:
        return self.team_salary > self._thresholds()["cap"]

    @property
    def over_tax(self) -> bool:
        return self.team_salary > self._thresholds()["tax"]

    @property
    def over_first_apron(self) -> bool:
        t = self._thresholds()
        return "apron1" in t and self.team_salary > t["apron1"]

    @property
    def over_second_apron(self) -> bool:
        t = self._thresholds()
        return "apron2" in t and self.team_salary > t["apron2"]

    @property
    def available_exceptions(self):
        if self.over_second_apron:
            return []  # hard-capped: no MLE, no bi-annual
        if self.over_first_apron:
            return ["taxpayer_mle"]
        if self.over_cap:
            return ["non_taxpayer_mle", "bi_annual"]
        return ["non_taxpayer_mle", "bi_annual", "room_mle"]

    @property
    def label(self) -> str:
        if self.over_second_apron:
            return "Second apron (hard cap)"
        if self.over_first_apron:
            return "First apron"
        if self.over_tax:
            return "Luxury tax"
        if self.over_cap:
            return "Over the cap"
        return "Cap room"


def classify_contract_type(player_row: dict) -> str:
    """Best-effort classification of contract 'type' from the data we have.
    This drives a lot of the qualitative read on a contract - a $13M salary
    means something completely different on a rookie-scale deal (great value
    by construction) vs. a 5th-year veteran deal (below-average pay).
    """
    yos = player_row.get("years_of_service")
    salary = player_row.get("salary_current")
    cap_pct = player_row.get("cap_pct")

    if yos is not None and yos <= 3 and cap_pct is not None and cap_pct < 0.12:
        return "Rookie scale"
    if cap_pct is not None and cap_pct >= 0.30:
        return "Max / Supermax"
    if cap_pct is not None and 0.18 <= cap_pct < 0.30:
        return "Veteran (above-MLE)"
    if cap_pct is not None and 0.08 <= cap_pct < 0.18:
        return "Mid-level range"
    if salary is not None and salary <= 3_500_000:
        return "Minimum / minimum-adjacent"
    return "Veteran"
