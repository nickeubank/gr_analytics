"""
GridRival fantasy F1 scoring and salary calculation.

score_event(scenario, round) scores a race for all drivers and constructors.

Driver output columns (appended):
  pts_qualifying, pts_race, pts_overtake, pts_improvement,
  pts_completion, pts_teammate, points_earned, salary_after_event

Constructor output columns (appended):
  pts_qualifying, pts_race, points_earned, salary_after_event
  (constructor qualifying/race pts are the sum across both drivers,
   using the constructor-specific point tables, not the driver tables)

Assumptions:
  - Grand Prix only (no sprint races)
  - All drivers finish (full completion bonus applied to drivers)
  - Each constructor has exactly 2 drivers in the input DataFrame
  - Constructors do not earn overtake, improvement, teammate, or completion bonuses
"""

import math
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def driver_data() -> pd.DataFrame:
    """Load and return driver_data.csv as a DataFrame."""
    return pd.read_csv(_DATA_DIR / "driver_data.csv")

# ---------------------------------------------------------------------------
# Lookup tables — Drivers
# ---------------------------------------------------------------------------

# Qualifying: P1=50, P2=48, ..., P22=8  (step -2)
DRIVER_QUAL_POINTS = {pos: 52 - 2 * pos for pos in range(1, 23)}

# Race finish: P1=100, P2=97, ..., P22=37  (step -3)
DRIVER_RACE_POINTS = {pos: 103 - 3 * pos for pos in range(1, 23)}

# Improvement over 8-race average (positions improved -> bonus points)
IMPROVEMENT_POINTS = {2: 2, 3: 4, 4: 6, 5: 9, 6: 12, 7: 16, 8: 20, 9: 25}
IMPROVEMENT_MAX = 30  # 10+ positions improved

# Beating teammate (margin in finish positions -> bonus points for the winner)
TEAMMATE_POINTS_THRESHOLDS = [(13, 12), (8, 8), (4, 5), (1, 2)]
# read as: margin >= threshold -> points

# Default driver salary table (rank -> salary in millions)
DRIVER_DEFAULT_SALARY = {
    1: 34.0,
    2: 32.4,
    3: 30.8,
    4: 29.2,
    5: 27.6,
    6: 26.0,
    7: 24.4,
    8: 22.8,
    9: 21.2,
    10: 19.6,
    11: 18.0,
    12: 16.4,
    13: 14.8,
    14: 13.2,
    15: 11.6,
    16: 10.0,
    17: 8.4,
    18: 6.8,
    19: 5.2,
    20: 3.6,
    21: 2.0,
    22: 0.4,
}

DRIVER_MAX_ADJUSTMENT = 2.0  # £M

# ---------------------------------------------------------------------------
# Lookup tables — Constructors
# ---------------------------------------------------------------------------

# Constructor qualifying: P1=30, P2=29, ..., P22=9  (step -1, per driver)
CONSTRUCTOR_QUAL_POINTS = {pos: 31 - pos for pos in range(1, 23)}

# Constructor race: P1=60, P2=58, ..., P22=18  (step -2, per driver)
CONSTRUCTOR_RACE_POINTS = {pos: 62 - 2 * pos for pos in range(1, 23)}

# Default constructor salary table (rank -> salary in millions)
CONSTRUCTOR_DEFAULT_SALARY = {
    1: 30.0,
    2: 27.4,
    3: 24.8,
    4: 22.2,
    5: 19.6,
    6: 17.0,
    7: 14.4,
    8: 11.8,
    9: 9.2,
    10: 6.6,
    11: 4.0,
}

CONSTRUCTOR_MAX_ADJUSTMENT = 3.0  # £M

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

SALARY_STEP = 0.1  # £M (100k)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _improvement_pts(positions_improved: float) -> int:
    """Points for finishing ahead of 8-race average."""
    n = math.floor(positions_improved)
    if n <= 1:
        return 0
    if n >= 10:
        return IMPROVEMENT_MAX
    return IMPROVEMENT_POINTS.get(n, 0)


def _teammate_pts(margin: int) -> int:
    """Points awarded to the driver who beats their teammate."""
    for threshold, pts in TEAMMATE_POINTS_THRESHOLDS:
        if margin >= threshold:
            return pts
    return 0


def _calc_adjustment(variation: float, max_adjustment: float) -> float:
    """
    Divide variation by 4, truncate toward zero to nearest £100k,
    then apply min/max caps.
    """
    raw = variation / 4
    sign = 1 if raw >= 0 else -1
    truncated = sign * math.floor(abs(raw) / SALARY_STEP) * SALARY_STEP
    truncated = round(truncated, 1)  # avoid float precision drift

    truncated = max(-max_adjustment, min(max_adjustment, truncated))

    # Minimum adjustment: if there's any variation, enforce at least SALARY_STEP
    if variation > 0 and truncated < SALARY_STEP:
        truncated = SALARY_STEP
    elif variation < 0 and truncated > -SALARY_STEP:
        truncated = -SALARY_STEP

    return truncated


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------


def _score_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Score all driver rows. Input must contain only type='driver' rows."""
    df = drivers.copy()

    df["_qual_pts"] = df["qualifying_position"].map(DRIVER_QUAL_POINTS)
    df["_race_pts"] = df["finishing_position"].map(DRIVER_RACE_POINTS)
    df["_overtake_pts"] = (df["qualifying_position"] - df["finishing_position"]).clip(
        lower=0
    ) * 3
    df["_improvement_pts"] = (
        df["eight_race_average"] - df["finishing_position"]
    ).apply(_improvement_pts)
    # All drivers finish -> all 4 completion milestones hit (4 x 3 = 12)
    df["_completion_pts"] = 12

    # Teammate beating points
    df["_teammate_pts"] = 0
    for team, group in df.groupby("driver_team"):
        if len(group) != 2:
            raise ValueError(
                f"Team '{team}' has {len(group)} driver(s) in the DataFrame; expected 2."
            )
        i0, i1 = group.index[0], group.index[1]
        p0 = df.at[i0, "finishing_position"]
        p1 = df.at[i1, "finishing_position"]
        margin = abs(p0 - p1)
        pts = _teammate_pts(margin)
        if p0 < p1:
            df.at[i0, "_teammate_pts"] = pts
        elif p1 < p0:
            df.at[i1, "_teammate_pts"] = pts

    point_cols = [
        "_qual_pts",
        "_race_pts",
        "_overtake_pts",
        "_improvement_pts",
        "_completion_pts",
        "_teammate_pts",
    ]
    df["points_earned"] = df[point_cols].sum(axis=1)

    # Salary adjustment — rank among all drivers
    df["_fantasy_rank"] = (
        df["points_earned"].rank(method="first", ascending=False).astype(int)
    )
    df["_default_salary"] = df["_fantasy_rank"].map(DRIVER_DEFAULT_SALARY)
    df["salary_after_event"] = df.apply(
        lambda row: round(
            row["starting_salary"]
            + _calc_adjustment(
                row["_default_salary"] - row["starting_salary"],
                DRIVER_MAX_ADJUSTMENT,
            ),
            1,
        ),
        axis=1,
    )

    df = df.drop(columns=["_fantasy_rank", "_default_salary"])
    df = df.rename(
        columns={
            "_qual_pts": "pts_qualifying",
            "_race_pts": "pts_race",
            "_overtake_pts": "pts_overtake",
            "_improvement_pts": "pts_improvement",
            "_completion_pts": "pts_completion",
            "_teammate_pts": "pts_teammate",
        }
    )
    df["salary_change"] = (df["salary_after_event"] - df["starting_salary"]).round(1)

    return df


def _score_constructors(
    constructors: pd.DataFrame, scored_drivers: pd.DataFrame
) -> pd.DataFrame:
    """
    Score all constructor rows.

    Constructor points = sum of each driver's constructor qualifying pts
                       + sum of each driver's constructor race pts.
    Uses constructor-specific point tables (different from driver tables).
    """
    if constructors.empty:
        return constructors

    df = constructors.copy()

    # Build per-driver constructor points, then aggregate to team level
    driver_con_pts = scored_drivers[
        ["driver_team", "qualifying_position", "finishing_position"]
    ].copy()
    driver_con_pts["_con_qual"] = driver_con_pts["qualifying_position"].map(
        CONSTRUCTOR_QUAL_POINTS
    )
    driver_con_pts["_con_race"] = driver_con_pts["finishing_position"].map(
        CONSTRUCTOR_RACE_POINTS
    )

    team_pts = (
        driver_con_pts.groupby("driver_team")[["_con_qual", "_con_race"]]
        .sum()
        .rename(columns={"_con_qual": "pts_qualifying", "_con_race": "pts_race"})
    )
    team_pts["points_earned"] = team_pts["pts_qualifying"] + team_pts["pts_race"]

    # driver_name holds the team code for constructor rows (e.g. "MER")
    df = df.join(team_pts, on="driver_name")

    # Salary adjustment — rank among all constructors
    df["_fantasy_rank"] = (
        df["points_earned"].rank(method="first", ascending=False).astype(int)
    )
    df["_default_salary"] = df["_fantasy_rank"].map(CONSTRUCTOR_DEFAULT_SALARY)
    df["salary_after_event"] = df.apply(
        lambda row: round(
            row["starting_salary"]
            + _calc_adjustment(
                row["_default_salary"] - row["starting_salary"],
                CONSTRUCTOR_MAX_ADJUSTMENT,
            ),
            1,
        ),
        axis=1,
    )

    df = df.drop(columns=["_fantasy_rank", "_default_salary"])
    df["salary_change"] = (df["salary_after_event"] - df["starting_salary"]).round(1)
    return df


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def score_event(scenario: pd.DataFrame, round: int = None) -> pd.DataFrame:
    """
    Score a Grand Prix event for drivers and constructors.

    Parameters
    ----------
    scenario : DataFrame with columns:
        driver_abbr, qualifying_position, race_position
        (any additional columns are passed through)
    round : int, optional
        Race round number. Defaults to the maximum round in driver_data.

    Returns
    -------
    DataFrame with scoring columns appended.
    Drivers get: pts_qualifying, pts_race, pts_overtake, pts_improvement,
                 pts_completion, pts_teammate, points_earned, salary_after_event
    Constructors get: pts_qualifying, pts_race, points_earned, salary_after_event
    """
    dd = driver_data()
    if round is None:
        round = dd["round"].max()
    dd = dd[dd["round"] == round].copy()

    drivers_dd = dd[dd["type"] == "driver"].copy()
    teams_dd = dd[dd["type"] == "team"].copy()

    drivers_merged = drivers_dd.merge(
        scenario,
        left_on="abbreviations",
        right_on="driver_abbr",
        how="left",
    ).rename(columns={"race_position": "finishing_position"})

    df = pd.concat([drivers_merged, teams_dd], ignore_index=True)

    drivers = df[df["type"] == "driver"].copy()
    constructors = df[df["type"] == "team"].copy()

    scored_drivers = _score_drivers(drivers)
    scored_constructors = _score_constructors(constructors, drivers)

    result = pd.concat([scored_drivers, scored_constructors])

    if "held" in result.columns:
        held = result[result["held"] == 1]
        print(f"total points: {held['points_earned'].sum():.0f}")
        print(f"total salary change: {held['salary_change'].sum():.1f}")

    return result


def score_my_team(
    scenario: pd.DataFrame,
    drivers: list,
    team: str,
    star_driver: str,
    round: int = None,
) -> tuple:
    """
    Score a specific team selection for a race.

    Parameters
    ----------
    scenario : DataFrame with driver_abbr, qualifying_position, race_position
    drivers : list of 5 driver_abbr strings
    team : team abbreviation (e.g. 'MER')
    star_driver : driver_abbr whose points_earned is doubled
    round : race round number; defaults to max round in driver_data

    Returns
    -------
    (total_points, total_salary_change)
    """
    result = score_event(scenario, round)

    my_drivers = result[
        (result["type"] == "driver") & (result["abbreviations"].isin(drivers))
    ].copy()
    my_team = result[
        (result["type"] == "team") & (result["driver_name"] == team)
    ].copy()

    my_picks = pd.concat([my_drivers, my_team])
    my_picks.loc[my_picks["abbreviations"] == star_driver, "points_earned"] *= 2

    total_points = my_picks["points_earned"].sum()
    total_salary_change = my_picks["salary_change"].sum()

    print(f"total points: {total_points:.0f}")
    print(f"total salary change: {total_salary_change:.1f}")

    return total_points, total_salary_change
