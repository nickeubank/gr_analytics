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

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

_DATA_DIR = Path(__file__).parent / "data"
_DRIVER_DATA_PATH = _DATA_DIR / "driver_data.csv"

# Read driver_data.csv once at import time so repeated driver_data() calls
# don't re-read and re-parse the file from disk.
try:
    _DRIVER_DATA = pd.read_csv(_DRIVER_DATA_PATH)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"gr_analytics could not find its bundled driver data at "
        f"{_DRIVER_DATA_PATH}. The packaged data file appears to be missing; "
        f"reinstalling gr_analytics should restore it."
    ) from exc
except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
    raise RuntimeError(
        f"gr_analytics could not parse its bundled driver data at "
        f"{_DRIVER_DATA_PATH}: {exc}"
    ) from exc


def driver_data() -> pd.DataFrame:
    """Return driver_data.csv as a DataFrame.

    The CSV is read into memory once at module import; each call returns a
    fresh copy so callers can safely mutate the result without corrupting
    the shared in-memory copy.
    """
    return _DRIVER_DATA.copy()


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
    truncated = sign * math.floor(round(abs(raw) / SALARY_STEP, 8)) * SALARY_STEP
    truncated = round(truncated, 1)  # avoid float precision drift

    truncated = max(-max_adjustment, min(max_adjustment, truncated))

    return truncated


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------


def _score_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Score all driver rows. Input must contain only type='driver' rows."""
    df = drivers.copy()

    # DNQ drivers (completed_qualifying=0) get 0 qual pts but overtakes use their qualifying_position
    qual_pts = df["qualifying_position"].map(DRIVER_QUAL_POINTS)
    if "completed_qualifying" in df.columns:
        qual_pts = qual_pts.where(df["completed_qualifying"] == 1, other=0)
    df["_qual_pts"] = qual_pts
    # DNS drivers (completed_pct="DNS") get 0 for all race-related points
    is_dns = pd.Series(False, index=df.index)
    if "completed_pct" in df.columns:
        is_dns = df["completed_pct"] == "DNS"

    df["_race_pts"] = df["finishing_position"].map(DRIVER_RACE_POINTS)
    df["_overtake_pts"] = (df["qualifying_position"] - df["finishing_position"]).clip(
        lower=0
    ) * 3
    df["_improvement_pts"] = (
        df["eight_race_average"] - df["finishing_position"]
    ).apply(_improvement_pts)
    # Completion bonus: 3 pts each at 25%, 50%, 75%, 90% of race distance
    if "completed_pct" in df.columns:

        def _completion_pts(pct):
            if pct >= 0.90:
                return 12
            if pct >= 0.75:
                return 9
            if pct >= 0.50:
                return 6
            if pct >= 0.25:
                return 3
            return 0

        pct_numeric = pd.to_numeric(df["completed_pct"], errors="coerce").fillna(0)
        df["_completion_pts"] = pct_numeric.apply(_completion_pts)
    else:
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

    # DNS drivers get 0 for all race-related points
    race_cols = [
        "_race_pts",
        "_overtake_pts",
        "_improvement_pts",
        "_completion_pts",
        "_teammate_pts",
    ]
    df.loc[is_dns, race_cols] = 0

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


def eight_race_average(initial_average: float, finishing_positions: list) -> int:
    """
    GridRival's eight-race average after a sequence of races.

    GridRival seeds each driver's season with 8 "slots" holding a
    hard-coded initial average. Each race replaces one slot with the
    driver's official classified finishing position, so after 8 races the
    initial average has fully rolled off and the value is a true rolling
    average of the last 8 finishes. The displayed value is the ceiling of
    the slot mean (verified against GridRival's displayed values for
    rounds 1-4 of 2026).

    Parameters
    ----------
    initial_average : float
        The hard-coded season-start average (round 0 of driver_data).
    finishing_positions : list of int
        Classified finishing positions in every race so far, oldest first.

    Returns
    -------
    int
        The eight-race average as GridRival displays it.
    """
    slots = [initial_average] * 8 + [int(p) for p in finishing_positions]
    return math.ceil(sum(slots[-8:]) / 8)


def calculate_eight_race_averages(through_round: int = None) -> pd.Series:
    """
    Compute every driver's eight-race average after ``through_round``.

    Reads everything from driver_data.csv: the round-0 ``eight_race_average``
    seeds and the per-race ``finishing_position`` recorded on each driver's
    round row (the round-N row holds the finishing position from race N,
    parallel to ``points_from_last_race``). Matches the convention of
    driver_data, where the round-N row holds the state *after* race N (so
    the returned values are what GridRival displays going into race N+1).

    Parameters
    ----------
    through_round : int, optional
        Include races 1 through this round. Defaults to the latest round
        whose finishing positions are recorded in driver_data. Pass 0 to
        get the season-start seeds.

    Returns
    -------
    Series
        Eight-race average indexed by driver_abbr.
    """
    dd = driver_data()
    drivers = dd[dd["type"] == "driver"]
    seeds = drivers[drivers["round"] == 0].set_index("driver_abbr")[
        "eight_race_average"
    ]

    finishes = drivers[(drivers["round"] >= 1) & drivers["finishing_position"].notna()]
    if through_round is None:
        through_round = int(finishes["round"].max())
    finishes = finishes[finishes["round"] <= through_round].sort_values("round")

    averages = {
        abbr: eight_race_average(
            seed,
            finishes.loc[finishes["driver_abbr"] == abbr, "finishing_position"],
        )
        for abbr, seed in seeds.items()
    }
    return pd.Series(averages, name="eight_race_average").rename_axis("driver_abbr")


def _validate_scenario(scenario: pd.DataFrame) -> None:
    """Check that qualifying and race positions are sequential and unique."""
    errors = []
    n = len(scenario)

    # Qualifying: all positions required, must be exactly 1..n
    qual = scenario["qualifying_position"]
    qual_vals = sorted(qual.astype(int).tolist())
    expected_qual = list(range(1, n + 1))
    if qual_vals != expected_qual:
        dupes = sorted({v for v in qual_vals if qual_vals.count(v) > 1})
        missing = sorted(set(expected_qual) - set(qual_vals))
        extra = sorted(set(qual_vals) - set(expected_qual))
        parts = []
        if dupes:
            parts.append(f"duplicates: {dupes}")
        if missing:
            parts.append(f"missing from sequence: {missing}")
        if extra:
            parts.append(f"unexpected values: {extra}")
        errors.append(
            "Qualifying positions are not sequential and unique — " + "; ".join(parts)
        )

    # Race: all positions required, must be exactly 1..n
    race_col = (
        "race_position" if "race_position" in scenario.columns else "finishing_position"
    )
    race = scenario[race_col]
    if race.isna().any():
        missing_idx = scenario.index[race.isna()].tolist()
        errors.append(f"Missing race positions for rows: {missing_idx}")
    else:
        race_vals = sorted(race.astype(int).tolist())
        expected_race = list(range(1, n + 1))
        if race_vals != expected_race:
            dupes = sorted({v for v in race_vals if race_vals.count(v) > 1})
            missing = sorted(set(expected_race) - set(race_vals))
            extra = sorted(set(race_vals) - set(expected_race))
            parts = []
            if dupes:
                parts.append(f"duplicates: {dupes}")
            if missing:
                parts.append(f"missing from sequence: {missing}")
            if extra:
                parts.append(f"unexpected values: {extra}")
            errors.append(
                "Race positions are not sequential and unique — " + "; ".join(parts)
            )

    if errors:
        raise ValueError("Invalid scenario:\n" + "\n".join(f"  - {e}" for e in errors))


def score_event(scenario: pd.DataFrame, round: int = None) -> pd.DataFrame:
    """
    Score a Grand Prix event for drivers and constructors.

    Merges ``scenario`` with the built-in ``driver_data.csv`` for the given
    round to obtain starting salaries, team affiliations, eight-race
    averages, and constructor rows. Then scores every driver and
    constructor and computes post-event salary adjustments.

    Parameters
    ----------
    scenario : DataFrame
        One row per driver with the following columns:

        - **driver_abbr** : str — three-letter driver code (e.g. "VER").
        - **qualifying_position** : int — grid position (1–22).
        - **race_position** : int — finishing position (1–22).
        - **completed_qualifying** : int (0 or 1), optional — whether the
          driver set a qualifying time. DNQ drivers (0) receive 0
          qualifying points but overtake points still use their official
          qualifying_position. Defaults to 1 if column is absent.
        - **completed_pct** : float or "DNS", optional — fraction of race
          distance completed (0.0–1.0), or the string "DNS" for drivers
          who did not start. Used to calculate completion bonus (3 pts
          each at 25%/50%/75%/90%). DNS drivers receive 0 for all
          race-related points. Defaults to 1.0 (full completion) if
          column is absent.

        Qualifying and race positions must each be a complete 1…n
        sequence with no duplicates.

    round : int, optional
        Race round number used to look up driver/constructor data (salary,
        eight-race average, team mapping). Defaults to the maximum round
        in driver_data.

    Returns
    -------
    DataFrame
        All driver and constructor rows with scoring columns appended.

        **Drivers:** pts_qualifying, pts_race, pts_overtake,
        pts_improvement, pts_completion, pts_teammate, points_earned,
        salary_after_event, salary_change.

        **Constructors:** pts_qualifying, pts_race, points_earned,
        salary_after_event, salary_change (qualifying and race points
        are the sum of constructor-specific point tables across both
        drivers, not the driver tables).

    Raises
    ------
    ValueError
        If qualifying or race positions contain duplicates, gaps, or
        out-of-range values.

    Notes
    -----
    If the driver_data contains a ``held`` column with value 1, the
    function prints the total points and salary change for held picks.
    """
    _validate_scenario(scenario)

    dd = driver_data()
    if round is None:
        round = dd["round"].max()
    dd = dd[dd["round"] == round].copy()

    # `finishing_position` stored in driver_data is the historical race result
    # (used only for the eight-race average); the race being scored is supplied
    # by `scenario`, so drop the stored column to avoid a merge collision.
    dd = dd.drop(columns=["finishing_position"], errors="ignore")

    drivers_dd = dd[dd["type"] == "driver"].copy()
    teams_dd = dd[dd["type"] == "team"].copy()

    # Validate driver_abbr values before merging
    dd_abbrs = set(drivers_dd["driver_abbr"])
    scenario_abbrs = set(scenario["driver_abbr"])
    missing_from_scenario = dd_abbrs - scenario_abbrs
    missing_from_dd = scenario_abbrs - dd_abbrs
    errors = []
    if missing_from_scenario:
        errors.append(
            f"Drivers in driver_data (round {round}) but not in scenario: "
            f"{sorted(missing_from_scenario)}"
        )
    if missing_from_dd:
        errors.append(
            f"Drivers in scenario but not in driver_data (round {round}): "
            f"{sorted(missing_from_dd)}"
        )
    if errors:
        raise ValueError(
            "driver_abbr mismatch between scenario and driver_data:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    drivers_merged = drivers_dd.merge(
        scenario,
        on="driver_abbr",
        how="left",
    ).rename(columns={"race_position": "finishing_position"})

    # Improvement points compare each driver's finish to GridRival's eight-race
    # average *going into* this race — i.e. the average as of the end of `round`,
    # the state the scenario is scored against (round=0 scores race 1 off the
    # season-start seeds). Compute it rather than trust the stored
    # eight_race_average column, which is left blank for rounds whose GridRival
    # value hasn't been transcribed.
    avgs = calculate_eight_race_averages(through_round=round)
    drivers_merged["eight_race_average"] = drivers_merged["driver_abbr"].map(avgs)

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
        (result["type"] == "driver") & (result["driver_abbr"].isin(drivers))
    ].copy()
    my_team = result[
        (result["type"] == "team") & (result["driver_abbr"] == team)
    ].copy()

    my_picks = pd.concat([my_drivers, my_team])
    my_picks.loc[my_picks["driver_abbr"] == star_driver, "points_earned"] *= 2

    total_points = my_picks["points_earned"].sum()
    total_salary_change = my_picks["salary_change"].sum()

    print(f"total points: {total_points:.0f}")
    print(f"total salary change: {total_salary_change:.1f}")

    return total_points, total_salary_change


def optimal_lineup(
    scored: pd.DataFrame,
    locked_in: list = None,
    locked_out: list = None,
    optimize_for="points",
    budget: float = 100.0,
    star_salary_cap: float = 19.0,
) -> pd.DataFrame:
    """
    Find the optimal 5-driver + 1-constructor lineup.

    Parameters
    ----------
    scored : DataFrame
        Output of score_event().
    locked_in : list of str, optional
        driver_abbr values (drivers) or driver_name team codes (e.g. "MER")
        that must appear in the lineup. These count against the budget and
        the 5-driver / 1-constructor slot limits.
    locked_out : list of str, optional
        driver_abbr values (drivers) or driver_name team codes that must
        be excluded from the lineup. These are removed from the candidate
        pool before optimisation.
    optimize_for : {"points", "salary_change"} or float
        Objective to maximise. ``"points"`` is equivalent to balance=1
        (pure points with star doubling). ``"salary_change"`` is equivalent
        to balance=0 (pure salary change). A float between 0 and 1 blends:
        ``objective = (points_earned / 100) * balance + salary_change * (1 - balance)``
        The star driver (whose points component is doubled) is chosen
        optimally whenever balance > 0.
    budget : float
        Total salary budget in £M (default 100). The current salary
        (``starting_salary``) of every locked-in pick is subtracted from
        this to determine the budget remaining for non-locked-in picks.
    star_salary_cap : float
        Maximum starting_salary allowed for the star driver (default 19.0).
        Drivers above this threshold are excluded from star consideration.

    Returns
    -------
    DataFrame with the 6 selected rows plus a `star` column (1 = starred
    driver, set when optimize_for="points" or a float balance > 0).
    """
    if locked_in is None:
        locked_in = []
    if locked_out is None:
        locked_out = []

    # Determine balance: 1.0 = pure points, 0.0 = pure salary_change
    if isinstance(optimize_for, (int, float)) and not isinstance(optimize_for, bool):
        balance = float(optimize_for)
        if not 0.0 <= balance <= 1.0:
            raise ValueError("optimize_for as a float must be between 0 and 1")
    elif optimize_for == "points":
        balance = 1.0
    elif optimize_for == "salary_change":
        balance = 0.0
    else:
        raise ValueError(
            "optimize_for must be 'points', 'salary_change', or a float between 0 and 1"
        )

    df = scored.copy()

    # Remove locked-out entries from the pool entirely
    _is_locked_out = df["driver_abbr"].isin(locked_out) | (
        (df["type"] == "team") & df["driver_name"].isin(locked_out)
    )
    df = df[~_is_locked_out].copy()

    df["_locked"] = df["driver_abbr"].isin(locked_in) | (
        (df["type"] == "team") & df["driver_name"].isin(locked_in)
    )

    locked = df[df["_locked"]].copy()
    free = df[~df["_locked"]].copy()

    n_locked_drivers = (locked["type"] == "driver").sum()
    n_locked_teams = (locked["type"] == "team").sum()
    drivers_needed = 5 - n_locked_drivers
    teams_needed = 1 - n_locked_teams

    remaining_budget = budget - locked["starting_salary"].sum()

    # Restrict free pool to only the slot types still needed
    if teams_needed == 0:
        free = free[free["type"] != "team"]
    if drivers_needed == 0:
        free = free[free["type"] != "driver"]

    # Drivers first, teams last (stable ordering for constraint indexing)
    free = free.sort_values("type").reset_index(drop=True)

    salaries_arr = free["starting_salary"].values.astype(float)
    pts_arr = free["points_earned"].values.astype(float)
    sal_arr = free["salary_change"].values.astype(float)
    obj_arr = pts_arr / 100.0 * balance + sal_arr * (1.0 - balance)

    bounds = Bounds(0, 1)
    integrality = np.ones(len(free))
    constraints = [LinearConstraint(A=salaries_arr, lb=0, ub=remaining_budget)]

    if drivers_needed > 0:
        driver_mask = (free["type"] == "driver").values.astype(float)
        constraints.append(
            LinearConstraint(A=driver_mask, lb=drivers_needed, ub=drivers_needed)
        )

    if teams_needed == 1:
        team_mask = (free["type"] == "team").values.astype(float)
        constraints.append(LinearConstraint(A=team_mask, lb=1, ub=1))

    def _run_milp(objective):
        result = milp(
            c=-objective,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        return result

    # Locked contribution for total comparison
    if not locked.empty:
        locked_obj = (
            locked["points_earned"].values.astype(float) / 100.0 * balance
            + locked["salary_change"].values.astype(float) * (1.0 - balance)
        ).sum()
    else:
        locked_obj = 0.0

    # --- star optimization: loop over star candidates ---
    free_drivers = free[free["type"] == "driver"]
    if star_salary_cap is not None:
        free_drivers = free_drivers[free_drivers["starting_salary"] <= star_salary_cap]
    free_driver_idx = free_drivers.index.tolist()
    # -1 sentinel: best star comes from locked_in set
    star_candidates = free_driver_idx + [-1]

    best_total = -np.inf
    best_full = None

    for i in star_candidates:
        obj_copy = obj_arr.copy()
        if i != -1:
            # Star doubles points_earned only → add the points component again
            obj_copy[i] += pts_arr[i] / 100.0 * balance

        result = _run_milp(obj_copy)
        picked = free[result.x.astype(bool)].copy()
        full = pd.concat([picked, locked], ignore_index=True)
        full["star"] = 0

        if i != -1:
            star_abbr = free.at[i, "driver_abbr"]
            full.loc[full["driver_abbr"] == star_abbr, "star"] = 1
            star_bonus = pts_arr[i] / 100.0 * balance
            total = -result.fun + locked_obj + star_bonus
        else:
            # Star is the highest-points locked_in driver (within salary cap)
            locked_drivers = locked[locked["type"] == "driver"]
            if star_salary_cap is not None:
                locked_drivers = locked_drivers[
                    locked_drivers["starting_salary"] <= star_salary_cap
                ]
            if locked_drivers.empty:
                total = -result.fun + locked_obj
            else:
                best_row = locked_drivers.loc[locked_drivers["points_earned"].idxmax()]
                star_abbr = best_row["driver_abbr"]
                full.loc[full["driver_abbr"] == star_abbr, "star"] = 1
                star_bonus = best_row["points_earned"] / 100.0 * balance
                total = -result.fun + locked_obj + star_bonus

        if total > best_total:
            best_total = total
            best_full = full

    # When balance == 0 the star bonus is zero so the loop picks
    # arbitrarily.  Override to star the highest-points eligible driver.
    if balance == 0.0:
        best_full["star"] = 0
        eligible = best_full[best_full["type"] == "driver"]
        if star_salary_cap is not None:
            eligible = eligible[eligible["starting_salary"] <= star_salary_cap]
        if not eligible.empty:
            best_full.loc[eligible["points_earned"].idxmax(), "star"] = 1

    # Double points_earned for the starred driver so the returned
    # DataFrame reflects actual fantasy points with star doubling.
    best_full.loc[best_full["star"] == 1, "points_earned"] *= 2

    return best_full.drop(columns=["_locked"])
