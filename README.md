# gr_analytics

![Tests](https://github.com/nickeubank/gr_analytics/actions/workflows/tests.yml/badge.svg)

Python package for scoring and salary calculation in [GridRival](https://www.gridrival.com/) fantasy F1.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd
from gr_analytics import score_event, score_my_team, optimal_lineup

# Load your race scenario
scenario = pd.read_csv("my_race.csv")

# Score the event (defaults to the latest round in driver_data)
result = score_event(scenario)

# Or score a specific race: pass the round one less than the race
# (round=1 is the post-race-1 state, so this scores race 2). See "Driver Data".
result = score_event(scenario, round=1)

# Score your specific team selection
points, salary_change = score_my_team(
    scenario,
    drivers=["RUS", "ANT", "LEC", "BEA", "LIN"],
    team="MER",
    star_driver="BEA",
    round=1,
)

# Find the optimal lineup (maximise points, £100M budget)
lineup = optimal_lineup(result)

# With locked-in drivers (already under contract, cost nothing)
# and a budget for the remaining open spots
lineup = optimal_lineup(
    result,
    locked_in=["HAM", "LEC"],   # driver_abbr or team code
    optimize_for="points",       # or "salary_change"
    budget=60.0,                 # £M available for non-locked picks
)
```

## Scenario Format

The scenario DataFrame must have one row per driver with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `driver_abbr` | str | Driver abbreviation (e.g. `"RUS"`, `"VER"`) |
| `qualifying_position` | int | Official qualifying position (1–22) |
| `completed_qualifying` | int | `1` if driver completed qualifying, `0` if DNQ |
| `finishing_position` | int | Race finishing position (1–22) |
| `completed_pct` | float or `"DNS"` | Fraction of race completed (0.0–1.0), or `"DNS"` if driver did not start |

**DNQ drivers** get 0 qualifying points but their `qualifying_position` is still used to calculate overtake points.

**DNS drivers** (`completed_pct="DNS"`) get 0 for all race-related points (race, overtake, improvement, completion, teammate).

**DNF drivers** (e.g. `completed_pct=0.3`) receive partial completion bonus points and their finishing position is used normally.

Qualifying and race positions must each form a complete sequence `1..n` with no duplicates or gaps.

## Output

`score_event` returns a DataFrame with all drivers and constructors, with scoring columns appended:

**Drivers:**

- `pts_qualifying`, `pts_race`, `pts_overtake`, `pts_improvement`, `pts_completion`, `pts_teammate`
- `points_earned` — total fantasy points
- `salary_after_event`, `salary_change`

**Constructors:**

- `pts_qualifying`, `pts_race` (sum across both drivers, using constructor-specific tables)
- `points_earned`, `salary_after_event`, `salary_change`

## Scoring Rules

All scoring follows GridRival's rules for Grand Prix events (no sprint races).

### Drivers

| Bonus | Rule |
|-------|------|
| Qualifying | P1=50, P2=48, … P22=8 (step −2) |
| Race finish | P1=100, P2=97, … P22=37 (step −3) |
| Overtake | (qualifying pos − finishing pos) × 3, min 0 |
| Improvement | Points for finishing ahead of 8-race average (2 pos=2 pts, 3=4, 4=6, 5=9, 6=12, 7=16, 8=20, 9=25, 10+=30) |
| Completion | 3 pts each at 25%, 50%, 75%, 90% of race distance (max 12) |
| Teammate | Points for beating teammate by margin: ≥1 pos=2 pts, ≥4=5, ≥8=8, ≥13=12 |

### Constructors

Constructor qualifying and race points use separate tables (P1=30/60, step −1/−2 per driver) summed across both drivers. No overtake, improvement, completion, or teammate bonuses.

### Salary Adjustment

After each race, salaries adjust based on the gap between a driver's actual starting salary and the default salary for their points-ranking position:

```
adjustment = truncate(variation / 4, to nearest £100K)
```

Capped at ±£2M for drivers, ±£3M for constructors.

## Driver Data

Bundled driver data (`gr_analytics/data/driver_data.csv`) contains starting salaries and 8-race averages by round:

- `round=0` — pre-season (before Australia 2026)
- `round=1` — post-Australia 2026

Each `round=N` row holds the state **after** race N: the post-race salaries,
the finishing position from race N, and the 8-race average including race N.

**Round semantics for `score_event`:** the `round` argument is that
post-race state, so `score_event(scenario, round=N)` scores the race that
*follows* it. Pass the round **one less** than the race you're scoring:
`round=0` scores race 1 (off the pre-season seeds), `round=1` scores race 2,
and so on. Improvement points therefore compare each finish to the 8-race
average *going into* that race (`calculate_eight_race_averages(through_round=round)`).

## Eight-Race Average

GridRival's "8 race average" can be computed instead of entered by hand.
GridRival seeds the season with 8 slots holding a hard-coded initial
average (the `round=0` values in driver_data); each race replaces one
slot with the driver's classified finishing position, and the displayed
value is the **ceiling** of the slot mean. Each driver's finishing
position per race is recorded in the `finishing_position` column of
`driver_data.csv` (the `round=N` row holds the finish from race N), so
the average is derived entirely from `driver_data`.

```python
from gr_analytics import calculate_eight_race_averages, eight_race_average

# All drivers, after the latest round with recorded finishing positions
calculate_eight_race_averages()

# All drivers, after round 2
calculate_eight_race_averages(through_round=2)

# Single driver from scratch: seed 1, finished P6 then P16
eight_race_average(1, [6, 16])
```

This reproduces GridRival's displayed values exactly for all rounds so
far (verified in `tests/test_eight_race_average.py`).

`score_event` uses this computed average for improvement points (the value
*going into* the race, i.e. `calculate_eight_race_averages(through_round=round)`)
rather than the stored `eight_race_average` column. That column is kept only as
GridRival's transcribed ground-truth/test oracle and may be blank for recent
rounds; scoring no longer depends on it.

## Lineup Optimisation

`optimal_lineup` uses mixed-integer linear programming (via `scipy.optimize.milp`) to find the best 5-driver + 1-constructor lineup within a salary budget.

```python
lineup = optimal_lineup(
    scored,                  # DataFrame from score_event()
    locked_in=None,          # list of driver_abbr / team codes already on your team
    optimize_for="points",   # "points" or "salary_change"
    budget=100.0,            # £M for non-locked picks
)
```

- **`locked_in`** picks are included free (already under contract) and don't count against the budget.
- **`optimize_for="points"`** selects the optimal star driver (who earns double points) across all candidates.
- **`optimize_for="salary_change"`** maximises total salary gain for the next race's team valuation.

The returned DataFrame has a `star` column (`1` = starred driver, points mode only).

## Running Tests

```bash
python -m pytest tests/test_scoring.py -v
```

Tests include hand-calculated unit tests for all scoring components and a full integration test against real GridRival results from Australia 2026 (22 drivers + 3 constructors).
