"""
Manual verification of scoring logic against hand-calculated values.

Run with:  pytest tests/test_scoring.py -v
       or:  python tests/test_scoring.py
"""

from pathlib import Path

import pandas as pd
import pytest

from gr_analytics import (
    _score_constructors,
    _score_drivers,
    calculate_eight_race_averages,
    driver_data,
    optimal_lineup,
    score_event,
)

_TESTS_DIR = Path(__file__).parent


def _score_full(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: run the full scoring pipeline on a raw DataFrame (old-style interface)."""
    drivers = df[df["type"] == "driver"].copy()
    constructors = df[df["type"] == "team"].copy()
    scored_drivers = _score_drivers(drivers)
    scored_constructors = _score_constructors(constructors, drivers)
    return pd.concat([scored_drivers, scored_constructors])


# ---------------------------------------------------------------------------
# Shared fixture: 2 teams x 2 drivers + 2 constructors
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_race():
    """
    Two teams, four drivers, two constructors.

    Team Mercedes: Hamilton (P5 -> P2), Russell (P3 -> P6)
    Team RedBull:  Verstappen (P1 -> P1), Perez (P4 -> P4)
    """
    return pd.DataFrame(
        {
            "type": ["driver", "driver", "driver", "driver", "team", "team"],
            "driver_name": [
                "Hamilton",
                "Russell",
                "Verstappen",
                "Perez",
                "Mercedes",
                "RedBull",
            ],
            "driver_team": ["Mercedes", "Mercedes", "RedBull", "RedBull", "", ""],
            "eight_race_average": [4.0, 5.5, 1.5, 4.5, 1.0, 3.0],
            "starting_salary": [26.0, 24.4, 34.0, 22.8, 28.0, 25.0],
            "qualifying_position": [5, 3, 1, 4, None, None],
            "finishing_position": [2, 6, 1, 4, None, None],
        }
    )


# ---------------------------------------------------------------------------
# Driver points tests (hand-calculated)
# ---------------------------------------------------------------------------


class TestDriverPoints:
    """
    Hamilton (qual=5, finish=2, 8ra=4.0):
      qual=42, race=97, overtakes=(5-2)*3=9, improvement=floor(4-2)=2->2pts,
      completion=12, teammate: beats Russell by 4 positions -> 5pts
      Total = 167

    Russell (qual=3, finish=6, 8ra=5.5):
      qual=46, race=85, overtakes=0, improvement=floor(5.5-6)=-1->0pts,
      completion=12, teammate: loses -> 0pts
      Total = 143

    Verstappen (qual=1, finish=1, 8ra=1.5):
      qual=50, race=100, overtakes=0, improvement=floor(1.5-1)=0->0pts,
      completion=12, teammate: beats Perez by 3 positions -> 2pts
      Total = 164

    Perez (qual=4, finish=4, 8ra=4.5):
      qual=44, race=91, overtakes=0, improvement=floor(4.5-4)=0->0pts,
      completion=12, teammate: loses -> 0pts
      Total = 147
    """

    def test_hamilton_points(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Hamilton", "points_earned"].iloc[0] == 167
        )

    def test_russell_points(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Russell", "points_earned"].iloc[0] == 143
        )

    def test_verstappen_points(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Verstappen", "points_earned"].iloc[0]
            == 164
        )

    def test_perez_points(self, basic_race):
        result = _score_full(basic_race)
        assert result.loc[result.driver_name == "Perez", "points_earned"].iloc[0] == 147


# ---------------------------------------------------------------------------
# Constructor points tests (hand-calculated)
# ---------------------------------------------------------------------------


class TestConstructorPoints:
    """
    Mercedes = Hamilton (P5 qual, P2 finish) + Russell (P3 qual, P6 finish)
      Hamilton con_qual = 31-5=26, con_race = 62-2*2=58
      Russell  con_qual = 31-3=28, con_race = 62-2*6=50
      pts_qualifying = 26+28 = 54
      pts_race       = 58+50 = 108
      Total = 162

    RedBull = Verstappen (P1 qual, P1 finish) + Perez (P4 qual, P4 finish)
      Verstappen con_qual = 31-1=30, con_race = 62-2*1=60
      Perez      con_qual = 31-4=27, con_race = 62-2*4=54
      pts_qualifying = 30+27 = 57
      pts_race       = 60+54 = 114
      Total = 171
    """

    def test_mercedes_points(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Mercedes", "points_earned"].iloc[0] == 162
        )

    def test_redbull_points(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "RedBull", "points_earned"].iloc[0] == 171
        )

    def test_constructor_qualifying_component(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "RedBull", "pts_qualifying"].iloc[0] == 57
        )

    def test_constructor_race_component(self, basic_race):
        result = _score_full(basic_race)
        assert result.loc[result.driver_name == "RedBull", "pts_race"].iloc[0] == 114


# ---------------------------------------------------------------------------
# Constructor salary tests (hand-calculated)
# ---------------------------------------------------------------------------


class TestConstructorSalary:
    """
    Points ranking among constructors: RedBull(171)=1st, Mercedes(162)=2nd

    RedBull:  default[1]=30.0, starting=25.0, variation=+5.0,
              adj=truncate(1.25 to 0.1)=+1.2 -> 26.2
    Mercedes: default[2]=27.4, starting=28.0, variation=-0.6,
              adj=truncate(-0.15 to 0.1)=-0.1 -> 27.9
    """

    def test_redbull_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "RedBull", "salary_after_event"].iloc[0]
            == 26.2
        )

    def test_mercedes_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Mercedes", "salary_after_event"].iloc[0]
            == 27.9
        )

    def test_constructor_salary_cap_at_3m(self):
        """
        Constructor at minimum salary (£4.0M) scoring best should be capped at +£3M,
        not +£2M like drivers.

        Alpha (P1+P2 qual, P1+P2 finish), starting £4.0M:
          con_qual = (31-1)+(31-2) = 59
          con_race = (62-2)+(62-4) = 118
          total = 177, ranks 1st -> default = £30.0M
          variation = +26.0M, raw adj = +6.5M -> capped at +3.0M
          new salary = 4.0 + 3.0 = 7.0
        """
        df = pd.DataFrame(
            {
                "type": ["driver", "driver", "driver", "driver", "team", "team"],
                "driver_name": ["D1", "D2", "D3", "D4", "Alpha", "Beta"],
                "driver_team": ["Alpha", "Alpha", "Beta", "Beta", "", ""],
                "eight_race_average": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                "starting_salary": [10.0, 10.0, 10.0, 10.0, 4.0, 30.0],
                "qualifying_position": [1, 2, 3, 4, None, None],
                "finishing_position": [1, 2, 3, 4, None, None],
            }
        )
        result = _score_full(df)
        assert (
            result.loc[result.driver_name == "Alpha", "salary_after_event"].iloc[0]
            == 7.0
        )


# ---------------------------------------------------------------------------
# Driver salary tests (hand-calculated)
# ---------------------------------------------------------------------------


class TestDriverSalary:
    """
    Points ranking: Hamilton(167)=1st, Verstappen(164)=2nd,
                    Perez(147)=3rd, Russell(143)=4th

    Hamilton:   default[1]=34.0, variation=+8.0, adj=+2.0 (capped) -> 28.0
    Verstappen: default[2]=32.4, variation=-1.6, adj=-0.4           -> 33.6
    Perez:      default[3]=30.8, variation=+8.0, adj=+2.0 (capped)  -> 24.8
    Russell:    default[4]=29.2, variation=+4.8, adj=+1.2           -> 25.6
    """

    def test_hamilton_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Hamilton", "salary_after_event"].iloc[0]
            == 28.0
        )

    def test_verstappen_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Verstappen", "salary_after_event"].iloc[0]
            == 33.6
        )

    def test_perez_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Perez", "salary_after_event"].iloc[0]
            == 24.8
        )

    def test_russell_salary(self, basic_race):
        result = _score_full(basic_race)
        assert (
            result.loc[result.driver_name == "Russell", "salary_after_event"].iloc[0]
            == 25.6
        )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_driver_output_columns(self, basic_race):
        result = _score_full(basic_race)
        for col in [
            "pts_qualifying",
            "pts_race",
            "pts_overtake",
            "pts_improvement",
            "pts_completion",
            "pts_teammate",
            "points_earned",
            "salary_after_event",
        ]:
            assert col in result.columns

    def test_no_internal_columns_leaked(self, basic_race):
        result = _score_full(basic_race)
        assert not any(c.startswith("_") for c in result.columns)

    def test_input_unchanged(self, basic_race):
        original = basic_race.copy()
        _score_full(basic_race)
        pd.testing.assert_frame_equal(basic_race, original)

    def test_wrong_team_size_raises(self):
        df = pd.DataFrame(
            {
                "type": ["driver"],
                "driver_name": ["OnlyDriver"],
                "driver_team": ["Lonely"],
                "eight_race_average": [5.0],
                "starting_salary": [20.0],
                "qualifying_position": [5],
                "finishing_position": [5],
            }
        )
        with pytest.raises(ValueError, match="expected 2"):
            _score_full(df)

    def test_big_improvement(self):
        """Driver massively overperforming their average caps at 30 improvement pts."""
        df = pd.DataFrame(
            {
                "type": ["driver", "driver"],
                "driver_name": ["Backmarker", "Teammate"],
                "driver_team": ["Slow", "Slow"],
                "eight_race_average": [20.0, 18.0],
                "starting_salary": [5.0, 6.0],
                "qualifying_position": [20, 19],
                "finishing_position": [8, 10],
            }
        )
        result = _score_full(df)
        # qual=12, race=79, overtakes=(20-8)*3=36, improvement=30, completion=12, teammate(margin=2)=2
        assert (
            result.loc[result.driver_name == "Backmarker", "points_earned"].iloc[0]
            == 171
        )

    def test_overtake_clamps_at_zero(self, basic_race):
        """Russell went backwards (P3 qual -> P6 finish): no negative overtake pts."""
        result = _score_full(basic_race)
        assert result.loc[result.driver_name == "Russell", "pts_overtake"].iloc[0] == 0

    def test_score_event_unknown_driver_abbr_raises(self):
        """score_event raises ValueError when scenario contains a driver_abbr not in driver_data."""
        scenario = pd.read_csv(_TESTS_DIR / "test_australia.csv")
        # Replace one valid abbreviation with a bogus one
        scenario.loc[scenario["driver_abbr"] == "VER", "driver_abbr"] = "ZZZ"
        with pytest.raises(ValueError, match="driver_abbr mismatch") as exc_info:
            score_event(scenario, round=0)
        # Both directions should be reported
        msg = str(exc_info.value)
        assert "ZZZ" in msg, "error should mention the unknown abbreviation"
        assert "VER" in msg, "error should mention the missing driver from driver_data"


# ---------------------------------------------------------------------------
# Australia 2026 round-0 integration test
# ---------------------------------------------------------------------------


@pytest.fixture
def australia_result():
    scenario = pd.read_csv(_TESTS_DIR / "test_australia.csv")
    return score_event(scenario, round=0)


def _get_row(result, abbr):
    """Look up a row by driver_abbr (drivers) or driver_name (teams)."""
    driver_row = result[result["driver_abbr"] == abbr]
    if not driver_row.empty:
        return driver_row.iloc[0]
    team_row = result[(result["type"] == "team") & (result["driver_name"] == abbr)]
    return team_row.iloc[0]


@pytest.mark.parametrize(
    "abbr,exp_pts,exp_salary,exp_change",
    [
        # Drivers (from GridRival Australia 2026 results)
        ("RUS", 164, 29.6, 0.9),
        ("VER", 151, 28.2, -1.8),
        ("NOR", 155, 26.7, -0.7),
        ("ANT", 161, 25.9, 1.1),
        ("LEC", 159, 24.5, 1.0),
        ("PIA", 42, 24.1, -2.0),
        ("HAM", 156, 22.1, 1.2),
        ("ALO", 70, 20.2, -2.0),
        ("GAS", 126, 18.3, 0.0),
        ("HAD", 89, 17.6, -2.0),
        ("STR", 78, 15.0, -2.0),
        ("LAW", 112, 14.5, 0.1),
        ("SAI", 88, 13.9, -1.8),
        ("ALB", 114, 13.9, 0.8),
        ("BOR", 151, 12.5, 2.0),
        ("BEA", 172, 11.2, 2.0),
        ("OCO", 130, 9.9, 2.0),
        ("HUL", 30, 9.8, -2.0),
        ("COL", 108, 6.7, 2.0),
        ("LIN", 163, 6.6, 2.0),
        ("PER", 95, 6.4, 1.7),
        ("BOT", 63, 4.5, -0.2),
        # Constructors
        ("MER", 177, 28.8, 0.3),
        ("FER", 161, 23.7, 1.2),
        ("HAS", 125, 8.0, 3.0),
    ],
)
class TestAustraliaRound0:
    def test_points_earned(
        self, australia_result, abbr, exp_pts, exp_salary, exp_change
    ):
        assert _get_row(australia_result, abbr)["points_earned"] == exp_pts

    def test_salary_after_event(
        self, australia_result, abbr, exp_pts, exp_salary, exp_change
    ):
        assert _get_row(australia_result, abbr)["salary_after_event"] == exp_salary

    def test_salary_change(
        self, australia_result, abbr, exp_pts, exp_salary, exp_change
    ):
        assert _get_row(australia_result, abbr)["salary_change"] == exp_change


# ---------------------------------------------------------------------------
# score_event derives the eight-race average from driver_data
# ---------------------------------------------------------------------------


class TestScoreEventEightRaceAverage:
    """score_event computes the eight-race average from the recorded
    finishing positions (calculate_eight_race_averages) rather than trusting
    the hard-coded eight_race_average column. So it works for rounds whose
    GridRival value was never transcribed (the column is blank), and uses the
    average going *into* the race being scored (through_round == round)."""

    @staticmethod
    def _scenario_from_finishes(rnd):
        d = driver_data()
        d = d[(d["type"] == "driver") & (d["round"] == rnd)][
            ["driver_abbr", "finishing_position"]
        ].copy()
        d["finishing_position"] = d["finishing_position"].astype(int)
        return pd.DataFrame(
            {
                "driver_abbr": d["driver_abbr"],
                "qualifying_position": d["finishing_position"],
                "race_position": d["finishing_position"],
            }
        )

    @pytest.mark.parametrize("rnd", [1, 2, 3, 4, 5, 6, 7])
    def test_uses_average_through_round(self, rnd):
        """The average score_event applies equals calculate_eight_race_averages
        through that round — the state going into the race scored at `round`."""
        result = score_event(self._scenario_from_finishes(rnd), round=rnd)
        used = (
            result[result["type"] == "driver"]
            .set_index("driver_abbr")["eight_race_average"]
            .astype(int)
            .sort_index()
        )
        expected = (
            calculate_eight_race_averages(through_round=rnd).astype(int).sort_index()
        )
        pd.testing.assert_series_equal(used, expected, check_names=False)

    @pytest.mark.parametrize("rnd", [5, 6, 7])
    def test_blank_column_rounds_score_finite(self, rnd):
        """Rounds whose eight_race_average column is blank must still score —
        before the average was computed, this raised on NaN improvement."""
        result = score_event(self._scenario_from_finishes(rnd), round=rnd)
        drivers = result[result["type"] == "driver"]
        assert drivers["pts_improvement"].notna().all()
        assert drivers["points_earned"].notna().all()


# ---------------------------------------------------------------------------
# optimal_lineup star_salary_cap tests
# ---------------------------------------------------------------------------


@pytest.fixture
def scored_pool():
    """
    Five drivers + one constructor, total salary exactly 100 (at budget).

    D1: salary=30, points=200  (highest points, but salary > 19 cap)
    D2: salary=25, points=180  (second best, but salary > 19 cap)
    D3: salary=15, points=150  (best eligible star under default cap=19)
    D4: salary=10, points=120
    D5: salary= 8, points=100
    TEAM: salary=12, points=160

    With cap=19 star is D3 → total = 200+180+(2*150)+120+100+160 = 1060
    Without cap   star is D1 → total = (2*200)+180+150+120+100+160 = 1110
    """
    return pd.DataFrame(
        {
            "type": ["driver", "driver", "driver", "driver", "driver", "team"],
            "driver_abbr": ["D1", "D2", "D3", "D4", "D5", None],
            "driver_name": ["D1", "D2", "D3", "D4", "D5", "TEAM"],
            "starting_salary": [30.0, 25.0, 15.0, 10.0, 8.0, 12.0],
            "points_earned": [200.0, 180.0, 150.0, 120.0, 100.0, 160.0],
            "salary_change": [2.0, 1.5, 1.0, 0.5, 0.0, 1.0],
        }
    )


class TestOptimalLineupStarCap:

    def test_default_cap_star_salary_at_most_19(self, scored_pool):
        """Default cap=19 means the starred driver has salary ≤ 19."""
        result = optimal_lineup(scored_pool)
        star_row = result[result["star"] == 1]
        assert len(star_row) == 1
        assert star_row.iloc[0]["starting_salary"] <= 19.0

    def test_default_cap_best_eligible_star_is_d3(self, scored_pool):
        """Under cap=19 the optimal star is D3 (highest points among ≤19 drivers)."""
        result = optimal_lineup(scored_pool)
        star_row = result[result["star"] == 1]
        assert star_row.iloc[0]["driver_abbr"] == "D3"

    def test_no_cap_best_star_is_d1(self, scored_pool):
        """With star_salary_cap=None, D1 (most points) becomes the star."""
        result = optimal_lineup(scored_pool, star_salary_cap=None)
        star_row = result[result["star"] == 1]
        assert star_row.iloc[0]["driver_abbr"] == "D1"

    def test_custom_cap_respected(self, scored_pool):
        """star_salary_cap=25 excludes D1 (30) but allows D2 (25)."""
        result = optimal_lineup(scored_pool, star_salary_cap=25.0)
        star_row = result[result["star"] == 1]
        assert star_row.iloc[0]["starting_salary"] <= 25.0
        assert star_row.iloc[0]["driver_abbr"] == "D2"

    def test_locked_in_high_salary_not_starred_under_cap(self, scored_pool):
        """A locked-in driver with salary > cap should not be the star."""
        # Lock in D1 (salary=30) — it's in the lineup but cap=19 blocks it as star
        result = optimal_lineup(scored_pool, locked_in=["D1"])
        star_rows = result[result["star"] == 1]
        # Either no star or the star is not D1
        if not star_rows.empty:
            assert star_rows.iloc[0]["driver_abbr"] != "D1"

    def test_locked_in_low_salary_can_be_star(self):
        """A locked-in driver under the cap is starred when no free driver qualifies."""
        # All free drivers have salary > 19, so the only star candidate is locked D3 (salary=15)
        pool = pd.DataFrame(
            {
                "type": ["driver", "driver", "driver", "driver", "driver", "team"],
                "driver_abbr": ["D1", "D2", "D3", "D4", "D5", None],
                "driver_name": ["D1", "D2", "D3", "D4", "D5", "TEAM"],
                "starting_salary": [25.0, 22.0, 15.0, 21.0, 20.0, 12.0],
                "points_earned": [200.0, 180.0, 150.0, 120.0, 100.0, 160.0],
                "salary_change": [2.0, 1.5, 1.0, 0.5, 0.0, 1.0],
            }
        )
        result = optimal_lineup(pool, locked_in=["D3"])
        star_row = result[result["star"] == 1]
        assert star_row.iloc[0]["driver_abbr"] == "D3"

    def test_budget_constraint_drops_expensive_driver(self):
        """With budget=120 the solver must skip drivere (too expensive) and pick driverf."""
        df = pd.DataFrame(
            {
                "type": [
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "team",
                ],
                "driver_abbr": [
                    "drivera",
                    "driverb",
                    "driverc",
                    "driverd",
                    "drivere",
                    "driverf",
                    "has",
                ],
                "driver_name": [
                    "drivera",
                    "driverb",
                    "driverc",
                    "driverd",
                    "drivere",
                    "driverf",
                    "has",
                ],
                "points_earned": [30, 30, 30, 30, 10, 7, 30],
                "starting_salary": [20, 20, 20, 20, 20, 15, 1],
                "salary_change": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = optimal_lineup(
            df, optimize_for="points", budget=120, star_salary_cap=19.0
        )
        picked = set(result["driver_abbr"].dropna().tolist()) | set(
            result.loc[result["type"] == "team", "driver_name"].tolist()
        )
        assert picked == {"drivera", "driverb", "driverc", "driverd", "driverf", "has"}


# ---------------------------------------------------------------------------
# locked_out tests
# ---------------------------------------------------------------------------


class TestLockedOut:

    def test_locked_out_excludes_high_value_driver(self):
        """A driver worth 999 points at cost 1 is excluded when locked_out."""
        df = pd.DataFrame(
            {
                "type": [
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "team",
                ],
                "driver_abbr": ["GOD", "D1", "D2", "D3", "D4", "D5", None],
                "driver_name": ["GOD", "D1", "D2", "D3", "D4", "D5", "TEAM"],
                "starting_salary": [1.0, 15.0, 15.0, 15.0, 15.0, 15.0, 10.0],
                "points_earned": [999.0, 50.0, 40.0, 30.0, 20.0, 10.0, 60.0],
                "salary_change": [9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        result = optimal_lineup(df, locked_out=["GOD"], star_salary_cap=None)
        picked_abbrs = set(result["driver_abbr"].dropna())
        assert "GOD" not in picked_abbrs

    def test_locked_out_team_excluded(self):
        """A constructor is excluded when its team code is in locked_out."""
        df = pd.DataFrame(
            {
                "type": [
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "team",
                    "team",
                ],
                "driver_abbr": ["D1", "D2", "D3", "D4", "D5", None, None],
                "driver_name": ["D1", "D2", "D3", "D4", "D5", "GOOD", "BAD"],
                "starting_salary": [15.0, 15.0, 15.0, 15.0, 15.0, 10.0, 1.0],
                "points_earned": [50.0, 40.0, 30.0, 20.0, 10.0, 60.0, 999.0],
                "salary_change": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0],
            }
        )
        result = optimal_lineup(df, locked_out=["BAD"], star_salary_cap=None)
        picked_teams = set(result.loc[result["type"] == "team", "driver_name"])
        assert "BAD" not in picked_teams


# ---------------------------------------------------------------------------
# optimize_for balance (float) tests
# ---------------------------------------------------------------------------


@pytest.fixture
def balance_pool():
    """
    Pool where points-optimal and salary_change-optimal lineups differ.

    D1: high points, low salary_change
    D2: low points, high salary_change
    D3–D5: filler
    """
    return pd.DataFrame(
        {
            "type": ["driver", "driver", "driver", "driver", "driver", "team"],
            "driver_abbr": ["D1", "D2", "D3", "D4", "D5", None],
            "driver_name": ["D1", "D2", "D3", "D4", "D5", "TEAM"],
            "starting_salary": [18.0, 18.0, 10.0, 10.0, 10.0, 10.0],
            "points_earned": [200.0, 50.0, 100.0, 100.0, 100.0, 100.0],
            "salary_change": [0.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


class TestOptimizeForBalance:

    def test_balance_1_matches_points(self, balance_pool):
        """optimize_for=1.0 should pick same drivers as optimize_for='points'."""
        result_pts = optimal_lineup(balance_pool, optimize_for="points")
        result_bal = optimal_lineup(balance_pool, optimize_for=1.0)
        pts_abbrs = sorted(result_pts["driver_abbr"].dropna().tolist())
        bal_abbrs = sorted(result_bal["driver_abbr"].dropna().tolist())
        assert bal_abbrs == pts_abbrs

    def test_balance_0_matches_salary_change(self, balance_pool):
        """optimize_for=0.0 should pick same drivers as optimize_for='salary_change'."""
        result_sal = optimal_lineup(balance_pool, optimize_for="salary_change")
        result_bal = optimal_lineup(balance_pool, optimize_for=0.0)
        sal_abbrs = sorted(result_sal["driver_abbr"].dropna().tolist())
        bal_abbrs = sorted(result_bal["driver_abbr"].dropna().tolist())
        assert bal_abbrs == sal_abbrs

    def test_balance_star_assigned_when_balance_positive(self, balance_pool):
        """When optimize_for is a positive float, a star driver should be assigned
        and their points_earned should be doubled."""
        result = optimal_lineup(balance_pool, optimize_for=0.5)
        assert (result["star"] == 1).sum() == 1
        star_row = result[result["star"] == 1].iloc[0]
        orig_pts = balance_pool.loc[
            balance_pool["driver_abbr"] == star_row["driver_abbr"], "points_earned"
        ].iloc[0]
        assert star_row["points_earned"] == orig_pts * 2

    def test_balance_0_star_assigned_to_highest_pts(self, balance_pool):
        """When balance=0, the highest-points eligible driver is still starred
        and their points_earned is doubled."""
        result = optimal_lineup(balance_pool, optimize_for=0.0)
        assert (result["star"] == 1).sum() == 1
        star_row = result[result["star"] == 1].iloc[0]
        # D1 has the highest points (200) in the pool
        assert star_row["driver_abbr"] == "D1"
        assert star_row["points_earned"] == 400.0

    def test_balance_out_of_range_raises(self, balance_pool):
        """optimize_for outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            optimal_lineup(balance_pool, optimize_for=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            optimal_lineup(balance_pool, optimize_for=-0.1)

    def test_balance_midpoint_blends_objectives(self):
        """
        At balance=0.5 the solver should prefer the candidate that best
        combines normalised points and salary change.

        D1: pts=200, sal_change=0 → obj = (200/100)*0.5 + 0*0.5 = 1.0
        D2: pts=50,  sal_change=5 → obj = (50/100)*0.5 + 5*0.5  = 2.75
        With budget=68 only one of D1/D2 fits alongside D3–D6+TEAM.
        D2 has higher blended obj → solver picks D2 over D1.
        """
        df = pd.DataFrame(
            {
                "type": [
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "driver",
                    "team",
                ],
                "driver_abbr": ["D1", "D2", "D3", "D4", "D5", "D6", None],
                "driver_name": ["D1", "D2", "D3", "D4", "D5", "D6", "TEAM"],
                "starting_salary": [18.0, 18.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "points_earned": [200.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "salary_change": [0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        result = optimal_lineup(df, optimize_for=0.5, budget=68.0)
        picked = set(result["driver_abbr"].dropna())
        assert "D2" in picked
        assert "D1" not in picked


# ---------------------------------------------------------------------------
# Pretty-print for manual inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "type": ["driver", "driver", "driver", "driver", "team", "team"],
            "driver_name": [
                "Hamilton",
                "Russell",
                "Verstappen",
                "Perez",
                "Mercedes",
                "RedBull",
            ],
            "driver_team": ["Mercedes", "Mercedes", "RedBull", "RedBull", "", ""],
            "eight_race_average": [4.0, 5.5, 1.5, 4.5, 1.0, 3.0],
            "starting_salary": [26.0, 24.4, 34.0, 22.8, 28.0, 25.0],
            "qualifying_position": [5, 3, 1, 4, None, None],
            "finishing_position": [2, 6, 1, 4, None, None],
        }
    )

    result = _score_full(df)
    print(
        result[
            [
                "type",
                "driver_name",
                "pts_qualifying",
                "pts_race",
                "points_earned",
                "starting_salary",
                "salary_after_event",
            ]
        ].to_string(index=False)
    )
