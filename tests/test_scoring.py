"""
Manual verification of scoring logic against hand-calculated values.

Run with:  pytest tests/test_scoring.py -v
       or:  python tests/test_scoring.py
"""

from pathlib import Path

import pandas as pd
import pytest

from gr_analytics import _score_constructors, _score_drivers, score_event

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
        ("RUS", 164, 29.6, 0.9),
        ("HAM", 156, 22.1, 1.2),
        ("ANT", 161, 25.9, 1.1),
        ("BEA", 172, 11.2, 2.0),
        ("COL", 108, 6.7, 2.0),
        ("VER", 151, 28.2, -1.8),
        ("LIN", 163, 6.6, 2),
        ("PIA", 42, 24.1, -2),
        ("FER", 161, 23.7, 1.2),
        ("MER", 177, 28.8, 0.3),
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
