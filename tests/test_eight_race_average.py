"""
Tests for GridRival eight-race average calculation.

Run with:  pytest tests/test_eight_race_average.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

from gr_analytics import (
    calculate_eight_race_averages,
    driver_data,
    eight_race_average,
    race_results,
)

_TESTS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Pure function tests (hand-calculated)
# ---------------------------------------------------------------------------


class TestEightRaceAverage:
    def test_no_races_returns_seed(self):
        assert eight_race_average(5, []) == 5

    def test_one_race_hand_calculated(self):
        # VER, Australia 2026: seed 1, finished P6
        # (7*1 + 6) / 8 = 1.625 -> ceil -> 2
        assert eight_race_average(1, [6]) == 2

    def test_uses_ceiling_not_rounding(self):
        # NOR, Australia 2026: seed 3, finished P5
        # (7*3 + 5) / 8 = 3.25 -> ceil -> 4 (rounding would give 3)
        assert eight_race_average(3, [5]) == 4

    def test_exact_integer_unchanged(self):
        # GAS, Australia 2026: seed 10, finished P10
        # (7*10 + 10) / 8 = 10.0 exactly
        assert eight_race_average(10, [10]) == 10

    def test_two_races_hand_calculated(self):
        # LIN, 2026: seed 19, finished P8 then P12
        # (6*19 + 8 + 12) / 8 = 16.75 -> ceil -> 17
        assert eight_race_average(19, [8, 12]) == 17

    def test_seed_fully_rolls_off_after_eight_races(self):
        # After 8 races the seed should not matter at all
        positions = [1, 2, 3, 4, 5, 6, 7, 8]  # mean 4.5 -> ceil 5
        assert eight_race_average(22, positions) == 5
        assert eight_race_average(1, positions) == 5

    def test_only_last_eight_races_count(self):
        # First race (P22) falls out of the window on race 9
        positions = [22] + [4] * 8
        assert eight_race_average(10, positions) == 4


# ---------------------------------------------------------------------------
# Agreement with GridRival's displayed values (driver_data.csv)
# ---------------------------------------------------------------------------


def _rounds_with_recorded_averages():
    """Rounds whose driver rows have GridRival eight_race_average values.

    Later rounds may be entered with only salary/points (eight_race_average
    left blank until the corresponding race_results are added), so we only
    check rounds where GridRival's own values exist to compare against.
    """
    dd = driver_data()
    drivers = dd[dd["type"] == "driver"]
    have_values = drivers.groupby("round")["eight_race_average"].apply(
        lambda s: s.notna().all()
    )
    return sorted(have_values[have_values].index)


class TestAgreementWithGridRival:
    @pytest.mark.parametrize("rnd", _rounds_with_recorded_averages())
    def test_matches_driver_data_sheet(self, rnd):
        """Computed averages must equal GridRival's values for every round."""
        dd = driver_data()
        sheet = (
            dd[(dd["type"] == "driver") & (dd["round"] == rnd)]
            .set_index("driver_abbr")["eight_race_average"]
            .sort_index()
        )
        computed = calculate_eight_race_averages(through_round=rnd).sort_index()

        mismatches = sheet[sheet != computed]
        assert mismatches.empty, (
            f"Round {rnd} mismatches (sheet vs computed):\n"
            f"{pd.DataFrame({'sheet': mismatches, 'computed': computed[mismatches.index]})}"
        )

    def test_default_round_is_latest(self):
        latest = race_results()["round"].max()
        pd.testing.assert_series_equal(
            calculate_eight_race_averages(),
            calculate_eight_race_averages(through_round=latest),
        )


# ---------------------------------------------------------------------------
# race_results.csv data integrity
# ---------------------------------------------------------------------------


class TestRaceResultsData:
    def test_positions_complete_each_round(self):
        rr = race_results()
        for rnd, group in rr.groupby("round"):
            positions = sorted(group["finishing_position"])
            assert positions == list(range(1, len(group) + 1)), (
                f"Round {rnd} positions are not a complete 1..n sequence"
            )

    def test_round_one_matches_australia_fixture(self):
        """Race 1 positions must agree with the Australia scenario fixture."""
        australia = pd.read_csv(_TESTS_DIR / "test_australia.csv").set_index(
            "driver_abbr"
        )
        rr = race_results()
        round_one = rr[rr["round"] == 1].set_index("driver_abbr")
        for abbr, expected in australia["finishing_position"].items():
            assert round_one.at[abbr, "finishing_position"] == expected
