"""
Fetch official F1 classified finishing positions via FastF1.

These feed the ``finishing_position`` column of ``driver_data.csv`` (the
round-N row holds each driver's finishing position in race N), which
``calculate_eight_race_averages()`` rolls into GridRival's eight-race average.

FastF1 reads the official F1 timing data, so ``Position`` is the *classified*
finishing position — DNF/DNS drivers keep the position they are classified in,
matching GridRival's convention. Driver codes are FastF1's three-letter
``Abbreviation`` values, which are the FIA codes used as ``driver_abbr`` in
driver_data.csv (VER, RUS, ...).

``fastf1`` is an optional dependency (``pip install fastf1``) and is imported
lazily, so importing :mod:`gr_analytics` never requires it. Fetching a session
needs network access; results are cached so repeat calls are offline-fast.

Examples
--------
>>> from gr_analytics.get_positions import race_finishing_positions
>>> race_finishing_positions(2026, 1)          # round 1 (season opener)
driver_abbr
RUS     1
ANT     2
...
Name: finishing_position, dtype: int64

>>> # Long-format rows for new rounds, shaped like driver_data's per-race data
>>> from gr_analytics.get_positions import finishing_positions_frame
>>> finishing_positions_frame(2026, [5, 6])
"""

from pathlib import Path

import pandas as pd

# Cache outside the repo so fetched timing data isn't committed or shipped.
_DEFAULT_CACHE = Path.home() / ".cache" / "fastf1"


def _enable_cache(cache_dir=None) -> None:
    """Point FastF1 at a cache directory, creating it if needed."""
    import fastf1

    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def race_finishing_positions(year: int, rnd: int, cache_dir=None) -> pd.Series:
    """
    Classified finishing positions for a single race.

    Parameters
    ----------
    year : int
        Season, e.g. ``2026``.
    rnd : int
        Championship round number (1 = season opener).
    cache_dir : path-like, optional
        FastF1 cache location. Defaults to ``~/.cache/fastf1``.

    Returns
    -------
    pandas.Series
        Finishing position (int) indexed by three-letter ``driver_abbr``,
        sorted by position. Drivers without a classified position (e.g. DNS
        with no time) are dropped. Series name is ``finishing_position`` to
        match driver_data.csv.
    """
    import fastf1

    _enable_cache(cache_dir)

    session = fastf1.get_session(year, rnd, "R")
    # Results are always loaded; skip the heavy laps/telemetry/weather data.
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    positions = (
        session.results.set_index("Abbreviation")["Position"]
        .dropna()
        .astype(int)
        .sort_values()
    )
    positions.index.name = "driver_abbr"
    positions.name = "finishing_position"
    return positions


def finishing_positions_frame(year: int, rounds, cache_dir=None) -> pd.DataFrame:
    """
    Long-format finishing positions for one or more rounds.

    Parameters
    ----------
    year : int
        Season, e.g. ``2026``.
    rounds : int or iterable of int
        Round number(s) to fetch.
    cache_dir : path-like, optional
        FastF1 cache location. Defaults to ``~/.cache/fastf1``.

    Returns
    -------
    pandas.DataFrame
        Columns ``round``, ``driver_abbr``, ``finishing_position`` — the shape
        of the per-race data recorded in driver_data.csv, handy for filling in
        new rounds.
    """
    if isinstance(rounds, int):
        rounds = [rounds]

    frames = []
    for rnd in rounds:
        frame = race_finishing_positions(year, rnd, cache_dir=cache_dir).reset_index()
        frame.insert(0, "round", rnd)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
