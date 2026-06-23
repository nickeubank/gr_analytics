#!/usr/bin/env python
"""
Fetch official F1 finishing positions via FastF1 and write them into the
``finishing_position`` column of ``gr_analytics/data/driver_data.csv``.

Each round-N driver row gets that driver's classified finishing position in
race N, which feeds ``calculate_eight_race_averages()``. Only the
``finishing_position`` field of matching driver rows is touched; every other
byte of the CSV (including its BOM) is preserved.

Usage
-----
    python scripts/update_finishing_positions.py            # 2026 rounds 5 6
    python scripts/update_finishing_positions.py 5 6 7      # 2026 rounds 5,6,7
    python scripts/update_finishing_positions.py --year 2025 1 2
    python scripts/update_finishing_positions.py 5 --dry-run

Requires the optional FastF1 dependency:  pip install "gr_analytics[fastf1]"
"""

import argparse
from pathlib import Path

import gr_analytics
from gr_analytics.get_positions import race_finishing_positions

DRIVER_DATA = Path(gr_analytics.__file__).parent / "data" / "driver_data.csv"


def update_finishing_positions(
    year: int, rounds, path: Path = DRIVER_DATA, dry_run: bool = False
) -> None:
    """Fetch ``rounds`` of ``year`` and write them into ``path``."""
    rounds = sorted(set(rounds))

    raw = path.read_text(encoding="utf-8-sig")
    has_bom = path.read_bytes().startswith(b"\xef\xbb\xbf")
    lines = raw.splitlines()
    header = lines[0].split(",")
    col = {name: i for i, name in enumerate(header)}
    for required in ("type", "driver_abbr", "round", "finishing_position"):
        if required not in col:
            raise SystemExit(f"driver_data.csv is missing a '{required}' column")

    # Fetch every round up front so a network/data error aborts before we write.
    fetched = {rnd: race_finishing_positions(year, rnd) for rnd in rounds}

    target = set(rounds)
    filled = {rnd: set() for rnd in rounds}      # abbrs we wrote a position for
    data_abbrs = {rnd: set() for rnd in rounds}  # driver rows present in the CSV

    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        fields = line.split(",")
        if len(fields) != len(header):
            raise SystemExit(
                f"Unexpected column count ({len(fields)} != {len(header)}): {line!r}"
            )
        rnd = int(fields[col["round"]])
        if fields[col["type"]] == "driver" and rnd in target:
            abbr = fields[col["driver_abbr"]]
            data_abbrs[rnd].add(abbr)
            positions = fetched[rnd]
            if abbr in positions.index:
                fields[col["finishing_position"]] = str(int(positions[abbr]))
                filled[rnd].add(abbr)
        out.append(",".join(fields))

    # --- report + validate every requested round -------------------------
    problems = []
    for rnd in rounds:
        positions = fetched[rnd]
        ff1_abbrs = set(positions.index)
        no_data_row = ff1_abbrs - data_abbrs[rnd]   # FastF1 code, no CSV row
        not_in_ff1 = data_abbrs[rnd] - ff1_abbrs    # CSV row, FastF1 had no code
        seq = sorted(int(positions[a]) for a in filled[rnd])
        complete = seq == list(range(1, len(seq) + 1))

        print(f"round {rnd}: filled {len(filled[rnd])}/{len(data_abbrs[rnd])} driver rows")
        if no_data_row:
            problems.append(f"round {rnd}: FastF1 codes with no driver row: {sorted(no_data_row)}")
        if not_in_ff1:
            problems.append(f"round {rnd}: driver rows FastF1 didn't return: {sorted(not_in_ff1)}")
        if not complete:
            problems.append(f"round {rnd}: positions are not a complete 1..n sequence: {seq}")

    if problems:
        print("\nWARNING:")
        for p in problems:
            print("  - " + p)

    if dry_run:
        print("\n(dry run — no file written)")
        return

    path.write_text("\n".join(out) + "\n", encoding="utf-8-sig" if has_bom else "utf-8")
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rounds", nargs="*", type=int, default=[5, 6],
        help="round numbers to fetch (default: 5 6)",
    )
    parser.add_argument("--year", type=int, default=2026, help="season (default: 2026)")
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch and report without writing"
    )
    args = parser.parse_args()
    update_finishing_positions(args.year, args.rounds or [5, 6], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
