#!/usr/bin/env python3
"""
Parse DEFINITION_DV entries of the form:

DEFINITION_DV= ( WING, 0, 0, 0, 0.0, 0.0, 1.0 ); ( WING, 1, 0, 0, 0.0, 0.0, 1.0 ); ...

and convert them into a list of records.
"""

import re
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class DVEntry:
    marker: str
    i: int
    j: int
    k: int
    dx: float
    dy: float
    dz: float


# Regex to capture things like:
# ( WING, 0, 0, 0, 0.0, 0.0, 1.0 )
DV_PATTERN = re.compile(
    r"""
    \(\s*                 # opening parenthesis
    (?P<marker>[^,]+?)\s*,\s*
    (?P<i>\d+)\s*,\s*
    (?P<j>\d+)\s*,\s*
    (?P<k>\d+)\s*,\s*
    (?P<dx>[-+0-9.eE]+)\s*,\s*
    (?P<dy>[-+0-9.eE]+)\s*,\s*
    (?P<dz>[-+0-9.eE]+)\s*
    \)                    # closing parenthesis
    """,
    re.VERBOSE,
)


def parse_definition_dv(text: str) -> List[DVEntry]:
    """
    Parse all ( MARKER, i, j, k, dx, dy, dz ) entries from a string.
    Returns a list of DVEntry.
    """
    entries: List[DVEntry] = []

    for m in DV_PATTERN.finditer(text):
        marker = m.group("marker").strip()
        i = int(m.group("i"))
        j = int(m.group("j"))
        k = int(m.group("k"))
        dx = float(m.group("dx"))
        dy = float(m.group("dy"))
        dz = float(m.group("dz"))

        entries.append(DVEntry(marker, i, j, k, dx, dy, dz))

    return entries

from typing import List

def remove_range_by_i_for_j(entries, j_target, i_min, i_max):
    """
    Remove all DVEntry items whose j == j_target AND whose i is between
    i_min and i_max (inclusive).

    Parameters
    ----------
    entries : list[DVEntry]
        Original list of DVEntry objects.
    j_target : int
        The j-index at which removal applies.
    i_min : int
        Minimum i-index to remove (inclusive).
    i_max : int
        Maximum i-index to remove (inclusive).

    Returns
    -------
    list[DVEntry]
        Filtered list with those entries removed.
    """
    return [
        e for e in entries
        if not (e.j == j_target and i_min <= e.i <= i_max)
    ]

def dv_entries_to_su2_string(entries):
    """
    Convert a list of DVEntry objects into a single SU2-style
    DEFINITION_DV= (...) ; (...) ; ... string.
    """
    parts = []
    for e in entries:
        parts.append(
            f"( {e.marker}, {e.i}, {e.j}, {e.k}, {e.dx:.1f}, {e.dy:.1f}, {e.dz:.1f} )"
        )
    return "DEFINITION_DV= " + " ; ".join(parts)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r") as f:
        text = f.read()

    # If the file has multiple lines, we don't care which one;
    # we just parse every tuple in the entire file.
    dv_entries = parse_definition_dv(text)

    print(f"Parsed {len(dv_entries)} DV entries before removal.")

    # Remove entries for j = 1
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=1, i_min=0, i_max=1)

    # Remove entries for j = 2
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=2, i_min=0, i_max=3)

    # Remove entries for j = 3
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=3, i_min=0, i_max=6)

    # Remove entries for j = 4
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=4, i_min=0, i_max=7)

    # Remove entries for j = 5
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=5, i_min=0, i_max=8)

    # Remove entries for j = 6
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=6, i_min=0, i_max=9)

    # Remove entries for j = 7
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=7, i_min=0, i_max=9)

    # Remove entries for j = 8
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=8, i_min=0, i_max=9)

    # Remove entries for j = 9
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=9, i_min=0, i_max=9)

    # Remove entries for j = 10
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=10, i_min=0, i_max=9)

    # Remove entries for j = 11
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=11, i_min=0, i_max=7)

    # Remove entries for j = 12
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=12, i_min=0, i_max=9)

    # Remove entries for j = 13
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=13, i_min=0, i_max=8)

    # Remove entries for j = 14
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=14, i_min=0, i_max=10)

    # Remove entries for j = 15
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=15, i_min=0, i_max=11)

    # Remove entries for j = 16
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=16, i_min=0, i_max=11)

    # Remove control points from the TE region


    # Remove entries for j = 0
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=0, i_min=20, i_max=20)

    # Remove entries for j = 1
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=1, i_min=20, i_max=20)

    # Remove entries for j = 2
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=2, i_min=20, i_max=20)

    # Remove entries for j = 3
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=3, i_min=18, i_max=20)

    # Remove entries for j = 4
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=4, i_min=18, i_max=20)

    # Remove entries for j = 5
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=5, i_min=16, i_max=20)

    # Remove entries for j = 6
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=6, i_min=15, i_max=20)

    # Remove entries for j = 7
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=7, i_min=14, i_max=20)

    # Remove entries for j = 8
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=8, i_min=14, i_max=20)

    # Remove entries for j = 9
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=9, i_min=14, i_max=20)

    # Remove entries for j = 10
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=10, i_min=14, i_max=20)

    # Remove entries for j = 11
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=11, i_min=14, i_max=20)

    # Remove entries for j = 12
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=12, i_min=14, i_max=20)

    # Remove entries for j = 13
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=13, i_min=15, i_max=20)

    # Remove entries for j = 14
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=14, i_min=16, i_max=20)

    # Remove entries for j = 15
    dv_entries = remove_range_by_i_for_j(dv_entries, j_target=14, i_min=17, i_max=20)


    print(f"{len(dv_entries)} DV entries after removal.")



    print(f"Parsed {len(dv_entries)} DV entries.")

    su2_output = dv_entries_to_su2_string(dv_entries)

    # Print rewritten DEFINITION_DV line
    print("\n================ REWRITTEN DEFINITION_DV ================\n")
    print(su2_output)
    print("\n=========================================================\n")


    # Print a few as a sanity check
    #for e in dv_entries[:30]:
    #    print(e)

    # If you want to do something more structured, e.g. build a dict:
    # dv_dict = {(e.i, e.j, e.k): e for e in dv_entries}
    # print(dv_dict[(0, 0, 0)])


if __name__ == "__main__":
    main()