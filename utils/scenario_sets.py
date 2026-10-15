"""
Scenario sets used to tune and evaluate GASPLAT on the VGR dataset.

Hyperparameters are tuned on TUNING only; HELD_OUT comes from other houses and
is kept for the final evaluation, so that the tuned values are shown to
generalize. Both sets cover the same kinds of distribution: compact sources,
plumes along corridors, gas spread over several rooms, and peaks from a few
ppm to several hundred.

Every scenario is the last snapshot of a simulation (t = 1000 s), sliced at
HEIGHT above the floor. scripts/make_scenarios.py extracts them from the
house archives.
"""
import os

from utils.gaden_utils import DEFAULT_ROOT, scenario_name

HEIGHT = 0.85       # Beam plane above the floor (m)
ITERATION = 1999    # Last snapshot of every VGR simulation

TUNING = [
    ("House01", "2,4-1_slow"),        # compact, confined to the source room
    ("House02", "3,5-1_slow"),        # broad, fills a room
    ("House04", "7-3,8_slow"),        # very sharp spike (~400 ppm)
    ("House14", "1,2-4,5_fast"),      # two separate regions
    ("House15", "1,2-3,4_fast"),      # carried away from the source into another room
    ("House21", "1,2,5-9,10_slow"),   # faint (~6 ppm), spread over many rooms
    ("House26", "5,6-3_fast"),        # long plume along a corridor
    ("House30", "7,8-3,4_slow"),      # plume leaving a room along a corridor
]

HELD_OUT = [
    ("House03", "5-3_slow"),          # source spike and a second blob
    ("House07", "6-1_fast"),          # faint (~7 ppm), elongated
    ("House10", "2,3-1_slow"),        # very sharp spike (~470 ppm)
    ("House13", "2,3-4,6_slow"),      # elongated plume
    ("House16", "1,2-4,5,6_fast"),    # two separate regions
    ("House25", "3,4-1,7_slow"),      # compact, in a room
    ("House27", "3,4-9_slow"),        # plume along a corridor
    ("House28", "8-1,3_slow"),        # broad spread from the source room
]

SETS = {"tuning": TUNING, "held_out": HELD_OUT, "all": TUNING + HELD_OUT}


def scenario_dirs(set_name: str, root: str = DEFAULT_ROOT) -> list[str]:
    """Scenario package directories of a set, in the order listed above."""
    return [os.path.join(root, scenario_name(house, sim, HEIGHT, ITERATION))
            for house, sim in SETS[set_name]]
