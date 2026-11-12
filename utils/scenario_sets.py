"""
Scenario sets used to tune and evaluate GASPLAT.

Hyperparameters are tuned on TUNING only; HELD_OUT comes from other houses and
is kept for the final evaluation, so that the tuned values are shown to
generalize. Both sets cover the same kinds of distribution: compact sources,
plumes along corridors, gas spread over several rooms, and peaks from a few
ppm to several hundred.

Every scenario is the last snapshot of a simulation (t = 1000 s), sliced at
HEIGHT above the floor. scripts/tools/make_scenarios.py extracts them from the
house archives.

The procedural set holds fractal maps without obstacles, one per seed: a
non-Gaussian control and a cheap testbed. Experiment scripts take a set with
--set, procedural maps with --seeds, or a single scenario package with --scenario.
"""
import os
import copy
from dataclasses import dataclass

from config import Config
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

# GADEN's empty test room (10 x 6 m) with our own sources (gaden_sims/10x6_empty_room), simulated with
# gaden_core by scripts/tools/simulate_gaden.py, for the experiment on the number of measurements.
# Beam plane through the point source (z = 0.75)
SIMPLE = [
    "10x6_empty_room_side_plume_h0.75_t1000",     # source in the inflow by the left door: plume across the room
    "10x6_empty_room_center_plume_h0.75_t1000",   # source in the still center: compact blob with a faint plume
]

PROCEDURAL_SEEDS = list(range(10))

SETS = {"tuning": TUNING, "held_out": HELD_OUT, "all": TUNING + HELD_OUT, "simple": SIMPLE,
        "procedural": PROCEDURAL_SEEDS}


def scenario_dirs(set_name: str, root: str = DEFAULT_ROOT, iteration: int = ITERATION) -> list[str]:
    """Scenario package directories of a GADEN set, in the order listed above. The VGR
    houses use the snapshot `iteration` (every 0.5 s; the last, 1999, is the default)."""
    return [os.path.join(root, entry if isinstance(entry, str) else scenario_name(*entry, HEIGHT, iteration))
            for entry in SETS[set_name]]


@dataclass(frozen=True)
class Case:
    """One environment to run: a GADEN scenario package or a procedural map."""
    name: str
    scenario: str | None = None   # Scenario package directory (GADEN)
    seed: int | None = None       # Seed of the procedural map

    def apply(self, cfg: Config) -> Config:
        """Copy of cfg set up for this case."""
        cfg = copy.deepcopy(cfg)
        cfg.sim.scenario = self.scenario
        if self.seed is not None:
            cfg.sim.seed = self.seed
        return cfg


def case_name(cfg: Config) -> str:
    """Name of the case a configuration runs."""
    if cfg.sim.scenario:
        return os.path.basename(os.path.normpath(cfg.sim.scenario))
    return f"procedural_seed{cfg.sim.seed}"


def get_cases(set_name: str | None = None, seeds: list[int] | None = None, scenario: str | None = None,
              root: str = DEFAULT_ROOT, iteration: int = ITERATION) -> list[Case]:
    """Procedural maps for the given seeds, else the single scenario package, else the cases of a set."""
    if seeds:
        return [Case(f"procedural_seed{s}", seed=s) for s in seeds]
    if scenario:
        cases = [Case(os.path.basename(os.path.normpath(scenario)), scenario=scenario)]
    elif set_name == "procedural":
        return [Case(f"procedural_seed{s}", seed=s) for s in PROCEDURAL_SEEDS]
    else:
        cases = [Case(os.path.basename(d), scenario=d) for d in scenario_dirs(set_name, root, iteration)]
    missing = [c.name for c in cases if not os.path.exists(os.path.join(c.scenario, "scenario.npz"))]
    if missing:
        tool = "simulate_gaden.py" if set_name == "simple" else "make_scenarios.py"
        hint = f" --iteration {iteration}" if iteration != ITERATION else ""
        raise SystemExit(f"Missing scenarios: {', '.join(missing)}. Create them with scripts/tools/{tool}{hint}")
    return cases


def cases_label(set_name: str, seeds: list[int] | None = None, scenario: str | None = None,
                iteration: int = ITERATION) -> str:
    """Name of what get_cases runs, for the result and plot files."""
    if seeds:
        return "procedural"
    if scenario:
        return os.path.basename(os.path.normpath(scenario))
    return set_name if iteration == ITERATION else f"{set_name}_it{iteration}"


def add_case_arguments(parser, default_set: str = "tuning"):
    """--set, --seeds and --iteration options shared by the experiment scripts (pass them
    to get_cases and cases_label). A single scenario is run with --scenario (an option of
    the configuration), which replaces the set."""
    parser.add_argument("--set", default=default_set, choices=list(SETS),
                        help="Scenario set to run (ignored with --scenario)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Run procedural maps with these seeds instead of a set")
    parser.add_argument("--iteration", type=int, default=ITERATION,
                        help="Snapshot of the VGR simulations (one every 0.5 s; default: the last). "
                             "Extract it first with scripts/tools/make_scenarios.py --iteration N")
