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
--set, or procedural maps with --seeds.
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

PROCEDURAL_SEEDS = list(range(10))

SETS = {"tuning": TUNING, "held_out": HELD_OUT, "all": TUNING + HELD_OUT, "procedural": PROCEDURAL_SEEDS}


def scenario_dirs(set_name: str, root: str = DEFAULT_ROOT) -> list[str]:
    """Scenario package directories of a GADEN set, in the order listed above."""
    return [os.path.join(root, scenario_name(house, sim, HEIGHT, ITERATION))
            for house, sim in SETS[set_name]]


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


def get_cases(set_name: str | None = None, seeds: list[int] | None = None, root: str = DEFAULT_ROOT) -> list[Case]:
    """Cases of a set, or procedural maps for the given seeds (which take precedence)."""
    if seeds:
        return [Case(f"procedural_seed{s}", seed=s) for s in seeds]
    if set_name == "procedural":
        return [Case(f"procedural_seed{s}", seed=s) for s in PROCEDURAL_SEEDS]
    cases = [Case(os.path.basename(d), scenario=d) for d in scenario_dirs(set_name, root)]
    missing = [c.name for c in cases if not os.path.exists(os.path.join(c.scenario, "scenario.npz"))]
    if missing:
        raise SystemExit(f"Missing scenarios: {', '.join(missing)}. "
                         "Extract them with scripts/tools/make_scenarios.py")
    return cases


def add_case_arguments(parser, default_set: str = "tuning"):
    """--set and --seeds options shared by the experiment scripts."""
    parser.add_argument("--set", default=default_set, choices=list(SETS), help="Scenario set to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Run procedural maps with these seeds instead of a set")
