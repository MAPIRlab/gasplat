"""
Extracts the scenario packages of the tuning and held-out sets (see
utils/scenario_sets.py) from the VGR house archives into data/scenarios.

    python scripts/tools/make_scenarios.py --vgr_dir ~/GADEN_files/scenarios [--set all] [--iteration 1999]

--iteration chooses the snapshot (one every 0.5 s of simulated time; 1999, the last, by default).
"""
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.gaden_utils import DEFAULT_ROOT, extract_scenario
from utils.scenario_sets import HEIGHT, ITERATION, SETS, scenario_dirs


def extract(args):
    zip_path, simulation, root, iteration = args
    out_dir, _ = extract_scenario(zip_path, simulation, root, HEIGHT, iteration)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Extract the scenario sets from the VGR dataset.")
    parser.add_argument("--vgr_dir", default=os.path.expanduser("~/GADEN_files/scenarios"),
                        help="Directory with the HouseNN.zip archives")
    parser.add_argument("--set", default="all", choices=["tuning", "held_out", "all"])
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Where to write the scenario packages")
    parser.add_argument("--overwrite", action="store_true", help="Extract again scenarios that already exist")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iteration", type=int, default=ITERATION,
                        help="Snapshot of the simulations (one every 0.5 s; default: the last)")
    args = parser.parse_args()

    tasks = []
    for (house, simulation), out_dir in zip(SETS[args.set], scenario_dirs(args.set, args.root, args.iteration)):
        if os.path.exists(os.path.join(out_dir, "scenario.npz")) and not args.overwrite:
            print(f"[=] {os.path.basename(out_dir)} already exists")
            continue
        tasks.append((os.path.join(args.vgr_dir, f"{house}.zip"), simulation, args.root, args.iteration))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for out_dir in pool.map(extract, tasks):
            print(f"[+] {os.path.basename(out_dir)}")


if __name__ == "__main__":
    main()
