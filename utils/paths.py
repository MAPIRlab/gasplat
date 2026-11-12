"""
Folders of the repository. Scripts write their results and plots here whatever
the directory they are run from.
"""
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")   # Experiment results (JSON, run logs)
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")       # Figures of the experiments and tools
DATA_DIR = os.path.join(REPO_ROOT, "data")
SCENARIOS_DIR = os.path.join(DATA_DIR, "scenarios")   # Scenario packages (utils/gaden_utils.py)
