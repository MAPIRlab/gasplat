"""Figure of the noise experiment (scripts/experiments/noise_sensitivity.py), from its results.

Top row: NRMSE of every method over the noise level, one panel per number of poses;
bottom row: the ratio of the reconstructed to the true peak (left out with --nrmse_only,
the column-width paper figure). Lines are means over the runs, shaded bands one standard
deviation.

    python scripts/figures/noise_figure.py results/noise_sensitivity_<case>_<date>.json \\
        --output plots/noise_sensitivity [--copy_to ~/paper/figs]
    python scripts/figures/noise_figure.py RESULTS --nrmse_only --noise_aware_only --output plots/noise_sensitivity_paper
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from methods.registry import AVAILABLE_METHODS

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})
ORDER = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]
NAMES = {"Gas Splatting": "GASPLAT", "RBF Coupled SART": "RBF-SART",
         "Gas Splatting (discrepancy)": "GASPLAT, noise-aware stop"}


def main():
    parser = argparse.ArgumentParser(description="Figure of the noise experiment")
    parser.add_argument("results", nargs="+", help="Results of noise_sensitivity.py (JSON); later files add methods")
    parser.add_argument("--output", default="plots/noise_sensitivity", help="Output path without extension")
    parser.add_argument("--copy_to", default=None, help="Also copy the PDF here")
    parser.add_argument("--nrmse_only", action="store_true", help="Only the NRMSE row, at the width of a column")
    parser.add_argument("--noise_aware_only", action="store_true",
                        help="Show GASPLAT only with its noise-aware stop, labelled GASPLAT (the method of the paper)")
    args = parser.parse_args()

    runs, meta = [], None
    for path in args.results:
        data = json.load(open(path))
        meta = meta or data["metadata"]
        runs += data["results"]["runs"]
    poses = sorted({r["poses"] for r in runs})
    levels = sorted({r["noise_level"] for r in runs})
    methods = [m for m in ORDER if m in runs[0]["metrics"]] + [m for m in runs[0]["metrics"] if m not in ORDER]
    if args.noise_aware_only and "Gas Splatting (discrepancy)" in methods:
        # The noise-aware stop is part of the method: its runs replace the plain ones, under the name GASPLAT
        for r in runs:
            r["metrics"]["Gas Splatting"] = r["metrics"].pop("Gas Splatting (discrepancy)")
        methods.remove("Gas Splatting (discrepancy)")

    rows = [("nrmse", 100, "NRMSE (% of peak)")]
    if not args.nrmse_only:
        rows.append(("peak_ratio", 1, "Reconstructed / true peak"))
    size = (3.5, 2.35) if args.nrmse_only else (7.16, 4.4)
    fig, axes = plt.subplots(len(rows), len(poses), figsize=size, squeeze=False, sharex=True, sharey=args.nrmse_only)
    for j, p in enumerate(poses):
        for i, (key, factor, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for m in methods:
                vals = np.array([[factor * r["metrics"][m][key] for r in runs
                                  if r["poses"] == p and r["noise_level"] == lv and m in r["metrics"]] for lv in levels])
                mean, std = vals.mean(1), vals.std(1)
                style = AVAILABLE_METHODS.get(m, AVAILABLE_METHODS["Gas Splatting"])["style"]
                gs = m.startswith("Gas Splatting")
                ax.plot(levels, mean, color=style["color"], marker=style["marker"], lw=1.8 if gs else 1.1,
                        ms=7 if gs else 4.5, ls="--" if gs and m != "Gas Splatting" else "-",
                        mfc="white" if gs and m != "Gas Splatting" else style["color"], label=NAMES.get(m, m))
                ax.fill_between(levels, np.maximum(mean - std, 0), mean + std, color=style["color"], alpha=0.15, lw=0)
            if key == "peak_ratio":
                ax.axhline(1, color="0.4", lw=0.8, ls=":")
            if i == len(rows) - 1:
                ax.set_xlabel("Noise level $\\sigma_{\\mathrm{noise}}$ (%)" if args.nrmse_only else
                              "Noise level $\\sigma_{\\mathrm{noise}}$ (% of each measurement)")
            if key == "nrmse":
                beams = np.mean([r["beams"] for r in runs if r["poses"] == p])
                ax.set_title(f"{p} poses ({beams:.0f} beams)", pad=4)
            ax.set_ylim(bottom=0)
            # Narrow panels: label only the levels that do not crowd each other
            ticks = [lv for lv in levels if lv == 0 or lv >= 5] if args.nrmse_only else levels
            ax.set_xticks(ticks)
            ax.grid(True, ls="--", alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if args.nrmse_only:
        for ax in axes.flat:
            for line in ax.get_lines():
                line.set_markersize(line.get_markersize() * 0.7)
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13),
                   fontsize=7.5, handlelength=2.2, columnspacing=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output}.{ext}", dpi=300, bbox_inches="tight")
    if args.copy_to:
        shutil.copy(f"{args.output}.pdf", os.path.expanduser(args.copy_to))
    print(f"Saved {args.output}.{{pdf,png}}")


if __name__ == "__main__":
    main()
