"""
Style of the figures: the paper figures (scripts/figures: fonts, the gas colour maps,
the methods' order, labels and line styles, and saving), and the larger style of the
working plots that the tools, main.py and the experiments save for themselves
(apply_plot_style).
"""
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from methods.registry import AVAILABLE_METHODS
from utils.paths import PLOTS_DIR

def apply_plot_style():
    """Matplotlib style of the working plots, meant to be looked at on screen rather than
    printed: sans-serif, large text, saved tight at 300 dpi. The paper figures use
    apply_paper_style instead."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'grid.alpha': 0.7,
        'grid.linestyle': '--',
        # --- Saving configuration ---
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.05
    })


# STIX (Times-like), shipped with matplotlib: heavier strokes than Computer Modern at the printed size
STIX = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}
# Font sizes of the plots of results; the drawings of the method use 8.5 pt text,
# its other sizes relative to it (matplotlib's "medium")
FONT_SIZES = {
    "plot": {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 9,
             "xtick.labelsize": 8.5, "ytick.labelsize": 8.5},
    "drawing": {"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": "medium", "legend.fontsize": "medium",
                "xtick.labelsize": "medium", "ytick.labelsize": "medium"},
}

# Gas concentration: white at zero, through yellow to orange at the peak
GAS_STOPS = [(0.0, (1, 1, 1, 1)), (0.05, (1, 0.97, 0.55, 1)), (0.5, (0.98, 0.82, 0.15, 1)),
             (1.0, (0.90, 0.45, 0.05, 1))]
GAS_CMAP = LinearSegmentedColormap.from_list("gas", GAS_STOPS)
# The same, transparent at zero (drawings where the field is drawn everywhere, walls included)
GAS_CMAP_CLEAR = LinearSegmentedColormap.from_list("gas", [(0.0, (1, 1, 1, 0))] + GAS_STOPS[1:])
# Yellow only, transparent at zero (the injection figure, where orange and red mark the beams)
GAS_CMAP_YELLOW = LinearSegmentedColormap.from_list("gas", [(1, 1, 1, 0), (1, 0.96, 0.35, 1), (0.93, 0.85, 0.10, 1)])
WALL_GREY = "0.6"          # walls of the maps of the experiments
WALL_DARK = "#4a4a4a"      # walls of the drawings of the method
PRIMITIVE_GREY = "#7a7a7a"   # primitive centers in the drawings

# Methods in the order of the paper, under their names there; colours and markers from methods/registry.py
METHOD_ORDER = ["SART", "RBF Coupled SART", "LFD", "LTD", "Gas Splatting"]
METHOD_LABEL = {"RBF Coupled SART": "RBF-SART", "Gas Splatting": "GASPLAT"}


def apply_paper_style(kind: str = "plot", **overrides):
    """Sets the rcParams of a paper figure: STIX and the font sizes of `kind` ("plot" or
    "drawing", FONT_SIZES), then the overrides, named as rcParams keys with underscores
    for the dots (axes_titlesize=8.5 sets "axes.titlesize")."""
    params = {**STIX, **FONT_SIZES[kind]}
    for name, value in overrides.items():
        key = name.replace("_", ".")
        if key not in plt.rcParams:
            raise KeyError(f"{name}: no rcParams key {key}")
        params[key] = value
    plt.rcParams.update(params)


def method_label(method: str) -> str:
    return METHOD_LABEL.get(method, method)


def method_color(method: str) -> str:
    return AVAILABLE_METHODS[method]["style"]["color"]


def line_style(method: str) -> dict:
    """Line of a method in the plots: its colour and marker in the registry (variants of
    GASPLAT take its own), GASPLAT drawn heavier than the baselines."""
    style = AVAILABLE_METHODS.get(method, AVAILABLE_METHODS["Gas Splatting"])["style"]
    gs = method.startswith("Gas Splatting")
    return dict(color=style["color"], marker=style["marker"], lw=1.8 if gs else 1.1, ms=7 if gs else 4.5,
                label=method_label(method))


def draw_gas_map(ax, img, obstacles, extent, vmax, title="", cmap=GAS_CMAP, vmin=0, interpolation="bilinear",
                 **title_kw):
    """A map over the free cells (obstacles: boolean mask) with the walls in WALL_GREY,
    without ticks or frame."""
    ax.imshow(np.where(obstacles, np.nan, img), origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation=interpolation)
    ax.imshow(np.where(obstacles, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([WALL_GREY]),
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, pad=3, **title_kw)


def figure_path(output: str) -> str:
    """Where a figure goes: an absolute path as given, a bare name or a path under plots/
    in PLOTS_DIR (whatever the current directory), any other relative path as given."""
    output = os.path.expanduser(output)
    if os.path.isabs(output):
        return output
    parts = os.path.normpath(output).split(os.sep)
    if len(parts) == 1:
        return os.path.join(PLOTS_DIR, output)
    if parts[0] == "plots":
        return os.path.join(PLOTS_DIR, *parts[1:])
    return output


def save_figure(fig, output: str, formats=("pdf", "png"), dpi: float | dict | None = 300, bbox_inches="tight",
                copy_to: str | None = None, copy_name: str | None = None) -> str:
    """Saves fig as <output>.<format> (figure_path) and closes it. dpi: one for every format,
    or per format (a dict; None leaves matplotlib's default). With copy_to, also copies
    the PDF there (e.g. the paper's figs folder), as copy_name if given. Returns the path
    without extension."""
    output = figure_path(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    for ext in formats:
        fig.savefig(f"{output}.{ext}", dpi=dpi.get(ext) if isinstance(dpi, dict) else dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Saved {output}.{{{','.join(formats)}}}")
    if copy_to:
        copy_to = os.path.expanduser(copy_to)
        os.makedirs(copy_to, exist_ok=True)
        shutil.copy(f"{output}.pdf", os.path.join(copy_to, copy_name or os.path.basename(output) + ".pdf"))
    return output
