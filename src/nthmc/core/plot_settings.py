"""Settings for visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Color settings
grey = "#808080"
red = "#FF6F6F"
peach = "#FF9E6F"
orange = "#FFBC6F"
sunkist = "#FFDF6F"
yellow = "#FFEE6F"
lime = "#CBF169"
green = "#5CD25C"
turquoise = "#4AAB89"
blue = "#508EAD"
grape = "#635BB1"
violet = "#7C5AB8"
fuschia = "#C3559F"

color_ls = [
    blue,
    orange,
    green,
    red,
    violet,
    fuschia,
    turquoise,
    grape,
    lime,
    peach,
    sunkist,
    yellow,
]

# Marker settings
marker_ls = [
    ".",
    "o",
    "s",
    "P",
    "X",
    "*",
    "p",
    "D",
    "<",
    ">",
    "^",
    "v",
    "1",
    "2",
    "3",
    "4",
    "+",
    "x",
]

# Font settings
font_config = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}

rcParams.update(font_config)

# Figure size settings
fig_width = 6.75
gr = 1.618034333
fig_size = (fig_width, fig_width / gr)

# Default plot axes for general plots
plt_axes = [0.15, 0.15, 0.8, 0.8]
fs_p = {"fontsize": 20}
fs_small_p = {"fontsize": 18}
ls_p = {"labelsize": 20}

# Errorbar plot settings
errorb = {
    "markersize": 8,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 2,
}

errorb_line = {
    "markersize": 5,
    "mfc": "none",
    "capsize": 3,
    "elinewidth": 2,
}

errorb_circle = {
    "marker": "o",
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 2,
}

# Common labels
tmin_label = r"$t_{\mathrm{min}}~/~a$"
tmax_label = r"$t_{\mathrm{max}}~/~a$"
tau_center_label = r"$(\tau - t_{\rm{sep}}/2)~/~a$"
tsep_label = r"${t_{\mathrm{sep}}~/~a}$"
z_label = r"${z~/~a}$"
lambda_label = r"$\lambda = z P^z$"
meff_label = r"${m}_{\mathrm{eff}}~/~\mathrm{GeV}$"


def auto_ylim(y_data, yerr_data, y_range_ratio=4):
    """Calculate y-axis limits from values and errors."""
    all_y = np.concatenate(
        [y + yerr for y, yerr in zip(y_data, yerr_data)]
        + [y - yerr for y, yerr in zip(y_data, yerr_data)]
    )
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_range = y_max - y_min
    return y_min - y_range / y_range_ratio, y_max + y_range / y_range_ratio


def default_plot():
    """Create a default plot."""
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes()
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    return fig, ax


def default_sub_plot(height_ratio=3):
    """Create a two-panel plot with shared x-axis."""
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=fig_size,
        gridspec_kw={"height_ratios": [height_ratio, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", top="on", right="on", **ls_p)
        ax.grid(linestyle=":")

    return fig, (ax1, ax2)
