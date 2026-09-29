"""
Plot theme for ScanStudio.

Matches the web UI palette so embedded plots don't look like a different app.
Call `use_theme("dark")` or `use_theme("light")` before plotting.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PALETTES = {
    "dark": {
        "bg":       "#0f0f14",
        "panel":    "#1a1a24",
        "grid":     "#2a2a38",
        "text":     "#e2e8f0",
        "muted":    "#8892a4",
        "accent":   "#5b7cff",
        "accent_2": "#22d3a7",
        "green":    "#22c55e",
        "yellow":   "#f59e0b",
        "pink":     "#ec4899",
        "blue":     "#3b82f6",
        "purple":   "#a855f7",
        "red":      "#ef4444",
        "gray":     "#64748b",
    },
    "light": {
        "bg":       "#ffffff",
        "panel":    "#f8f9fa",
        "grid":     "#e5e7eb",
        "text":     "#1a1a2e",
        "muted":    "#6c757d",
        "accent":   "#4361ee",
        "accent_2": "#0d9488",
        "green":    "#16a34a",
        "yellow":   "#d97706",
        "pink":     "#db2777",
        "blue":     "#2563eb",
        "purple":   "#9333ea",
        "red":      "#dc2626",
        "gray":     "#94a3b8",
    },
}

C = PALETTES["dark"]


def _pick_font():
    """Prefer a clean UI font, fall back gracefully."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "SF Pro Display", "Helvetica Neue", "Segoe UI",
                 "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def use_theme(mode: str = "dark"):
    """Apply the palette globally. Returns the color dict."""
    global C
    C = PALETTES[mode]
    f = _pick_font()

    plt.rcParams.update({
        "figure.facecolor":  C["bg"],
        "axes.facecolor":    C["bg"],
        "savefig.facecolor": C["bg"],
        "savefig.edgecolor": "none",

        "font.family":     f,
        "font.size":       10,
        "axes.titlesize":  12,
        "axes.titleweight": "600",
        "axes.titlepad":   14,
        "axes.labelsize":  9.5,
        "axes.labelcolor": C["muted"],
        "axes.titlecolor": C["text"],
        "text.color":      C["text"],

        # only the axes you actually need
        "axes.edgecolor":  C["grid"],
        "axes.linewidth":  1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        "xtick.color":     C["muted"],
        "ytick.color":     C["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        "grid.color":     C["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha":     0.55,
        "axes.grid":      True,
        "axes.grid.axis": "y",

        "legend.frameon":    False,
        "legend.fontsize":   9,
        "legend.labelcolor": C["muted"],

        "figure.dpi":     130,
        "savefig.dpi":    130,
        "savefig.bbox":   "tight",
        "savefig.pad_inches": 0.3,
    })
    return C


def style_axes(ax, title=None, subtitle=None, xlabel=None, ylabel=None):
    """Left-aligned title + optional subtitle, the way dashboards read."""
    if title:
        ax.set_title(title, loc="left", pad=18 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=9, color=C["muted"], va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    return ax


def annotate_stat(ax, x, y, text, color=None):
    """Small callout label with a soft background."""
    ax.annotate(text, xy=(x, y), fontsize=8.5,
                color=color or C["muted"],
                bbox=dict(boxstyle="round,pad=0.35", fc=C["panel"],
                          ec="none", alpha=0.9))
