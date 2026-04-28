"""
flu/utils/viz_magic.py
======================
High-quality visualizations for FLU FM-Dance magic hypercubes.

Four complementary views for odd-order magic constructs (any dimension d):

- plot_simplex:          Unit-circle symmetry view ("Einheitskreisfraktal")
- plot_spectral_grid:    Spectral band visualization
- plot_bones_grid:       Centered "bones" view (value - mean)
- plot_blur_reduction:   Collapse one axis to reveal higher-level fractal pattern
- plot_overview:         Convenient 2×2 overview

Includes safety limits, improved legends, better performance handling and clean code.
"""

from __future__ import annotations
import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _require_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for flu.utils.viz_magic.\n"
            "Install with: pip install flu[viz] or pip install matplotlib"
        )


# ── Mathematical Helpers ────────────────────────────────────────────────────
def _magic_sum(n: int, d: int) -> int:
    """Magic constant M = n * (n^d + 1) // 2"""
    return n * (n**d + 1) // 2


def _mean(n: int, d: int) -> float:
    """Center value μ = (n^d + 1) / 2"""
    return (n**d + 1) / 2.0


def _spectral_band(val: int, n: int, d: int) -> int:
    """Spectral band index: (val - 1) // n^(d-1)"""
    return (int(val) - 1) // (n ** (d - 1))


def _spectral_band_array(arr: np.ndarray, n: int, d: int) -> np.ndarray:
    return (arr - 1) // (n ** (d - 1))


def _unit_circle_pos(v: int, total: int, r: float = 1.0) -> Tuple[float, float]:
    """Position on unit circle, starting at top (-π/2)"""
    angle = 2 * np.pi * (v - 1) / total - np.pi / 2
    return r * np.cos(angle), r * np.sin(angle)


# ── Safety & Validation ─────────────────────────────────────────────────────
def _check_size(n: int, d: int, max_cells: int = 20000, warn_only: bool = False) -> None:
    """Warn or raise for very large constructs."""
    total = n ** d
    if total > max_cells:
        msg = (f"Large magic construct: n={n}, d={d} → {total:,} cells. "
               f"Visualizations may be slow or cluttered. "
               f"Recommended limit is {max_cells:,} cells.")
        if warn_only:
            warnings.warn(msg, UserWarning)
        else:
            raise ValueError(msg)


# ── Colour & Label Helpers ──────────────────────────────────────────────────
_N3_SPECTRAL_COLOURS = ["#ef4444", "#94a3b8", "#22c55e"]


def _spectral_cmap(n: int):
    """Return nice colours for spectral bands."""
    if n == 3:
        return _N3_SPECTRAL_COLOURS
    cmap = plt.cm.RdYlGn
    return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]


def _spectral_labels(n: int) -> List[str]:
    """Human readable spectral band labels."""
    if n == 3:
        return ["negative (−)", "balanced (0)", "positive (+)"]
    half = n // 2
    return [f"band {i} ({'+' if i > half else '-' if i < half else ''}{abs(i - half)})"
            for i in range(n)]


# ── Plot Functions ──────────────────────────────────────────────────────────

def plot_simplex(
    n: int,
    d: int,
    show_axes: Optional[List[int]] = None,
    show_diags: bool = True,
    alpha_fill: float = 0.12,
    alpha_edge: float = 0.75,
    label_values: bool = True,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["Figure", "Axes"]:
    """Unit-circle simplex showing symmetry structure of the magic hypercube."""
    _require_matplotlib()
    from flu.core.fm_dance import generate_magic

    _check_size(n, d, max_cells=15000)

    total = n ** d
    arr = generate_magic(n, d)

    if show_axes is None:
        show_axes = list(range(d))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    ax.axis("off")

    # Background circle
    ax.add_patch(plt.Circle((0, 0), 1.02, fill=False, color="#475569", linewidth=1.1))

    def draw_polygon(values: List[int], colour: str):
        pts = np.array([_unit_circle_pos(v, total) for v in values])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=colour,
                                 alpha=alpha_fill, edgecolor="none"))
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False,
                                 edgecolor=colour, alpha=alpha_edge, linewidth=1.1))

    axis_colours = plt.cm.tab10.colors

    # Axis-aligned lines
    for axis_idx in show_axes:
        colour = axis_colours[axis_idx % len(axis_colours)]
        other_axes = [a for a in range(d) if a != axis_idx]
        for combo in itertools.product(*[range(n) for _ in other_axes]):
            idx = list(combo)
            idx.insert(axis_idx, slice(None))
            line = arr[tuple(idx)].tolist()
            draw_polygon(line, colour)

    # Space diagonals
    if show_diags:
        diag_colour = "#f59e0b"
        for signs in itertools.product([-1, 1], repeat=d - 1):
            full_signs = [1] + list(signs)
            diag = [
                int(arr[tuple(i if full_signs[dim] == 1 else n - 1 - i 
                             for dim in range(d))])
                for i in range(n)
            ]
            draw_polygon(diag, diag_colour)

    # Nodes and optional labels
    fontsize = max(4, min(10, int(85 / (total ** 0.45))))
    for v in range(1, total + 1):
        x, y = _unit_circle_pos(v, total, 1.02)
        ax.plot(x, y, "o", color="#1e293b", markersize=3.2, zorder=5)
        if label_values and total <= 200:
            xl, yl = _unit_circle_pos(v, total, 1.16)
            ax.text(xl, yl, str(v), ha="center", va="center",
                    fontsize=fontsize, color="#334155",
                    fontfamily="monospace", zorder=6)

    # Legend
    handles = []
    for axis_idx in show_axes:
        colour = axis_colours[axis_idx % len(axis_colours)]
        handles.append(mpatches.Patch(color=colour, alpha=0.75, label=f"Axis {axis_idx}"))
    if show_diags:
        handles.append(mpatches.Patch(color="#f59e0b", alpha=0.75, label="Space diagonals"))

    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
                  ncol=min(5, len(handles)), fontsize=9, framealpha=0.85)

    default_title = f"FM-Dance Magic Simplex • n={n}  d={d}  •  Magic sum = {_magic_sum(n, d)}"
    ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=20)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    return fig, ax


def plot_spectral_grid(
    n: int,
    d: int,
    slice_axes: Optional[Dict[int, int]] = None,
    annotate: bool = True,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["Figure", "Axes"]:
    """Spectral band grid showing coarse value bands."""
    _require_matplotlib()
    from flu.core.fm_dance import generate_magic

    _check_size(n, d, max_cells=10000, warn_only=True)

    arr = generate_magic(n, d)

    if slice_axes is None:
        slice_axes = {a: n // 2 for a in range(max(0, d - 2))}

    idx = [slice(None)] * d
    for axis, coord in slice_axes.items():
        idx[axis] = coord

    slice_2d = arr[tuple(idx)]
    if slice_2d.n != 2:
        raise ValueError(f"Expected 2D slice, got shape {slice_2d.shape}")

    bands = _spectral_band_array(slice_2d, n, d)
    colours = _spectral_cmap(n)

    if ax is None:
        rows, cols = slice_2d.shape
        fig, ax = plt.subplots(figsize=(max(5, cols * 0.7), max(5, rows * 0.7)))
    else:
        fig = ax.get_figure()

    rows, cols = slice_2d.shape
    for r in range(rows):
        for c in range(cols):
            val = int(slice_2d[r, c])
            band = int(bands[r, c])
            rect = mpatches.FancyBboxPatch(
                (c, rows - 1 - r), 1, 1, boxstyle="square,pad=0",
                facecolor=colours[band], edgecolor="#e2e8f0", linewidth=0.8, alpha=0.8
            )
            ax.add_patch(rect)

            if annotate:
                fs = max(5, min(12, int(110 / max(rows, cols))))
                ax.text(c + 0.5, rows - 0.5 - r, str(val),
                        ha="center", va="center", fontsize=fs,
                        fontfamily="monospace", fontweight="bold", color="#1e293b")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    band_labels = _spectral_labels(n)
    handles = [mpatches.Patch(color=colours[i], alpha=0.8, label=band_labels[i])
               for i in range(n)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(n, 4), fontsize=9, framealpha=0.9)

    slice_desc = " · ".join(f"axis{a}={v}" for a, v in sorted(slice_axes.items())) or "central slice"
    default_title = f"Spectral Bands • n={n} d={d} • {slice_desc}"
    ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

    return fig, ax


def plot_bones_grid(
    n: int,
    d: int,
    slice_axes: Optional[Dict[int, int]] = None,
    annotate: bool = True,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["Figure", "Axes"]:
    """Centered bones view (value - mean)."""
    _require_matplotlib()
    from flu.core.fm_dance import generate_magic

    _check_size(n, d, max_cells=10000, warn_only=True)

    arr = generate_magic(n, d)
    mu = _mean(n, d)

    if slice_axes is None:
        slice_axes = {a: n // 2 for a in range(max(0, d - 2))}

    idx = [slice(None)] * d
    for axis, coord in slice_axes.items():
        idx[axis] = coord

    slice_2d = arr[tuple(idx)].astype(float)
    bones = slice_2d - mu

    if ax is None:
        rows, cols = bones.shape
        fig, ax = plt.subplots(figsize=(max(5, cols * 0.7), max(5, rows * 0.7)))
    else:
        fig = ax.get_figure()

    vabs = max(abs(bones.min()), abs(bones.max()), 1)
    im = ax.imshow(bones, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="equal", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="value − μ")

    if annotate:
        rows, cols = bones.shape
        fs = max(5, min(11, int(110 / max(rows, cols))))
        for r in range(rows):
            for c in range(cols):
                b = bones[r, c]
                label = f"{int(b):+d}" if b != 0 else "0"
                color = "white" if abs(b) > 0.6 * vabs else "#1e293b"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=fs, color=color,
                        fontfamily="monospace", fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    slice_desc = " · ".join(f"axis{a}={v}" for a, v in sorted(slice_axes.items())) or "central slice"
    default_title = f"Bones View • n={n} d={d} • μ={mu:.1f} • {slice_desc}"
    ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

    return fig, ax


def plot_blur_reduction(
    n: int,
    d: int,
    blur_axis: int = 0,
    annotate: bool = True,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["Figure", "Axes"]:
    """Blur reduction: collapse one axis to reveal higher fractal pattern."""
    _require_matplotlib()
    from flu.core.fm_dance import generate_magic

    _check_size(n, d, max_cells=30000, warn_only=True)

    if not (0 <= blur_axis < d):
        raise ValueError(f"blur_axis must be between 0 and {d-1}")

    arr = generate_magic(n, d)
    bands = _spectral_band_array(arr, n, d)

    center = n // 2
    other_axes = [a for a in range(d) if a != blur_axis]
    reduced = np.zeros([n] * (d - 1), dtype=int)

    for combo in itertools.product(*[range(n) for _ in other_axes]):
        idx = list(combo)
        idx.insert(blur_axis, center)
        reduced[combo] = bands[tuple(idx)]

    if reduced.ndim > 2:
        reduced = reduced.reshape(-1, reduced.shape[-1])
    if reduced.ndim == 1:
        reduced = reduced.reshape(1, -1)

    rows, cols = reduced.shape
    colours = _spectral_cmap(n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.5, cols * 0.75), max(4, rows * 0.75)))
    else:
        fig = ax.get_figure()

    # Symbol mapping
    sym = {i: str(i) for i in range(n)}
    if n == 3:
        sym = {0: "−", 1: "0", 2: "+"}
    elif n == 5:
        sym = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    for r in range(rows):
        for c in range(cols):
            band = int(reduced[r, c])
            rect = mpatches.FancyBboxPatch(
                (c, rows - 1 - r), 1, 1, boxstyle="square,pad=0",
                facecolor=colours[band], edgecolor="#e2e8f0",
                linewidth=1.0, alpha=0.78
            )
            ax.add_patch(rect)

            if annotate:
                fs = max(7, min(16, int(130 / max(rows, cols))))
                ax.text(c + 0.5, rows - 0.5 - r, sym[band],
                        ha="center", va="center", fontsize=fs,
                        fontfamily="monospace", fontweight="bold", color="#1e293b")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    band_labels = _spectral_labels(n)
    handles = [mpatches.Patch(color=colours[i], alpha=0.78,
                             label=f"{sym[i]} {band_labels[i]}") for i in range(n)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(n, 4), fontsize=9, framealpha=0.9)

    default_title = f"Blur Reduction • n={n} d={d} → d={d-1} (axis {blur_axis} collapsed)"
    ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

    return fig, ax


def plot_overview(
    n: int,
    d: int,
    slice_axes: Optional[Dict[int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> "Figure":
    """Convenient 2×2 overview of all magic visualizations."""
    _require_matplotlib()
    _check_size(n, d, max_cells=8000, warn_only=True)

    if figsize is None:
        figsize = (15, 14)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"FLU FM-Dance Magic Analysis • n={n}  d={d}  •  "
        f"Cells = {n**d:,}  •  Magic sum = {_magic_sum(n, d)}",
        fontsize=14, fontweight="bold", y=0.98
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_simplex(n, d, ax=ax1, label_values=(n**d <= 81))
    plot_spectral_grid(n, d, slice_axes=slice_axes, ax=ax2)
    plot_bones_grid(n, d, slice_axes=slice_axes, ax=ax3)
    plot_blur_reduction(n, d, blur_axis=0, ax=ax4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
