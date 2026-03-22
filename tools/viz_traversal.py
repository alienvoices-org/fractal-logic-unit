"""
tools/viz_traversal.py
=======================
FM-Dance Traversal Visualisation — V12 Sprint Item 12 / OD-12.

Generates static figures showing the FM-Dance path through lattice space.

FIGURES PRODUCED
────────────────
  fm_dance_n{n}_d2.png   : 2D path coloured by step index
  fm_dance_n{n}_d3.png   : 3D path (optional, d=3)
  nary_comparison.png     : FM-Dance vs Morton vs Gray step quality
  step_bound_regimes.png  : T4 step bound crossover (L4 lemma visualisation)

SAFE LIMITS: n ≤ 11, d ≤ 3 for visualisation. No 13^4 extremes.

STATUS: V12 Sprint Item 12 / OD-12.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import List, Optional, Tuple
import numpy as np

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from flu.core.fm_dance import index_to_coords
from flu.core.n_ary import nary_step_bound, nary_comparison_table


def _check_matplotlib():
    if not _MPL_AVAILABLE:
        print("WARNING: matplotlib not available. Install it for visualisation.")
        return False
    return True


def _fm_dance_path_2d(n: int) -> Tuple[List[int], List[int], List[int]]:
    """Extract 2D FM-Dance path coordinates and step indices."""
    total = n ** 2
    xs, ys = [], []
    for k in range(total):
        coord = index_to_coords(k, n, 2)
        xs.append(coord[0])
        ys.append(coord[1])
    return xs, ys, list(range(total))


def plot_fm_dance_2d(
    n: int = 5,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot the FM-Dance traversal path in 2D.

    Produces a scatter+line plot of the path through Z_n^2,
    coloured by step index (blue→red = early→late).

    SAFE LIMIT: n ≤ 13 (n^2 ≤ 169).
    """
    if not _check_matplotlib():
        return None
    if n > 13:
        raise ValueError(f"n={n} too large for 2D visualisation (n^2={n**2}). Use n ≤ 13.")

    xs, ys, steps = _fm_dance_path_2d(n)
    total = len(xs)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("coolwarm")

    # Draw path segments coloured by step
    for i in range(total - 1):
        colour = cmap(i / total)
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color=colour, alpha=0.7, linewidth=1.2)

    # Draw points
    scatter = ax.scatter(xs, ys, c=steps, cmap="coolwarm", s=80,
                         zorder=5, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="Step index k")

    # Mark start and end
    ax.scatter([xs[0]], [ys[0]], c="green", s=200, marker="*",
               zorder=10, label="Start (k=0)")
    ax.scatter([xs[-1]], [ys[-1]], c="red", s=200, marker="X",
               zorder=10, label=f"End (k={total-1})")

    ax.set_xlabel(f"x₀  ∈ {{{-n//2},...,{n//2}}}")
    ax.set_ylabel(f"x₁  ∈ {{{-n//2},...,{n//2}}}")
    ax.set_title(title or f"FM-Dance Path  n={n}, D=2  ({total} steps)\nT4 step bound = {nary_step_bound(n, 2)}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if output_path is None:
        os.makedirs("docs/assets", exist_ok=True)
        output_path = f"docs/assets/fm_dance_n{n}_d2.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_step_bound_regimes(
    n_values: Optional[List[int]] = None,
    d_max: int = 8,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot the T4 / L4 step bound crossover regimes.

    Shows how step_bound = min(d, floor(n/2)) transitions between
    dimension-limited (d < n/2) and radix-limited (d > n/2) regimes.
    """
    if not _check_matplotlib():
        return None

    if n_values is None:
        n_values = [3, 5, 7, 9, 11]

    d_range = range(1, d_max + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = plt.get_cmap("Set1").colors

    for i, n in enumerate(n_values):
        bounds = [nary_step_bound(n, d) for d in d_range]
        dstar = n // 2
        ax.plot(list(d_range), bounds, "-o", color=colours[i % len(colours)],
                label=f"n={n}  (D*={dstar})", linewidth=2, markersize=5)
        # Mark D* crossover
        if dstar <= d_max:
            ax.axvline(x=dstar, color=colours[i % len(colours)],
                       linestyle="--", alpha=0.3)

    ax.set_xlabel("Dimension D")
    ax.set_ylabel("Step Bound  min(D, ⌊n/2⌋)")
    ax.set_title("L4 Step-Bound Crossover Regimes\n"
                 "Left of D*: dimension-limited  |  Right of D*: radix-limited (saturated)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(d_range))

    if output_path is None:
        os.makedirs("docs/assets", exist_ok=True)
        output_path = "docs/assets/step_bound_regimes.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_nary_comparison(
    n_values: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot the n-ary comparison table: step bound vs D for different n values.
    Visualises the alignment principle from N-ARY-1.
    """
    if not _check_matplotlib():
        return None
    if n_values is None:
        n_values = [2, 3, 5, 7, 9, 11]

    rows = nary_comparison_table(n_values, [2, 3, 4])

    # Group by d
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, d_show in zip(axes, [2, 3]):
        d_rows = [r for r in rows if r["d"] == d_show]
        ns = [r["n"] for r in d_rows]
        bounds = [r["step_bound"] for r in d_rows]
        regimes = [r["regime"] for r in d_rows]
        colours = ["#2196F3" if reg == "dim-limited" else "#FF5722" for reg in regimes]

        bars = ax.bar([str(n) for n in ns], bounds, color=colours, alpha=0.8,
                      edgecolor="black", linewidth=0.7)
        ax.set_xlabel("Base n")
        ax.set_ylabel("Step Bound")
        ax.set_title(f"N-ary Step Bound (D={d_show})\n"
                     f"Blue=dimension-limited  |  Orange=radix-limited")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate odd/even
        for bar, row in zip(bars, d_rows):
            label = "odd" if row["odd_n"] else "even"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    label, ha="center", va="bottom", fontsize=7, color="gray")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.8, label="dimension-limited (D < D*)"),
        Patch(facecolor="#FF5722", alpha=0.8, label="radix-limited (D > D*, saturated)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    if output_path is None:
        os.makedirs("docs/assets", exist_ok=True)
        output_path = "docs/assets/nary_comparison.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def generate_all_figures(output_dir: str = "docs/assets") -> List[str]:
    """
    Generate all standard V12 documentation figures.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # 2D path for n=5 (primary README figure)
    p = plot_fm_dance_2d(n=5, output_path=f"{output_dir}/fm_dance_n5_d2.png")
    if p:
        saved.append(p)

    # 2D path for n=7
    p = plot_fm_dance_2d(n=7, output_path=f"{output_dir}/fm_dance_n7_d2.png")
    if p:
        saved.append(p)

    # Step bound regimes (L4)
    p = plot_step_bound_regimes(output_path=f"{output_dir}/step_bound_regimes.png")
    if p:
        saved.append(p)

    # N-ary comparison (N-ARY-1)
    p = plot_nary_comparison(output_path=f"{output_dir}/nary_comparison.png")
    if p:
        saved.append(p)

    return saved


if __name__ == "__main__":
    print("FLU V12 — Generating visualisation figures...")
    saved = generate_all_figures()
    print(f"\nGenerated {len(saved)} figures:")
    for path in saved:
        print(f"  {path}")
    if not saved:
        print("No figures generated (matplotlib not available or path issues).")
