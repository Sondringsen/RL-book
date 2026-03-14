"""Experiment 2: DP runtime scaling with state/action space cardinality.

Shows how value iteration wall-clock time grows as we expand:
  (a) inventory space  — I_max = 2 … 100
  (b) price grid       — n_price_bins = 5 … 81  (the key DP bottleneck)
  (c) action space     — n_spread_options = 2 … 8
  (d) all three simultaneously — the curse of dimensionality

All curves are plotted against |S| × |A| to make the comparison direct.
Because price is a *continuous* quantity that must be discretised for DP to
work, panel (b) highlights the core DP vs RL trade-off: the price transition
is a dense Gaussian matrix, so cost per VI sweep is O(n_inv · n_price² · |A|)
— quadratic in the grid size.  RL needs no grid at all.

Panel (d) shows the curse of dimensionality: when inventory, price bins, and
action space all grow together each dimension multiplies the cost, producing a
much steeper slope than any single dimension alone.
"""

import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from project.market_making.params import MarketParams
from project.market_making.dp_solver import value_iteration, value_iteration_2d

# ── timing helper ─────────────────────────────────────────────────────────────

BENCH_ITERS = 300   # fixed number of VI sweeps per configuration


def time_1d(I_max: int, n_spread: int = 5) -> tuple[float, int, int]:
    """Return (seconds, |S|, |A|) for 1-D value iteration."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration(params, max_iter=BENCH_ITERS, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    return elapsed, params.n_inventory_states, params.n_actions


def time_2d(n_price_bins: int, I_max: int = 5, n_spread: int = 5) -> tuple[float, int, int]:
    """Return (seconds, |S|, |A|) for 2-D value iteration."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_2d(
        params,
        n_price_bins=n_price_bins,
        max_iter=BENCH_ITERS,
        tol=0.0,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    n_states = params.n_inventory_states * n_price_bins
    return elapsed, n_states, params.n_actions


def time_joint(I_max: int, n_price_bins: int, n_spread: int) -> tuple[float, int, int]:
    """Return (seconds, |S|, |A|) growing all three dimensions at once."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_2d(
        params,
        n_price_bins=n_price_bins,
        max_iter=BENCH_ITERS,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    n_states = params.n_inventory_states * n_price_bins
    return elapsed, n_states, params.n_actions


def time_action(n_spread: int, I_max: int = 10) -> tuple[float, int, int]:
    """Return (seconds, |S|, |A|) varying action-space size."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration(params, max_iter=BENCH_ITERS, verbose=False)
    elapsed = time.perf_counter() - t0
    return elapsed, params.n_inventory_states, params.n_actions


# ── sweep configurations ──────────────────────────────────────────────────────

print("Benchmarking 1-D VI over inventory size …")
inv_results = []
for I_max in [2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
    elapsed, nS, nA = time_1d(I_max)
    inv_results.append((nS * nA, elapsed))
    print(f"  I_max={I_max:>3d}  |S|={nS:>4d}  |A|={nA:>3d}  |S||A|={nS*nA:>6d}  t={elapsed:.3f}s")

print("\nBenchmarking 2-D VI over price grid size …")
price_results = []
for n_price in [25, 31, 41, 51, 61, 71, 81, 100, 125, 150]:
    elapsed, nS, nA = time_2d(n_price)
    price_results.append((nS * nA, elapsed))
    print(f"  n_price={n_price:>3d}  |S|={nS:>4d}  |A|={nA:>3d}  |S||A|={nS*nA:>6d}  t={elapsed:.3f}s")

print("\nBenchmarking 1-D VI over action space size …")
action_results = []
for n_spread in [2, 3, 4, 5, 6, 7, 8]:
    elapsed, nS, nA = time_action(n_spread)
    action_results.append((nS * nA, elapsed))
    print(f"  n_spread={n_spread}  |S|={nS:>3d}  |A|={nA:>4d}  |S||A|={nS*nA:>6d}  t={elapsed:.3f}s")

# Curse of dimensionality: all three grow together along a single scale axis.
# Schedule chosen so |S|×|A| spans a wide range without blowing up runtime.
JOINT_CONFIGS = [
    (2,  5,  2),   # scale=1: tiny
    (3,  7,  3),
    (5, 11,  4),
    (8, 15,  5),
    (10, 21, 6),
    (15, 25, 7),
    (20, 31, 8),
    (30, 41, 10),
    # (50, 70, 20),
    # (70, 90, 40),
]

print("\nBenchmarking 2-D VI with all dimensions growing (curse of dimensionality) …")
joint_results = []
for I_max, n_price, n_spread in JOINT_CONFIGS:
    elapsed, nS, nA = time_joint(I_max, n_price, n_spread)
    joint_results.append((nS * nA, elapsed))
    print(f"  I_max={I_max:>2d}  n_price={n_price:>2d}  n_spread={n_spread}"
          f"  |S|={nS:>5d}  |A|={nA:>4d}  |S||A|={nS*nA:>7d}  t={elapsed:.3f}s")


# ── plotting ──────────────────────────────────────────────────────────────────

fig, axes_2d = plt.subplots(2, 2, figsize=(13, 10))
axes = axes_2d.flat
fig.suptitle(
    f"Dynamic Programming: Wall-clock Time vs State–Action Space Size\n"
    f"({BENCH_ITERS} VI sweeps per configuration)",
    fontsize=13,
)

COLORS = {"inv": "#2563EB", "price": "#DC2626", "action": "#16A34A", "joint": "#7C3AED"}
MARKERS = {"inv": "o", "price": "s", "action": "^", "joint": "D"}


def _fit_line(xs, ys):
    """Fit log-log line and return (slope, fitted_ys)."""
    lx, ly = np.log10(xs), np.log10(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, 10 ** (intercept + slope * lx)


def _plot_panel(ax, results, color, marker, label, title):
    xs = np.array([r[0] for r in results], dtype=float)
    ys = np.array([r[1] for r in results], dtype=float)

    ax.scatter(xs, ys, color=color, marker=marker, zorder=5, s=60, label=label)
    ax.plot(xs, ys, color=color, lw=1.5, alpha=0.7)

    slope, fitted = _fit_line(xs, ys)
    ax.plot(xs, fitted, color=color, lw=1, ls="--", alpha=0.5,
            label=f"log-log fit (slope ≈ {slope:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|S| × |A|  (log scale)", fontsize=10)
    ax.set_ylabel("Wall-clock time (s, log scale)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, which="both", ls=":", alpha=0.4)


_plot_panel(
    axes[0], inv_results,
    COLORS["inv"], MARKERS["inv"],
    "1-D VI (inventory)",
    "Inventory space scaling\n(I_max = 2 … 100, fixed 5×5 actions)",
)

_plot_panel(
    axes[1], price_results,
    COLORS["price"], MARKERS["price"],
    "2-D VI (inv + price grid)",
    "Price grid scaling  ← DP bottleneck\n(n_price = 5 … 81, I_max=5, 5×5 actions)",
)


_plot_panel(
    axes[2], action_results,
    COLORS["action"], MARKERS["action"],
    "1-D VI (action space)",
    "Action space scaling\n(n_spread = 2 … 8 per side, I_max=10)",
)

# ── Panel 4: curse of dimensionality — all dimensions grow together ───────────
ax4 = axes[3]
xs = np.array([r[0] for r in joint_results], dtype=float)
ys = np.array([r[1] for r in joint_results], dtype=float)
slope, fitted = _fit_line(xs, ys)
ax4.scatter(xs, ys, color=COLORS["joint"], marker=MARKERS["joint"], zorder=5, s=60)
ax4.plot(xs, ys, color=COLORS["joint"], lw=1.5, alpha=0.6,
            label=f"All three together (2-D VI) (slope≈{slope:.2f})")
ax4.plot(xs, fitted, color=COLORS["joint"], lw=1, ls="--", alpha=0.4)

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("|S| × |A|  (log scale)", fontsize=10)
ax4.set_ylabel("Wall-clock time (s, log scale)", fontsize=10)
ax4.set_title(
    "Curse of Dimensionality\n(all dimensions grow simultaneously vs. one at a time)",
    fontsize=11,
)
ax4.legend(fontsize=8, loc="upper left")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax4.grid(True, which="both", ls=":", alpha=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "results/experiment_2_dp_scaling.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")

# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Key takeaway:")
print("  - 1-D inventory DP: transitions are sparse (≤3 next states).")
print("    Cost per iter: O(|S|·|A|)  — linear, tractable up to I_max~100.")
print("  - 2-D price DP: price transition is a *dense* n_price×n_price")
print("    Gaussian matrix.  Cost per iter: O(n_inv · n_price² · |A|).")
print("    Doubling the price grid quadruples the per-iteration work.")
print("    Accurate price modelling needs ~50–100 bins → quickly prohibitive.")
print("  - Action space: |A| grows as n_spread², blows up multiplicatively.")
print("  - Joint scaling (curse of dimensionality): each new dimension")
print("    multiplies |S|×|A|, producing a steeper log-log slope than")
print("    any single dimension alone.  RL sidesteps this entirely.")
print("  - RL agents learn directly in the continuous state space —")
print("    no discretisation needed, constant memory, constant per-step cost.")
print("=" * 65)
