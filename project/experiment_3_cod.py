"""Experiment 3: DP runtime scaling with the 3-D state space (inventory, price, volatility).

Shows how value_iteration_3d wall-clock time grows as we expand:
  (a) inventory space  — I_max = 2 … 30
  (b) price grid       — n_price_bins = 5 … 25
  (c) volatility grid  — n_vol_bins = 3 … 19   ← the new dimension
  (d) all four simultaneously — the full curse of dimensionality

All panels plot wall-clock time against |S| × |A| on log-log axes.

Complexity recap
----------------
  2-D VI: cost per iter ≈ O(n_inv · n_price² · |A|)
  3-D VI: cost per iter ≈ O(n_inv · n_vol · n_price² · |A|)
             + O(n_inv · n_price · n_vol²)   [einsum over vol transition]

Adding the volatility dimension multiplies the per-iteration work by n_vol,
so what was already expensive in experiment 2 becomes dramatically worse here.
Panel (d) shows the combined effect: growing all four axes together causes
super-multiplicative blowup — the defining symptom of the curse of
dimensionality.
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
from project.market_making.dp_solver import value_iteration_3d

# ── constants ─────────────────────────────────────────────────────────────────

BENCH_ITERS = 30   # fixed VI sweeps per config (3-D VI is much slower than 2-D)


# ── timing helpers ────────────────────────────────────────────────────────────

def time_inv(I_max: int, n_price: int = 9, n_vol: int = 5, n_spread: int = 4):
    """Vary inventory size, fix everything else."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_3d(params, n_price_bins=n_price, n_vol_bins=n_vol,
                       max_iter=BENCH_ITERS, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    nS = params.n_inventory_states * n_price * n_vol
    return elapsed, nS, params.n_actions


def time_price(n_price: int, I_max: int = 5, n_vol: int = 5, n_spread: int = 4):
    """Vary price bins, fix everything else."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_3d(params, n_price_bins=n_price, n_vol_bins=n_vol,
                       max_iter=BENCH_ITERS, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    nS = params.n_inventory_states * n_price * n_vol
    return elapsed, nS, params.n_actions


def time_vol(n_vol: int, I_max: int = 5, n_price: int = 9, n_spread: int = 4):
    """Vary volatility bins, fix everything else."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_3d(params, n_price_bins=n_price, n_vol_bins=n_vol,
                       max_iter=BENCH_ITERS, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    nS = params.n_inventory_states * n_price * n_vol
    return elapsed, nS, params.n_actions


def time_joint(I_max: int, n_price: int, n_vol: int, n_spread: int):
    """All four dimensions grow together."""
    params = MarketParams(
        max_inventory=I_max,
        spread_options=tuple(np.linspace(0.5, 2.5, n_spread)),
    )
    t0 = time.perf_counter()
    value_iteration_3d(params, n_price_bins=n_price, n_vol_bins=n_vol,
                       max_iter=BENCH_ITERS, tol=0.0, verbose=False)
    elapsed = time.perf_counter() - t0
    nS = params.n_inventory_states * n_price * n_vol
    return elapsed, nS, params.n_actions


# ── sweep configurations ──────────────────────────────────────────────────────

print("Benchmarking 3-D VI over inventory size …")
inv_results = []
for I_max in [2, 3, 5, 7, 10, 15, 20, 25, 30]:
    elapsed, nS, nA = time_inv(I_max)
    inv_results.append((nS * nA, elapsed))
    print(f"  I_max={I_max:>2d}  |S|={nS:>5d}  |A|={nA:>3d}  |S||A|={nS*nA:>7d}  t={elapsed:.3f}s")

print("\nBenchmarking 3-D VI over price grid size …")
price_results = []
for n_price in [25, 31, 41, 51, 61, 71, 81, 100, 125, 150]:
    elapsed, nS, nA = time_price(n_price)
    price_results.append((nS * nA, elapsed))
    print(f"  n_price={n_price:>2d}  |S|={nS:>5d}  |A|={nA:>3d}  |S||A|={nS*nA:>7d}  t={elapsed:.3f}s")

print("\nBenchmarking 3-D VI over volatility grid size …")
vol_results = []
for n_vol in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
    elapsed, nS, nA = time_vol(n_vol)
    vol_results.append((nS * nA, elapsed))
    print(f"  n_vol={n_vol:>2d}  |S|={nS:>5d}  |A|={nA:>3d}  |S||A|={nS*nA:>7d}  t={elapsed:.3f}s")

# All four dimensions grow together.
# Each step multiplies |S| and |A|, demonstrating the curse of dimensionality.
JOINT_CONFIGS = [
    # (I_max, n_price, n_vol, n_spread)
    (2,  5,  3, 2),
    (3,  7,  4, 3),
    (5,  9,  5, 4),
    (7, 11,  6, 5),
    (10, 13,  7, 5),
    (12, 15,  8, 6),
    (15, 17,  9, 7),
]

print("\nBenchmarking 3-D VI with all dimensions growing (curse of dimensionality) …")
joint_results = []
for I_max, n_price, n_vol, n_spread in JOINT_CONFIGS:
    elapsed, nS, nA = time_joint(I_max, n_price, n_vol, n_spread)
    joint_results.append((nS * nA, elapsed))
    print(f"  I_max={I_max:>2d}  n_price={n_price:>2d}  n_vol={n_vol:>2d}  n_spread={n_spread}"
          f"  |S|={nS:>6d}  |A|={nA:>4d}  |S||A|={nS*nA:>8d}  t={elapsed:.3f}s")


# ── plotting ──────────────────────────────────────────────────────────────────

fig, axes_2d = plt.subplots(2, 2, figsize=(13, 10))
axes = axes_2d.flat
fig.suptitle(
    f"Dynamic Programming (3-D): Wall-clock Time vs State–Action Space Size\n"
    f"State = (inventory, price, volatility)   —   {BENCH_ITERS} VI sweeps per config",
    fontsize=13,
)

COLORS = {"inv": "#2563EB", "price": "#DC2626", "vol": "#EA580C", "joint": "#7C3AED"}
MARKERS = {"inv": "o", "price": "s", "vol": "v", "joint": "D"}


def _fit_line(xs, ys):
    """Fit log-log line, return (slope, fitted_ys)."""
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
    "3-D VI (inventory)",
    "Inventory scaling\n(I_max = 2 … 30, fixed n_price=9, n_vol=5, 4×4 actions)",
)

_plot_panel(
    axes[1], price_results,
    COLORS["price"], MARKERS["price"],
    "3-D VI (price grid)",
    "Price grid scaling  ← O(n_price²) per iter\n(n_price = 5 … 25, I_max=5, n_vol=5, 4×4 actions)",
)

_plot_panel(
    axes[2], vol_results,
    COLORS["vol"], MARKERS["vol"],
    "3-D VI (volatility grid)",
    "Volatility grid scaling  ← multiplies price cost\n(n_vol = 3 … 19, I_max=5, n_price=9, 4×4 actions)",
)

# ── Panel 4: curse of dimensionality ─────────────────────────────────────────
ax4 = axes[3]
xs = np.array([r[0] for r in joint_results], dtype=float)
ys = np.array([r[1] for r in joint_results], dtype=float)
slope, fitted = _fit_line(xs, ys)

ax4.scatter(xs, ys, color=COLORS["joint"], marker=MARKERS["joint"], zorder=5, s=70,
            label=f"3-D VI — all dims grow  (slope ≈ {slope:.2f})")
ax4.plot(xs, ys, color=COLORS["joint"], lw=2, alpha=0.8)
ax4.plot(xs, fitted, color=COLORS["joint"], lw=1, ls="--", alpha=0.5, label="log-log fit")

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("|S| × |A|  (log scale)", fontsize=10)
ax4.set_ylabel("Wall-clock time (s, log scale)", fontsize=10)
ax4.set_title(
    "Curse of Dimensionality\n"
    "(inventory + price + volatility + actions all grow together)",
    fontsize=11,
)
ax4.legend(fontsize=9)
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax4.grid(True, which="both", ls=":", alpha=0.4)

# Annotate the config at each data point so readers can see what drives growth
# for (I_max, n_p, n_v, n_sp), (sa, t) in zip(JOINT_CONFIGS, zip(xs, ys)):
#     ax4.annotate(
#         f"I={I_max}\np={n_p}\nv={n_v}\na={n_sp}²",
#         xy=(sa, t), xytext=(sa * 1.05, t * 0.6),
#         fontsize=6, color="#4B5563",
#         arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.6),
#     )

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "results/experiment_3_dp_scaling.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Key takeaway:")
print("  - 3-D VI cost per iter: O(n_inv · n_vol · n_price² · |A|)")
print("    Adding the vol dimension multiplies the 2-D cost by n_vol.")
print("  - Inventory: sparse transitions → linear in n_inv.")
print("  - Price: dense Gaussian matrix → quadratic in n_price.")
print("  - Volatility: also dense → linear multiplier on the price cost,")
print("    plus O(n_inv · n_price · n_vol²) for the vol einsum.")
print("  - Joint scaling: every dimension multiplies the others.")
print("    RL sidesteps all of this — no grid, constant per-step cost.")
print("=" * 65)
