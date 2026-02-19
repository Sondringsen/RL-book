"""Phase 1: MDP formulation for market making.

The market maker posts symmetric bid/ask quotes at  mid - δ  and  mid + δ.
Customer orders arrive as independent Poisson processes with intensity
    λ(δ) = A · exp(-k · δ)
where tighter spreads attract more flow but expose the agent to inventory risk.

Simplified (DP-solvable) MDP — Phase 2
    State:   I_t ∈ {-I_max, …, I_max}           (inventory only)
    Action:  δ_t ∈ {δ_1, …, δ_n}                (half-spread)
    Reward:  spread_captured − α I²              (profit vs. inventory penalty)

    Mid-price is a random walk with constant σ; because E[ΔP]=0 the price
    dimension drops out and value iteration is tractable in O(|I|·|A|).

Extended (RL-solvable) MDP — Phase 3
    State:   (I_t, σ_t)  with σ_t following an OU process
    Action:  same discrete set
    Reward:  spread_captured + I·Δp − α I²       (includes price-move P&L)

    The stochastic volatility makes the optimal spread depend on σ_t,
    which DP cannot capture without blowing up the state space.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MarketParams:
    """All tuneable knobs for both the DP and RL formulations."""

    # ── Price dynamics ───────────────────────────────────────────────
    initial_price: float = 100.0
    sigma_base: float = 1.0          # constant vol used in Phase 2

    # OU volatility dynamics (Phase 3 only)
    vol_mean_reversion: float = 0.15  # κ
    vol_long_run_mean: float = 1.0    # σ̄
    vol_of_vol: float = 0.2           # ξ

    # ── Order arrival  λ(δ) = A · exp(−k · δ) ───────────────────────
    arrival_base: float = 0.5         # A
    arrival_decay: float = 1.5        # k

    # ── Inventory ────────────────────────────────────────────────────
    max_inventory: int = 5

    # ── Reward ───────────────────────────────────────────────────────
    inventory_penalty: float = 0.01   # α  (quadratic penalty per step)
    terminal_penalty: float = 1.0     # per-unit liquidation cost at T

    # ── Discrete action set (half-spreads in ticks) ──────────────────
    spread_options: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)

    # ── Episode / discount ───────────────────────────────────────────
    episode_length: int = 200
    discount: float = 0.99

    # ── Derived helpers ──────────────────────────────────────────────
    @property
    def n_actions(self) -> int:
        return len(self.spread_options)

    @property
    def n_inventory_states(self) -> int:
        return 2 * self.max_inventory + 1

    @property
    def inventory_states(self) -> np.ndarray:
        return np.arange(-self.max_inventory, self.max_inventory + 1)

    def fill_probability(self, spread: float) -> float:
        """P(fill on one side in one time-step) = 1 − exp(−λ(δ))."""
        intensity = self.arrival_base * np.exp(-self.arrival_decay * spread)
        return 1.0 - np.exp(-intensity)

    def inventory_to_index(self, inventory: int) -> int:
        return inventory + self.max_inventory

    def index_to_inventory(self, idx: int) -> int:
        return idx - self.max_inventory
