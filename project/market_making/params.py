"""Phase 1: MDP formulation for market making.

The market maker posts asymmetric bid/ask quotes at
    bid = mid − δ_bid,   ask = mid + δ_ask.
Customer orders arrive as independent Poisson processes with intensity
    λ(δ) = A · exp(−k · δ)
where tighter spreads attract more flow but expose the agent to
inventory risk.

Adverse selection: fills carry information about the subsequent price
move.  When someone buys from us (ask fill), the price tends to rise;
when someone sells to us (bid fill), the price tends to fall.

Simplified (DP-solvable) MDP — Phase 2
    State:   I_t ∈ {−I_max, …, I_max}           (inventory only)
    Action:  (δ_bid, δ_ask) pair                  (asymmetric half-spreads)
    Reward:  spread_captured + E[inv_pnl] − α I²

Extended (RL-solvable) MDP — Phase 3
    State:   (I_t, σ_t)  with σ_t following an OU process
    Action:  same discrete set of (δ_bid, δ_ask) pairs
    Reward:  spread_captured + I·Δp − α I²
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MarketParams:
    """All tuneable knobs for both the DP and RL formulations."""

    # ── Price dynamics ───────────────────────────────────────────────
    initial_price: float = 100.0
    sigma_base: float = 0.25

    # OU volatility dynamics (Phase 3 only)
    vol_mean_reversion: float = 0.15
    vol_long_run_mean: float = 1.0
    vol_of_vol: float = 0.2

    # ── Order arrival  λ(δ) = A · exp(−k · δ) ───────────────────────
    arrival_base: float = 0.5
    arrival_decay: float = 1.5

    # ── Adverse selection ────────────────────────────────────────────
    adverse_selection: float = 0.2   # E[Δp] shift per fill direction

    # ── Inventory ────────────────────────────────────────────────────
    max_inventory: int = 5

    # ── Reward ───────────────────────────────────────────────────────
    inventory_penalty: float = 0.01
    terminal_penalty: float = 1.0

    # ── Discrete spread options (each side picks independently) ──────
    spread_options: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)

    # ── Episode / discount ───────────────────────────────────────────
    episode_length: int = 500
    discount: float = 0.99

    # ── Derived helpers ──────────────────────────────────────────────

    @property
    def n_spread_options(self) -> int:
        return len(self.spread_options)

    @property
    def n_actions(self) -> int:
        """Action space = all (δ_bid, δ_ask) pairs."""
        return len(self.spread_options) ** 2

    @property
    def n_inventory_states(self) -> int:
        return 2 * self.max_inventory + 1

    @property
    def inventory_states(self) -> np.ndarray:
        return np.arange(-self.max_inventory, self.max_inventory + 1)

    def action_to_spreads(self, action_idx: int) -> Tuple[float, float]:
        """Map flat action index → (δ_bid, δ_ask)."""
        n = len(self.spread_options)
        bid_idx = action_idx // n
        ask_idx = action_idx % n
        return self.spread_options[bid_idx], self.spread_options[ask_idx]

    def spreads_to_action(self, bid_idx: int, ask_idx: int) -> int:
        """Map (bid_index, ask_index) → flat action index."""
        return bid_idx * len(self.spread_options) + ask_idx

    def fill_probability(self, spread: float) -> float:
        """P(≥1 arrival on one side) = 1 − exp(−λ(δ)), Poisson discretised."""
        intensity = self.arrival_base * np.exp(-self.arrival_decay * spread)
        return 1.0 - np.exp(-intensity)

    def inventory_to_index(self, inventory: int) -> int:
        return inventory + self.max_inventory

    def index_to_inventory(self, idx: int) -> int:
        return idx - self.max_inventory
