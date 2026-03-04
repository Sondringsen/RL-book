"""Synthetic market-making environment (gym-style API).

Shared by Phase 2 (constant vol) and Phase 3 (OU vol).

Step logic
----------
1. Agent picks action index → (δ_bid, δ_ask) asymmetric half-spreads.
2. Bid/ask fill indicators are drawn from independent Bernoulli
   probabilities derived from Poisson arrival intensities, one per side.
   Fills are blocked when they would breach ±I_max.
3. Cash and inventory are updated from fills.
4. (Phase 3 only) Volatility evolves via a discretised OU process.
5. Mid-price moves with adverse selection:
      Δp = adverse · (ask_filled − bid_filled)  +  σ_t · ε,   ε ~ N(0,1)
   An ask fill signals informed buying (price tends up);
   a bid fill signals informed selling (price tends down).
6. Reward = spread P&L  +  inventory × Δp  −  α · I²
   (terminal step also subtracts  terminal_penalty · |I|).
"""

import numpy as np

from .params import MarketParams


class MarketMakingEnv:

    def __init__(
        self,
        params: MarketParams,
        use_volatility_dynamics: bool = False,
        include_price: bool = False,
        price_scale: float = 10.0,
        seed: int = None,
        # Align RL state space with DP for closer policy match
        discrete_inventory: bool = False,
        price_grid: np.ndarray = None,
        vol_grid: np.ndarray = None,
        # Randomize initial (I, price, vol) for uniform state coverage during training
        random_init: bool = False,
    ):
        self.params = params
        self.use_vol = use_volatility_dynamics
        self.include_price = include_price
        self.price_scale = price_scale
        self.discrete_inventory = discrete_inventory
        self.price_grid = np.asarray(price_grid) if price_grid is not None else None
        self.vol_grid = np.asarray(vol_grid) if vol_grid is not None else None
        self.random_init = random_init
        self.rng = np.random.RandomState(seed)

        self.mid_price: float = params.initial_price
        self.inventory: int = 0
        self.cash: float = 0.0
        self.volatility: float = params.sigma_base
        self.step_count: int = 0

        self.pnl_history: list[float] = []
        self.inventory_history: list[int] = []
        self.spread_history: list[tuple] = []

    # ── gym interface ────────────────────────────────────────────────

    def reset(
        self,
        seed: int = None,
        inventory: int = None,
        price_dev: float = None,
        vol: float = None,
    ) -> np.ndarray:
        """Reset to initial state. If seed is provided, reset RNG for reproducible trajectories.
        If random_init=True was set at construction, (I, price_dev, vol) are sampled from the grid.
        Otherwise, optional inventory/price_dev/vol override defaults (0, 0, sigma_base)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if self.random_init:
            # Always randomize inventory for full state coverage during training
            inv_idx = self.rng.randint(0, self.params.n_inventory_states)
            self.inventory = self.params.index_to_inventory(inv_idx)
            if self.price_grid is not None:
                price_idx = self.rng.randint(0, len(self.price_grid))
                self.mid_price = self.params.initial_price + self.price_grid[price_idx]
            if self.vol_grid is not None:
                vol_idx = self.rng.randint(0, len(self.vol_grid))
                self.volatility = self.vol_grid[vol_idx]
            else:
                self.volatility = self.params.sigma_base
        elif inventory is not None or price_dev is not None or vol is not None:
            self.inventory = inventory if inventory is not None else 0
            self.mid_price = (
                self.params.initial_price + (price_dev if price_dev is not None else 0.0)
            )
            self.volatility = vol if vol is not None else self.params.sigma_base
        else:
            self.mid_price = self.params.initial_price
            self.inventory = 0
            self.volatility = self.params.sigma_base

        # Cash = -inventory * mid_price so initial MtM = 0 (consistent with DP)
        self.cash = -self.inventory * self.mid_price
        self.step_count = 0
        self.pnl_history = [0.0]
        self.inventory_history = [self.inventory]
        self.spread_history = []
        return self._get_obs()

    def step(self, action_idx: int):
        p = self.params
        delta_bid, delta_ask = p.action_to_spreads(action_idx)
        self.spread_history.append((delta_bid, delta_ask))

        # ── order flow (Poisson-derived, independent per side) ───────
        p_bid = p.fill_probability(delta_bid)
        p_ask = p.fill_probability(delta_ask)

        bid_fill = (self.inventory < p.max_inventory) and (self.rng.random() < p_bid)
        ask_fill = (self.inventory > -p.max_inventory) and (self.rng.random() < p_ask)

        spread_pnl = 0.0
        if bid_fill:                       # someone sells to us (hits our bid)
            self.cash -= (self.mid_price - delta_bid)
            self.inventory += 1
            spread_pnl += delta_bid
        if ask_fill:                       # someone buys from us (lifts our ask)
            self.cash += (self.mid_price + delta_ask)
            self.inventory -= 1
            spread_pnl += delta_ask

        # ── volatility dynamics (Phase 3) ────────────────────────────
        if self.use_vol:
            dv = (p.vol_mean_reversion * (p.vol_long_run_mean - self.volatility)
                  + p.vol_of_vol * self.rng.randn())
            self.volatility = max(0.1, self.volatility + dv)

        # ── price move with adverse selection ────────────────────────
        adverse = p.adverse_selection * (int(ask_fill) - int(bid_fill))
        dp = adverse + self.volatility * self.rng.randn()
        self.mid_price += dp

        # ── reward ───────────────────────────────────────────────────
        inventory_pnl = self.inventory * dp
        penalty = p.inventory_penalty * self.inventory ** 2
        reward = spread_pnl + inventory_pnl - penalty

        self.step_count += 1
        done = self.step_count >= p.episode_length

        if done:
            reward -= p.terminal_penalty * abs(self.inventory)

        mtm = self.cash + self.inventory * self.mid_price
        self.pnl_history.append(mtm)
        self.inventory_history.append(self.inventory)

        info = {
            "spread_pnl": spread_pnl,
            "inventory_pnl": inventory_pnl,
            "penalty": penalty,
            "mtm_pnl": mtm,
        }
        return self._get_obs(), reward, done, info

    # ── helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        if self.discrete_inventory:
            inv_onehot = np.zeros(self.params.n_inventory_states, dtype=np.float32)
            inv_onehot[self.params.inventory_to_index(self.inventory)] = 1.0
            parts = [inv_onehot]
        else:
            parts = [np.array([self.inventory / self.params.max_inventory], dtype=np.float32)]

        if self.include_price:
            if self.price_grid is not None:
                price_dev = self.mid_price - self.params.initial_price
                idx = int(np.clip(np.argmin(np.abs(self.price_grid - price_dev)), 0, len(self.price_grid) - 1))
                ph = np.zeros(len(self.price_grid), dtype=np.float32)
                ph[idx] = 1.0
                parts.append(ph)
            else:
                parts.append(np.array([(self.mid_price - self.params.initial_price) / self.price_scale], dtype=np.float32))
        if self.use_vol:
            if self.vol_grid is not None:
                idx = int(np.clip(np.argmin(np.abs(self.vol_grid - self.volatility)), 0, len(self.vol_grid) - 1))
                vh = np.zeros(len(self.vol_grid), dtype=np.float32)
                vh[idx] = 1.0
                parts.append(vh)
            else:
                parts.append(np.array([self.volatility / self.params.vol_long_run_mean], dtype=np.float32))
        return np.concatenate(parts)

    def obs_for_state(self, inventory: int, price_dev: float = 0.0, vol: float = None) -> np.ndarray:
        """Build observation for a given (I, price_dev, vol) — for policy plotting."""
        if self.discrete_inventory:
            inv_onehot = np.zeros(self.params.n_inventory_states, dtype=np.float32)
            inv_onehot[self.params.inventory_to_index(inventory)] = 1.0
            parts = [inv_onehot]
        else:
            parts = [np.array([inventory / self.params.max_inventory], dtype=np.float32)]

        if self.include_price:
            if self.price_grid is not None:
                idx = int(np.clip(np.argmin(np.abs(self.price_grid - price_dev)), 0, len(self.price_grid) - 1))
                ph = np.zeros(len(self.price_grid), dtype=np.float32)
                ph[idx] = 1.0
                parts.append(ph)
            else:
                parts.append(np.array([price_dev / self.price_scale], dtype=np.float32))
        if self.use_vol:
            v = vol if vol is not None else self.params.vol_long_run_mean
            if self.vol_grid is not None:
                idx = int(np.clip(np.argmin(np.abs(self.vol_grid - v)), 0, len(self.vol_grid) - 1))
                vh = np.zeros(len(self.vol_grid), dtype=np.float32)
                vh[idx] = 1.0
                parts.append(vh)
            else:
                parts.append(np.array([v / self.params.vol_long_run_mean], dtype=np.float32))
        return np.concatenate(parts)

    @property
    def state_dim(self) -> int:
        dim = self.params.n_inventory_states if self.discrete_inventory else 1
        if self.include_price:
            dim += len(self.price_grid) if self.price_grid is not None else 1
        if self.use_vol:
            dim += len(self.vol_grid) if self.vol_grid is not None else 1
        return dim

    @property
    def n_actions(self) -> int:
        return self.params.n_actions
