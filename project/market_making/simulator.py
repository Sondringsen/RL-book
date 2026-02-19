"""Synthetic market-making environment (gym-style API).

Shared by Phase 2 (constant vol) and Phase 3 (OU vol).

Step logic
----------
1. Agent picks action index  →  half-spread  δ.
2. Bid/ask fill indicators are drawn from independent Bernoulli(p(δ)).
   - Fills are blocked when they would breach ±I_max.
3. Cash and inventory are updated from fills.
4. (Phase 3 only) Volatility evolves via a discretised OU process.
5. Mid-price moves:  Δp = σ_t · ε,   ε ~ N(0,1).
6. Reward = spread P&L  +  inventory × Δp  −  α · I_new²
   (terminal step also subtracts  terminal_penalty · |I|).
"""

import numpy as np

from .params import MarketParams


class MarketMakingEnv:

    def __init__(
        self,
        params: MarketParams,
        use_volatility_dynamics: bool = False,
        seed: int = None,
    ):
        self.params = params
        self.use_vol = use_volatility_dynamics
        self.rng = np.random.RandomState(seed)

        self.mid_price: float = params.initial_price
        self.inventory: int = 0
        self.cash: float = 0.0
        self.volatility: float = params.sigma_base
        self.step_count: int = 0

        self.pnl_history: list[float] = []
        self.inventory_history: list[int] = []
        self.spread_history: list[float] = []

    # ── gym interface ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.mid_price = self.params.initial_price
        self.inventory = 0
        self.cash = 0.0
        self.volatility = self.params.sigma_base
        self.step_count = 0
        self.pnl_history = [0.0]
        self.inventory_history = [0]
        self.spread_history = []
        return self._get_obs()

    def step(self, action_idx: int):
        p = self.params
        delta = p.spread_options[action_idx]
        self.spread_history.append(delta)

        # ── fills ────────────────────────────────────────────────────
        fill_prob = p.fill_probability(delta)
        bid_fill = (self.inventory < p.max_inventory) and (self.rng.random() < fill_prob)
        ask_fill = (self.inventory > -p.max_inventory) and (self.rng.random() < fill_prob)

        spread_pnl = 0.0
        if bid_fill:                       # someone sells to us
            self.cash -= (self.mid_price - delta)
            self.inventory += 1
            spread_pnl += delta
        if ask_fill:                       # someone buys from us
            self.cash += (self.mid_price + delta)
            self.inventory -= 1
            spread_pnl += delta

        # ── price dynamics ───────────────────────────────────────────
        if self.use_vol:
            dv = (p.vol_mean_reversion * (p.vol_long_run_mean - self.volatility)
                  + p.vol_of_vol * self.rng.randn())
            self.volatility = max(0.1, self.volatility + dv)

        dp = self.volatility * self.rng.randn()
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
        norm_inv = self.inventory / self.params.max_inventory
        if self.use_vol:
            norm_vol = self.volatility / self.params.vol_long_run_mean
            return np.array([norm_inv, norm_vol], dtype=np.float32)
        return np.array([norm_inv], dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 2 if self.use_vol else 1

    @property
    def n_actions(self) -> int:
        return self.params.n_actions
