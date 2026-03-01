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
    ):
        self.params = params
        self.use_vol = use_volatility_dynamics
        self.include_price = include_price
        self.price_scale = price_scale
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
        norm_inv = self.inventory / self.params.max_inventory
        obs = [norm_inv]
        if self.include_price:
            obs.append((self.mid_price - self.params.initial_price) / self.price_scale)
        if self.use_vol:
            obs.append(self.volatility / self.params.vol_long_run_mean)
        return np.array(obs, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        dim = 1
        if self.include_price:
            dim += 1
        if self.use_vol:
            dim += 1
        return dim

    @property
    def n_actions(self) -> int:
        return self.params.n_actions
