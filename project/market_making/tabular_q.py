"""Tabular Q-learning agent for the (inventory, price) market-making MDP.

With only 231 states, tabular Q-learning can converge to the same optimum as DP
(given sufficient visits). No function approximation — stores Q(s,a) exactly.
"""

import random
from typing import Optional

import numpy as np

from .params import MarketParams


def _find_nearest(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


class TabularQAgent:
    """Tabular Q-learning for state = (inventory_idx, price_idx)."""

    def __init__(
        self,
        params: MarketParams,
        price_grid: np.ndarray,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.params = params
        self.price_grid = np.asarray(price_grid)
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

        n_inv = params.n_inventory_states
        n_price = len(price_grid)
        # Q[si, sp, a] — initialized to 0
        self.Q = np.zeros((n_inv, n_price, n_actions), dtype=np.float64)

    def _obs_to_state(self, obs: np.ndarray) -> tuple[int, int]:
        """Decode one-hot obs (inv_onehot | price_onehot) to (si, sp)."""
        n_inv = self.params.n_inventory_states
        n_price = len(self.price_grid)
        si = int(np.argmax(obs[:n_inv]))
        sp = int(np.argmax(obs[n_inv : n_inv + n_price]))
        return si, sp

    def _state_from_env(self, inventory: int, price_dev: float) -> tuple[int, int]:
        """Map (inventory, price_dev) to (si, sp)."""
        si = self.params.inventory_to_index(inventory)
        sp = _find_nearest(self.price_grid, price_dev)
        sp = int(np.clip(sp, 0, len(self.price_grid) - 1))
        return si, sp

    def select_action(
        self,
        state_or_obs,
        epsilon: float = 0.0,
        valid_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select action. Accepts (si, sp) tuple or one-hot obs array."""
        if isinstance(state_or_obs, (tuple, list)) and len(state_or_obs) == 2:
            si, sp = int(state_or_obs[0]), int(state_or_obs[1])
        else:
            obs = np.asarray(state_or_obs)
            si, sp = self._obs_to_state(obs)

        q_vals = self.Q[si, sp, :].copy()
        if valid_mask is not None:
            q_vals[~valid_mask] = -np.inf

        if self.rng.random() < epsilon:
            valid = np.where(valid_mask)[0] if valid_mask is not None else np.arange(self.n_actions)
            if len(valid) == 0:
                return 0
            return int(self.rng.choice(valid))
        return int(np.argmax(q_vals))

    @staticmethod
    def boundary_mask(inventory: int, params: MarketParams) -> np.ndarray:
        """Same as DQNAgent — valid actions at inventory boundaries."""
        n = params.n_spread_options
        n_actions = n * n
        mask = np.ones(n_actions, dtype=bool)
        if inventory >= params.max_inventory:
            for a in range(n_actions):
                if a // n != n - 1:
                    mask[a] = False
        elif inventory <= -params.max_inventory:
            for a in range(n_actions):
                if a % n != n - 1:
                    mask[a] = False
        return mask

    def update(self, si: int, sp: int, action: int, reward: float, next_si: int, next_sp: int, done: bool):
        """Q-learning update: Q(s,a) += α * (r + γ max_a' Q(s',a') - Q(s,a))."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_si, next_sp, :])
        td_error = target - self.Q[si, sp, action]
        self.Q[si, sp, action] += self.alpha * td_error

    def train(
        self,
        env,
        n_episodes: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 500_000,
        verbose: bool = True,
        use_boundary_mask: bool = True,
        log_interval: int = 100,
    ) -> tuple[list[float], dict]:
        """Train with Q-learning. Env must have inventory, mid_price, params, price_grid."""
        episode_rewards: list[float] = []
        total_steps = 0

        for ep in range(n_episodes):
            obs = env.reset()
            si, sp = self._obs_to_state(obs)
            total_reward = 0.0
            done = False

            while not done:
                mask = (
                    self.boundary_mask(env.inventory, env.params)
                    if use_boundary_mask
                    else None
                )
                decay_frac = min(1.0, total_steps / max(epsilon_decay_steps, 1))
                epsilon = epsilon_end + (1.0 - decay_frac) * (epsilon_start - epsilon_end)

                action = self.select_action((si, sp), epsilon=epsilon, valid_mask=mask)
                next_obs, reward, done, _ = env.step(action)

                next_si, next_sp = self._obs_to_state(next_obs)
                self.update(si, sp, action, reward, next_si, next_sp, done)

                si, sp = next_si, next_sp
                total_reward += reward
                total_steps += 1

            episode_rewards.append(total_reward)

            if verbose and (ep + 1) % log_interval == 0:
                recent = episode_rewards[-log_interval:]
                print(
                    f"  ep {ep+1:>5d}/{n_episodes}  "
                    f"reward(last {log_interval})={np.mean(recent):+.2f}  "
                    f"ε={epsilon:.3f}  steps={total_steps}"
                )

        return episode_rewards, {}

    def get_policy(self) -> np.ndarray:
        """Return greedy policy: policy[si, sp] = argmax_a Q[si, sp, a]."""
        return np.argmax(self.Q, axis=-1)

    def evaluate(
        self,
        env,
        n_episodes: int = 1000,
        episode_seed_base: Optional[int] = None,
    ) -> dict:
        """Evaluate greedy policy. Same return format as DQNAgent.evaluate."""
        episode_rewards: list[float] = []
        episode_pnls: list[float] = []
        final_inventories: list[int] = []
        all_inventories: list[int] = []

        for i in range(n_episodes):
            if episode_seed_base is not None:
                obs = env.reset(seed=episode_seed_base + i)
            else:
                obs = env.reset()
            si, sp = self._obs_to_state(obs)
            total_reward = 0.0
            done = False

            while not done:
                mask = self.boundary_mask(env.inventory, env.params)
                action = self.select_action((si, sp), epsilon=0.0, valid_mask=mask)
                all_inventories.append(env.inventory)
                obs, reward, done, _ = env.step(action)
                si, sp = self._obs_to_state(obs)
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_pnls.append(env.pnl_history[-1])
            final_inventories.append(env.inventory)

        rew = np.array(episode_rewards)
        pnl = np.array(episode_pnls)

        return {
            "mean_reward": np.mean(rew),
            "std_reward": np.std(rew),
            "sharpe": np.mean(rew) / (np.std(rew) + 1e-8),
            "mean_pnl": np.mean(pnl),
            "std_pnl": np.std(pnl),
            "mean_abs_inventory": np.mean(np.abs(all_inventories)),
            "max_abs_inventory": int(np.max(np.abs(all_inventories))),
            "mean_final_inventory": np.mean(np.abs(final_inventories)),
            "episode_rewards": rew,
            "episode_pnls": pnl,
            "final_inventories": np.array(final_inventories),
        }
