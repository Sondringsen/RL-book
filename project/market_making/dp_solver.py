"""Phase 2: value-iteration solver for the simplified (inventory-only) MDP.

Bellman equation
----------------
    V(I) = max_δ  Σ_{I'} P(I'|I,δ) · [r(I,δ,I') + γ V(I')]

Transition probabilities (p = fill_probability(δ)):
    Interior I:
        P(I  |I,δ) = p² + (1-p)²     (both fill or neither)
        P(I-1|I,δ) = p(1-p)           (ask fills only → sell 1)
        P(I+1|I,δ) = (1-p)p           (bid fills only → buy 1)
    Boundary I = I_max:   only the ask side is active.
    Boundary I = -I_max:  only the bid side is active.

Rewards include spread capture and a quadratic inventory penalty
applied to the *resulting* inventory I'.
"""

import numpy as np

from .params import MarketParams


def value_iteration(
    params: MarketParams,
    tol: float = 1e-8,
    max_iter: int = 10000,
    verbose: bool = True,
):
    """Run value iteration.

    Returns
    -------
    V : ndarray, shape (n_states,)
    policy : ndarray[int], shape (n_states,)  — action indices
    residuals : list[float]
    """
    n_s = params.n_inventory_states
    n_a = params.n_actions
    I_max = params.max_inventory
    gamma = params.discount
    alpha = params.inventory_penalty

    V = np.zeros(n_s)
    policy = np.zeros(n_s, dtype=int)
    residuals: list[float] = []

    for it in range(max_iter):
        V_new = np.full(n_s, -np.inf)

        for s in range(n_s):
            I = params.index_to_inventory(s)

            best_q, best_a = -np.inf, 0
            for a in range(n_a):
                delta = params.spread_options[a]
                p = params.fill_probability(delta)

                at_upper = I >= I_max
                at_lower = I <= -I_max

                if at_upper and at_lower:               # degenerate I_max == 0
                    q = -alpha * I**2 + gamma * V[s]

                elif at_upper:                           # only ask active
                    s_ask = params.inventory_to_index(I - 1)
                    q = (p     * (delta - alpha * (I - 1)**2 + gamma * V[s_ask])
                         + (1 - p) * (      - alpha * I**2      + gamma * V[s]))

                elif at_lower:                           # only bid active
                    s_bid = params.inventory_to_index(I + 1)
                    q = (p     * (delta - alpha * (I + 1)**2 + gamma * V[s_bid])
                         + (1 - p) * (      - alpha * I**2      + gamma * V[s]))

                else:                                    # both sides active
                    s_bid = params.inventory_to_index(I + 1)
                    s_ask = params.inventory_to_index(I - 1)
                    q = (p**2       * (2*delta - alpha * I**2       + gamma * V[s])
                         + p*(1-p)  * (delta   - alpha * (I-1)**2   + gamma * V[s_ask])
                         + (1-p)*p  * (delta   - alpha * (I+1)**2   + gamma * V[s_bid])
                         + (1-p)**2 * (        - alpha * I**2       + gamma * V[s]))

                if q > best_q:
                    best_q, best_a = q, a

            V_new[s] = best_q
            policy[s] = best_a

        residual = np.max(np.abs(V_new - V))
        residuals.append(residual)
        V = V_new

        if verbose and (it + 1) % 200 == 0:
            print(f"  iter {it+1:>5d}  residual = {residual:.2e}")

        if residual < tol:
            if verbose:
                print(f"  converged at iter {it+1}  (residual {residual:.2e})")
            break

    return V, policy, residuals


def simulate_dp_policy(
    env,
    policy: np.ndarray,
    params: MarketParams,
    n_episodes: int = 1000,
):
    """Roll out the DP policy in any MarketMakingEnv (ignores volatility)."""
    episode_rewards = []
    episode_pnls = []
    final_inventories = []
    all_inventories: list[int] = []

    for _ in range(n_episodes):
        env.reset()
        total_reward = 0.0
        done = False
        while not done:
            s_idx = params.inventory_to_index(env.inventory)
            action = int(policy[s_idx])
            _, reward, done, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_pnls.append(env.pnl_history[-1])
        final_inventories.append(env.inventory)
        all_inventories.extend(env.inventory_history)

    rewards_arr = np.array(episode_rewards)
    pnls_arr = np.array(episode_pnls)

    return {
        "mean_reward": np.mean(rewards_arr),
        "std_reward": np.std(rewards_arr),
        "sharpe": np.mean(rewards_arr) / (np.std(rewards_arr) + 1e-8),
        "mean_pnl": np.mean(pnls_arr),
        "std_pnl": np.std(pnls_arr),
        "mean_abs_inventory": np.mean(np.abs(all_inventories)),
        "max_abs_inventory": int(np.max(np.abs(all_inventories))),
        "mean_final_inventory": np.mean(np.abs(final_inventories)),
        "episode_rewards": rewards_arr,
        "episode_pnls": pnls_arr,
        "final_inventories": np.array(final_inventories),
    }
