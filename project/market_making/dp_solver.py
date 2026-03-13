"""Finite-horizon backward induction for the market-making MDP.

Solves V_t(s) = max_a E[r + γ V_{t+1}(s')] for t = T-1, …, 0
with terminal value  V_T(I) = −terminal_penalty · |I|.

The policy is time-dependent: π_t(s) may differ across t, allowing the
agent to reduce inventory as the episode end approaches.

Transitions (p_b = fill_prob(δ_bid), p_a = fill_prob(δ_ask)):
    Interior I:
        P(I  |I,δ) = p_b·p_a + (1−p_b)(1−p_a)
        P(I−1|I,δ) = (1−p_b)·p_a             (ask fills only → sell 1)
        P(I+1|I,δ) = p_b·(1−p_a)             (bid fills only → buy 1)
    Boundary I = I_max:   only the ask side is active.
    Boundary I = −I_max:  only the bid side is active.

Adverse selection: E[Δp | fills] = adverse · (ask_fill − bid_fill).
The expected inventory P&L (post-fill I' × E[Δp]) is folded into
the immediate reward for each fill scenario.
"""

import numpy as np
from scipy.stats import norm as _norm

from .params import MarketParams


def _gaussian_transition_matrix(grid, shift, sigma):
    """P(land in bin k | start at bin j) for  X' = grid[j] + shift + sigma*N(0,1)."""
    n = len(grid)
    T = np.zeros((n, n))
    for j in range(n):
        mu = grid[j] + shift
        edges = np.empty(n + 1)
        edges[0] = -1e10
        edges[-1] = 1e10
        edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
        cdf_vals = _norm.cdf(edges, loc=mu, scale=sigma)
        T[j] = np.diff(cdf_vals)
    return T


def value_iteration(
    params: MarketParams,
    verbose: bool = True,
):
    """Finite-horizon backward induction (1-D: inventory only).

    Returns
    -------
    V : ndarray, shape (n_states,)        — value at t=0
    policy : ndarray[int], shape (T, n_states) — time-dependent policy
    """
    n_s = params.n_inventory_states
    n_a = params.n_actions
    I_max = params.max_inventory
    gamma = params.discount
    alpha = params.inventory_penalty
    adv = params.adverse_selection
    T = params.episode_length
    tp = params.terminal_penalty

    V_next = np.zeros(n_s)
    for s in range(n_s):
        V_next[s] = -tp * abs(params.index_to_inventory(s))

    policy = np.zeros((T, n_s), dtype=int)

    for t in range(T - 1, -1, -1):
        V_new = np.full(n_s, -np.inf)

        for s in range(n_s):
            I = params.index_to_inventory(s)

            best_q, best_a = -np.inf, 0
            for a in range(n_a):
                d_bid, d_ask = params.action_to_spreads(a)
                p_b = params.fill_probability(d_bid)
                p_a = params.fill_probability(d_ask)

                at_upper = I >= I_max
                at_lower = I <= -I_max

                if at_upper and at_lower:
                    q = -alpha * I**2 + gamma * V_next[s]

                elif at_upper:
                    s_ask = params.inventory_to_index(I - 1)
                    r_ask  = d_ask + (I - 1) * adv - alpha * (I - 1)**2
                    r_none = -alpha * I**2
                    q = (p_a     * (r_ask  + gamma * V_next[s_ask])
                         + (1 - p_a) * (r_none + gamma * V_next[s]))

                elif at_lower:
                    s_bid = params.inventory_to_index(I + 1)
                    r_bid  = d_bid - (I + 1) * adv - alpha * (I + 1)**2
                    r_none = -alpha * I**2
                    q = (p_b     * (r_bid  + gamma * V_next[s_bid])
                         + (1 - p_b) * (r_none + gamma * V_next[s]))

                else:
                    s_bid = params.inventory_to_index(I + 1)
                    s_ask = params.inventory_to_index(I - 1)

                    r_both = d_bid + d_ask - alpha * I**2
                    r_bid  = d_bid - (I + 1) * adv - alpha * (I + 1)**2
                    r_ask  = d_ask + (I - 1) * adv - alpha * (I - 1)**2
                    r_none = -alpha * I**2

                    q = (p_b * p_a         * (r_both + gamma * V_next[s])
                         + p_b * (1 - p_a) * (r_bid  + gamma * V_next[s_bid])
                         + (1 - p_b) * p_a * (r_ask  + gamma * V_next[s_ask])
                         + (1 - p_b) * (1 - p_a) * (r_none + gamma * V_next[s]))

                if q >= best_q:
                    best_q, best_a = q, a

            V_new[s] = best_q
            policy[t, s] = best_a

        V_next = V_new

        if verbose and (T - t) % 50 == 0:
            print(f"  backward step {T - t:>5d}/{T}")

    if verbose:
        print(f"  finite-horizon backward induction complete (T={T})")

    return V_next, policy


def simulate_dp_policy(
    env,
    policy: np.ndarray,
    params: MarketParams,
    n_episodes: int = 1000,
    episode_seed_base: int = None,
):
    """Roll out the time-dependent DP policy in any MarketMakingEnv."""
    episode_rewards = []
    episode_pnls = []
    final_inventories = []
    all_inventories: list[int] = []

    for i in range(n_episodes):
        if episode_seed_base is not None:
            env.reset(seed=episode_seed_base + i)
        else:
            env.reset()
        total_reward = 0.0
        done = False
        while not done:
            t = env.step_count
            s_idx = params.inventory_to_index(env.inventory)
            action = int(policy[t, s_idx])
            _, reward, done, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_pnls.append(env.pnl_history[-1])
        final_inventories.append(env.inventory)
        all_inventories.extend(env.inventory_history)

    rewards_arr = np.array(episode_rewards)
    pnls_arr = np.array(episode_pnls)

    return _pack_stats(rewards_arr, pnls_arr, final_inventories, all_inventories)


def _pack_stats(rewards_arr, pnls_arr, final_inventories, all_inventories):
    rewards_arr = np.asarray(rewards_arr, dtype=float)
    pnls_arr = np.asarray(pnls_arr, dtype=float)
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


def _find_nearest(grid, value):
    return int(np.argmin(np.abs(grid - value)))


# ═══════════════════════════════════════════════════════════════════════
#  2-D value iteration  — state = (inventory, price)
# ═══════════════════════════════════════════════════════════════════════

def value_iteration_2d(
    params: MarketParams,
    n_price_bins: int = 21,
    price_half_range: float = 10.0,
    verbose: bool = True,
):
    """Finite-horizon backward induction (2-D: inventory × price).

    Returns
    -------
    V : ndarray, shape (n_inv, n_price)           — value at t=0
    policy : ndarray[int], shape (T, n_inv, n_price) — time-dependent policy
    price_grid : ndarray
    """
    n_inv = params.n_inventory_states
    n_a = params.n_actions
    I_max = params.max_inventory
    gamma = params.discount
    alpha = params.inventory_penalty
    adv = params.adverse_selection
    sigma = params.sigma_base
    T = params.episode_length
    tp = params.terminal_penalty

    price_grid = np.linspace(-price_half_range, price_half_range, n_price_bins)

    shifts = sorted({-adv, 0.0, adv})
    trans = {s: _gaussian_transition_matrix(price_grid, s, sigma) for s in shifts}
    exp_dp = {s: trans[s] @ price_grid - price_grid for s in shifts}

    V_next = np.zeros((n_inv, n_price_bins))
    for si in range(n_inv):
        V_next[si, :] = -tp * abs(params.index_to_inventory(si))

    policy = np.zeros((T, n_inv, n_price_bins), dtype=int)

    for t in range(T - 1, -1, -1):
        V_new = np.full_like(V_next, -np.inf)

        for si in range(n_inv):
            I = params.index_to_inventory(si)
            at_upper = I >= I_max
            at_lower = I <= -I_max

            best_Q = np.full(n_price_bins, -np.inf)
            best_A = np.zeros(n_price_bins, dtype=int)

            for a in range(n_a):
                db, da = params.action_to_spreads(a)
                pb = params.fill_probability(db)
                pa = params.fill_probability(da)

                Q = np.zeros(n_price_bins)

                if at_upper and at_lower:
                    Q = -alpha * I**2 + I * exp_dp[0.0] + gamma * (trans[0.0] @ V_next[si])

                elif at_upper:
                    Ia, sa = I - 1, params.inventory_to_index(I - 1)
                    Q_ask = da - alpha * Ia**2 + Ia * exp_dp[adv] + gamma * (trans[adv] @ V_next[sa])
                    Q_no = -alpha * I**2 + I * exp_dp[0.0] + gamma * (trans[0.0] @ V_next[si])
                    Q = pa * Q_ask + (1 - pa) * Q_no

                elif at_lower:
                    Ib, sb = I + 1, params.inventory_to_index(I + 1)
                    Q_bid = db - alpha * Ib**2 + Ib * exp_dp[-adv] + gamma * (trans[-adv] @ V_next[sb])
                    Q_no = -alpha * I**2 + I * exp_dp[0.0] + gamma * (trans[0.0] @ V_next[si])
                    Q = pb * Q_bid + (1 - pb) * Q_no

                else:
                    Ib, sb = I + 1, params.inventory_to_index(I + 1)
                    Ia, sa = I - 1, params.inventory_to_index(I - 1)

                    Q_both = db + da - alpha * I**2 + I * exp_dp[0.0] + gamma * (trans[0.0] @ V_next[si])
                    Q_bid = db - alpha * Ib**2 + Ib * exp_dp[-adv] + gamma * (trans[-adv] @ V_next[sb])
                    Q_ask = da - alpha * Ia**2 + Ia * exp_dp[adv] + gamma * (trans[adv] @ V_next[sa])
                    Q_no = -alpha * I**2 + I * exp_dp[0.0] + gamma * (trans[0.0] @ V_next[si])

                    Q = (pb * pa * Q_both
                         + pb * (1 - pa) * Q_bid
                         + (1 - pb) * pa * Q_ask
                         + (1 - pb) * (1 - pa) * Q_no)

                better = Q >= best_Q
                best_Q = np.where(better, Q, best_Q)
                best_A = np.where(better, a, best_A)

            V_new[si] = best_Q
            policy[t, si] = best_A

        V_next = V_new.copy()

        if verbose and (T - t) % 50 == 0:
            print(f"  backward step {T - t:>5d}/{T}")

    if verbose:
        print(f"  finite-horizon backward induction complete (T={T})")

    return V_next, policy, price_grid


def simulate_dp_policy_2d(env, policy, params, price_grid, n_episodes=1000, episode_seed_base=None):
    episode_rewards, episode_pnls, final_inventories = [], [], []
    all_inventories: list[int] = []

    for i in range(n_episodes):
        if episode_seed_base is not None:
            env.reset(seed=episode_seed_base + i)
        else:
            env.reset()
        total_reward = 0.0
        done = False
        while not done:
            t = env.step_count
            si = params.inventory_to_index(env.inventory)
            sp = _find_nearest(price_grid, env.mid_price - params.initial_price)
            action = int(policy[t, si, sp])
            _, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        episode_pnls.append(env.pnl_history[-1])
        final_inventories.append(env.inventory)
        all_inventories.extend(env.inventory_history)

    return _pack_stats(episode_rewards, episode_pnls, final_inventories, all_inventories)


# ═══════════════════════════════════════════════════════════════════════
#  3-D value iteration  — state = (inventory, price, volatility)
# ═══════════════════════════════════════════════════════════════════════

def value_iteration_3d(
    params: MarketParams,
    n_price_bins: int = 15,
    price_half_range: float = 10.0,
    n_vol_bins: int = 9,
    vol_lo: float = 0.3,
    vol_hi: float = 2.0,
    verbose: bool = True,
):
    """Finite-horizon backward induction (3-D: inventory × price × vol).

    Returns
    -------
    V : ndarray, shape (n_inv, n_price, n_vol)                — value at t=0
    policy : ndarray[int], shape (T, n_inv, n_price, n_vol)   — time-dependent
    price_grid : ndarray
    vol_grid : ndarray
    """
    n_inv = params.n_inventory_states
    n_a = params.n_actions
    I_max = params.max_inventory
    gamma = params.discount
    alpha = params.inventory_penalty
    adv = params.adverse_selection
    T_horizon = params.episode_length
    tp = params.terminal_penalty

    price_grid = np.linspace(-price_half_range, price_half_range, n_price_bins)
    vol_grid = np.linspace(vol_lo, vol_hi, n_vol_bins)

    kappa = params.vol_mean_reversion
    theta = params.vol_long_run_mean
    xi = params.vol_of_vol
    T_vol = np.zeros((n_vol_bins, n_vol_bins))
    for v in range(n_vol_bins):
        mu_v = vol_grid[v] + kappa * (theta - vol_grid[v])
        T_vol[v] = _gaussian_transition_matrix(vol_grid, mu_v - vol_grid[v], xi)[v]

    shifts = sorted({-adv, 0.0, adv})
    T_price = {}
    exp_dp = {}
    for vi in range(n_vol_bins):
        sig = vol_grid[vi]
        for s in shifts:
            Tp = _gaussian_transition_matrix(price_grid, s, sig)
            T_price[(vi, s)] = Tp
            exp_dp[(vi, s)] = Tp @ price_grid - price_grid

    V_next = np.zeros((n_inv, n_price_bins, n_vol_bins))
    for si in range(n_inv):
        V_next[si, :, :] = -tp * abs(params.index_to_inventory(si))

    policy = np.zeros((T_horizon, n_inv, n_price_bins, n_vol_bins), dtype=int)

    for t in range(T_horizon - 1, -1, -1):
        V_new = np.full_like(V_next, -np.inf)

        EV_vol = np.einsum("ipw,vw->ipv", V_next, T_vol)

        for si in range(n_inv):
            I = params.index_to_inventory(si)
            at_upper = I >= I_max
            at_lower = I <= -I_max

            for vi in range(n_vol_bins):
                best_Q = np.full(n_price_bins, -np.inf)
                best_A = np.zeros(n_price_bins, dtype=int)

                for a in range(n_a):
                    db, da = params.action_to_spreads(a)
                    pb = params.fill_probability(db)
                    pa = params.fill_probability(da)

                    def _q(s_next, shift, base_r, Iprime):
                        return (base_r
                                + Iprime * exp_dp[(vi, shift)]
                                + gamma * (T_price[(vi, shift)] @ EV_vol[s_next, :, vi]))

                    if at_upper and at_lower:
                        Q = _q(si, 0.0, -alpha * I**2, I)

                    elif at_upper:
                        Ia, sa = I - 1, params.inventory_to_index(I - 1)
                        Q = (pa * _q(sa, adv, da - alpha * Ia**2, Ia)
                             + (1 - pa) * _q(si, 0.0, -alpha * I**2, I))

                    elif at_lower:
                        Ib, sb = I + 1, params.inventory_to_index(I + 1)
                        Q = (pb * _q(sb, -adv, db - alpha * Ib**2, Ib)
                             + (1 - pb) * _q(si, 0.0, -alpha * I**2, I))

                    else:
                        Ib, sb = I + 1, params.inventory_to_index(I + 1)
                        Ia, sa = I - 1, params.inventory_to_index(I - 1)
                        Q = (pb * pa * _q(si, 0.0, db + da - alpha * I**2, I)
                             + pb * (1 - pa) * _q(sb, -adv, db - alpha * Ib**2, Ib)
                             + (1 - pb) * pa * _q(sa, adv, da - alpha * Ia**2, Ia)
                             + (1 - pb) * (1 - pa) * _q(si, 0.0, -alpha * I**2, I))

                    better = Q >= best_Q
                    best_Q = np.where(better, Q, best_Q)
                    best_A = np.where(better, a, best_A)

                V_new[si, :, vi] = best_Q
                policy[t, si, :, vi] = best_A

        V_next = V_new.copy()

        if verbose and (T_horizon - t) % 50 == 0:
            print(f"  backward step {T_horizon - t:>5d}/{T_horizon}")

    if verbose:
        print(f"  finite-horizon backward induction complete (T={T_horizon})")

    return V_next, policy, price_grid, vol_grid


def simulate_dp_policy_3d(env, policy, params, price_grid, vol_grid, n_episodes=1000, episode_seed_base=None):
    episode_rewards, episode_pnls, final_inventories = [], [], []
    all_inventories: list[int] = []

    for i in range(n_episodes):
        if episode_seed_base is not None:
            env.reset(seed=episode_seed_base + i)
        else:
            env.reset()
        total_reward = 0.0
        done = False
        while not done:
            t = env.step_count
            si = params.inventory_to_index(env.inventory)
            sp = _find_nearest(price_grid, env.mid_price - params.initial_price)
            sv = _find_nearest(vol_grid, env.volatility)
            action = int(policy[t, si, sp, sv])
            _, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        episode_pnls.append(env.pnl_history[-1])
        final_inventories.append(env.inventory)
        all_inventories.extend(env.inventory_history)

    return _pack_stats(episode_rewards, episode_pnls, final_inventories, all_inventories)
