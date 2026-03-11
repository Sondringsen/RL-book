#!/usr/bin/env python3
"""Experiment 3 — State = (inventory, price, volatility), OU vol dynamics.

Compares DP vs RL (regular/continuous) vs RL (discrete) vs RL (distillation).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_3d, simulate_dp_policy_3d

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
N_PRICE_BINS = 15
N_VOL_BINS = 9
PRICE_HALF_RANGE = 10.0
VOL_LO, VOL_HI = 0.3, 2.0
PRICE_SCALE = PRICE_HALF_RANGE
SEED = 42
EVAL_SEED = 123


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 3: (Inventory, Price, Vol) — DP vs RL (regular, discrete, distillation)")
    print("=" * 60)
    print(f"  State dim   : 3 (inventory, price, volatility)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  Vol bins: {N_VOL_BINS}")
    print(f"  Volatility  : OU process")

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/5] DP 3-D value iteration …")
    V_dp, policy_dp, residuals, price_grid, vol_grid = value_iteration_3d(
        params, n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
        n_vol_bins=N_VOL_BINS, vol_lo=VOL_LO, vol_hi=VOL_HI,
    )

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS, terminal_penalty=0.0,
        episode_length=500,
    )

    # ── RL Regular (continuous: inv_norm, price_norm, vol_norm) ──────────
    print("\n[2/5] Training DQN (continuous state) …")
    train_env_reg = MarketMakingEnv(
        train_params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE, vol_grid=None,
        discrete_inventory=False, price_grid=None,
        random_init=True, use_continuous_state=True, seed=SEED,
    )
    agent_reg = DQNAgent(
        state_dim=train_env_reg.state_dim, n_actions=train_env_reg.n_actions,
        lr=2e-4, gamma=train_params.discount, batch_size=256, hidden_dim=256,
        learning_starts=5_000, tau=0.005, seed=SEED,
    )
    agent_reg.train(train_env_reg, n_episodes=1500, epsilon_decay_steps=300_000, verbose=True)

    # ── RL Discrete (one-hot inv + price + vol) ────────────────────────
    print("\n[3/5] Training DQN (discrete state) …")
    train_env_disc = MarketMakingEnv(
        train_params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, vol_grid=vol_grid,
        random_init=True, use_continuous_state=False, seed=SEED,
    )
    agent_disc = DQNAgent(
        state_dim=train_env_disc.state_dim, n_actions=train_env_disc.n_actions,
        lr=2e-4, gamma=train_params.discount, batch_size=256, hidden_dim=256,
        learning_starts=5_000, tau=0.005, seed=SEED,
    )
    agent_disc.train(train_env_disc, n_episodes=1500, epsilon_decay_steps=300_000, verbose=True)

    # ── RL Distillation ───────────────────────────────────────────────
    print("\n[4/5] Training DQN via policy distillation …")
    env_obs = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, vol_grid=vol_grid,
        use_continuous_state=False, seed=SEED,
    )
    agent_dist = DQNAgent(
        state_dim=env_obs.state_dim, n_actions=env_obs.n_actions,
        lr=1e-3, gamma=params.discount, batch_size=128, hidden_dim=128, seed=SEED,
    )
    agent_dist.train_distillation(
        env_obs, policy_dp, params, price_grid=price_grid, vol_grid=vol_grid,
        n_epochs=300, batch_size=128, verbose=True, log_interval=30,
    )

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[5/5] Evaluating (1 000 episodes each) …")
    eval_base = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, vol_grid=vol_grid,
        seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy_3d(
        eval_base, policy_dp, params, price_grid, vol_grid,
        n_episodes=1000, episode_seed_base=EVAL_SEED,
    )

    eval_reg = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE, vol_grid=None,
        discrete_inventory=False, use_continuous_state=True, seed=EVAL_SEED,
    )
    reg_stats = agent_reg.evaluate(eval_reg, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_disc = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, vol_grid=vol_grid,
        use_continuous_state=False, seed=EVAL_SEED,
    )
    disc_stats = agent_disc.evaluate(eval_disc, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_dist = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, vol_grid=vol_grid,
        use_continuous_state=False, seed=EVAL_SEED,
    )
    dist_stats = agent_dist.evaluate(eval_dist, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, reg_stats, disc_stats, dist_stats)

    # ── Plots (policy at mid price, mid vol) ───────────────────────────
    print("\nPlotting …")
    mid_p = N_PRICE_BINS // 2
    mid_v = N_VOL_BINS // 2
    dp_bids = [params.action_to_spreads(policy_dp[i, mid_p, mid_v])[0] for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i, mid_p, mid_v])[1] for i in range(params.n_inventory_states)]

    def get_rl_spreads(agent, env):
        bids, asks = [], []
        price_dev = float(price_grid[mid_p])
        vol = float(vol_grid[mid_v])
        for I in inventories:
            obs = env.obs_for_state(I, price_dev=price_dev, vol=vol)
            mask = DQNAgent.boundary_mask(I, params)
            a = agent.select_action(obs, valid_mask=mask)
            db, da = params.action_to_spreads(a)
            bids.append(db)
            asks.append(da)
        return bids, asks

    reg_bids, reg_asks = get_rl_spreads(agent_reg, eval_reg)
    disc_bids, disc_asks = get_rl_spreads(agent_disc, eval_disc)
    dist_bids, dist_asks = get_rl_spreads(agent_dist, eval_dist)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, reg_bids, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 0].step(inventories, disc_bids, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 0].step(inventories, dist_bids, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads (mid price, mid vol)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, reg_asks, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 1].step(inventories, disc_asks, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 1].step(inventories, dist_asks, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads (mid price, mid vol)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [dp_stats["episode_rewards"], reg_stats["episode_rewards"],
                   disc_stats["episode_rewards"], dist_stats["episode_rewards"]]
    lo, hi = min(r.min() for r in all_rewards), max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.35, label="DP", edgecolor="k")
    axes[1, 0].hist(reg_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (regular)", edgecolor="k")
    axes[1, 0].hist(disc_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (discrete)", edgecolor="k")
    axes[1, 0].hist(dist_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (distill)", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(reg_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (regular)", edgecolor="k", align="left")
    axes[1, 1].hist(disc_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (discrete)", edgecolor="k", align="left")
    axes[1, 1].hist(dist_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (distill)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Experiment 3: (Inventory, Price, Vol) — DP vs RL (regular, discrete, distillation)")
    plt.tight_layout()
    plt.savefig("results/exp3_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_comparison.png")
    plt.show()


def _print_table(dp_stats, reg_stats, disc_stats, dist_stats):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'Regular':>10s}  {'Discrete':>10s}  {'Distill':>10s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        v = [dp_stats[key], reg_stats[key], disc_stats[key], dist_stats[key]]
        print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))


if __name__ == "__main__":
    main()
