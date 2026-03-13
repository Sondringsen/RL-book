#!/usr/bin/env python3
"""Experiment 1 — State = inventory only (1D).

Compares DP vs RL (regular/continuous) vs RL (distillation).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, VecMarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration, simulate_dp_policy
from market_making.gpu_config import gpu_batch_size, gpu_hidden_dim, gpu_info

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
SIGMA_BASE = 0.25
SEED = 42
EVAL_SEED = 123


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS, sigma_base=SIGMA_BASE)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 1: Inventory-only state — DP vs RL (regular, distillation)")
    print("=" * 60)
    print(f"  State dim   : 1 (inventory)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Volatility  : constant σ = {params.sigma_base}")
    print(f"  Device      : {gpu_info()}")

    BATCH_TRAIN = gpu_batch_size(256)
    BATCH_DISTILL = gpu_batch_size(64)
    HIDDEN_TRAIN = gpu_hidden_dim(128)
    HIDDEN_DISTILL = gpu_hidden_dim(64)

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/5] DP value iteration …")
    V_dp, policy_dp = value_iteration(params)

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        sigma_base=SIGMA_BASE,
        episode_length=300,
    )

    # ── RL Regular (continuous state, vectorized for GPU) ───────────────
    print("\n[2/5] Training DQN (continuous state, vectorized) …")
    def _make_env_reg():
        return MarketMakingEnv(
            train_params, use_volatility_dynamics=False,
            include_price=False, discrete_inventory=False,
            random_init=True, use_continuous_state=True, seed=SEED,
        )
    train_env_reg = VecMarketMakingEnv(32, _make_env_reg)
    agent_reg = DQNAgent(
        state_dim=train_env_reg.state_dim, n_actions=train_env_reg.n_actions,
        lr=2e-4, gamma=train_params.discount, batch_size=BATCH_TRAIN, hidden_dim=HIDDEN_TRAIN,
        learning_starts=5_000, tau=0.005, seed=SEED,
    )
    agent_reg.train(train_env_reg, n_episodes=500, epsilon_decay_steps=150_000, verbose=True)

    # ── RL Distillation ────────────────────────────────────────────────
    print("\n[3/4] Training DQN via policy distillation …")
    env_obs = MarketMakingEnv(
        params, use_volatility_dynamics=False, include_price=False,
        discrete_inventory=True, use_continuous_state=False, seed=SEED,
    )
    agent_dist = DQNAgent(
        state_dim=env_obs.state_dim, n_actions=env_obs.n_actions,
        lr=1e-3, gamma=params.discount, batch_size=BATCH_DISTILL, hidden_dim=HIDDEN_DISTILL, seed=SEED,
    )
    agent_dist.train_distillation(env_obs, policy_dp[0], params, n_epochs=300, verbose=True)

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[4/4] Evaluating (1 000 episodes each) …")
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False, include_price=False,
        discrete_inventory=False, seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy(eval_env, policy_dp, params, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_reg = MarketMakingEnv(params, include_price=False, discrete_inventory=False, use_continuous_state=True, seed=EVAL_SEED)
    reg_stats = agent_reg.evaluate(eval_reg, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_dist = MarketMakingEnv(params, include_price=False, discrete_inventory=True, use_continuous_state=False, seed=EVAL_SEED)
    dist_stats = agent_dist.evaluate(eval_dist, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, reg_stats, dist_stats)

    print("\n  Experiment 1 — Mean MtM PnL:")
    print(f"    DP: {dp_stats['mean_pnl']:.2f}  RL (regular): {reg_stats['mean_pnl']:.2f}  RL (distill): {dist_stats['mean_pnl']:.2f}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nPlotting …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dp_bids = [params.action_to_spreads(policy_dp[0, i])[0] for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[0, i])[1] for i in range(params.n_inventory_states)]

    def get_rl_spreads(agent, env):
        bids, asks = [], []
        for I in inventories:
            obs = env.obs_for_state(I, price_dev=0.0)
            mask = DQNAgent.boundary_mask(I, params)
            a = agent.select_action(obs, valid_mask=mask)
            db, da = params.action_to_spreads(a)
            bids.append(db)
            asks.append(da)
        return bids, asks

    reg_bids, reg_asks = get_rl_spreads(agent_reg, eval_reg)
    dist_bids, dist_asks = get_rl_spreads(agent_dist, eval_dist)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, reg_bids, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 0].step(inventories, dist_bids, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, reg_asks, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 1].step(inventories, dist_asks, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [dp_stats["episode_rewards"], reg_stats["episode_rewards"], dist_stats["episode_rewards"]]
    lo = min(r.min() for r in all_rewards)
    hi = max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.4, label="DP", edgecolor="k")
    axes[1, 0].hist(reg_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (regular)", edgecolor="k")
    axes[1, 0].hist(dist_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (distill)", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.4, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(reg_stats["final_inventories"], bins=inv_bins, alpha=0.4, label="RL (regular)", edgecolor="k", align="left")
    axes[1, 1].hist(dist_stats["final_inventories"], bins=inv_bins, alpha=0.4, label="RL (distill)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend(fontsize=8)

    # Cumulative PnL
    n_ep = len(dp_stats["episode_pnls"])
    fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(10, 4))
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(reg_stats["episode_pnls"]), label="RL (regular)", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dist_stats["episode_pnls"]), label="RL (distill)", lw=1)
    ax_pnl.set_xlabel("Episode")
    ax_pnl.set_ylabel("Cumulative MtM PnL")
    ax_pnl.set_title("Experiment 1: Cumulative PnL")
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp1_pnl.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_pnl.png")

    fig.suptitle("Experiment 1: Inventory-only — DP vs RL (regular, distillation)")
    fig.tight_layout()
    fig.savefig("results/exp1_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_comparison.png")
    plt.show()


def _print_table(dp_stats, reg_stats, dist_stats):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'Regular':>10s}  {'Distill':>10s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        v = [dp_stats[key], reg_stats[key], dist_stats[key]]
        print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))


if __name__ == "__main__":
    main()
