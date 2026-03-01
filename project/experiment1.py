#!/usr/bin/env python3
"""Experiment 1 — Inventory-only state, limited actions, constant vol.

State : inventory  I ∈ {−I_max, …, I_max}
Action: (δ_bid, δ_ask) from a small spread grid
Vol   : constant σ

Compares DP (value iteration) vs RL (DQN) side by side.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration, simulate_dp_policy

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
SEED = 42
EVAL_SEED = 123  # Same seed for DP/RL evaluation → identical trajectories


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 1: Inventory-only state, DP vs RL")
    print("=" * 60)
    print(f"  State dim   : 1 (inventory)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Volatility  : constant σ = {params.sigma_base}")

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/4] DP value iteration …")
    V_dp, policy_dp, residuals = value_iteration(params)

    # ── RL (discrete inventory + more training → align with DP) ────────
    print("\n[2/4] Training DQN (state = one-hot inventory, aligned with DP) …")
    train_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        discrete_inventory=True, seed=SEED,
    )

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=2e-4,
        gamma=params.discount,
        batch_size=64,
        hidden_dim=128,
        learning_starts=10_000,
        tau=0.005,
        seed=SEED,
    )
    ep_rewards, _ = agent.train(
        train_env, n_episodes=8000,
        epsilon_start=1.0, epsilon_end=0.02,
        epsilon_decay_steps=500_000, verbose=True,
    )

    # ── evaluate (same env config + per-episode seeds → identical trajectories) ─
    print("\n[3/4] Evaluating (1 000 episodes, constant σ, paired trajectories) …")
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        discrete_inventory=True, seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy(
        eval_env, policy_dp, params, n_episodes=1000, episode_seed_base=EVAL_SEED
    )
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        discrete_inventory=True, seed=EVAL_SEED,
    )
    rl_stats = agent.evaluate(eval_env, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, rl_stats)

    # ── plots ─────────────────────────────────────────────────────────
    print("\n[4/4] Plotting …")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Optimal policy: bids only (DP vs RL)
    dp_bids = [params.action_to_spreads(policy_dp[i])[0] for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i])[1] for i in range(params.n_inventory_states)]
    rl_bids, rl_asks = [], []
    for I in inventories:
        obs = eval_env.obs_for_state(I)
        a = agent.select_action(obs)
        db, da = params.action_to_spreads(a)
        rl_bids.append(db)
        rl_asks.append(da)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP bid")
    axes[0, 0].step(inventories, rl_bids, "r-s", where="mid", ms=5, label="RL bid", ls="--")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Half-spread")
    axes[0, 0].set_title("Optimal Policy: Bid Spreads")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Optimal policy: asks only (DP vs RL)
    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP ask")
    axes[0, 1].step(inventories, rl_asks, "r-s", where="mid", ms=5, label="RL ask", ls="--")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Half-spread")
    axes[0, 1].set_title("Optimal Policy: Ask Spreads")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # (0,2) Reward distribution
    lo = min(dp_stats["episode_rewards"].min(), rl_stats["episode_rewards"].min())
    hi = max(dp_stats["episode_rewards"].max(), rl_stats["episode_rewards"].max())
    bins = np.linspace(lo, hi, 50)
    axes[0, 2].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.5, label="DP", edgecolor="k")
    axes[0, 2].hist(rl_stats["episode_rewards"], bins=bins, alpha=0.5, label="RL", edgecolor="k")
    axes[0, 2].axvline(dp_stats["mean_reward"], color="C0", ls="--")
    axes[0, 2].axvline(rl_stats["mean_reward"], color="C1", ls="--")
    axes[0, 2].set_xlabel("Episode Reward")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Reward Distribution")
    axes[0, 2].legend()

    # (1,0) Cumulative PnL
    n_ep = len(dp_stats["episode_pnls"])
    cum_dp = np.cumsum(dp_stats["episode_pnls"])
    cum_rl = np.cumsum(rl_stats["episode_pnls"])
    axes[1, 0].plot(range(1, n_ep + 1), cum_dp, label="DP", lw=1)
    axes[1, 0].plot(range(1, n_ep + 1), cum_rl, label="RL", lw=1)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cumulative MtM PnL")
    axes[1, 0].set_title("Cumulative PnL")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Final inventory distribution
    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.5,
                     label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(rl_stats["final_inventories"], bins=inv_bins, alpha=0.5,
                     label="RL", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend()

    # Hide unused (1,2)
    axes[1, 2].set_visible(False)

    fig.suptitle("Experiment 1: Inventory-Only State (DP vs RL)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("results/exp1_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_comparison.png")
    plt.show()


def _print_table(dp_stats, rl_stats):
    rows = [
        ("Mean Reward",       "mean_reward",          ".2f"),
        ("Std Reward",        "std_reward",           ".2f"),
        ("Reward Sharpe",     "sharpe",               ".3f"),
        ("Mean MtM PnL",     "mean_pnl",             ".2f"),
        ("Mean |Inventory|",  "mean_abs_inventory",   ".2f"),
        ("Max |Inventory|",   "max_abs_inventory",    "d"),
        ("Mean |Final Inv|",  "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<22s}  {'DP':>14s}  {'RL (DQN)':>14s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        print(f"  {name:<22s}  {format(dp_stats[key], fmt):>14s}  {format(rl_stats[key], fmt):>14s}")


if __name__ == "__main__":
    main()
