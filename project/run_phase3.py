#!/usr/bin/env python3
"""Phase 3 — DQN market maker with stochastic volatility vs. DP baseline.

Pipeline
--------
1. Solve the Phase 2 DP (inventory-only state) — serves as baseline.
2. Train a DQN agent on the Phase 3 environment (state = [I, σ]).
3. Evaluate both agents in the stochastic-volatility environment.
4. Compare performance and visualise the learned policy.

Outputs
-------
  results/phase3_comparison.png       training curve, reward dists, skew, inv
  results/phase3_policy_heatmap.png   bid & ask spread over (I, σ) grid
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration, simulate_dp_policy


def print_table(dp_stats: dict, rl_stats: dict):
    rows = [
        ("Mean Reward",       "mean_reward",          ".2f"),
        ("Std Reward",        "std_reward",           ".2f"),
        ("Reward Sharpe",     "sharpe",               ".3f"),
        ("Mean MtM PnL",     "mean_pnl",             ".2f"),
        ("Std MtM PnL",      "std_pnl",              ".2f"),
        ("Mean |Inventory|",  "mean_abs_inventory",   ".2f"),
        ("Max |Inventory|",   "max_abs_inventory",    "d"),
        ("Mean |Final Inv|",  "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<22s}  {'DP baseline':>14s}  {'DQN agent':>14s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        d = format(dp_stats[key], fmt)
        r = format(rl_stats[key], fmt)
        print(f"  {name:<22s}  {d:>14s}  {r:>14s}")


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams()

    print("=" * 60)
    print("Phase 3: DQN (with stochastic σ) vs DP baseline")
    print("=" * 60)

    # ── Step 1: DP baseline ──────────────────────────────────────────
    print("\n[1/4] Solving DP baseline (inventory-only state) …")
    V_dp, policy_dp, _ = value_iteration(params, verbose=False)
    print("  DP converged.")

    # ── Step 2: train DQN (Double DQN, Huber loss, soft targets, step ε-decay) ─
    print("\n[2/4] Training DQN (state = [inventory, volatility]) …")
    train_env = MarketMakingEnv(params, use_volatility_dynamics=True, seed=42)

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=5e-4,
        gamma=params.discount,
        batch_size=64,
        hidden_dim=128,
        learning_starts=10_000,
        tau=0.005,
        clip_reward=False,
        seed=42,
    )

    ep_rewards, losses = agent.train(
        train_env,
        n_episodes=3000,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=400_000,
        verbose=True,
    )

    # ── Step 3: evaluate both ────────────────────────────────────────
    print("\n[3/4] Evaluating both agents (1 000 episodes, stochastic σ) …")
    eval_env_dp = MarketMakingEnv(params, use_volatility_dynamics=True, seed=123)
    dp_stats = simulate_dp_policy(eval_env_dp, policy_dp, params, n_episodes=1000)

    eval_env_rl = MarketMakingEnv(params, use_volatility_dynamics=True, seed=123)
    rl_stats = agent.evaluate(eval_env_rl, n_episodes=1000)

    # ── Step 4: report ───────────────────────────────────────────────
    print("\n[4/4] Results\n")
    print_table(dp_stats, rl_stats)

    # ── comparison plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # training curve
    win = 100
    smoothed = np.convolve(ep_rewards, np.ones(win) / win, mode="valid")
    axes[0, 0].plot(smoothed, lw=0.8)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward (smoothed)")
    axes[0, 0].set_title("DQN Training Curve")
    axes[0, 0].grid(True, alpha=0.3)

    # reward distributions
    lo = min(dp_stats["episode_rewards"].min(), rl_stats["episode_rewards"].min())
    hi = max(dp_stats["episode_rewards"].max(), rl_stats["episode_rewards"].max())
    bins = np.linspace(lo, hi, 50)
    axes[0, 1].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.5,
                     label="DP", edgecolor="k")
    axes[0, 1].hist(rl_stats["episode_rewards"], bins=bins, alpha=0.5,
                     label="DQN", edgecolor="k")
    axes[0, 1].axvline(dp_stats["mean_reward"], color="C0", ls="--")
    axes[0, 1].axvline(rl_stats["mean_reward"], color="C1", ls="--")
    axes[0, 1].set_xlabel("Episode Reward")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Reward Distribution: DP vs DQN")
    axes[0, 1].legend()

    # learned policy skew vs DP for several volatility levels
    vol_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    inv_norm = np.linspace(-1.0, 1.0, 11)
    for vol in vol_levels:
        skews = []
        for ni in inv_norm:
            obs = np.array([ni, vol / params.vol_long_run_mean], dtype=np.float32)
            a = agent.select_action(obs)
            db, da = params.action_to_spreads(a)
            skews.append(db - da)
        axes[1, 0].plot(inv_norm * params.max_inventory, skews,
                        "-o", ms=4, label=f"σ = {vol:.2f}")
    dp_skew = []
    for i in range(params.n_inventory_states):
        db, da = params.action_to_spreads(policy_dp[i])
        dp_skew.append(db - da)
    axes[1, 0].plot(params.inventory_states, dp_skew,
                    "k--s", ms=5, lw=2, label="DP (no σ)")
    axes[1, 0].axhline(0, color="gray", ls=":", lw=0.8)
    axes[1, 0].set_xlabel("Inventory")
    axes[1, 0].set_ylabel("Skew  (δ_bid − δ_ask)")
    axes[1, 0].set_title("Learned Skew: DQN vs DP")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # final inventory distributions
    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins,
                     alpha=0.5, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(rl_stats["final_inventories"], bins=inv_bins,
                     alpha=0.5, label="DQN", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results/phase3_comparison.png", dpi=150)
    print("\n→ saved results/phase3_comparison.png")

    # ── policy heatmaps (bid and ask) ────────────────────────────────
    fig2, (ax_bid, ax_ask) = plt.subplots(1, 2, figsize=(14, 6))
    inv_grid = np.linspace(-1.0, 1.0, 21)
    vol_grid = np.linspace(0.3, 2.0, 21)
    bid_map = np.zeros((len(vol_grid), len(inv_grid)))
    ask_map = np.zeros((len(vol_grid), len(inv_grid)))
    for i, v in enumerate(vol_grid):
        for j, ni in enumerate(inv_grid):
            obs = np.array([ni, v / params.vol_long_run_mean], dtype=np.float32)
            a = agent.select_action(obs)
            db, da = params.action_to_spreads(a)
            bid_map[i, j] = db
            ask_map[i, j] = da

    extent = [inv_grid[0] * params.max_inventory,
              inv_grid[-1] * params.max_inventory,
              vol_grid[0], vol_grid[-1]]

    im1 = ax_bid.imshow(bid_map, aspect="auto", origin="lower",
                        extent=extent, cmap="viridis")
    ax_bid.set_xlabel("Inventory")
    ax_bid.set_ylabel("Volatility  σ")
    ax_bid.set_title("DQN Policy: Bid Half-Spread  δ*_bid")
    plt.colorbar(im1, ax=ax_bid, label="δ*_bid")

    im2 = ax_ask.imshow(ask_map, aspect="auto", origin="lower",
                        extent=extent, cmap="viridis")
    ax_ask.set_xlabel("Inventory")
    ax_ask.set_ylabel("Volatility  σ")
    ax_ask.set_title("DQN Policy: Ask Half-Spread  δ*_ask")
    plt.colorbar(im2, ax=ax_ask, label="δ*_ask")

    plt.tight_layout()
    plt.savefig("results/phase3_policy_heatmap.png", dpi=150)
    print("→ saved results/phase3_policy_heatmap.png")

    plt.show()


if __name__ == "__main__":
    main()
