#!/usr/bin/env python3
"""Experiment 2 — Policy distillation: training a DQN agent to mimic a DP-derived (optimal) policy.

This script demonstrates policy distillation from dynamic programming (DP) to deep Q-learning (DQN)
using supervised learning. The process works as follows:

1. **Solve with DP ("Teacher")**:  
   - The market-making environment is configured with a discrete state space: inventory levels and price bins.
   - Dynamic programming (value iteration) computes the optimal state-value function and the optimal policy for each discrete (inventory, price) pair.
   - The resulting optimal policy grid assigns a (bid, ask) spread action to every state.

2. **Create the RL agent ("Student")**:  
   - A DQNAgent is constructed with the same state and action structure as the DP grid to facilitate direct imitation.

3. **Policy Distillation (Supervised Training):**  
   - Rather than learning from environment interaction, the DQN is trained purely via supervised learning:
       - The "training set" consists of (state, optimal action) pairs from the DP policy grid.
       - For each training example, the DQN’s predicted action probabilities (logits) over all actions are compared to the one-hot label given by the DP policy, using a cross-entropy loss.
       - The DQN is optimized to predict the same action as the DP policy for any given state.
   - No environment steps are taken during this phase; learning is entirely data-driven from the policy table.

4. **Evaluation:**  
   - Both the teacher (DP) and the student (DQN) are evaluated on 1,000 common market-making episodes with identical volatility and starting conditions.
   - Statistics (rewards, PnL, inventory behavior) and policy outputs are compared.

5. **Visualization:**  
   - The script generates comparison plots showing:
       - Policy agreement (bid/ask profiles as a function of inventory at mid-price)
       - Reward distributions and cumulative PnL
       - Final inventory histograms
       - Heatmaps of bid/ask actions across the full state space for both teacher and student policies

The result: If distillation is successful, the DQN learns to closely replicate the DP policy’s actions in all visited states, even without direct environment exploration.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_2d, simulate_dp_policy_2d

# Larger action space: 5×5 = 25 spread combinations
SPREAD_OPTIONS = (0.5, 1.0, 1.5, 2.0, 2.5)
N_PRICE_BINS = 21
PRICE_HALF_RANGE = 10.0
PRICE_SCALE = PRICE_HALF_RANGE
SIGMA_BASE = 0.25
SEED = 42
EVAL_SEED = 123


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS, sigma_base=SIGMA_BASE)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 2: Policy Distillation (DQN → mimic DP)")
    print("=" * 60)
    print(f"  State dim   : discrete (one-hot inv + price)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  range=[{-PRICE_HALF_RANGE}, {PRICE_HALF_RANGE}]")
    print(f"  Volatility  : constant σ = {params.sigma_base}")

    # 1. --- DP (teacher): Optimal policy via value iteration ---
    print("\n[1/4] DP 2-D value iteration (teacher policy) …")
    V_dp, policy_dp, price_grid = value_iteration_2d(
        params, n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
    )

    # 2. --- RL (student): Initialize DQNAgent to imitate DP policy via distillation ---
    env_for_obs = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid,
        use_continuous_state=False,  # Discrete to match DP grid
        seed=SEED,
    )

    agent = DQNAgent(
        state_dim=env_for_obs.state_dim,
        n_actions=env_for_obs.n_actions,
        lr=1e-3,
        gamma=params.discount,
        batch_size=128,
        hidden_dim=128,
        learning_starts=0,
        tau=0.005,
        seed=SEED,
    )

    # 3. --- Supervised policy distillation (DQN "student" fitted to DP "teacher") ---
    print("\n[2/4] Training DQN via policy distillation (supervised) …")
    agent.train_distillation(
        env_for_obs,
        policy_teacher=policy_dp[0],
        params=params,
        price_grid=price_grid,
        n_epochs=500,
        batch_size=128,
        lr=1e-3,
        verbose=True,
        log_interval=50,
    )

    # 4. --- Evaluation: Compare student and teacher on the same environment and seeds ---
    print("\n[3/4] Evaluating (1 000 episodes, paired trajectories) …")
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid,
        use_continuous_state=False,
        seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy_2d(
        eval_env, policy_dp, params, price_grid,
        n_episodes=1000, episode_seed_base=EVAL_SEED,
    )
    rl_stats = agent.evaluate(eval_env, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, rl_stats)

    # ── plots ─────────────────────────────────────────────────────────
    print("\n[4/4] Plotting …")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    mid_idx = N_PRICE_BINS // 2
    dp_bids = [params.action_to_spreads(policy_dp[0, i, mid_idx])[0]
               for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[0, i, mid_idx])[1]
               for i in range(params.n_inventory_states)]
    rl_bids, rl_asks = [], []
    for I in inventories:
        obs = eval_env.obs_for_state(I, price_dev=0.0)
        mask = DQNAgent.boundary_mask(I, params)
        a = agent.select_action(obs, valid_mask=mask)
        db, da = params.action_to_spreads(a)
        rl_bids.append(db)
        rl_asks.append(da)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP (teacher)")
    axes[0, 0].step(inventories, rl_bids, "r--s", where="mid", ms=5, label="RL (distilled)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Optimal Policy: Bid Spreads (Price = Mid)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP (teacher)")
    axes[0, 1].step(inventories, rl_asks, "r--s", where="mid", ms=5, label="RL (distilled)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Optimal Policy: Ask Spreads (Price = Mid)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    lo = min(dp_stats["episode_rewards"].min(), rl_stats["episode_rewards"].min())
    hi = max(dp_stats["episode_rewards"].max(), rl_stats["episode_rewards"].max())
    bins = np.linspace(lo, hi, 50)
    axes[0, 2].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.5, label="DP", edgecolor="k")
    axes[0, 2].hist(rl_stats["episode_rewards"], bins=bins, alpha=0.5, label="RL (distilled)", edgecolor="k")
    axes[0, 2].axvline(dp_stats["mean_reward"], color="C0", ls="--")
    axes[0, 2].axvline(rl_stats["mean_reward"], color="C1", ls="--")
    axes[0, 2].set_xlabel("Episode Reward")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Reward Distribution")
    axes[0, 2].legend()

    n_ep = len(dp_stats["episode_pnls"])
    cum_dp = np.cumsum(dp_stats["episode_pnls"])
    cum_rl = np.cumsum(rl_stats["episode_pnls"])
    axes[1, 0].plot(range(1, n_ep + 1), cum_dp, label="DP", lw=1)
    axes[1, 0].plot(range(1, n_ep + 1), cum_rl, label="RL (distilled)", lw=1)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cumulative MtM PnL")
    axes[1, 0].set_title("Cumulative PnL")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.5,
                     label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(rl_stats["final_inventories"], bins=inv_bins, alpha=0.5,
                     label="RL (distilled)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend()

    axes[1, 2].set_visible(False)

    fig.suptitle("Experiment 2: Policy Distillation — DP vs RL (mimic DP)")
    plt.tight_layout()
    plt.savefig("results/exp2_distillation_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_distillation_comparison.png")

    # ── policy heatmaps ───────────────────────────────────────────────
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    inv_norm = np.linspace(-1, 1, 21)
    price_norm = np.linspace(-1, 1, 21)

    dp_bid_map = np.zeros((len(price_norm), len(inv_norm)))
    dp_ask_map = np.zeros_like(dp_bid_map)
    for i, pn in enumerate(price_norm):
        sp = int(np.clip(np.round((pn * PRICE_HALF_RANGE - price_grid[0])
                 / (price_grid[1] - price_grid[0])), 0, N_PRICE_BINS - 1))
        for j, ni in enumerate(inv_norm):
            si = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory),
                     0, params.n_inventory_states - 1))
            db, da = params.action_to_spreads(policy_dp[0, si, sp])
            dp_bid_map[i, j] = db
            dp_ask_map[i, j] = da

    rl_bid_map = np.zeros_like(dp_bid_map)
    rl_ask_map = np.zeros_like(dp_bid_map)
    for i, pn in enumerate(price_norm):
        price_dev = pn * PRICE_HALF_RANGE
        for j, ni in enumerate(inv_norm):
            inv = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory),
                      0, params.n_inventory_states - 1))
            inv = params.index_to_inventory(inv)
            obs = eval_env.obs_for_state(inv, price_dev=price_dev)
            mask = DQNAgent.boundary_mask(inv, params)
            a = agent.select_action(obs, valid_mask=mask)
            db, da = params.action_to_spreads(a)
            rl_bid_map[i, j] = db
            rl_ask_map[i, j] = da

    extent = [inventories[0], inventories[-1], -PRICE_HALF_RANGE, PRICE_HALF_RANGE]
    vmin, vmax = min(SPREAD_OPTIONS), max(SPREAD_OPTIONS)
    for ax, data, title in [
        (ax1, dp_bid_map, "DP (teacher): Bid δ*"),
        (ax2, dp_ask_map, "DP (teacher): Ask δ*"),
        (ax3, rl_bid_map, "RL (distilled): Bid δ*"),
        (ax4, rl_ask_map, "RL (distilled): Ask δ*"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent,
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Inventory")
        ax.set_ylabel("Price deviation")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig2.suptitle("Experiment 2: Policy Heatmaps — DP vs RL (Distillation)")
    plt.tight_layout()
    plt.savefig("results/exp2_distillation_policy_heatmap.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_distillation_policy_heatmap.png")
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
    hdr = f"  {'Metric':<22s}  {'DP':>14s}  {'RL (distilled)':>14s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        print(f"  {name:<22s}  {format(dp_stats[key], fmt):>14s}  {format(rl_stats[key], fmt):>14s}")


if __name__ == "__main__":
    main()
