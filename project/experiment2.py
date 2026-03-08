#!/usr/bin/env python3
"""Experiment 2 — State = (inventory, price), limited actions, constant vol.

State : (I, S)  with S = mid-price deviation from initial price
Action: (δ_bid, δ_ask) from a small spread grid
Vol   : constant σ

Compares DP (2-D value iteration) vs RL (DQN) side by side.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_2d, simulate_dp_policy_2d

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
N_PRICE_BINS = 21
PRICE_HALF_RANGE = 10.0
PRICE_SCALE = PRICE_HALF_RANGE
SEED = 42
EVAL_SEED = 123  # Same seed for DP/RL → identical trajectories, same problem


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 2: (Inventory, Price) state, DP vs RL")
    print("=" * 60)
    print(f"  State dim   : 2 (inventory, price)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  range=[{-PRICE_HALF_RANGE}, {PRICE_HALF_RANGE}]")
    print(f"  Volatility  : constant σ = {params.sigma_base}")

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/4] DP 2-D value iteration …")
    V_dp, policy_dp, residuals, price_grid = value_iteration_2d(
        params, n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
    )

    # ── RL (discrete inv+price + random init for full state coverage) ───
    # train_params: terminal_penalty=0.0 to match the DP (infinite-horizon, no terminal cost)
    # episode_length=1000 so γ^1000 ≈ 0 → eliminates terminal truncation bias
    # (γ^200 = 0.134 with the default episode_length=200 systematically underestimates Q-values)
    # Note: sigma_base is kept at default (1.0) unlike exp 1, because sigma IS part of the 2D DP problem.
    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        # sigma_base=1.0,       # remove I·σ·ε noise that swamps spread signal (SNR ≈ 0.5 at σ=1)
        episode_length=1000,  # γ^1000 ≈ 0 → no terminal truncation bias
    )
    print("\n[2/4] Training DQN (state = one-hot inv+price, random init for full coverage) …")
    train_env = MarketMakingEnv(
        train_params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid,
        random_init=True,  # Start from random (I, price) → visit all 231 states
        seed=SEED,
    )

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=2e-4,
        gamma=train_params.discount,
        batch_size=256,
        hidden_dim=256,
        learning_starts=15_000,
        tau=0.005,
        seed=SEED,
    )
    ep_rewards, _ = agent.train(
        train_env,
        n_episodes=1500,  # 3 000 × 1 000 = 3M steps (same budget as before)
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=800_000,
        verbose=True,
    )

    # ── evaluate (same env config + per-episode seeds → identical trajectories) ─
    print("\n[3/4] Evaluating (1 000 episodes, constant σ, paired trajectories) …")
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy_2d(
        eval_env, policy_dp, params, price_grid,
        n_episodes=1000, episode_seed_base=EVAL_SEED,
    )
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, seed=EVAL_SEED,
    )
    rl_stats = agent.evaluate(eval_env, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, rl_stats)

    # ── plots ─────────────────────────────────────────────────────────
    print("\n[4/4] Plotting …")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # shared policy data at price = mid (centre of grid)
    mid_idx = N_PRICE_BINS // 2
    dp_bids = [params.action_to_spreads(policy_dp[i, mid_idx])[0]
               for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i, mid_idx])[1]
               for i in range(params.n_inventory_states)]
    rl_bids, rl_asks = [], []
    for I in inventories:
        obs = eval_env.obs_for_state(I, price_dev=0.0)
        mask = DQNAgent.boundary_mask(I, params)
        a = agent.select_action(obs, valid_mask=mask)
        db, da = params.action_to_spreads(a)
        rl_bids.append(db)
        rl_asks.append(da)

    # (0,0) Bid spreads at mid-price
    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, rl_bids, "r-s", where="mid", ms=5, label="RL", ls="--")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Optimal Policy: Bid Spreads (Price = Mid)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Ask spreads at mid-price
    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, rl_asks, "r-s", where="mid", ms=5, label="RL", ls="--")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Optimal Policy: Ask Spreads (Price = Mid)")
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

    # (1,2) unused
    axes[1, 2].set_visible(False)

    fig.suptitle("Experiment 2: (Inventory, Price) State — DP vs RL",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("results/exp2_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_comparison.png")

    # ── policy heatmaps ───────────────────────────────────────────────
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    inv_norm = np.linspace(-1, 1, 21)
    price_norm = np.linspace(-1, 1, 21)

    # DP heatmaps
    dp_bid_map = np.zeros((len(price_norm), len(inv_norm)))
    dp_ask_map = np.zeros_like(dp_bid_map)
    for i, pn in enumerate(price_norm):
        sp = int(np.clip(np.round((pn * PRICE_HALF_RANGE - price_grid[0])
                 / (price_grid[1] - price_grid[0])), 0, N_PRICE_BINS - 1))
        for j, ni in enumerate(inv_norm):
            si = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory),
                     0, params.n_inventory_states - 1))
            db, da = params.action_to_spreads(policy_dp[si, sp])
            dp_bid_map[i, j] = db
            dp_ask_map[i, j] = da

    # RL heatmaps
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

    extent = [inventories[0], inventories[-1],
              -PRICE_HALF_RANGE, PRICE_HALF_RANGE]
    vmin = min(SPREAD_OPTIONS)
    vmax = max(SPREAD_OPTIONS)

    for ax, data, title in [
        (ax1, dp_bid_map, "DP: Bid δ*"),
        (ax2, dp_ask_map, "DP: Ask δ*"),
        (ax3, rl_bid_map, "RL: Bid δ*"),
        (ax4, rl_ask_map, "RL: Ask δ*"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent,
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Inventory")
        ax.set_ylabel("Price deviation")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig2.suptitle("Experiment 2: Policy Heatmaps (Inventory × Price)",
                  fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("results/exp2_policy_heatmap.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_policy_heatmap.png")
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
