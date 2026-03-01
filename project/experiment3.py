#!/usr/bin/env python3
"""Experiment 3 — State = (inventory, price, volatility), limited actions.

State : (I, S, σ)  with σ following an OU process
Action: (δ_bid, δ_ask) from a small spread grid
Vol   : stochastic (Ornstein–Uhlenbeck)

Compares DP (3-D value iteration) vs RL (DQN) side by side.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_3d, simulate_dp_policy_3d

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
N_PRICE_BINS = 15
PRICE_HALF_RANGE = 10.0
PRICE_SCALE = PRICE_HALF_RANGE
N_VOL_BINS = 9
VOL_LO, VOL_HI = 0.3, 2.0
SEED = 42


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 3: (Inventory, Price, Vol) state, DP vs RL")
    print("=" * 60)
    print(f"  State dim   : 3 (inventory, price, volatility)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  range=[{-PRICE_HALF_RANGE}, {PRICE_HALF_RANGE}]")
    print(f"  Vol bins    : {N_VOL_BINS}  range=[{VOL_LO}, {VOL_HI}]")
    print(f"  Vol dynamics: OU  κ={params.vol_mean_reversion}  "
          f"θ={params.vol_long_run_mean}  ξ={params.vol_of_vol}")

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/4] DP 3-D value iteration …")
    V_dp, policy_dp, residuals, price_grid, vol_grid = value_iteration_3d(
        params,
        n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
        n_vol_bins=N_VOL_BINS, vol_lo=VOL_LO, vol_hi=VOL_HI,
    )

    # ── RL ────────────────────────────────────────────────────────────
    print("\n[2/4] Training DQN (state = [inventory, price, vol]) …")
    train_env = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE, seed=SEED,
    )

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=5e-4,
        gamma=params.discount,
        batch_size=64,
        hidden_dim=128,
        learning_starts=10_000,
        tau=0.005,
        seed=SEED,
    )
    ep_rewards, _ = agent.train(
        train_env, n_episodes=5000,
        epsilon_start=1.0, epsilon_end=0.02,
        epsilon_decay_steps=400_000, verbose=True,
    )

    # ── evaluate ──────────────────────────────────────────────────────
    print("\n[3/4] Evaluating (1 000 episodes, stochastic σ) …")
    eval_dp = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE, seed=123,
    )
    dp_stats = simulate_dp_policy_3d(
        eval_dp, policy_dp, params, price_grid, vol_grid, n_episodes=1000,
    )

    eval_rl = MarketMakingEnv(
        params, use_volatility_dynamics=True,
        include_price=True, price_scale=PRICE_SCALE, seed=123,
    )
    rl_stats = agent.evaluate(eval_rl, n_episodes=1000)

    _print_table(dp_stats, rl_stats)

    # ── plots ─────────────────────────────────────────────────────────
    print("\n[4/4] Plotting …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Policy slice at price=0, vol=θ
    mid_p = N_PRICE_BINS // 2
    mid_v = np.argmin(np.abs(vol_grid - params.vol_long_run_mean))
    dp_bids = [params.action_to_spreads(policy_dp[i, mid_p, mid_v])[0]
               for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i, mid_p, mid_v])[1]
               for i in range(params.n_inventory_states)]
    rl_bids, rl_asks = [], []
    for I in inventories:
        obs = np.array([I / params.max_inventory, 0.0,
                        params.vol_long_run_mean / params.vol_long_run_mean],
                       dtype=np.float32)
        a = agent.select_action(obs)
        db, da = params.action_to_spreads(a)
        rl_bids.append(db)
        rl_asks.append(da)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP bid")
    axes[0, 0].step(inventories, dp_asks, "b-s", where="mid", ms=5, label="DP ask", ls="--")
    axes[0, 0].step(inventories, rl_bids, "r-o", where="mid", ms=5, label="RL bid")
    axes[0, 0].step(inventories, rl_asks, "r-s", where="mid", ms=5, label="RL ask", ls="--")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Half-spread")
    axes[0, 0].set_title("Policy Slice at Price=0, σ=θ")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Reward distribution
    lo = min(dp_stats["episode_rewards"].min(), rl_stats["episode_rewards"].min())
    hi = max(dp_stats["episode_rewards"].max(), rl_stats["episode_rewards"].max())
    bins = np.linspace(lo, hi, 50)
    axes[0, 1].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.5, label="DP", edgecolor="k")
    axes[0, 1].hist(rl_stats["episode_rewards"], bins=bins, alpha=0.5, label="RL", edgecolor="k")
    axes[0, 1].axvline(dp_stats["mean_reward"], color="C0", ls="--")
    axes[0, 1].axvline(rl_stats["mean_reward"], color="C1", ls="--")
    axes[0, 1].set_xlabel("Episode Reward")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Reward Distribution")
    axes[0, 1].legend()

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

    fig.suptitle("Experiment 3: (Inventory, Price, Vol) State — DP vs RL",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("results/exp3_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_comparison.png")

    # ── policy heatmaps: bid & ask over (inventory, vol) at price=0 ──
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    inv_norm = np.linspace(-1, 1, 21)
    vol_vals = np.linspace(VOL_LO, VOL_HI, 21)

    dp_bid_map = np.zeros((len(vol_vals), len(inv_norm)))
    dp_ask_map = np.zeros_like(dp_bid_map)
    rl_bid_map = np.zeros_like(dp_bid_map)
    rl_ask_map = np.zeros_like(dp_bid_map)

    for i, v in enumerate(vol_vals):
        sv = int(np.clip(np.round((v - vol_grid[0]) / (vol_grid[1] - vol_grid[0])),
                 0, N_VOL_BINS - 1))
        for j, ni in enumerate(inv_norm):
            si = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory),
                     0, params.n_inventory_states - 1))
            db, da = params.action_to_spreads(policy_dp[si, mid_p, sv])
            dp_bid_map[i, j] = db
            dp_ask_map[i, j] = da

            obs = np.array([ni, 0.0, v / params.vol_long_run_mean], dtype=np.float32)
            a = agent.select_action(obs)
            db, da = params.action_to_spreads(a)
            rl_bid_map[i, j] = db
            rl_ask_map[i, j] = da

    extent = [inventories[0], inventories[-1], VOL_LO, VOL_HI]
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
        ax.set_ylabel("Volatility σ")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig2.suptitle("Experiment 3: Policy Heatmaps (Inventory × Vol) at Price=0",
                  fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("results/exp3_policy_heatmap.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_policy_heatmap.png")
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
