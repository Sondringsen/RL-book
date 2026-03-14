#!/usr/bin/env python3
"""Experiment 3 — State = (inventory, price, volatility), OU vol dynamics.

Compares DP vs RL (regular/continuous).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, VecMarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_3d, simulate_dp_policy_3d
from market_making.gpu_config import gpu_batch_size, gpu_hidden_dim, gpu_info

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
    print("Experiment 3: (Inventory, Price, Vol) — DP vs RL (regular)")
    print("=" * 60)
    print(f"  State dim   : 3 (inventory, price, volatility)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  Vol bins: {N_VOL_BINS}")
    print(f"  Volatility  : OU process")
    print(f"  Device      : {gpu_info()}")

    BATCH_TRAIN = gpu_batch_size(256)
    HIDDEN_TRAIN = gpu_hidden_dim(256)

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/4] DP 3-D value iteration …")
    V_dp, policy_dp, price_grid, vol_grid = value_iteration_3d(
        params, n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
        n_vol_bins=N_VOL_BINS, vol_lo=VOL_LO, vol_hi=VOL_HI,
    )

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS, terminal_penalty=0.0,
        episode_length=300,
    )

    # ── RL Regular (continuous, vectorized for GPU) ─────────────────────
    print("\n[2/4] Training DQN (continuous state, vectorized) …")
    def _make_env_reg():
        return MarketMakingEnv(
            train_params, use_volatility_dynamics=True,
            include_price=True, price_scale=PRICE_SCALE, vol_grid=None,
            discrete_inventory=False, price_grid=None,
            random_init=True, use_continuous_state=False, seed=SEED,
        )
    train_env_reg = VecMarketMakingEnv(32, _make_env_reg)
    agent_reg = DQNAgent(
        state_dim=train_env_reg.state_dim, n_actions=train_env_reg.n_actions,
        lr=2e-4, gamma=train_params.discount, batch_size=BATCH_TRAIN, hidden_dim=HIDDEN_TRAIN,
        learning_starts=5_000, tau=0.005, seed=SEED,
    )
    agent_reg.train(train_env_reg, n_episodes=500, epsilon_decay_steps=150_000, verbose=True)

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[3/4] Evaluating (1 000 episodes each) …")
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
        discrete_inventory=False, use_continuous_state=False, seed=EVAL_SEED,
    )
    reg_stats = agent_reg.evaluate(eval_reg, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, reg_stats)

    print("\n  Experiment 3 — Mean MtM PnL:")
    print(f"    DP: {dp_stats['mean_pnl']:.2f}  RL (regular): {reg_stats['mean_pnl']:.2f}")

    # ── Plots (policy at mid price, mid vol) ───────────────────────────
    print("\nPlotting …")
    mid_p = N_PRICE_BINS // 2
    mid_v = N_VOL_BINS // 2
    dp_bids = [params.action_to_spreads(policy_dp[0, i, mid_p, mid_v])[0] for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[0, i, mid_p, mid_v])[1] for i in range(params.n_inventory_states)]

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, reg_bids, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads (mid price, mid vol)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, reg_asks, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads (mid price, mid vol)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [dp_stats["episode_rewards"], reg_stats["episode_rewards"]]
    lo, hi = min(r.min() for r in all_rewards), max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.35, label="DP", edgecolor="k")
    axes[1, 0].hist(reg_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (regular)", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(reg_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (regular)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Experiment 3: (Inventory, Price, Vol) — DP vs RL (regular)")
    plt.tight_layout()
    plt.savefig("results/exp3_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_comparison.png")

    # Cumulative PnL
    n_ep = len(dp_stats["episode_pnls"])
    fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(10, 4))
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(reg_stats["episode_pnls"]), label="RL (regular)", lw=1)
    ax_pnl.set_xlabel("Episode")
    ax_pnl.set_ylabel("Cumulative MtM PnL")
    ax_pnl.set_title("Experiment 3: Cumulative PnL")
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp3_pnl.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_pnl.png")

    # Episode reward over time
    fig_rew, ax_rew = plt.subplots(1, 1, figsize=(10, 4))
    window = 50
    for rewards, label, color in [
        (dp_stats["episode_rewards"], "DP", "C0"),
        (reg_stats["episode_rewards"], "RL (regular)", "C1"),
    ]:
        ax_rew.plot(range(1, n_ep + 1), rewards, alpha=0.15, color=color)
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax_rew.plot(range(window, n_ep + 1), rolling, lw=2, color=color, label=label)
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Episode Reward")
    ax_rew.set_title("Experiment 3: Episode Reward over Time")
    ax_rew.legend(fontsize=8)
    ax_rew.grid(True, alpha=0.3)
    fig_rew.tight_layout()
    fig_rew.savefig("results/exp3_reward_over_time.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_reward_over_time.png")

    # ── Policy heatmaps at mid vol (DP, regular) ───────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    mid_v = N_VOL_BINS // 2
    vol_mid = float(vol_grid[mid_v])
    inv_norm = np.linspace(-1, 1, 21)
    price_norm = np.linspace(-1, 1, 21)

    def build_heatmap_3d(agent, env):
        bid_map = np.zeros((len(price_norm), len(inv_norm)))
        ask_map = np.zeros_like(bid_map)
        for i, pn in enumerate(price_norm):
            price_dev = pn * PRICE_HALF_RANGE
            for j, ni in enumerate(inv_norm):
                inv = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory), 0, params.n_inventory_states - 1))
                inv = params.index_to_inventory(inv)
                obs = env.obs_for_state(inv, price_dev=price_dev, vol=vol_mid)
                mask = DQNAgent.boundary_mask(inv, params)
                a = agent.select_action(obs, valid_mask=mask)
                db, da = params.action_to_spreads(a)
                bid_map[i, j] = db
                ask_map[i, j] = da
        return bid_map, ask_map

    dp_bid_map = np.zeros((len(price_norm), len(inv_norm)))
    dp_ask_map = np.zeros_like(dp_bid_map)
    for i, pn in enumerate(price_norm):
        sp = int(np.clip(np.round((pn * PRICE_HALF_RANGE - price_grid[0]) / (price_grid[1] - price_grid[0])), 0, N_PRICE_BINS - 1))
        for j, ni in enumerate(inv_norm):
            si = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory), 0, params.n_inventory_states - 1))
            db, da = params.action_to_spreads(policy_dp[0, si, sp, mid_v])
            dp_bid_map[i, j] = db
            dp_ask_map[i, j] = da

    reg_bid_map, reg_ask_map = build_heatmap_3d(agent_reg, eval_reg)

    extent = [inventories[0], inventories[-1], -PRICE_HALF_RANGE, PRICE_HALF_RANGE]
    vmin, vmax = min(SPREAD_OPTIONS), max(SPREAD_OPTIONS)

    for ax, data, title in [
        (axes2[0, 0], dp_bid_map, "DP: Bid δ*"),
        (axes2[0, 1], reg_bid_map, "RL (regular): Bid δ*"),
        (axes2[1, 0], dp_ask_map, "DP: Ask δ*"),
        (axes2[1, 1], reg_ask_map, "RL (regular): Ask δ*"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Inventory")
        ax.set_ylabel("Price deviation")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig2.suptitle(f"Experiment 3: Policy Heatmaps at mid vol (σ={vol_mid:.2f})")
    plt.tight_layout()
    plt.savefig("results/exp3_policy_heatmap.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp3_policy_heatmap.png")

    plt.show()


def _print_table(dp_stats, reg_stats):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'Regular':>10s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        v = [dp_stats[key], reg_stats[key]]
        print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))


if __name__ == "__main__":
    main()
