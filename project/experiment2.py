#!/usr/bin/env python3
"""Experiment 2 — State = (inventory, price), limited actions, constant vol.

Compares DP vs RL (regular/continuous) vs RL (distillation).
Increased action space: 5×5 = 25 spread combinations.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, VecMarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_2d, simulate_dp_policy_2d
from market_making.gpu_config import gpu_batch_size, gpu_hidden_dim, gpu_info

# Larger action space: 5 spread options → 25 (δ_bid, δ_ask) pairs
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
    print("Experiment 2: (Inventory, Price) — DP vs RL (regular, distillation)")
    print("=" * 60)
    print(f"  State dim   : 2 (inventory, price)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  range=[{-PRICE_HALF_RANGE}, {PRICE_HALF_RANGE}]")
    print(f"  Volatility  : constant σ = {params.sigma_base}")
    print(f"  Device      : {gpu_info()}")

    BATCH_TRAIN = gpu_batch_size(256)
    BATCH_DISTILL = gpu_batch_size(128)
    HIDDEN_TRAIN = gpu_hidden_dim(256)
    HIDDEN_DISTILL = gpu_hidden_dim(128)

    timings = {}  # method -> seconds

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/5] DP 2-D value iteration …")
    t0 = time.perf_counter()
    V_dp, policy_dp, residuals, price_grid = value_iteration_2d(
        params, n_price_bins=N_PRICE_BINS, price_half_range=PRICE_HALF_RANGE,
    )
    timings["DP"] = time.perf_counter() - t0
    print(f"  DP converged in {timings['DP']:.1f} s")

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS, terminal_penalty=0.0, sigma_base=SIGMA_BASE,
        episode_length=300,
    )

    # ── RL Regular (continuous state) — vectorized for GPU ─────────────
    print("\n[2/5] Training DQN (continuous state, vectorized) …")
    t0 = time.perf_counter()
    def _make_env_reg():
        return MarketMakingEnv(
            train_params, use_volatility_dynamics=False,
            include_price=True, price_scale=PRICE_SCALE, discrete_inventory=True, price_grid=price_grid,
            random_init=True, use_continuous_state=True, seed=SEED,
        )
    train_env_reg = VecMarketMakingEnv(32, _make_env_reg)
    agent_reg = DQNAgent(
        state_dim=train_env_reg.state_dim, n_actions=train_env_reg.n_actions,
        lr=2e-4, gamma=train_params.discount, batch_size=BATCH_TRAIN, hidden_dim=HIDDEN_TRAIN,
        learning_starts=5_000, tau=0.005, seed=SEED,
        use_prioritized_replay=True, rare_priority=3.0,
    )
    agent_reg.train(
        train_env_reg, n_episodes=500, epsilon_decay_steps=150_000,
        uniform_state_interval=8, extreme_state_prob=0.3, verbose=True,
    )
    timings["RL (regular)"] = time.perf_counter() - t0
    print(f"  RL (regular) trained in {timings['RL (regular)']:.1f} s")

    # ── RL Distillation ────────────────────────────────────────────────
    print("\n[3/5] Training DQN via policy distillation …")
    t0 = time.perf_counter()
    env_obs = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE, discrete_inventory=True, price_grid=price_grid,
        use_continuous_state=False, seed=SEED,
    )
    agent_dist = DQNAgent(
        state_dim=env_obs.state_dim, n_actions=env_obs.n_actions,
        lr=1e-3, gamma=params.discount, batch_size=BATCH_DISTILL, hidden_dim=HIDDEN_DISTILL, seed=SEED,
    )
    agent_dist.train_distillation(
        env_obs, policy_dp, params, price_grid=price_grid,
        n_epochs=500, batch_size=BATCH_DISTILL, verbose=True, log_interval=50,
    )
    timings["RL (distillation)"] = time.perf_counter() - t0
    print(f"  RL (distillation) trained in {timings['RL (distillation)']:.1f} s")

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[4/5] Evaluating (1 000 episodes each) …")
    eval_base = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE,
        discrete_inventory=True, price_grid=price_grid, seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy_2d(
        eval_base, policy_dp, params, price_grid,
        n_episodes=1000, episode_seed_base=EVAL_SEED,
    )

    eval_reg = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE, discrete_inventory=True, price_grid=price_grid,
        use_continuous_state=True, seed=EVAL_SEED,
    )
    reg_stats = agent_reg.evaluate(eval_reg, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_dist = MarketMakingEnv(
        params, use_volatility_dynamics=False,
        include_price=True, price_scale=PRICE_SCALE, discrete_inventory=True, price_grid=price_grid,
        use_continuous_state=False, seed=EVAL_SEED,
    )
    dist_stats = agent_dist.evaluate(eval_dist, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, reg_stats, dist_stats)

    print("\n  Experiment 2 — Mean MtM PnL:")
    print(f"    DP: {dp_stats['mean_pnl']:.2f}  RL (regular): {reg_stats['mean_pnl']:.2f}  RL (distill): {dist_stats['mean_pnl']:.2f}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nPlotting …")
    mid_idx = N_PRICE_BINS // 2
    dp_bids = [params.action_to_spreads(policy_dp[i, mid_idx])[0] for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i, mid_idx])[1] for i in range(params.n_inventory_states)]

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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, reg_bids, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 0].step(inventories, dist_bids, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads (Price = Mid)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, reg_asks, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 1].step(inventories, dist_asks, "m--d", where="mid", ms=4, label="RL (distill)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads (Price = Mid)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [dp_stats["episode_rewards"], reg_stats["episode_rewards"], dist_stats["episode_rewards"]]
    lo, hi = min(r.min() for r in all_rewards), max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 50)
    axes[0, 2].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.35, label="DP", edgecolor="k")
    axes[0, 2].hist(reg_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (regular)", edgecolor="k")
    axes[0, 2].hist(dist_stats["episode_rewards"], bins=bins, alpha=0.35, label="RL (distill)", edgecolor="k")
    axes[0, 2].set_xlabel("Episode Reward")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Reward Distribution")
    axes[0, 2].legend(fontsize=8)

    n_ep = len(dp_stats["episode_pnls"])
    axes[1, 0].plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    axes[1, 0].plot(range(1, n_ep + 1), np.cumsum(reg_stats["episode_pnls"]), label="RL (regular)", lw=1)
    axes[1, 0].plot(range(1, n_ep + 1), np.cumsum(dist_stats["episode_pnls"]), label="RL (distill)", lw=1)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cumulative MtM PnL")
    axes[1, 0].set_title("Cumulative PnL")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(reg_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (regular)", edgecolor="k", align="left")
    axes[1, 1].hist(dist_stats["final_inventories"], bins=inv_bins, alpha=0.35, label="RL (distill)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].set_visible(False)

    fig.suptitle("Experiment 2: (Inventory, Price) — DP vs RL (regular, distillation)")
    plt.tight_layout()
    plt.savefig("results/exp2_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_comparison.png")

    # Cumulative PnL (standalone)
    fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(10, 4))
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(reg_stats["episode_pnls"]), label="RL (regular)", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dist_stats["episode_pnls"]), label="RL (distill)", lw=1)
    ax_pnl.set_xlabel("Episode")
    ax_pnl.set_ylabel("Cumulative MtM PnL")
    ax_pnl.set_title("Experiment 2: Cumulative PnL")
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp2_pnl.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_pnl.png")

    # ── Policy heatmaps (DP, regular, distill) ──────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    inv_norm = np.linspace(-1, 1, 21)
    price_norm = np.linspace(-1, 1, 21)

    def build_heatmap(agent, env):
        bid_map = np.zeros((len(price_norm), len(inv_norm)))
        ask_map = np.zeros_like(bid_map)
        for i, pn in enumerate(price_norm):
            price_dev = pn * PRICE_HALF_RANGE
            for j, ni in enumerate(inv_norm):
                inv = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory), 0, params.n_inventory_states - 1))
                inv = params.index_to_inventory(inv)
                obs = env.obs_for_state(inv, price_dev=price_dev)
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
            db, da = params.action_to_spreads(policy_dp[si, sp])
            dp_bid_map[i, j] = db
            dp_ask_map[i, j] = da

    reg_bid_map, reg_ask_map = build_heatmap(agent_reg, eval_reg)
    dist_bid_map, dist_ask_map = build_heatmap(agent_dist, eval_dist)

    extent = [inventories[0], inventories[-1], -PRICE_HALF_RANGE, PRICE_HALF_RANGE]
    vmin, vmax = min(SPREAD_OPTIONS), max(SPREAD_OPTIONS)

    for ax, data, title in [
        (axes2[0, 0], dp_bid_map, "DP: Bid δ*"),
        (axes2[0, 1], reg_bid_map, "RL (regular): Bid δ*"),
        (axes2[0, 2], dist_bid_map, "RL (distill): Bid δ*"),
        (axes2[1, 0], dp_ask_map, "DP: Ask δ*"),
        (axes2[1, 1], reg_ask_map, "RL (regular): Ask δ*"),
        (axes2[1, 2], dist_ask_map, "RL (distill): Ask δ*"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Inventory")
        ax.set_ylabel("Price deviation")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig2.suptitle("Experiment 2: Policy Heatmaps (DP, regular, distill)")
    plt.tight_layout()
    plt.savefig("results/exp2_policy_heatmap.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_policy_heatmap.png")

    # ── Convergence time plot ──────────────────────────────────────────
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    methods = list(timings.keys())
    times_s = [timings[m] for m in methods]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]
    bars = ax3.bar(methods, times_s, color=colors, edgecolor="k")
    ax3.set_ylabel("Time to converge (seconds)")
    ax3.set_title(f"Experiment 2: Convergence Time by Method ({params.n_actions} actions)")
    ax3.set_ylim(0, max(times_s) * 1.15)
    for bar, t in zip(bars, times_s):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times_s) * 0.02,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/exp2_convergence_time.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_convergence_time.png")

    print("\nConvergence times (seconds):")
    for m, t in timings.items():
        print(f"  {m}: {t:.1f} s")

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
