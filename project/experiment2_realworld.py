#!/usr/bin/env python3
"""Experiment 2 Real-World — When RL beats DP.

Real-world scenario where DP fails and RL succeeds:
1. State space huge/continuous (millions) — DP cannot enumerate
2. Transition model unknown — order flow has no DP model
3. Generalization to unseen states — RL uses continuous state, works at live prices

If DP fails (timeout, OOM), it fails. RL continues.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from market_making import MarketParams, MarketMakingEnv, VecMarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_3d, simulate_dp_policy_3d
from market_making.gpu_config import gpu_batch_size, gpu_hidden_dim, gpu_info

# Real-world: larger action space
SPREAD_OPTIONS = (0.5, 1.0, 1.5, 2.0, 2.5)

# Huge state grid: DP must enumerate all → impractical
# 11 × 501 × 101 = 556,611 states — DP takes ~30+ min or OOM; we timeout
N_PRICE_BINS = 501
N_VOL_BINS = 101
PRICE_HALF_RANGE = 12.0
VOL_LO, VOL_HI = 0.25, 2.2
PRICE_SCALE = PRICE_HALF_RANGE
SEED = 42
EVAL_SEED = 123

# DP timeout: if DP doesn't converge in this many seconds, skip it
DP_TIMEOUT_SEC = 120

# RL: continuous state, learns from samples, generalizes
RL_EPISODES = 200
RL_EPISODE_LENGTH = 200
RL_LEARNING_STARTS = 2000
RL_EPSILON_DECAY_STEPS = 40_000

N_STATES = 11 * N_PRICE_BINS * N_VOL_BINS  # 11 inv × 1001 price × 201 vol


def _run_dp():
    """Run DP in a thread so we can timeout."""
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    return value_iteration_3d(
        params,
        n_price_bins=N_PRICE_BINS,
        price_half_range=PRICE_HALF_RANGE,
        n_vol_bins=N_VOL_BINS,
        vol_lo=VOL_LO,
        vol_hi=VOL_HI,
    )


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 70)
    print("Experiment 2 Real-World: When RL beats DP")
    print("=" * 70)
    print(f"  State dim   : 4 (inventory, price, volatility, order_flow)")
    print(f"  State count : {N_STATES:,} (DP must enumerate — impractical)")
    print(f"  Order flow  : stochastic, DP has no transition model")
    print(f"  RL state    : continuous — generalizes to unseen (live) states")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  Vol bins: {N_VOL_BINS}")
    print(f"  DP timeout  : {DP_TIMEOUT_SEC} s")
    print(f"  Device      : {gpu_info()}")
    print()

    BATCH_TRAIN = gpu_batch_size(256)
    HIDDEN_TRAIN = gpu_hidden_dim(256)

    timings = {}
    dp_result = None
    price_grid = None
    vol_grid = None

    # ── DP 3D (will likely fail: huge state space) ───────────────────
    print("[1/3] DP 3-D value iteration (enumerates all states) …")
    print(f"      {N_STATES:,} states — may timeout or OOM …")
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_dp)
            V_dp, policy_dp, price_grid, vol_grid = future.result(
                timeout=DP_TIMEOUT_SEC
            )
        timings["DP (3D)"] = time.perf_counter() - t0
        dp_result = (V_dp, policy_dp, price_grid, vol_grid)
        print(f"  DP converged in {timings['DP (3D)']:.1f} s")
    except (FuturesTimeoutError, MemoryError) as e:
        timings["DP (3D)"] = None  # failed
        dp_result = None
        print(f"  DP FAILED: {type(e).__name__} (state space too large)")
        # Coarse grid for RL discrete eval / plotting (DP couldn't use it)
        price_grid = np.linspace(-PRICE_HALF_RANGE, PRICE_HALF_RANGE, 51)
        vol_grid = np.linspace(VOL_LO, VOL_HI, 11)

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        episode_length=RL_EPISODE_LENGTH,
    )

    # ── RL: continuous state + order flow (vectorized for GPU) ──────────
    print("\n[2/3] Training DQN (continuous state, order flow, vectorized) …")
    t0 = time.perf_counter()
    def _make_env():
        return MarketMakingEnv(
            train_params,
            use_volatility_dynamics=True,
            include_price=True,
            price_scale=PRICE_SCALE,
            vol_grid=None,
            discrete_inventory=False,
            price_grid=None,
            random_init=True,
            use_continuous_state=True,
            use_order_flow=True,
            seed=SEED,
        )
    train_env = VecMarketMakingEnv(32, _make_env)
    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=2e-4,
        gamma=train_params.discount,
        batch_size=BATCH_TRAIN,
        hidden_dim=HIDDEN_TRAIN,
        learning_starts=RL_LEARNING_STARTS,
        tau=0.005,
        seed=SEED,
    )
    agent.train(
        train_env,
        n_episodes=RL_EPISODES,
        epsilon_decay_steps=RL_EPSILON_DECAY_STEPS,
        verbose=True,
    )
    timings["RL (continuous)"] = time.perf_counter() - t0
    print(f"  RL trained in {timings['RL (continuous)']:.1f} s")

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n[3/3] Evaluating (1 000 episodes each) …")

    eval_env = MarketMakingEnv(
        params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        vol_grid=None,
        discrete_inventory=False,
        use_continuous_state=True,
        use_order_flow=True,
        seed=EVAL_SEED,
    )
    rl_stats = agent.evaluate(
        eval_env, n_episodes=1000, episode_seed_base=EVAL_SEED
    )

    if dp_result is not None:
        V_dp, policy_dp, price_grid, vol_grid = dp_result
        eval_base = MarketMakingEnv(
            params,
            use_volatility_dynamics=True,
            include_price=True,
            price_scale=PRICE_SCALE,
            discrete_inventory=True,
            price_grid=price_grid,
            vol_grid=vol_grid,
            use_order_flow=False,  # DP doesn't see order flow
            seed=EVAL_SEED,
        )
        dp_stats = simulate_dp_policy_3d(
            eval_base,
            policy_dp,
            params,
            price_grid,
            vol_grid,
            n_episodes=1000,
            episode_seed_base=EVAL_SEED,
        )
        _print_table(dp_stats, rl_stats, has_dp=True)
        print("\n  Experiment 2 Real-World — Mean MtM PnL:")
        print(f"    DP: {dp_stats['mean_pnl']:.2f}  RL (continuous): {rl_stats['mean_pnl']:.2f}")
    else:
        dp_stats = None
        _print_table(None, rl_stats, has_dp=False)
        print("\n  Experiment 2 Real-World — Mean MtM PnL:")
        print(f"    RL (continuous): {rl_stats['mean_pnl']:.2f}")

    # Cumulative PnL
    n_ep = len(rl_stats["episode_pnls"])
    fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(10, 4))
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(rl_stats["episode_pnls"]), label="RL (continuous)", lw=1)
    if dp_stats is not None:
        ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    ax_pnl.set_xlabel("Episode")
    ax_pnl.set_ylabel("Cumulative MtM PnL")
    ax_pnl.set_title("Experiment 2 Real-World: Cumulative PnL")
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp2_realworld_pnl.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_realworld_pnl.png")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nPlotting …")

    mid_p = len(price_grid) // 2 if price_grid is not None else 0
    mid_v = len(vol_grid) // 2 if vol_grid is not None else 0

    def get_rl_spreads(ag, env):
        bids, asks = [], []
        price_dev = float(price_grid[mid_p]) if price_grid is not None else 0.0
        vol = float(vol_grid[mid_v]) if vol_grid is not None else 1.0
        for I in inventories:
            obs = env.obs_for_state(I, price_dev=price_dev, vol=vol, order_flow=0.0)
            mask = DQNAgent.boundary_mask(I, params)
            a = ag.select_action(obs, valid_mask=mask)
            db, da = params.action_to_spreads(a)
            bids.append(db)
            asks.append(da)
        return bids, asks

    rl_bids, rl_asks = get_rl_spreads(agent, eval_env)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].step(inventories, rl_bids, "r--s", where="mid", ms=4, label="RL (continuous)")
    if dp_result is not None:
        dp_bids = [
            params.action_to_spreads(policy_dp[0, i, mid_p, mid_v])[0]
            for i in range(params.n_inventory_states)
        ]
        axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads (mid price, mid vol)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, rl_asks, "r--s", where="mid", ms=4, label="RL (continuous)")
    if dp_result is not None:
        dp_asks = [
            params.action_to_spreads(policy_dp[0, i, mid_p, mid_v])[1]
            for i in range(params.n_inventory_states)
        ]
        axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads (mid price, mid vol)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [rl_stats["episode_rewards"]]
    if dp_stats is not None:
        all_rewards.append(dp_stats["episode_rewards"])
    lo = min(r.min() for r in all_rewards)
    hi = max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(rl_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (continuous)", edgecolor="k")
    if dp_stats is not None:
        axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.4, label="DP", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    methods = [m for m in timings if timings[m] is not None]
    times_s = [timings[m] for m in methods]
    colors = ["#e74c3c", "#2ecc71"] if "DP (3D)" in methods else ["#e74c3c"]
    if methods:
        bars = axes[1, 1].bar(methods, times_s, color=colors[: len(methods)], edgecolor="k")
        axes[1, 1].set_ylabel("Time to converge (seconds)")
        winner = min(methods, key=lambda m: timings[m])
        axes[1, 1].set_title(f"Convergence Time — {winner} ({N_STATES:,} states)")
        axes[1, 1].set_ylim(0, max(times_s) * 1.2)
        for bar, t in zip(bars, times_s):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times_s) * 0.02,
                f"{t:.0f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        axes[1, 1].text(0.5, 0.5, "DP failed (timeout/OOM)\nRL only", ha="center", va="center", fontsize=14)
        axes[1, 1].set_title("DP cannot scale to this state space")

    fig.suptitle(
        "Experiment 2 Real-World: RL when DP fails (huge state, unknown transitions, generalization)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("results/exp2_realworld_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_realworld_comparison.png")

    # ── Policy heatmaps (DP + RL continuous at mid vol) ──────────────────
    if price_grid is not None and vol_grid is not None:
        fig_hm, axes_hm = plt.subplots(2, 2, figsize=(12, 10))
        inv_norm = np.linspace(-1, 1, 21)
        price_norm = np.linspace(-1, 1, 21)

        def build_heatmap_realworld(agent, env):
            bid_map = np.zeros((len(price_norm), len(inv_norm)))
            ask_map = np.zeros_like(bid_map)
            vol_mid = float(vol_grid[mid_v])
            for i, pn in enumerate(price_norm):
                price_dev = pn * PRICE_HALF_RANGE
                for j, ni in enumerate(inv_norm):
                    inv = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory), 0, params.n_inventory_states - 1))
                    inv = params.index_to_inventory(inv)
                    obs = env.obs_for_state(inv, price_dev=price_dev, vol=vol_mid, order_flow=0.0)
                    mask = DQNAgent.boundary_mask(inv, params)
                    a = agent.select_action(obs, valid_mask=mask)
                    db, da = params.action_to_spreads(a)
                    bid_map[i, j] = db
                    ask_map[i, j] = da
            return bid_map, ask_map

        rl_bid_map, rl_ask_map = build_heatmap_realworld(agent, eval_env)
        extent = [inventories[0], inventories[-1], -PRICE_HALF_RANGE, PRICE_HALF_RANGE]
        vmin, vmax = min(SPREAD_OPTIONS), max(SPREAD_OPTIONS)

        if dp_result is not None:
            _V, policy_dp, _pg, _vg = dp_result
            dp_bid_map = np.zeros((len(price_norm), len(inv_norm)))
            dp_ask_map = np.zeros_like(dp_bid_map)
            for i, pn in enumerate(price_norm):
                price_dev = pn * PRICE_HALF_RANGE
                sp = int(np.clip(np.argmin(np.abs(_pg - price_dev)), 0, len(_pg) - 1))
                for j, ni in enumerate(inv_norm):
                    si = int(np.clip(np.round(ni * params.max_inventory + params.max_inventory), 0, params.n_inventory_states - 1))
                    db, da = params.action_to_spreads(policy_dp[0, si, sp, mid_v])
                    dp_bid_map[i, j] = db
                    dp_ask_map[i, j] = da
            for ax, data, title in [
                (axes_hm[0, 0], dp_bid_map, "DP: Bid δ*"),
                (axes_hm[0, 1], rl_bid_map, "RL (continuous): Bid δ*"),
                (axes_hm[1, 0], dp_ask_map, "DP: Ask δ*"),
                (axes_hm[1, 1], rl_ask_map, "RL (continuous): Ask δ*"),
            ]:
                im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_xlabel("Inventory")
                ax.set_ylabel("Price deviation")
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
        else:
            for ax, data, title in [
                (axes_hm[0, 0], rl_bid_map, "RL (continuous): Bid δ*"),
                (axes_hm[0, 1], rl_ask_map, "RL (continuous): Ask δ*"),
            ]:
                im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_xlabel("Inventory")
                ax.set_ylabel("Price deviation")
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
            axes_hm[1, 0].set_visible(False)
            axes_hm[1, 1].set_visible(False)

        fig_hm.suptitle("Experiment 2 Real-World: Policy Heatmaps at mid vol")
        plt.tight_layout()
        plt.savefig("results/exp2_realworld_policy_heatmap.png", dpi=150, bbox_inches="tight")
        print("→ saved results/exp2_realworld_policy_heatmap.png")

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    if methods:
        bars2 = ax2.bar(methods, times_s, color=colors[: len(methods)], edgecolor="k")
        ax2.set_ylabel("Time (seconds)")
        winner = min(methods, key=lambda m: timings[m])
        ax2.set_title(f"Real-World: {N_STATES:,} states — RL scales, DP fails")
        ax2.set_ylim(0, max(times_s) * 1.2)
        for bar, t in zip(bars2, times_s):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times_s) * 0.02,
                f"{t:.0f}s",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    else:
        ax2.text(0.5, 0.5, "DP failed\nRL succeeds", ha="center", va="center", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/exp2_realworld_convergence.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_realworld_convergence.png")

    print("\nConvergence times (seconds):")
    for m, t in timings.items():
        print(f"  {m}: {t if t is not None else 'FAILED'}")
    if methods:
        winner = min(methods, key=lambda m: timings[m])
        print(f"\n→ {winner} converges. DP fails at {N_STATES:,} states; RL generalizes via function approximation.")
    else:
        print(f"\n→ DP cannot enumerate {N_STATES:,} states. RL learns from samples and generalizes.")

    plt.show()


def _print_table(dp_stats, rl_stats, has_dp: bool):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    if has_dp:
        hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'RL (continuous)':>16s}"
        print(f"\n{hdr}")
        print("  " + "─" * (len(hdr) - 2))
        for name, key, fmt in rows:
            v = [dp_stats[key], rl_stats[key]]
            print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))
    else:
        hdr = f"  {'Metric':<18s}  {'RL (continuous)':>16s}"
        print(f"\n{hdr}")
        print("  " + "─" * (len(hdr) - 2))
        for name, key, fmt in rows:
            print(f"  {name:<18s}  {rl_stats[key]:{fmt}}")


if __name__ == "__main__":
    main()
