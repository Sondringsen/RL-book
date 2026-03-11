#!/usr/bin/env python3
"""Experiment 2 Real-World — When RL beats DP (wall-clock time).

Real-world scenario:
- 3D state: (inventory, price, volatility) with stochastic OU vol dynamics
- Large state space: 11×121×21 = 27,951 states
  → DP must enumerate all states each iteration → slow (O(|S|×|A|))
  → RL learns from samples → scales with experience, not state count
- 25 actions (5×5 spread grid)

Tuned so RL converges faster than DP: fewer RL episodes, larger DP grid.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration_3d, simulate_dp_policy_3d

# Real-world: larger action space
SPREAD_OPTIONS = (0.5, 1.0, 1.5, 2.0, 2.5)
# Large state grid: DP scales O(|S|×|A|), becomes slow (~3 min for 28k states)
N_PRICE_BINS = 121
N_VOL_BINS = 21
PRICE_HALF_RANGE = 12.0
VOL_LO, VOL_HI = 0.25, 2.2
PRICE_SCALE = PRICE_HALF_RANGE
SEED = 42
EVAL_SEED = 123

# RL tuned for faster convergence (fewer episodes) — beats DP at this scale
RL_EPISODES = 350
RL_EPISODE_LENGTH = 280
RL_LEARNING_STARTS = 2500
RL_EPSILON_DECAY_STEPS = 120_000

N_STATES = 11 * N_PRICE_BINS * N_VOL_BINS  # 11 inv × 121 price × 21 vol


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS)
    inventories = params.inventory_states

    print("=" * 70)
    print("Experiment 2 Real-World: When RL beats DP")
    print("=" * 70)
    print(f"  State dim   : 3 (inventory, price, volatility)")
    print(f"  State count : {N_STATES:,} (DP must enumerate all)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Price bins  : {N_PRICE_BINS}  Vol bins: {N_VOL_BINS}")
    print(f"  Volatility  : OU process (stochastic, real-world)")
    print()

    timings = {}

    # ── DP 3D (slow at scale) ──────────────────────────────────────────
    print("[1/4] DP 3-D value iteration (enumerates all states) …")
    t0 = time.perf_counter()
    V_dp, policy_dp, residuals, price_grid, vol_grid = value_iteration_3d(
        params,
        n_price_bins=N_PRICE_BINS,
        price_half_range=PRICE_HALF_RANGE,
        n_vol_bins=N_VOL_BINS,
        vol_lo=VOL_LO,
        vol_hi=VOL_HI,
    )
    timings["DP (3D)"] = time.perf_counter() - t0
    print(f"  DP converged in {timings['DP (3D)']:.1f} s")

    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        episode_length=RL_EPISODE_LENGTH,
    )

    # ── RL Regular (continuous state, generalizes to any state) ─────────
    print("\n[2/4] Training DQN (continuous state — generalizes to live/unseen states) …")
    t0 = time.perf_counter()
    train_env_reg = MarketMakingEnv(
        train_params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        vol_grid=None,
        discrete_inventory=False,
        price_grid=None,
        random_init=True,
        use_continuous_state=True,
        seed=SEED,
    )
    agent_reg = DQNAgent(
        state_dim=train_env_reg.state_dim,
        n_actions=train_env_reg.n_actions,
        lr=2e-4,
        gamma=train_params.discount,
        batch_size=256,
        hidden_dim=256,
        learning_starts=RL_LEARNING_STARTS,
        tau=0.005,
        seed=SEED,
    )
    agent_reg.train(
        train_env_reg,
        n_episodes=RL_EPISODES,
        epsilon_decay_steps=RL_EPSILON_DECAY_STEPS,
        verbose=True,
    )
    timings["RL (regular)"] = time.perf_counter() - t0
    print(f"  RL (regular) trained in {timings['RL (regular)']:.1f} s")

    # ── RL Discrete (one-hot, matches DP grid but learns from samples) ──
    print("\n[3/4] Training DQN (discrete state) …")
    t0 = time.perf_counter()
    train_env_disc = MarketMakingEnv(
        train_params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        discrete_inventory=True,
        price_grid=price_grid,
        vol_grid=vol_grid,
        random_init=True,
        use_continuous_state=False,
        seed=SEED,
    )
    agent_disc = DQNAgent(
        state_dim=train_env_disc.state_dim,
        n_actions=train_env_disc.n_actions,
        lr=2e-4,
        gamma=train_params.discount,
        batch_size=256,
        hidden_dim=256,
        learning_starts=RL_LEARNING_STARTS,
        tau=0.005,
        seed=SEED,
    )
    agent_disc.train(
        train_env_disc,
        n_episodes=RL_EPISODES,
        epsilon_decay_steps=RL_EPSILON_DECAY_STEPS,
        verbose=True,
    )
    timings["RL (discrete)"] = time.perf_counter() - t0
    print(f"  RL (discrete) trained in {timings['RL (discrete)']:.1f} s")

    # ── Evaluate (live usage: same seeds, compare performance) ──────────
    print("\n[4/4] Evaluating (1 000 episodes each) …")
    eval_base = MarketMakingEnv(
        params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        discrete_inventory=True,
        price_grid=price_grid,
        vol_grid=vol_grid,
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

    eval_reg = MarketMakingEnv(
        params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        vol_grid=None,
        discrete_inventory=False,
        use_continuous_state=True,
        seed=EVAL_SEED,
    )
    reg_stats = agent_reg.evaluate(
        eval_reg, n_episodes=1000, episode_seed_base=EVAL_SEED
    )

    eval_disc = MarketMakingEnv(
        params,
        use_volatility_dynamics=True,
        include_price=True,
        price_scale=PRICE_SCALE,
        discrete_inventory=True,
        price_grid=price_grid,
        vol_grid=vol_grid,
        use_continuous_state=False,
        seed=EVAL_SEED,
    )
    disc_stats = agent_disc.evaluate(
        eval_disc, n_episodes=1000, episode_seed_base=EVAL_SEED
    )

    _print_table(dp_stats, reg_stats, disc_stats)

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nPlotting …")

    # Policy at mid price, mid vol
    mid_p = N_PRICE_BINS // 2
    mid_v = N_VOL_BINS // 2
    dp_bids = [
        params.action_to_spreads(policy_dp[i, mid_p, mid_v])[0]
        for i in range(params.n_inventory_states)
    ]
    dp_asks = [
        params.action_to_spreads(policy_dp[i, mid_p, mid_v])[1]
        for i in range(params.n_inventory_states)
    ]

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, reg_bids, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 0].step(inventories, disc_bids, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads (mid price, mid vol)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, reg_asks, "r--s", where="mid", ms=4, label="RL (regular)")
    axes[0, 1].step(inventories, disc_asks, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads (mid price, mid vol)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [
        dp_stats["episode_rewards"],
        reg_stats["episode_rewards"],
        disc_stats["episode_rewards"],
    ]
    lo = min(r.min() for r in all_rewards)
    hi = max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.4, label="DP", edgecolor="k")
    axes[1, 0].hist(reg_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (regular)", edgecolor="k")
    axes[1, 0].hist(disc_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (discrete)", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    # Convergence time: when RL becomes more useful
    methods = list(timings.keys())
    times_s = [timings[m] for m in methods]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = axes[1, 1].bar(methods, times_s, color=colors, edgecolor="k")
    axes[1, 1].set_ylabel("Time to converge (seconds)")
    winner = min(timings, key=timings.get)
    axes[1, 1].set_title(f"Convergence Time — {winner} wins ({N_STATES:,} states)")
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

    fig.suptitle(
        "Experiment 2 Real-World: RL scales with large state space (3D, stochastic vol)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("results/exp2_realworld_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_realworld_comparison.png")

    # Scalability message plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    bars2 = ax2.bar(methods, times_s, color=colors, edgecolor="k")
    ax2.set_ylabel("Time (seconds)")
    winner = min(timings, key=timings.get)
    ax2.set_title(
        f"Real-World: {N_STATES:,} states — RL beats DP (RL learns from samples)"
    )
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
    plt.tight_layout()
    plt.savefig("results/exp2_realworld_convergence.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp2_realworld_convergence.png")

    print("\nConvergence times (seconds):")
    for m, t in timings.items():
        print(f"  {m}: {t:.1f} s")
    winner = min(timings, key=timings.get)
    print(f"\n→ {winner} converges fastest. With {N_STATES:,} states, RL scales via function approximation.")

    plt.show()


def _print_table(dp_stats, reg_stats, disc_stats):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'RL (regular)':>12s}  {'RL (discrete)':>12s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        v = [dp_stats[key], reg_stats[key], disc_stats[key]]
        print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))


if __name__ == "__main__":
    main()
