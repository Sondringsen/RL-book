#!/usr/bin/env python3
"""Experiment 1 — State = inventory only (1D).

Compares DP vs RL (discrete).
DP is optimal for the discrete MDP and serves as an upper bound for RL (discrete).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, VecMarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration, simulate_dp_policy
from market_making.gpu_config import gpu_batch_size, gpu_hidden_dim, gpu_info

SPREAD_OPTIONS = (0.5, 1.0, 1.5, 2.0, 2.5)
SIGMA_BASE = 0.25
SEED = 42
EVAL_SEED = 123


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams(spread_options=SPREAD_OPTIONS, sigma_base=SIGMA_BASE)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 1: Inventory-only state — DP vs RL (discrete)")
    print("=" * 60)
    print(f"  State dim   : 1 (inventory)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Volatility  : constant σ = {params.sigma_base}")
    print(f"  Device      : {gpu_info()}")

    BATCH_TRAIN = gpu_batch_size(256)
    HIDDEN_TRAIN = gpu_hidden_dim(128)

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/3] DP value iteration …")
    V_dp, policy_dp = value_iteration(params)

    # ── RL Discrete ────────────────────────────────────────────────────
    print("\n[2/3] Training DQN (discrete state, vectorized) …")
    def _make_env_disc():
        return MarketMakingEnv(
            params, use_volatility_dynamics=False,
            include_price=False, discrete_inventory=True,
            random_init=True, use_continuous_state=False, seed=SEED,
        )
    train_env_disc = VecMarketMakingEnv(32, _make_env_disc)
    agent_disc = DQNAgent(
        state_dim=train_env_disc.state_dim, n_actions=train_env_disc.n_actions,
        lr=2e-4, gamma=params.discount, batch_size=BATCH_TRAIN, hidden_dim=HIDDEN_TRAIN,
        learning_starts=5_000, tau=0.005, seed=SEED,
    )
    agent_disc.train(train_env_disc, n_episodes=1000, epsilon_decay_steps=150_000, verbose=True)

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\n[3/3] Evaluating (1 000 episodes each) …")
    eval_env = MarketMakingEnv(
        params, use_volatility_dynamics=False, include_price=False,
        discrete_inventory=False, seed=EVAL_SEED,
    )
    dp_stats = simulate_dp_policy(eval_env, policy_dp, params, n_episodes=1000, episode_seed_base=EVAL_SEED)

    eval_disc = MarketMakingEnv(
        params, use_volatility_dynamics=False, include_price=False,
        discrete_inventory=True, use_continuous_state=False, seed=EVAL_SEED,
    )
    disc_stats = agent_disc.evaluate(eval_disc, n_episodes=1000, episode_seed_base=EVAL_SEED)

    _print_table(dp_stats, disc_stats)
    _save_latex_table(
        [dp_stats, disc_stats],
        col_names=["DP", "RL (discrete)"],
        caption=(
            "Experiment 1 (inventory-only state): mean and standard deviation of "
            "episode reward over 1\\,000 evaluation episodes. "
            "DP is optimal for the discrete MDP and provides an upper bound for RL (discrete)."
        ),
        label="tab:exp1",
        path="results/exp1_table.tex",
    )

    print("\n  Experiment 1 — Mean MtM PnL:")
    print(f"    DP: {dp_stats['mean_pnl']:.2f}  RL (discrete): {disc_stats['mean_pnl']:.2f}")

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

    disc_bids, disc_asks = get_rl_spreads(agent_disc, eval_disc)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, disc_bids, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Bid half-spread")
    axes[0, 0].set_title("Bid Spreads")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, disc_asks, "g--^", where="mid", ms=4, label="RL (discrete)")
    axes[0, 1].set_xlabel("Inventory")
    axes[0, 1].set_ylabel("Ask half-spread")
    axes[0, 1].set_title("Ask Spreads")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    all_rewards = [dp_stats["episode_rewards"], disc_stats["episode_rewards"]]
    lo = min(r.min() for r in all_rewards)
    hi = max(r.max() for r in all_rewards)
    bins = np.linspace(lo, hi, 40)
    axes[1, 0].hist(dp_stats["episode_rewards"], bins=bins, alpha=0.4, label="DP", edgecolor="k")
    axes[1, 0].hist(disc_stats["episode_rewards"], bins=bins, alpha=0.4, label="RL (discrete)", edgecolor="k")
    axes[1, 0].set_xlabel("Episode Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].legend(fontsize=8)

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes[1, 1].hist(dp_stats["final_inventories"], bins=inv_bins, alpha=0.4, label="DP", edgecolor="k", align="left")
    axes[1, 1].hist(disc_stats["final_inventories"], bins=inv_bins, alpha=0.4, label="RL (discrete)", edgecolor="k", align="left")
    axes[1, 1].set_xlabel("Final Inventory")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Final Inventory Distribution")
    axes[1, 1].legend(fontsize=8)

    # Cumulative PnL
    n_ep = len(dp_stats["episode_pnls"])
    fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(10, 4))
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(dp_stats["episode_pnls"]), label="DP", lw=1)
    ax_pnl.plot(range(1, n_ep + 1), np.cumsum(disc_stats["episode_pnls"]), label="RL (discrete)", lw=1)
    ax_pnl.set_xlabel("Episode")
    ax_pnl.set_ylabel("Cumulative MtM PnL")
    ax_pnl.set_title("Experiment 1: Cumulative PnL")
    ax_pnl.legend(fontsize=8)
    ax_pnl.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/exp1_pnl.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_pnl.png")

    # Episode reward over time
    fig_rew, ax_rew = plt.subplots(1, 1, figsize=(10, 4))
    window = 50
    for rewards, label, color in [
        (dp_stats["episode_rewards"], "DP", "C0"),
        (disc_stats["episode_rewards"], "RL (discrete)", "C2"),
    ]:
        ax_rew.plot(range(1, n_ep + 1), rewards, alpha=0.15, color=color)
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax_rew.plot(range(window, n_ep + 1), rolling, lw=2, color=color, label=label)
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Episode Reward")
    ax_rew.set_title("Experiment 1: Episode Reward over Time")
    ax_rew.legend(fontsize=8)
    ax_rew.grid(True, alpha=0.3)
    fig_rew.tight_layout()
    fig_rew.savefig("results/exp1_reward_over_time.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_reward_over_time.png")

    fig.suptitle("Experiment 1: Inventory-only — DP vs RL (discrete)")
    fig.tight_layout()
    fig.savefig("results/exp1_comparison.png", dpi=150, bbox_inches="tight")
    print("→ saved results/exp1_comparison.png")
    plt.show()


def _print_table(dp_stats, disc_stats):
    rows = [
        ("Mean Reward", "mean_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Mean MtM PnL", "mean_pnl", ".2f"),
        ("Mean |Final Inv|", "mean_final_inventory", ".2f"),
    ]
    hdr = f"  {'Metric':<18s}  {'DP':>10s}  {'Discrete':>10s}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))
    for name, key, fmt in rows:
        v = [dp_stats[key], disc_stats[key]]
        print(f"  {name:<18s}  " + "  ".join(format(x, fmt) for x in v))


def _save_latex_table(all_stats, col_names, caption, label, path):
    """Write a booktabs LaTeX table with mean reward and std reward."""
    rows = [
        ("Mean Reward", "mean_reward"),
        ("Std Reward", "std_reward"),
    ]
    n_cols = len(col_names)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        "Metric & " + " & ".join(col_names) + r" \\",
        r"\midrule",
    ]
    for name, key in rows:
        vals = " & ".join(f"{s[key]:.2f}" for s in all_stats)
        lines.append(f"{name} & {vals} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ saved {path}")


if __name__ == "__main__":
    main()
