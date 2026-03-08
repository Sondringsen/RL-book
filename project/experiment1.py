#!/usr/bin/env python3
"""Experiment 1 — Inventory-only state, limited actions, constant vol.

State : inventory  I ∈ {−I_max, …, I_max}
Action: (δ_bid, δ_ask) from a small spread grid
Vol   : constant σ

Compares DP (value iteration) vs RL (DQN) side by side.
"""

import os
import time
import uuid

import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv, DQNAgent
from market_making.dp_solver import value_iteration, simulate_dp_policy

SPREAD_OPTIONS = (0.5, 1.0, 1.5)
SEED = 42
EVAL_SEED = 123  # Same seed for DP/RL evaluation → identical trajectories


def _smooth(arr, window=50):
    """Simple moving average."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def main():
    run_id = uuid.uuid4().hex[:8]
    plot_dir = os.path.join("plots", run_id)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print(f"  Run ID : {run_id}  →  plots saved to {plot_dir}/")

    params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        # sigma_base=0.2,   # DP ignores sigma entirely; setting it to 0 removes the I*σ*ε noise
    )                     # that swamps the spread signal (SNR ≈ 0.5 at σ=1)
    inventories = params.inventory_states

    print("=" * 60)
    print("Experiment 1: Inventory-only state, DP vs RL")
    print("=" * 60)
    print(f"  State dim   : 1 (inventory)")
    print(f"  Actions     : {params.n_actions}  spreads={SPREAD_OPTIONS}")
    print(f"  Inv range   : [{-params.max_inventory}, {params.max_inventory}]")
    print(f"  Discount γ  : {params.discount}")
    print(f"  Inv penalty : α = {params.inventory_penalty}")
    print(f"  Volatility  : constant σ = {params.sigma_base}")

    # ── DP ────────────────────────────────────────────────────────────
    print("\n[1/4] DP value iteration …")
    t0 = time.time()
    V_dp, policy_dp, residuals = value_iteration(params)
    dp_time = time.time() - t0

    dp_policy_bids = [params.action_to_spreads(policy_dp[i])[0]
                      for i in range(params.n_inventory_states)]
    dp_policy_asks = [params.action_to_spreads(policy_dp[i])[1]
                      for i in range(params.n_inventory_states)]
    print(f"  → converged in {len(residuals)} iters  "
          f"final residual={residuals[-1]:.2e}  time={dp_time:.1f}s")
    print(f"  → bid spreads : {dp_policy_bids}")
    print(f"  → ask spreads : {dp_policy_asks}")

    # ── RL (discrete inventory + more training → align with DP) ────────
    # Use longer episodes for training only: with episode_length=200 and γ=0.99,
    # done=True truncates 13.4% of future value (γ^200=0.134), systematically
    # underestimating Q-values at states visited near step 200 (i.e. the most
    # frequently visited states near I=0). Longer episodes (γ^1000≈0) eliminate
    # this bias. Evaluation still uses the original 200-step episodes.
    train_params = MarketParams(
        spread_options=SPREAD_OPTIONS,
        terminal_penalty=0.0,
        sigma_base=0.0,
        episode_length=1000,  # γ^1000 ≈ 0 → no terminal truncation bias
    )
    print("\n[2/4] Training DQN (state = one-hot inventory, aligned with DP) …")
    train_env = MarketMakingEnv(
        train_params, use_volatility_dynamics=False,
        discrete_inventory=True, seed=SEED,
        random_init=True,  # randomize starting inventory so all states are visited
    )

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=train_env.n_actions,
        lr=2e-4,
        gamma=train_params.discount,
        batch_size=64,
        hidden_dim=128,
        learning_starts=10_000,
        tau=0.005,
        seed=SEED,
    )
    t0 = time.time()
    ep_rewards, train_info = agent.train(
        train_env,
        n_episodes=400,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100_000,
        verbose=True,
        log_interval=100,
    )
    dqn_time = time.time() - t0
    print(f"  → training done in {dqn_time:.1f}s  "
          f"({agent.total_steps:,} env steps, {agent.train_steps:,} gradient steps)")

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

    # ── training curves ───────────────────────────────────────────────
    print("\n[4/4] Plotting …")
    fig_train, axes_t = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) DP convergence
    axes_t[0, 0].semilogy(residuals, color="steelblue", lw=1)
    axes_t[0, 0].set_xlabel("Iteration")
    axes_t[0, 0].set_ylabel("Max Bellman residual (log scale)")
    axes_t[0, 0].set_title(f"DP Convergence  ({len(residuals)} iters, {dp_time:.1f}s)")
    axes_t[0, 0].grid(True, alpha=0.3)

    # (0,1) DQN episode reward over training
    raw = np.array(ep_rewards)
    smoothed = _smooth(raw, window=50)
    xs_raw = np.arange(1, len(raw) + 1)
    xs_sm = np.arange(50, len(raw) + 1)
    axes_t[0, 1].plot(xs_raw, raw, color="tomato", alpha=0.25, lw=0.8, label="raw")
    axes_t[0, 1].plot(xs_sm, smoothed, color="tomato", lw=2, label="MA-50")
    axes_t[0, 1].axhline(dp_stats["mean_reward"], color="steelblue",
                         ls="--", lw=1.2, label="DP mean")
    axes_t[0, 1].set_xlabel("Episode")
    axes_t[0, 1].set_ylabel("Episode reward")
    axes_t[0, 1].set_title("DQN Training Reward")
    axes_t[0, 1].legend(fontsize=8)
    axes_t[0, 1].grid(True, alpha=0.3)

    # (1,0) DQN loss per gradient step (smoothed)
    losses = np.array(train_info["losses"])
    if len(losses) > 0:
        smoothed_loss = _smooth(losses, window=500)
        xs_loss = np.arange(500, len(losses) + 1)
        axes_t[1, 0].plot(np.arange(1, len(losses) + 1), losses,
                          color="orange", alpha=0.2, lw=0.5)
        axes_t[1, 0].plot(xs_loss, smoothed_loss, color="orange", lw=2)
        axes_t[1, 0].set_xlabel("Gradient step")
        axes_t[1, 0].set_ylabel("Huber loss")
        axes_t[1, 0].set_title("DQN Training Loss")
        axes_t[1, 0].grid(True, alpha=0.3)

    # (1,1) Epsilon decay over episodes
    epsilons = np.array(train_info["episode_epsilons"])
    axes_t[1, 1].plot(np.arange(1, len(epsilons) + 1), epsilons,
                      color="purple", lw=1.5)
    axes_t[1, 1].set_xlabel("Episode")
    axes_t[1, 1].set_ylabel("ε (exploration rate)")
    axes_t[1, 1].set_title("Epsilon Decay")
    axes_t[1, 1].grid(True, alpha=0.3)

    fig_train.suptitle(f"Experiment 1 — Training Curves  (run {run_id})",
                       fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(plot_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")

    # ── comparison plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Bid spreads
    dp_bids = [params.action_to_spreads(policy_dp[i])[0]
               for i in range(params.n_inventory_states)]
    dp_asks = [params.action_to_spreads(policy_dp[i])[1]
               for i in range(params.n_inventory_states)]
    rl_bids, rl_asks = [], []
    for I in inventories:
        obs = eval_env.obs_for_state(I)
        mask = DQNAgent.boundary_mask(I, params)
        a = agent.select_action(obs, valid_mask=mask)
        db, da = params.action_to_spreads(a)
        rl_bids.append(db)
        rl_asks.append(da)

    axes[0, 0].step(inventories, dp_bids, "b-o", where="mid", ms=5, label="DP")
    axes[0, 0].step(inventories, rl_bids, "r-s", where="mid", ms=5, label="RL", ls="--")
    axes[0, 0].set_xlabel("Inventory")
    axes[0, 0].set_ylabel("Half-spread")
    axes[0, 0].set_title("Optimal Policy: Bid Spreads")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Ask spreads
    axes[0, 1].step(inventories, dp_asks, "b-o", where="mid", ms=5, label="DP")
    axes[0, 1].step(inventories, rl_asks, "r-s", where="mid", ms=5, label="RL", ls="--")
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

    axes[1, 2].set_visible(False)

    fig.suptitle(f"Experiment 1: Inventory-Only State — DP vs RL  (run {run_id})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(plot_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    # also keep the results/ copy for backward compat
    plt.savefig("results/exp1_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
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
