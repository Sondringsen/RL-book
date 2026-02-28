#!/usr/bin/env python3
"""Phase 2 — solve the simplified market-making MDP with value iteration.

Outputs
-------
  results/phase2_dp_analysis.png          value fn, policy, convergence, skew sensitivity
  results/phase2_simulation.png           reward & inventory histograms from rollouts
  results/phase2_arrival_sensitivity.png  bid/ask spread vs arrival rate
  results/phase2_curse_of_dim.png         DP solve time vs state/action space size
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from market_making import MarketParams, MarketMakingEnv
from market_making.dp_solver import value_iteration, simulate_dp_policy


def main():
    os.makedirs("results", exist_ok=True)
    params = MarketParams()

    print("=" * 60)
    print("Phase 2: Dynamic Programming Solution")
    print("=" * 60)
    print(f"  Inventory grid      : I ∈ [{-params.max_inventory}, {params.max_inventory}]")
    print(f"  Spread options      : δ ∈ {params.spread_options}")
    print(f"  Action space        : {params.n_actions} (δ_bid, δ_ask) pairs")
    print(f"  Arrival rate        : λ(δ) = {params.arrival_base}·exp(−{params.arrival_decay}·δ)")
    print(f"  Adverse selection   : {params.adverse_selection}")
    print(f"  Inv. penalty α      : {params.inventory_penalty}")
    print(f"  Discount γ          : {params.discount}")
    print()
    print("  Fill probabilities per spread:")
    for d in params.spread_options:
        print(f"    δ = {d:.1f}  →  p = {params.fill_probability(d):.4f}")

    # ── value iteration ──────────────────────────────────────────────
    print("\nRunning value iteration …")
    V, policy, residuals = value_iteration(params)

    inventories = params.inventory_states

    print("\nOptimal policy (π*):")
    print(f"  {'I':>4s}  {'δ*_bid':>7s}  {'δ*_ask':>7s}  {'V(I)':>10s}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*10}")
    for i, I in enumerate(inventories):
        db, da = params.action_to_spreads(policy[i])
        print(f"  {I:>4d}  {db:>7.1f}  {da:>7.1f}  {V[i]:>10.4f}")

    # ── analysis plots ───────────────────────────────────────────────
    opt_bids = np.array([params.action_to_spreads(policy[i])[0]
                         for i in range(params.n_inventory_states)])
    opt_asks = np.array([params.action_to_spreads(policy[i])[1]
                         for i in range(params.n_inventory_states)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(inventories, V, "b-o", markersize=5)
    axes[0, 0].set_xlabel("Inventory  I")
    axes[0, 0].set_ylabel("V(I)")
    axes[0, 0].set_title("Value Function")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].step(inventories, opt_bids, "b-o", where="mid", ms=5, label="δ*_bid")
    axes[0, 1].step(inventories, opt_asks, "r-s", where="mid", ms=5, label="δ*_ask")
    axes[0, 1].set_xlabel("Inventory  I")
    axes[0, 1].set_ylabel("Half-spread")
    axes[0, 1].set_title("Optimal Policy  π*(I): Bid & Ask Spreads")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(residuals)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Bellman residual")
    axes[1, 0].set_title("Convergence of Value Iteration")
    axes[1, 0].grid(True, alpha=0.3)

    # α sensitivity — policy skew (δ_bid − δ_ask) vs I
    alphas = [0.0, 0.005, 0.01, 0.02, 0.05]
    for alpha_val in alphas:
        p_a = MarketParams(inventory_penalty=alpha_val)
        _, pol_a, _ = value_iteration(p_a, verbose=False)
        skew = []
        for i in range(p_a.n_inventory_states):
            db, da = p_a.action_to_spreads(pol_a[i])
            skew.append(db - da)
        axes[1, 1].plot(inventories, skew, "-o", ms=4, label=f"α = {alpha_val}")
    axes[1, 1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[1, 1].set_xlabel("Inventory  I")
    axes[1, 1].set_ylabel("Skew  (δ*_bid − δ*_ask)")
    axes[1, 1].set_title("Policy Skew Sensitivity to α")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/phase2_dp_analysis.png", dpi=150)
    print("\n→ saved results/phase2_dp_analysis.png")

    # ── simulate DP policy ───────────────────────────────────────────
    print("\nSimulating DP policy (1 000 episodes, constant σ) …")
    env = MarketMakingEnv(params, use_volatility_dynamics=False, seed=42)
    stats = simulate_dp_policy(env, policy, params, n_episodes=1000)

    print("\nPerformance summary:")
    print(f"  Mean episode reward : {stats['mean_reward']:+.2f} ± {stats['std_reward']:.2f}")
    print(f"  Reward Sharpe       : {stats['sharpe']:.3f}")
    print(f"  Mean MtM PnL        : {stats['mean_pnl']:+.2f} ± {stats['std_pnl']:.2f}")
    print(f"  Mean |inventory|    : {stats['mean_abs_inventory']:.2f}")
    print(f"  Max  |inventory|    : {stats['max_abs_inventory']}")
    print(f"  Mean |final inv|    : {stats['mean_final_inventory']:.2f}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    axes2[0].hist(stats["episode_rewards"], bins=50, edgecolor="k", alpha=0.7)
    axes2[0].axvline(stats["mean_reward"], color="red", ls="--", label="mean")
    axes2[0].set_xlabel("Episode Reward")
    axes2[0].set_ylabel("Frequency")
    axes2[0].set_title("DP Policy — Reward Distribution")
    axes2[0].legend()

    inv_bins = range(-params.max_inventory - 1, params.max_inventory + 2)
    axes2[1].hist(
        stats["final_inventories"], bins=inv_bins,
        edgecolor="k", alpha=0.7, align="left",
    )
    axes2[1].set_xlabel("Final Inventory")
    axes2[1].set_ylabel("Frequency")
    axes2[1].set_title("DP Policy — Final Inventory Distribution")

    plt.tight_layout()
    plt.savefig("results/phase2_simulation.png", dpi=150)
    print("→ saved results/phase2_simulation.png")

    # ── arrival-rate sensitivity ─────────────────────────────────────
    print("\nArrival-rate sensitivity …")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    for A in [0.3, 0.5, 0.8, 1.2]:
        p_ar = MarketParams(arrival_base=A)
        _, pol_ar, _ = value_iteration(p_ar, verbose=False)
        bids_ar, asks_ar = [], []
        for i in range(p_ar.n_inventory_states):
            db, da = p_ar.action_to_spreads(pol_ar[i])
            bids_ar.append(db)
            asks_ar.append(da)
        axes3[0].plot(inventories, bids_ar, "-o", ms=4, label=f"A = {A}")
        axes3[1].plot(inventories, asks_ar, "-o", ms=4, label=f"A = {A}")
    axes3[0].set_xlabel("Inventory  I")
    axes3[0].set_ylabel("δ*_bid")
    axes3[0].set_title("Bid Spread Sensitivity to Arrival Rate A")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    axes3[1].set_xlabel("Inventory  I")
    axes3[1].set_ylabel("δ*_ask")
    axes3[1].set_title("Ask Spread Sensitivity to Arrival Rate A")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase2_arrival_sensitivity.png", dpi=150)
    print("→ saved results/phase2_arrival_sensitivity.png")

    # ── curse of dimensionality ───────────────────────────────────────
    print("\nCurse of dimensionality experiment …")

    # Sweep 1: fix I_max=5, grow number of spread options
    spread_counts = [3, 5, 7, 10, 13, 16, 20, 25, 30]
    times_spread = []
    n_actions_list = []
    for n_sp in spread_counts:
        sp = tuple(np.linspace(0.5, 2.5, n_sp).round(2))
        p = MarketParams(spread_options=sp, max_inventory=5)
        t0 = time.perf_counter()
        value_iteration(p, verbose=False)
        elapsed = time.perf_counter() - t0
        times_spread.append(elapsed)
        n_actions_list.append(p.n_actions)
        print(f"  n_spread={n_sp:>2d}  actions={p.n_actions:>4d}  "
              f"states={p.n_inventory_states:>3d}  time={elapsed:.4f}s")

    # Sweep 2: fix n_spread=5, grow max inventory
    inv_maxes = [3, 5, 10, 15, 20, 30, 40, 50]
    times_inv = []
    n_states_list = []
    for im in inv_maxes:
        p = MarketParams(max_inventory=im)
        t0 = time.perf_counter()
        value_iteration(p, verbose=False)
        elapsed = time.perf_counter() - t0
        times_inv.append(elapsed)
        n_states_list.append(p.n_inventory_states)
        print(f"  I_max={im:>2d}  actions={p.n_actions:>4d}  "
              f"states={p.n_inventory_states:>3d}  time={elapsed:.4f}s")

    # Sweep 3: grow both simultaneously
    configs = [
        (3,  3),
        (5,  5),
        (7,  7),
        (10, 10),
        (13, 15),
        (16, 20),
        (20, 30),
        (25, 40),
        (30, 50),
    ]
    times_both = []
    total_sa_list = []
    for n_sp, im in configs:
        sp = tuple(np.linspace(0.5, 2.5, n_sp).round(2))
        p = MarketParams(spread_options=sp, max_inventory=im)
        t0 = time.perf_counter()
        value_iteration(p, verbose=False)
        elapsed = time.perf_counter() - t0
        times_both.append(elapsed)
        total_sa = p.n_inventory_states * p.n_actions
        total_sa_list.append(total_sa)
        print(f"  n_spread={n_sp:>2d}  I_max={im:>2d}  "
              f"|S|×|A|={total_sa:>6d}  time={elapsed:.4f}s")

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

    axes4[0].plot(n_actions_list, times_spread, "o-", color="C0", lw=1.5, ms=6)
    axes4[0].set_xlabel("|A|  (number of actions = n_spread²)")
    axes4[0].set_ylabel("Solve time (s)")
    axes4[0].set_title("DP Time vs Action Space\n(I_max = 5 fixed)")
    axes4[0].grid(True, alpha=0.3)

    axes4[1].plot(n_states_list, times_inv, "s-", color="C1", lw=1.5, ms=6)
    axes4[1].set_xlabel("|S|  (number of states = 2·I_max + 1)")
    axes4[1].set_ylabel("Solve time (s)")
    axes4[1].set_title("DP Time vs State Space\n(n_spread = 5 fixed)")
    axes4[1].grid(True, alpha=0.3)

    axes4[2].plot(total_sa_list, times_both, "D-", color="C2", lw=1.5, ms=6)
    axes4[2].set_xlabel("|S| × |A|")
    axes4[2].set_ylabel("Solve time (s)")
    axes4[2].set_title("DP Time vs |S| × |A|\n(both growing)")
    axes4[2].grid(True, alpha=0.3)

    for ax in axes4:
        ax.set_yscale("log")
        ax.set_xscale("log")

    fig4.suptitle("Curse of Dimensionality: Value Iteration Solve Time", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/phase2_curse_of_dim.png", dpi=150, bbox_inches="tight")
    print("→ saved results/phase2_curse_of_dim.png")

    plt.show()


if __name__ == "__main__":
    main()
