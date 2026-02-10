# Reinforcement Learning for Market Making: Project Proposal

## 1. Overview and Motivation

This project proposes to study market making as a sequential decision-making problem using the tools of Markov Decision Processes (MDPs), Dynamic Programming (DP), and Reinforcement Learning (RL). The goal is to train an agent that dynamically sets bid and ask quotes in an electronic limit order market while managing inventory risk, adverse selection from informed traders, tail events, regime changes, and operational constraints such as volume limits and overnight risk.

Market making is a canonical problem in quantitative finance and stochastic control. Classical solutions (e.g., Avellaneda–Stoikov) rely on stylized assumptions and closed-form approximations. In practice, markets exhibit nonstationarity, fat tails, informed order flow, and microstructure effects that violate these assumptions. We want to use RL to train agents to learn robust policies in a noisy, partially predictable, and evolving environment.

The project is structured to align with the three phases outlined in the course project description, with progressively richer models and environments.

---

## 2. Phase 1: Problem Definition (MDP Formulation)

### 2.1 Idealized (Full) Problem Definition

**Objective:**
Design a market-making agent that maximizes long-run risk-adjusted PnL while controlling inventory, tail risk, and drawdowns across changing market regimes.

#### State Space

At time $t$, the state $s_t$ includes:

* $p_t$: mid-price
* $I_t$: agent inventory
* $\sigma_t$: short-term volatility estimate
* $\lambda_t^{buy}, \lambda_t^{sell}$: order arrival intensities
* $R_t$: latent market regime (e.g., calm, volatile, trending)
* $T_t$: time of day (to capture intraday seasonality)

$$
s_t = (p_t, I_t, \sigma_t, \lambda_t^{buy}, \lambda_t^{sell}, R_t, T_t)
$$

#### Action Space

The agent chooses bid and ask offsets relative to the mid-price:

$$
a_t = (\delta_t^{bid}, \delta_t^{ask})
$$

Optionally extended to include quote sizes $q_t^{bid}, q_t^{ask}$.

#### Dynamics

* Mid-price follows a stochastic process with regime-dependent dynamics: $dp_t = \mu(R_t)dt + \sigma(R_t)dW_t + dJ_t$ where $dJ_t$ captures jump (tail) events.

* Order arrivals follow Hawkes or Poisson processes with intensities depending on spreads and regime: $\lambda^{buy}(\delta^{ask}), \quad \lambda^{sell}(\delta^{bid})$

* Inventory evolves based on executions.

* Regimes evolve as a hidden Markov chain.

#### Reward Function

Risk-adjusted PnL with penalties:

$$
r_t = \text{PnL}\cdot t - \alpha I_t^2 - \beta |\Delta I_t| - \eta \cdot \text{slippage}(\Delta I_t, \sigma_t) - \gamma \mathbb{I}_{\text{tail event}}
$$

Where:

* $\alpha$: inventory risk
* $\beta$: transaction/impact cost
* $\eta$: slippage cost (execution price degradation vs. quoted price; often modeled as $\propto \sigma_t |\Delta I_t|$ or $\propto |\Delta I_t|^2$ to capture market impact)
* $\gamma$: tail-risk penalty

#### Practical Risks Modeled

* Inventory risk via quadratic penalties
* Overnight risk via terminal inventory liquidation costs
* Informed traders via adverse selection (price moves after fills)
* Slippage via execution price degradation (quote is hit but fills occur at worse prices due to queue position, partial fills, or latency)
* Tail events via jump processes
* Regime changes via latent state
* Volume constraints via max quote size or participation limits

This version is **not solvable within the course**, but defines the long-term vision.

---

### 2.2 Course-Scope (RL-Solvable) Version

This is the version targeted in **Phase 3**.

Simplifications:

* Discrete time ($\Delta t$)
* Discrete inventory grid: $I_t \in \lbrace-I_{max}, ..., I_{max}\rbrace$
* Finite action set of spread pairs
* Observable volatility proxy instead of latent regime

State:

$$
s_t = (p_t, I_t, \sigma_t)
$$

Action:

$$
a_t \in \lbrace(\delta_i^{bid}, \delta_j^{ask})\rbrace
$$

Dynamics driven by simulated order flow calibrated to real data.

---

### 2.3 DP-Solvable (Phase 2) Version

This is the most simplified model, designed for DP/ADP.

Simplifications:

* Fixed mid-price or random walk with constant volatility
* Independent Poisson arrivals
* Symmetric spreads

State:

$$
s_t = I_t
$$

Action (because we assume symmetric spreads and known midprice we don't only need to quote one price as the other can be inferred):

$$
a_t = \delta_t \in \lbrace\delta_1, ..., \delta_n\rbrace
$$

Reward:

$$
r_t = \text{spread capture} - \alpha I_t^2 - \eta \cdot \text{slippage}(\Delta I_t)
$$

With slippage modeled simply as $\eta |\Delta I_t|$ or $\eta (\Delta I_t)^2$ to capture that larger trades incur worse average execution prices.

This version highlights the inventory–profit tradeoff and motivates RL by exposing the curse of dimensionality.

---

## 3. Phase 2: Solving a Simplified Version with DP

### 3.1 Approach

* Implement the simplified MDP with discrete inventory and spreads
* Use value iteration or policy iteration
* Analyze:

  * Optimal spread vs. inventory
  * Impact of inventory penalty $\alpha$
  * Sensitivity to arrival rates

### 3.2 Key Insights to Extract

* Inventory mean reversion induced by optimal policy
* Tradeoff between tight spreads and inventory risk
* Failure modes when volatility or arrival rates are misspecified

### 3.3 Mathematical Solution

Bellman equation:

$$
V(I) = \max_{\delta} \mathbb{E}\left[r(I, \delta) + \gamma V(I')\right]
$$

Where $I'$ depends on execution probabilities.

This phase demonstrates why DP breaks down as realism is added. The simplifications required one-dimensional state, known transition dynamics, and discrete actions which make value iteration tractable. Adding volatility, regimes, or adverse selection would blow up the state space and and make DP intractable.

---

## 4. Phase 3: Realistic Market Making with RL

### 4.1 Simulated Environment

* Calibrate mid-price dynamics and order flow to historical LOB data
* Introduce:

  * Stochastic volatility
  * Adverse selection (post-fill price drift)
  * Slippage (fills at worse-than-quoted prices)
  * Volume constraints
  * Regime switching

Synthetic data is used first, followed by out-of-sample testing on real data for evaluation only.

### 4.2 RL Algorithm

Candidate methods:

* Deep Q-Network (DQN) for discrete actions
* Actor–Critic (e.g., PPO) for continuous spreads
* Distributional RL to capture tail risk


### 4.3 Constraints and Extensions

* Inventory limits enforced via hard constraints or penalties
* Volume constraints via capped execution sizes
* Overnight risk via terminal liquidation cost
* Regime robustness via domain randomization

### 4.4 Evaluation Metrics

* Risk-adjusted PnL (Sharpe)
* Inventory distribution
* Drawdowns
* Performance across regimes
* Stability under tail events

