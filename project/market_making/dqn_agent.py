"""Phase 3: Deep Q-Network agent for market making.

Architecture: simple MLP  →  Q(s, ·) for each discrete spread action.
Training: Double DQN, Huber loss, soft target updates, step-based ε-decay,
optional reward clipping. Experience replay with learning_starts delay.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, priority: float = 1.0):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class PrioritizedReplayBuffer:
    """Oversamples transitions from rare (extreme inventory/price) states."""

    def __init__(self, capacity: int = 50_000, rare_priority: float = 3.0):
        self.buf: list = []
        self.priorities: list = []
        self.capacity = capacity
        self.rare_priority = rare_priority

    def push(self, state, action, reward, next_state, done, is_rare: bool = False):
        priority = self.rare_priority if is_rare else 1.0
        self.priorities.append(priority)
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) > self.capacity:
            self.buf.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size: int):
        n = len(self.buf)
        probs = np.array(self.priorities, dtype=np.float64) / sum(self.priorities)
        indices = np.random.choice(n, size=min(batch_size, n), replace=(batch_size > n), p=probs)
        batch = [self.buf[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        hidden_dim: int = 64,
        seed: int = None,
        # --- Double DQN + loss: no new args (Huber in code) ---
        # --- learning_starts: no gradient updates until total_steps > this ---
        learning_starts: int = 10_000,
        # --- soft target updates (Polyak) ---
        tau: float = 0.005,
        # --- optional reward clipping ---
        clip_reward: bool = False,
        # kept for API compatibility; ignored when using step-based decay
        target_update_freq: int = 200,
        # --- prioritized replay: oversample rare (extreme inv/price) states ---
        use_prioritized_replay: bool = False,
        rare_priority: float = 3.0,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts  # [CHANGE] no updates until total_steps > this
        self.tau = tau                          # [CHANGE] soft target update (Polyak)
        self.clip_reward = clip_reward          # [CHANGE] clip rewards to [-1, 1] when True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimiser = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = (
            PrioritizedReplayBuffer(buffer_size, rare_priority=rare_priority)
            if use_prioritized_replay
            else ReplayBuffer(buffer_size)
        )
        self.use_prioritized_replay = use_prioritized_replay
        self.train_steps = 0   # number of gradient steps taken
        self.total_steps = 0   # [CHANGE] total environment steps (for learning_starts + ε decay)

    # ── action selection ─────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        valid_mask: np.ndarray = None,
    ) -> int:
        """Select action with optional boolean mask (True = valid action).

        At inventory boundaries the environment blocks one fill side regardless
        of the spread chosen, so Q-values for that side become indistinguishable.
        Passing a valid_mask forces the agent to pick from the correct spread
        region instead of defaulting to the lowest-index (tightest) spread.
        """
        if random.random() < epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                return int(valid_actions[random.randrange(len(valid_actions))])
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.q_net(s)
            if valid_mask is not None:
                invalid = torch.tensor(~valid_mask, dtype=torch.bool, device=self.device)
                q_vals = q_vals.masked_fill(invalid.unsqueeze(0), float("-inf"))
            return int(q_vals.argmax(dim=1).item())

    @staticmethod
    def boundary_mask(inventory: int, params) -> np.ndarray:
        """Boolean mask of valid actions given current inventory boundary.

        At I == +I_max: bid fills are always blocked → force the widest bid
        spread (highest bid_idx) so the agent signals "don't buy".
        At I == -I_max: ask fills are always blocked → force the widest ask
        spread (highest ask_idx) so the agent signals "don't sell".
        Otherwise all actions are valid.
        """
        n = params.n_spread_options
        n_actions = n * n
        mask = np.ones(n_actions, dtype=bool)
        if inventory >= params.max_inventory:
            # only allow actions with bid_idx == n-1 (widest bid spread)
            for a in range(n_actions):
                if a // n != n - 1:
                    mask[a] = False
        elif inventory <= -params.max_inventory:
            # only allow actions with ask_idx == n-1 (widest ask spread)
            for a in range(n_actions):
                if a % n != n - 1:
                    mask[a] = False
        return mask

    # ── one gradient step (Double DQN, Huber, soft target, learning_starts) ─

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return None
        # [CHANGE] No gradient updates until total env steps exceed learning_starts
        if self.total_steps < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # [CHANGE] Double DQN: online net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        # [CHANGE] Huber loss (Smooth L1) instead of MSE
        loss = nn.SmoothL1Loss()(q_vals, targets)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimiser.step()

        self.train_steps += 1

        # [CHANGE] Soft target update (Polyak) every step instead of hard sync
        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return loss.item()

    # ── full training loop (step-based ε decay, optional reward clipping) ───

    def train(
        self,
        env,
        n_episodes: int = 3000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_episodes: int = 2000,
        epsilon_decay_steps: int = None,
        verbose: bool = True,
        use_boundary_mask: bool = True,
        log_interval: int = 100,
        # Uniform state sampling: every N steps, reset to random (inv, price) and take 1 step.
        # Ensures all 231 states get visited, reduces overfitting to trajectory distribution.
        uniform_state_interval: int = None,
        # With probability X, sample from extreme states (inv ±max, extreme price bins)
        extreme_state_prob: float = 0.0,
    ):
        # [CHANGE] Step-based epsilon decay: default decay over 400k steps if not set
        if epsilon_decay_steps is None:
            epsilon_decay_steps = 400_000
        episode_rewards: list[float] = []
        losses: list[float] = []
        episode_epsilons: list[float] = []
        episode_mean_losses: list[float] = []

        # For uniform state injection: need price_grid from env
        price_grid = getattr(env, "price_grid", None)
        params = env.params

        for ep in range(n_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            ep_losses: list[float] = []
            epsilon = epsilon_start

            while not done:
                # [CHANGE] Uniform state injection: every N steps, teleport to random state
                # Use set_state (not reset) so step_count is preserved and episode can end
                # With extreme_state_prob, oversample extreme (inv, price) for better coverage
                if (
                    uniform_state_interval is not None
                    and price_grid is not None
                    and self.total_steps > 0
                    and self.total_steps % uniform_state_interval == 0
                ):
                    if random.random() < extreme_state_prob:
                        # Targeted extreme states: inv in ±[max-1, max], price in outer bins
                        si = random.choice([0, 1, params.n_inventory_states - 2, params.n_inventory_states - 1])
                        sp = random.choice([0, 1, len(price_grid) - 2, len(price_grid) - 1])
                    else:
                        si = random.randint(0, params.n_inventory_states - 1)
                        sp = random.randint(0, len(price_grid) - 1)
                    inv = params.index_to_inventory(si)
                    price_dev = float(price_grid[sp])
                    obs = env.set_state(inv, price_dev)

                # [CHANGE] Epsilon from total_steps (linear decay over epsilon_decay_steps)
                decay_frac = min(1.0, self.total_steps / max(epsilon_decay_steps, 1))
                epsilon = epsilon_end + (1.0 - decay_frac) * (epsilon_start - epsilon_end)

                mask = (
                    self.boundary_mask(env.inventory, env.params)
                    if use_boundary_mask
                    else None
                )
                inv_before = env.inventory
                price_dev_before = env.mid_price - env.params.initial_price
                action = self.select_action(obs, epsilon, valid_mask=mask)
                next_obs, reward, done, _ = env.step(action)
                # [CHANGE] Optional reward clipping to [-1, 1]
                if self.clip_reward:
                    reward = float(np.clip(reward, -1.0, 1.0))
                price_scale = getattr(env, "price_scale", 10.0)
                is_rare = (
                    abs(inv_before) >= params.max_inventory - 1
                    or abs(price_dev_before) >= 0.8 * price_scale
                )
                if self.use_prioritized_replay:
                    self.buffer.push(obs, action, reward, next_obs, done, is_rare=is_rare)
                else:
                    self.buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                self.total_steps += 1
                loss = self._update()
                if loss is not None:
                    losses.append(loss)
                    ep_losses.append(loss)

            episode_rewards.append(total_reward)
            episode_epsilons.append(epsilon)
            episode_mean_losses.append(
                float(np.mean(ep_losses)) if ep_losses else float("nan")
            )

            if verbose and (ep + 1) % log_interval == 0:
                recent_rew = episode_rewards[-log_interval:]
                recent_loss = [
                    l for l in episode_mean_losses[-log_interval:] if not np.isnan(l)
                ]
                loss_str = (
                    f"loss={np.mean(recent_loss):.4f}  " if recent_loss else "loss=n/a  "
                )
                print(
                    f"  ep {ep+1:>5d}/{n_episodes}  "
                    f"reward(last {log_interval})={np.mean(recent_rew):+.2f}  "
                    f"{loss_str}"
                    f"ε={epsilon:.3f}  steps={self.total_steps}"
                )

        return episode_rewards, {
            "losses": losses,
            "episode_epsilons": episode_epsilons,
            "episode_mean_losses": episode_mean_losses,
        }

    # ── policy distillation: train to mimic a teacher (DP) policy ─────────

    def train_distillation(
        self,
        env,
        policy_teacher: np.ndarray,
        params,
        price_grid: np.ndarray,
        n_epochs: int = 500,
        batch_size: int = 128,
        lr: float = None,
        verbose: bool = True,
        log_interval: int = 50,
    ):
        """Train DQN to mimic a teacher policy (e.g. from DP) via supervised learning.

        policy_teacher: shape (n_inventory_states, n_price_bins), action indices.
        Samples (s, a*) uniformly from the grid and minimizes cross-entropy.
        """
        if lr is not None:
            for g in self.optimiser.param_groups:
                g["lr"] = lr

        n_inv = params.n_inventory_states
        n_price = len(price_grid)
        n_states = n_inv * n_price

        # Build full dataset: (obs, teacher_action) for every state
        dataset_states = []
        dataset_actions = []
        for si in range(n_inv):
            inv = params.index_to_inventory(si)
            for sp in range(n_price):
                price_dev = float(price_grid[sp])
                obs = env.obs_for_state(inv, price_dev=price_dev)
                a_star = int(policy_teacher[si, sp])
                dataset_states.append(obs)
                dataset_actions.append(a_star)

        dataset_states = np.array(dataset_states, dtype=np.float32)
        dataset_actions = np.array(dataset_actions, dtype=np.int64)

        losses = []
        for epoch in range(n_epochs):
            perm = np.random.permutation(n_states)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_states, batch_size):
                idx = perm[start : start + batch_size]
                states = torch.FloatTensor(dataset_states[idx]).to(self.device)
                targets = torch.LongTensor(dataset_actions[idx]).to(self.device)

                logits = self.q_net(states)
                loss = nn.CrossEntropyLoss()(logits, targets)

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
                self.optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Soft target update
            for target_param, param in zip(
                self.target_net.parameters(), self.q_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

            mean_loss = epoch_loss / max(n_batches, 1)
            losses.append(mean_loss)

            if verbose and (epoch + 1) % log_interval == 0:
                # Compute policy match accuracy
                with torch.no_grad():
                    all_logits = self.q_net(torch.FloatTensor(dataset_states).to(self.device))
                    pred_actions = all_logits.argmax(dim=1).cpu().numpy()
                    match = (pred_actions == dataset_actions).mean() * 100
                print(
                    f"  epoch {epoch+1:>4d}/{n_epochs}  loss={mean_loss:.4f}  "
                    f"policy_match={match:.1f}%"
                )

        return {"losses": losses}

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int = 1000, episode_seed_base: int = None):
        """Evaluate policy. If episode_seed_base is set, each episode i uses
        seed=episode_seed_base+i for identical trajectories (paired with DP)."""
        episode_rewards: list[float] = []
        episode_pnls: list[float] = []
        final_inventories: list[int] = []
        all_inventories: list[int] = []
        all_actions: list[int] = []
        all_vols: list[float] = []

        for i in range(n_episodes):
            if episode_seed_base is not None:
                obs = env.reset(seed=episode_seed_base + i)
            else:
                obs = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                mask = self.boundary_mask(env.inventory, env.params)
                action = self.select_action(obs, epsilon=0.0, valid_mask=mask)
                all_actions.append(action)
                all_inventories.append(env.inventory)
                if env.use_vol:
                    all_vols.append(env.volatility)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)
            episode_pnls.append(env.pnl_history[-1])
            final_inventories.append(env.inventory)

        rew = np.array(episode_rewards)
        pnl = np.array(episode_pnls)

        return {
            "mean_reward": np.mean(rew),
            "std_reward": np.std(rew),
            "sharpe": np.mean(rew) / (np.std(rew) + 1e-8),
            "mean_pnl": np.mean(pnl),
            "std_pnl": np.std(pnl),
            "mean_abs_inventory": np.mean(np.abs(all_inventories)),
            "max_abs_inventory": int(np.max(np.abs(all_inventories))),
            "mean_final_inventory": np.mean(np.abs(final_inventories)),
            "episode_rewards": rew,
            "episode_pnls": pnl,
            "final_inventories": np.array(final_inventories),
            "all_actions": np.array(all_actions),
            "all_inventories": np.array(all_inventories),
            "all_vols": np.array(all_vols) if all_vols else None,
        }
