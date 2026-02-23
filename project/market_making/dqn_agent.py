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

    def push(self, state, action, reward, next_state, done):
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
        self.buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0   # number of gradient steps taken
        self.total_steps = 0   # [CHANGE] total environment steps (for learning_starts + ε decay)

    # ── action selection ─────────────────────────────────────────────

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())

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
    ):
        # [CHANGE] Step-based epsilon decay: default decay over 400k steps if not set
        if epsilon_decay_steps is None:
            epsilon_decay_steps = 400_000
        episode_rewards: list[float] = []
        losses: list[float] = []

        for ep in range(n_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                # [CHANGE] Epsilon from total_steps (linear decay over epsilon_decay_steps)
                decay_frac = min(1.0, self.total_steps / max(epsilon_decay_steps, 1))
                epsilon = epsilon_end + (1.0 - decay_frac) * (epsilon_start - epsilon_end)

                action = self.select_action(obs, epsilon)
                next_obs, reward, done, _ = env.step(action)
                # [CHANGE] Optional reward clipping to [-1, 1]
                if self.clip_reward:
                    reward = float(np.clip(reward, -1.0, 1.0))
                self.buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                self.total_steps += 1
                loss = self._update()
                if loss is not None:
                    losses.append(loss)

            episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 500 == 0:
                recent = episode_rewards[-500:]
                print(
                    f"  ep {ep+1:>5d}/{n_episodes}  "
                    f"reward(last 500)={np.mean(recent):+.2f}  "
                    f"ε={epsilon:.3f}  steps={self.total_steps}"
                )

        return episode_rewards, losses

    # ── evaluation (unchanged signature and behaviour) ──────────────────────

    def evaluate(self, env, n_episodes: int = 1000):
        episode_rewards: list[float] = []
        episode_pnls: list[float] = []
        final_inventories: list[int] = []
        all_inventories: list[int] = []
        all_actions: list[int] = []
        all_vols: list[float] = []

        for _ in range(n_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs, epsilon=0.0)
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
