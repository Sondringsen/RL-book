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
        use_amp: bool = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts  # [CHANGE] no updates until total_steps > this
        self.tau = tau                          # [CHANGE] soft target update (Polyak)
        self.clip_reward = clip_reward          # [CHANGE] clip rewards to [-1, 1] when True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (use_amp if use_amp is not None else (self.device.type == "cuda"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # faster conv-like ops when input sizes fixed

        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        # torch.compile (PyTorch 2+) for faster forward/backward on GPU
        if hasattr(torch, "compile") and self.device.type == "cuda":
            self.q_net = torch.compile(self.q_net, mode="reduce-overhead")
            self.target_net = torch.compile(self.target_net, mode="reduce-overhead")

        self.optimiser = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0   # number of gradient steps taken
        self.total_steps = 0   # [CHANGE] total environment steps (for learning_starts + ε decay)
        self._scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

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
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
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
        # Pin memory and non_blocking for faster CPU→GPU transfer
        kwargs = {"device": self.device, "non_blocking": True} if self.device.type == "cuda" else {"device": self.device}
        states = torch.as_tensor(states, dtype=torch.float32).to(**kwargs)
        actions = torch.as_tensor(actions, dtype=torch.int64).to(**kwargs)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(**kwargs)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(**kwargs)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(**kwargs)

        self.optimiser.zero_grad(set_to_none=True)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = self.q_net(next_states).argmax(dim=1)
                    next_q = self.target_net(next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                    targets = rewards + self.gamma * next_q * (1 - dones)
                loss = nn.SmoothL1Loss()(q_vals, targets)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self._scaler.step(self.optimiser)
            self._scaler.update()
        else:
            q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                targets = rewards + self.gamma * next_q * (1 - dones)
            loss = nn.SmoothL1Loss()(q_vals, targets)
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

                mask = (
                    self.boundary_mask(env.inventory, env.params)
                    if use_boundary_mask
                    else None
                )
                action = self.select_action(obs, epsilon, valid_mask=mask)
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
