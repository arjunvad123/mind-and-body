"""
Executor Trainer

Trains the executor using DQN (Deep Q-Network) with experience replay.
The executor learns to act in the environment. It does not know it will
be observed. It simply learns a policy.

After training, the executor is frozen and used as a fixed "body" for
the observer to watch.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import gymnasium as gym
from pathlib import Path

from .network import TappableNetwork, DiscreteExecutor
from ..shared.state_packet import StatePacket, EpisodeRecord


class ReplayBuffer:
    """Standard experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """Trains the executor using DQN.

    The trained executor will be frozen and used as the "body" that the
    observer watches.
    """

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_capacity: int = 50000,
    ):
        self.env = gym.make(env_name)
        self.env_name = env_name
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        # Policy network (the executor)
        self.policy_net = TappableNetwork(obs_dim, act_dim, hidden_dim)
        # Target network (for stable DQN training)
        self.target_net = TappableNetwork(obs_dim, act_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def get_epsilon(self):
        """Exponential epsilon decay."""
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-self.steps_done / self.epsilon_decay)
        return eps

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        eps = self.get_epsilon()
        if random.random() < eps:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        """Single DQN update step."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes: int = 1000, print_every: int = 50,
              target_reward: float = None, save_path: str = None):
        """Train the executor.

        Args:
            n_episodes: maximum training episodes
            print_every: logging frequency
            target_reward: stop early if rolling average hits this
            save_path: where to save the trained model

        Returns:
            training_history: list of episode rewards
        """
        history = []
        rolling_reward = deque(maxlen=100)

        print(f"Training executor on {self.env_name}")
        print(f"Obs dim: {self.env.observation_space.shape[0]}, "
              f"Act dim: {self.env.action_space.n}")
        print("-" * 60)

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(state, action, reward, next_state, done)
                loss = self.update()

                state = next_state
                total_reward += reward
                self.steps_done += 1

                # Update target network
                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            history.append(total_reward)
            rolling_reward.append(total_reward)
            rolling_avg = np.mean(rolling_reward)

            if (ep + 1) % print_every == 0:
                print(f"Episode {ep+1}/{n_episodes} | "
                      f"Reward: {total_reward:.1f} | "
                      f"Rolling Avg: {rolling_avg:.1f} | "
                      f"Epsilon: {self.get_epsilon():.3f}")

            if target_reward and rolling_avg >= target_reward:
                print(f"\nTarget reward {target_reward} reached at episode {ep+1}!")
                break

        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'env_name': self.env_name,
                'obs_dim': self.env.observation_space.shape[0],
                'act_dim': self.env.action_space.n,
                'hidden_dim': self.policy_net.hidden_dim,
                'training_history': history,
            }, save_path)
            print(f"Executor saved to {save_path}")

        self.env.close()
        return history

    def collect_state_stream(self, n_episodes: int = 100,
                             model_path: str = None) -> list:
        """Run the trained executor and collect state packets for the observer.

        This is the data collection phase: the executor acts, and we record
        everything the observer will see.

        Args:
            n_episodes: how many episodes to collect
            model_path: if provided, load model from this path

        Returns:
            list of EpisodeRecord objects
        """
        if model_path:
            checkpoint = torch.load(model_path, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_net'])

        self.policy_net.eval()
        env = gym.make(self.env_name)
        episodes = []

        print(f"\nCollecting state stream: {n_episodes} episodes")
        print("-" * 60)

        for ep in range(n_episodes):
            state, _ = env.reset()
            record = EpisodeRecord(episode_id=ep)
            done = False
            t = 0

            while not done:
                # Forward pass through executor
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.policy_net(state_t)
                    logits_np = logits.squeeze(0).numpy()

                # Get hidden activations (the "neural tap")
                hidden_acts = self.policy_net.get_activations()

                # Select action (greedy during collection)
                action = np.argmax(logits_np)

                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Create state packet
                packet = StatePacket(
                    timestep=t,
                    episode=ep,
                    env_observation=state.copy(),
                    executor_hidden_states=hidden_acts,
                    executor_output_logits=logits_np.copy(),
                    action_taken=np.array([action]),
                    reward=reward,
                    done=terminated,
                    truncated=truncated,
                )
                record.add_packet(packet)

                state = next_state
                t += 1

            episodes.append(record)

            if (ep + 1) % 20 == 0:
                print(f"Episode {ep+1}/{n_episodes} | "
                      f"Length: {record.length} | "
                      f"Reward: {record.total_reward:.1f}")

        env.close()
        print(f"\nCollected {sum(e.length for e in episodes)} total timesteps "
              f"across {n_episodes} episodes")
        return episodes
