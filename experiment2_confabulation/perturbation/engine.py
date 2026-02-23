"""
Perturbation Engine

Injects controlled perturbations into the executor's behavior or the
observer's input stream. Each perturbation type tests a different aspect
of how the observer handles unexpected changes.

The key insight: the observer cannot access the executor's mechanisms.
When the executor changes, the observer MUST confabulate — it has no
access to the real reason for the change. This mirrors the split-brain
phenomenon in humans, where the left hemisphere narrator fabricates
explanations for actions it didn't initiate.

Perturbation Types:
1. Policy Swap — swap the executor for a different one mid-episode
2. Noise Injection — add noise to executor's actions
3. Reward Function Change — change what the executor is optimizing for
4. Environment Dynamics Change — alter physics mid-episode
5. Observation Masking — remove streams from observer's input
"""

import numpy as np
import torch
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

from experiment1_separation.shared.state_packet import StatePacket, EpisodeRecord
from experiment1_separation.executor.network import TappableNetwork


class PerturbationType(Enum):
    POLICY_SWAP = "policy_swap"
    NOISE_INJECTION = "noise_injection"
    OBSERVATION_MASK = "observation_mask"


@dataclass
class PerturbationEvent:
    """Record of when and how a perturbation was applied."""
    perturbation_type: PerturbationType
    start_timestep: int
    end_timestep: Optional[int]  # None = permanent
    parameters: dict  # type-specific parameters
    description: str  # human-readable description of what happened


@dataclass
class PerturbedEpisodeRecord:
    """An episode with perturbation metadata."""
    episode: EpisodeRecord
    perturbation: PerturbationEvent
    pre_perturbation_packets: list   # packets before perturbation
    post_perturbation_packets: list  # packets after perturbation


class PerturbationEngine:
    """Generates perturbed episodes for the confabulation test.

    Takes a trained executor and environment, runs episodes, and injects
    perturbations at specified points. Records complete state packets
    so the observer can be tested on them.
    """

    def __init__(self, env_name: str, primary_executor_path: str,
                 hidden_dim: int = 64):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.hidden_dim = hidden_dim

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        # Load primary executor
        self.primary_net = TappableNetwork(obs_dim, act_dim, hidden_dim)
        checkpoint = torch.load(primary_executor_path, weights_only=True)
        self.primary_net.load_state_dict(checkpoint['policy_net'])
        self.primary_net.eval()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def policy_swap(self, secondary_executor_path: str,
                    swap_timestep: int = 30,
                    n_episodes: int = 50) -> list:
        """Perturbation Type 1: Swap the executor mid-episode.

        At swap_timestep, the policy silently changes to a different
        trained policy. The observer sees a sudden behavioral shift
        with no external cause.

        This is the split-brain test: the "body" changes controllers,
        and the "mind" has to make sense of it.
        """
        # Load secondary executor
        secondary_net = TappableNetwork(self.obs_dim, self.act_dim, self.hidden_dim)
        checkpoint = torch.load(secondary_executor_path, weights_only=True)
        secondary_net.load_state_dict(checkpoint['policy_net'])
        secondary_net.eval()

        results = []

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            record = EpisodeRecord(episode_id=ep)
            pre_packets = []
            post_packets = []
            done = False
            t = 0

            while not done:
                # Choose which network to use
                if t < swap_timestep:
                    net = self.primary_net
                else:
                    net = secondary_net

                # Forward pass
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = net(state_t)
                    logits_np = logits.squeeze(0).numpy()
                hidden_acts = net.get_activations()

                action = np.argmax(logits_np)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                packet = StatePacket(
                    timestep=t, episode=ep,
                    env_observation=state.copy(),
                    executor_hidden_states=hidden_acts,
                    executor_output_logits=logits_np.copy(),
                    action_taken=np.array([action]),
                    reward=reward, done=terminated, truncated=truncated,
                )
                record.add_packet(packet)

                if t < swap_timestep:
                    pre_packets.append(packet)
                else:
                    post_packets.append(packet)

                state = next_state
                t += 1

            perturbation = PerturbationEvent(
                perturbation_type=PerturbationType.POLICY_SWAP,
                start_timestep=swap_timestep,
                end_timestep=None,
                parameters={'secondary_path': secondary_executor_path},
                description=f"Policy swapped to secondary executor at t={swap_timestep}",
            )

            results.append(PerturbedEpisodeRecord(
                episode=record,
                perturbation=perturbation,
                pre_perturbation_packets=pre_packets,
                post_perturbation_packets=post_packets,
            ))

        self.env.close()
        return results

    def noise_injection(self, noise_std: float = 2.0,
                        inject_timestep: int = 30,
                        duration: int = 20,
                        n_episodes: int = 50) -> list:
        """Perturbation Type 2: Add noise to executor's actions.

        The executor becomes "clumsy" — same policy, noisier output.
        The observer should detect degraded competence.

        This tests whether the observer can distinguish "different agent"
        from "same agent, impaired execution."
        """
        results = []

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            record = EpisodeRecord(episode_id=ep)
            pre_packets = []
            post_packets = []
            done = False
            t = 0

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.primary_net(state_t)
                    logits_np = logits.squeeze(0).numpy()
                hidden_acts = self.primary_net.get_activations()

                # Inject noise during the perturbation window
                if inject_timestep <= t < inject_timestep + duration:
                    noisy_logits = logits_np + np.random.normal(0, noise_std,
                                                                 size=logits_np.shape)
                    action = np.argmax(noisy_logits)
                    # Record the noisy logits as what the executor "intended"
                    logits_np = noisy_logits
                else:
                    action = np.argmax(logits_np)

                action = int(np.clip(action, 0, self.act_dim - 1))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                packet = StatePacket(
                    timestep=t, episode=ep,
                    env_observation=state.copy(),
                    executor_hidden_states=hidden_acts,
                    executor_output_logits=logits_np.copy(),
                    action_taken=np.array([action]),
                    reward=reward, done=terminated, truncated=truncated,
                )
                record.add_packet(packet)

                if t < inject_timestep:
                    pre_packets.append(packet)
                elif t < inject_timestep + duration:
                    post_packets.append(packet)

                state = next_state
                t += 1

            perturbation = PerturbationEvent(
                perturbation_type=PerturbationType.NOISE_INJECTION,
                start_timestep=inject_timestep,
                end_timestep=inject_timestep + duration,
                parameters={'noise_std': noise_std, 'duration': duration},
                description=f"Gaussian noise (std={noise_std}) injected at "
                           f"t={inject_timestep} for {duration} steps",
            )

            results.append(PerturbedEpisodeRecord(
                episode=record,
                perturbation=perturbation,
                pre_perturbation_packets=pre_packets,
                post_perturbation_packets=post_packets,
            ))

        self.env.close()
        return results

    def observation_mask(self, mask_layers: list = None,
                         mask_timestep: int = 30,
                         n_episodes: int = 50) -> list:
        """Perturbation Type 5: Remove streams from observer's input.

        This doesn't change the executor — it changes what the observer
        can see. Like suddenly losing peripheral vision. The observer
        must compensate using its internal model.

        If the observer fills in gaps from its model (confabulation),
        this is strong evidence of a generative internal model.
        """
        if mask_layers is None:
            mask_layers = [0]  # Mask first hidden layer by default

        results = []

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            record = EpisodeRecord(episode_id=ep)
            pre_packets = []
            post_packets = []
            done = False
            t = 0

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.primary_net(state_t)
                    logits_np = logits.squeeze(0).numpy()
                hidden_acts = self.primary_net.get_activations()

                # Mask specified layers after mask_timestep
                if t >= mask_timestep:
                    for layer_idx in mask_layers:
                        if layer_idx < len(hidden_acts):
                            hidden_acts[layer_idx] = np.zeros_like(
                                hidden_acts[layer_idx]
                            )

                action = np.argmax(logits_np)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                packet = StatePacket(
                    timestep=t, episode=ep,
                    env_observation=state.copy(),
                    executor_hidden_states=hidden_acts,
                    executor_output_logits=logits_np.copy(),
                    action_taken=np.array([action]),
                    reward=reward, done=terminated, truncated=truncated,
                )
                record.add_packet(packet)

                if t < mask_timestep:
                    pre_packets.append(packet)
                else:
                    post_packets.append(packet)

                state = next_state
                t += 1

            perturbation = PerturbationEvent(
                perturbation_type=PerturbationType.OBSERVATION_MASK,
                start_timestep=mask_timestep,
                end_timestep=None,
                parameters={'mask_layers': mask_layers},
                description=f"Hidden layers {mask_layers} masked at t={mask_timestep}",
            )

            results.append(PerturbedEpisodeRecord(
                episode=record,
                perturbation=perturbation,
                pre_perturbation_packets=pre_packets,
                post_perturbation_packets=post_packets,
            ))

        self.env.close()
        return results
