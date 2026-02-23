"""
State Packet Protocol

The fundamental data structure connecting executor and observer.
At each timestep, the executor emits a state packet containing everything
the observer is allowed to see. The observer NEVER gets access to weights,
gradients, loss, or any mechanism to influence the executor.

This one-way information flow is the architectural enforcement of the
Observer Hypothesis: consciousness watches, it does not control.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


@dataclass
class StatePacket:
    """A single observation from the executor at one timestep.

    This is what the observer "experiences" at each moment.
    """
    timestep: int
    episode: int

    # What the environment looks like
    env_observation: np.ndarray

    # The executor's internal processing (hidden layer activations)
    # These are the "thoughts" the observer witnesses
    executor_hidden_states: list  # list of np.ndarray, one per layer

    # What the executor decided (output logits before action selection)
    executor_output_logits: np.ndarray

    # What the executor actually did
    action_taken: np.ndarray

    # What happened as a result
    reward: float
    done: bool
    truncated: bool = False

    # Optional: value estimate from the executor (if using actor-critic)
    executor_value_estimate: Optional[float] = None


@dataclass
class EpisodeRecord:
    """A complete episode of state packets — one "experience" for the observer."""
    episode_id: int
    packets: list = field(default_factory=list)  # list of StatePacket
    total_reward: float = 0.0
    length: int = 0

    def add_packet(self, packet: StatePacket):
        self.packets.append(packet)
        self.total_reward += packet.reward
        self.length += 1


def packets_to_tensors(packets: list, device='cpu'):
    """Convert a sequence of StatePackets into batched tensors for the observer.

    Returns a dict of tensors, each with shape (seq_len, feature_dim).
    This is the input format the observer model expects.
    """
    seq_len = len(packets)
    if seq_len == 0:
        return None

    # Stack env observations
    env_obs = torch.tensor(
        np.stack([p.env_observation for p in packets]),
        dtype=torch.float32, device=device
    )

    # Stack hidden states per layer
    n_layers = len(packets[0].executor_hidden_states)
    hidden_states = []
    for layer_idx in range(n_layers):
        layer_acts = torch.tensor(
            np.stack([p.executor_hidden_states[layer_idx] for p in packets]),
            dtype=torch.float32, device=device
        )
        hidden_states.append(layer_acts)

    # Stack output logits
    output_logits = torch.tensor(
        np.stack([p.executor_output_logits for p in packets]),
        dtype=torch.float32, device=device
    )

    # Stack actions
    actions = torch.tensor(
        np.stack([p.action_taken for p in packets]),
        dtype=torch.float32, device=device
    )

    # Stack rewards
    rewards = torch.tensor(
        [p.reward for p in packets],
        dtype=torch.float32, device=device
    ).unsqueeze(-1)

    # Stack dones
    dones = torch.tensor(
        [float(p.done) for p in packets],
        dtype=torch.float32, device=device
    ).unsqueeze(-1)

    # Timesteps
    timesteps = torch.tensor(
        [p.timestep for p in packets],
        dtype=torch.long, device=device
    )

    return {
        'env_obs': env_obs,                # (seq_len, obs_dim)
        'hidden_states': hidden_states,     # list of (seq_len, hidden_dim)
        'output_logits': output_logits,     # (seq_len, act_dim)
        'actions': actions,                 # (seq_len, act_dim) or (seq_len,)
        'rewards': rewards,                 # (seq_len, 1)
        'dones': dones,                     # (seq_len, 1)
        'timesteps': timesteps,             # (seq_len,)
    }
