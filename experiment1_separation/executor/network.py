"""
Executor Network

A policy network with "tappable" hidden layers. At each forward pass,
the network records its internal activations so they can be streamed
to the observer.

The executor is the "body" — it acts in the world. It has no awareness
that it is being watched. It simply processes observations and produces
actions, like the brain's motor and decision circuits operating below
the threshold of consciousness.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TappableNetwork(nn.Module):
    """A neural network whose hidden activations can be read by the observer.

    Architecture is intentionally simple and modular:
    Input → Layer1 → ReLU → Layer2 → ReLU → Output

    Each layer's post-activation values are stored in self.layer_activations
    after every forward pass. This is the "neural tap" — the one-way channel
    from executor to observer.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 n_hidden_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # Build layers
        layers = []
        prev_dim = obs_dim
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation — raw logits for discrete, mean for continuous)
        layers.append(nn.Linear(prev_dim, act_dim))

        self.layers = nn.ModuleList()
        self.activations_layers = nn.ModuleList()

        # Separate linear and activation layers for tapping
        idx = 0
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(
                obs_dim if i == 0 else hidden_dim, hidden_dim
            ))
            self.activations_layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, act_dim)

        # Storage for activations — the "neural tap"
        self.layer_activations = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that records all internal activations.

        Args:
            x: observation tensor, shape (batch, obs_dim) or (obs_dim,)

        Returns:
            output logits, shape (batch, act_dim) or (act_dim,)
        """
        self.layer_activations = []

        h = x
        for linear, activation in zip(self.layers, self.activations_layers):
            h = linear(h)
            h = activation(h)
            # TAP: record this layer's activations
            self.layer_activations.append(h.detach().cpu().numpy().copy())

        output = self.output_layer(h)
        return output

    def get_activations(self) -> list:
        """Return the most recent layer activations.

        This is the read-only channel to the observer. The observer can
        call this after each forward pass to get the executor's "thoughts."
        """
        return [act.squeeze(0) if act.ndim > 1 and act.shape[0] == 1 else act
                for act in self.layer_activations]


class DiscreteExecutor:
    """Executor for discrete action spaces (CartPole, LunarLander).

    Wraps a TappableNetwork with epsilon-greedy action selection.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 epsilon: float = 0.1, lr: float = 1e-3):
        self.network = TappableNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.act_dim = act_dim
        self.epsilon = epsilon

    def select_action(self, observation: np.ndarray) -> tuple:
        """Select action and return (action, output_logits, hidden_activations).

        The returned tuple contains everything the observer will receive.
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = self.network(obs_tensor)
            logits_np = logits.squeeze(0).numpy()

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.act_dim)
        else:
            action = np.argmax(logits_np)

        hidden_activations = self.network.get_activations()

        return action, logits_np, hidden_activations

    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
