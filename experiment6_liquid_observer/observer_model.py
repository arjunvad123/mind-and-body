"""
CfC Observer Model

A Closed-form Continuous-time observer that watches executor hidden states
and predicts the next hidden state. The observer exists in the same
continuous-time domain as the executor.

Also includes LinearBaseline and SelfObservingWrapper.
"""

import torch
import torch.nn as nn
import numpy as np

from ncps.torch import CfC
from ncps.wirings import AutoNCP

from . import config


class LiquidObserver(nn.Module):
    """CfC observer that watches executor hidden states and predicts next state.

    Architecture: Input projection -> CfC with AutoNCP -> Prediction head

    The observer's hidden state evolves via ODEs, naturally integrating
    temporal information about the executor's computational trajectory.
    """

    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        output_size=None,
    ):
        super().__init__()
        input_size = input_size or config.EXECUTOR_HIDDEN_SIZE
        hidden_size = hidden_size or config.OBSERVER_DEFAULT_SIZE
        output_size = output_size or config.OBSERVER_OUTPUT_SIZE

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # CfC output neurons (motor neurons in NCP)
        # AutoNCP requires output_size < units - 2
        self.cfc_output = min(output_size, hidden_size - 3)

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # NCP-structured CfC
        self.wiring = AutoNCP(hidden_size, self.cfc_output)
        self.cfc = CfC(
            hidden_size,
            self.wiring,
            batch_first=True,
            return_sequences=True,
        )

        # Prediction head: map from CfC output to executor hidden size
        self.prediction_head = nn.Sequential(
            nn.Linear(self.cfc_output, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        self.config = {
            'type': 'liquid_observer',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
        }

    def forward(self, executor_hidden, h0=None, return_hidden=False):
        """Forward pass.

        Args:
            executor_hidden: (batch, seq_len, input_size) — executor hidden states
            h0: optional initial hidden state
            return_hidden: if True, also return observer hidden states

        Returns:
            predictions: (batch, seq_len, output_size)
            hidden (optional): (batch, seq_len, hidden_size)
        """
        projected = self.input_proj(executor_hidden)

        if h0 is not None:
            cfc_out, hn = self.cfc(projected, h0)
        else:
            cfc_out, hn = self.cfc(projected)

        predictions = self.prediction_head(cfc_out)

        if return_hidden:
            # For return_hidden, we need step-by-step hidden states
            # Do a step-by-step pass
            _, all_hidden = self.forward_with_states(executor_hidden, h0)
            return predictions, all_hidden

        return predictions

    def forward_with_states(self, executor_hidden, h0=None):
        """Step-by-step forward, capturing observer hidden at every timestep.

        Returns:
            predictions: (batch, seq_len, output_size)
            all_hidden: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = executor_hidden.shape
        projected = self.input_proj(executor_hidden)

        all_hidden = []
        h = h0

        for t in range(seq_len):
            inp = projected[:, t:t + 1, :]
            if h is not None:
                out, h = self.cfc(inp, h)
            else:
                out, h = self.cfc(inp)
            all_hidden.append(h.detach())

        all_hidden = torch.stack(all_hidden, dim=1)  # (batch, seq_len, hidden_size)

        # Full forward for predictions
        predictions = self.forward(executor_hidden, h0)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return predictions, all_hidden

    def get_neuron_groups(self):
        """Inspect AutoNCP adjacency to classify neurons.

        Returns:
            dict: {'sensory': [...], 'inter': [...], 'command': [...], 'motor': [...]}

        AutoNCP convention: motor neurons are the first output_size indices.
        Sensory receive from input. Command have recurrent connections.
        """
        adj = self.wiring.adjacency_matrix
        n_neurons = adj.shape[0]

        # Motor neurons: first cfc_output indices (AutoNCP convention)
        motor = list(range(self.cfc_output))

        # Sensory neurons: neurons that receive input (non-zero in adjacency from input)
        # In AutoNCP, sensory neurons are those in the "sensory" layer
        # Use wiring attributes if available, otherwise infer
        if hasattr(self.wiring, 'sensory_indices'):
            sensory = list(self.wiring.sensory_indices)
        else:
            # Neurons that are not motor and have high in-degree
            sensory = []

        # Command neurons: have recurrent connections (self-loops or mutual)
        command = []
        inter = []

        for i in range(n_neurons):
            if i in motor:
                continue
            if i in sensory:
                continue
            # Check for recurrent connections
            has_recurrent = adj[i, i] != 0 or any(
                adj[i, j] != 0 and adj[j, i] != 0
                for j in range(n_neurons) if j != i
            )
            if has_recurrent:
                command.append(i)
            else:
                inter.append(i)

        # If sensory is empty, assign first non-motor neurons as sensory
        if not sensory:
            remaining = [i for i in range(n_neurons) if i not in motor]
            n_sensory = max(1, len(remaining) // 3)
            sensory = remaining[:n_sensory]
            rest = remaining[n_sensory:]
            command = rest[:len(rest) // 2]
            inter = rest[len(rest) // 2:]

        return {
            'sensory': sensory,
            'inter': inter,
            'command': command,
            'motor': motor,
        }

    def count_parameters(self):
        """Total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class LinearBaseline(nn.Module):
    """Linear baseline: h(t+1) = W @ h(t).

    Tests whether the observer's temporal modeling adds value beyond
    a simple linear transformation.
    """

    def __init__(self, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or config.EXECUTOR_HIDDEN_SIZE
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.config = {
            'type': 'linear_baseline',
            'hidden_size': hidden_size,
        }

    def forward(self, executor_hidden, h0=None, return_hidden=False):
        """Prediction from each timestep independently (no temporal integration).

        Args:
            executor_hidden: (batch, seq_len, hidden_size)
        """
        predictions = self.linear(executor_hidden)

        if return_hidden:
            # No meaningful hidden state for linear baseline
            return predictions, executor_hidden

        return predictions

    def forward_with_states(self, executor_hidden, h0=None):
        """Returns predictions and input as "hidden states"."""
        predictions = self.linear(executor_hidden)
        return predictions, executor_hidden

    def get_neuron_groups(self):
        """Linear baseline has no neuron groups."""
        return {'sensory': [], 'inter': [], 'command': [], 'motor': []}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class SelfObservingWrapper(nn.Module):
    """Wraps a LiquidObserver to feed its own output back as input with delay.

    For the self-observation experiment: does reflexive observation
    produce different dynamics than other-observation?

    At each timestep: input = concat(executor_h[t], observer_h[t-delta])
    """

    def __init__(self, observer, delay_steps=5):
        super().__init__()
        self.observer = observer
        self.delay_steps = delay_steps

        # Expand input projection to accept concatenated input
        old_input_size = observer.input_size
        new_input_size = old_input_size + observer.hidden_size
        self.expanded_proj = nn.Linear(new_input_size, observer.hidden_size)

        # Initialize: first half copies original weights, second half is small random
        with torch.no_grad():
            self.expanded_proj.weight[:, :old_input_size] = observer.input_proj.weight
            self.expanded_proj.weight[:, old_input_size:] *= 0.01
            self.expanded_proj.bias.copy_(observer.input_proj.bias)

    def forward(self, executor_hidden):
        """Process with self-feedback loop.

        Args:
            executor_hidden: (batch, seq_len, input_size)

        Returns:
            predictions: (batch, seq_len, output_size)
            all_hidden: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = executor_hidden.shape
        hidden_size = self.observer.hidden_size

        all_hidden = []
        all_preds = []
        h = None
        hidden_buffer = torch.zeros(batch_size, hidden_size,
                                    device=executor_hidden.device)

        for t in range(seq_len):
            # Get delayed observer hidden state
            if t >= self.delay_steps and len(all_hidden) > 0:
                delayed_h = all_hidden[t - self.delay_steps]
            else:
                delayed_h = hidden_buffer

            # Concatenate executor input with delayed self-observation
            combined = torch.cat([executor_hidden[:, t, :], delayed_h], dim=-1)
            projected = self.expanded_proj(combined).unsqueeze(1)

            if h is not None:
                out, h = self.observer.cfc(projected, h)
            else:
                out, h = self.observer.cfc(projected)

            pred = self.observer.prediction_head(out.squeeze(1))
            all_hidden.append(h.detach())
            all_preds.append(pred)

        all_hidden = torch.stack(all_hidden, dim=1)
        all_preds = torch.stack(all_preds, dim=1)

        return all_preds, all_hidden
