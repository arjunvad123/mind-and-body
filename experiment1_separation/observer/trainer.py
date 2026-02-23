"""
Observer Trainer

Trains the observer on state streams collected from the executor.
The observer's ONLY objective is prediction: given what has happened so far,
what will happen next?

Everything else — self-models, surprise, preferences — is emergent.
We measure those properties AFTER training, in the analysis phase.

The observer never sees the executor's weights, gradients, or loss.
It only sees the state stream. It learns to model the executor purely
from watching its behavior — like consciousness learning about the brain
by observing thoughts as they arise.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .model import ObserverTransformer
from ..shared.state_packet import EpisodeRecord, packets_to_tensors


class StateStreamDataset:
    """Dataset of state packet sequences for observer training.

    Converts episode records into fixed-length windows for training.
    Each window is a sequence of state packets the observer processes
    to predict the next step.
    """

    def __init__(self, episodes: list, window_size: int = 64, stride: int = 16):
        """
        Args:
            episodes: list of EpisodeRecord from executor
            window_size: number of timesteps per training sequence
            stride: step size between windows (< window_size = overlapping)
        """
        self.windows = []
        self.window_size = window_size

        for episode in episodes:
            packets = episode.packets
            if len(packets) < window_size + 1:  # need at least 1 extra for target
                # Pad short episodes
                if len(packets) > 1:
                    self.windows.append(packets)
                continue

            for start in range(0, len(packets) - window_size, stride):
                window = packets[start:start + window_size + 1]  # +1 for targets
                self.windows.append(window)

        print(f"Created dataset: {len(self.windows)} windows from "
              f"{len(episodes)} episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


class ObserverTrainer:
    """Trains the observer to predict the executor's behavior.

    The only loss: can you predict what the executor will do next,
    and what the environment will look like after?

    This is pure prediction. The observer develops its internal
    representations entirely in service of this objective.
    What those representations MEAN — whether they constitute
    a self-model, whether they carry valence, whether they integrate
    information in consciousness-like ways — that's for the analysis
    phase to determine.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        n_hidden_layers: int = 2,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        max_seq_len: int = 512,
        lr: float = 3e-4,
        device: str = 'cpu',
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.model = ObserverTransformer(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            n_hidden_layers=n_hidden_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            max_seq_len=max_seq_len,
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr / 10
        )

        # Loss weights
        self.action_loss_weight = 1.0
        self.obs_loss_weight = 0.5

    def _prepare_batch(self, windows: list):
        """Convert a list of packet windows into batched tensors.

        Separates input (0..T-1) and target (1..T) for prediction.
        """
        batch_tensors = []
        for window in windows:
            tensors = packets_to_tensors(window, device=self.device)
            if tensors is not None:
                batch_tensors.append(tensors)

        if not batch_tensors:
            return None, None

        # Stack into batch
        def stack_key(key):
            if key == 'hidden_states':
                n_layers = len(batch_tensors[0]['hidden_states'])
                return [
                    torch.stack([t['hidden_states'][i] for t in batch_tensors])
                    for i in range(n_layers)
                ]
            return torch.stack([t[key] for t in batch_tensors])

        batched = {
            'env_obs': stack_key('env_obs'),
            'hidden_states': stack_key('hidden_states'),
            'output_logits': stack_key('output_logits'),
            'actions': stack_key('actions'),
            'rewards': stack_key('rewards'),
            'dones': stack_key('dones'),
        }

        # Input is 0..T-1, target is 1..T
        input_data = {
            'env_obs': batched['env_obs'][:, :-1],
            'hidden_states': [h[:, :-1] for h in batched['hidden_states']],
            'output_logits': batched['output_logits'][:, :-1],
            'actions': batched['actions'][:, :-1],
            'rewards': batched['rewards'][:, :-1],
            'dones': batched['dones'][:, :-1],
        }

        target_data = {
            'actions': batched['actions'][:, 1:],
            'env_obs': batched['env_obs'][:, 1:],
        }

        return input_data, target_data

    def train(self, dataset: StateStreamDataset, n_epochs: int = 50,
              batch_size: int = 16, save_path: str = None,
              print_every: int = 5):
        """Train the observer.

        Args:
            dataset: StateStreamDataset of executor state streams
            n_epochs: training epochs
            batch_size: windows per batch
            save_path: where to save the trained observer
            print_every: logging frequency
        """
        print(f"Training observer on {len(dataset)} windows")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        history = {'action_loss': [], 'obs_loss': [], 'total_loss': []}

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = {'action': [], 'obs': [], 'total': []}

            # Shuffle and batch
            indices = np.random.permutation(len(dataset))

            for batch_start in range(0, len(indices), batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                windows = [dataset[i] for i in batch_idx]

                input_data, target_data = self._prepare_batch(windows)
                if input_data is None:
                    continue

                # Forward pass
                pred_actions, pred_obs, _ = self.model(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                )

                # Compute losses
                action_loss = nn.MSELoss()(pred_actions, target_data['actions'])
                obs_loss = nn.MSELoss()(pred_obs, target_data['env_obs'])

                total_loss = (self.action_loss_weight * action_loss +
                              self.obs_loss_weight * obs_loss)

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses['action'].append(action_loss.item())
                epoch_losses['obs'].append(obs_loss.item())
                epoch_losses['total'].append(total_loss.item())

            self.scheduler.step()

            # Log
            avg_action = np.mean(epoch_losses['action'])
            avg_obs = np.mean(epoch_losses['obs'])
            avg_total = np.mean(epoch_losses['total'])

            history['action_loss'].append(avg_action)
            history['obs_loss'].append(avg_obs)
            history['total_loss'].append(avg_total)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Action Loss: {avg_action:.6f} | "
                      f"Obs Loss: {avg_obs:.6f} | "
                      f"Total: {avg_total:.6f}")

        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model.config,
                'training_history': history,
            }, save_path)
            print(f"\nObserver saved to {save_path}")

        return history
