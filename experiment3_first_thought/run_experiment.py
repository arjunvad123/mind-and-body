"""
Experiment 3: First Thought vs. Reasoned Explanation
=====================================================

Tests Arjun's key insight: "First thought retrieval is perfect,
but reasoning for it fails."

Pipeline:
1. Load state streams from Experiment 1
2. Train Dual-Path Observer (fast path + slow path)
3. Compare fast vs. slow accuracy across all episodes
4. Run K-pass deliberation analysis (overthinking curve)
5. Measure confidence calibration
6. Determine whether consciousness (slow path) helps or hurts

Usage:
    python -m experiment3_first_thought.run_experiment --exp1-dir experiment1_separation/data
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path

from experiment1_separation.executor.trainer import DQNTrainer
from experiment1_separation.observer.trainer import StateStreamDataset
from experiment1_separation.shared.state_packet import packets_to_tensors
from .capture.dual_path_observer import DualPathObserver
from .comparison.accuracy_analysis import FirstThoughtAnalyzer


class DualPathTrainer:
    """Train the dual-path observer."""

    def __init__(self, obs_dim, hidden_dim, act_dim, n_hidden_layers=2,
                 embed_dim=128, lr=3e-4, device='cpu'):
        self.device = device
        self.model = DualPathObserver(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            n_hidden_layers=n_hidden_layers,
            embed_dim=embed_dim,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self, dataset, n_epochs=50, batch_size=16, print_every=5):
        """Train both paths jointly on the same prediction objective."""
        import torch.nn as nn

        print(f"Training dual-path observer on {len(dataset)} windows")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {total_params:,}")
        print("-" * 60)

        history = {'fast_loss': [], 'slow_loss': [], 'total_loss': []}

        for epoch in range(n_epochs):
            self.model.train()
            epoch_fast_loss = []
            epoch_slow_loss = []
            epoch_total_loss = []

            indices = np.random.permutation(len(dataset))

            for batch_start in range(0, len(indices), batch_size):
                batch_idx = indices[batch_start:batch_start + batch_size]
                windows = [dataset[i] for i in batch_idx]

                # Convert to tensors
                batch_tensors = []
                for window in windows:
                    t = packets_to_tensors(window, self.device)
                    if t is not None:
                        batch_tensors.append(t)
                if not batch_tensors:
                    continue

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

                # Input (0..T-1) and target (1..T)
                inp = {
                    'env_obs': batched['env_obs'][:, :-1],
                    'hidden_states': [h[:, :-1] for h in batched['hidden_states']],
                    'output_logits': batched['output_logits'][:, :-1],
                    'actions': batched['actions'][:, :-1],
                    'rewards': batched['rewards'][:, :-1],
                    'dones': batched['dones'][:, :-1],
                }
                target_actions = batched['actions'][:, 1:]

                # Forward: both paths
                fast_pred, slow_pred, _, _ = self.model(
                    inp['env_obs'], inp['hidden_states'], inp['output_logits'],
                    inp['actions'], inp['rewards'], inp['dones'],
                )

                fast_loss = nn.MSELoss()(fast_pred, target_actions)
                slow_loss = nn.MSELoss()(slow_pred, target_actions)
                total_loss = fast_loss + slow_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_fast_loss.append(fast_loss.item())
                epoch_slow_loss.append(slow_loss.item())
                epoch_total_loss.append(total_loss.item())

            avg_fast = np.mean(epoch_fast_loss)
            avg_slow = np.mean(epoch_slow_loss)
            avg_total = np.mean(epoch_total_loss)

            history['fast_loss'].append(avg_fast)
            history['slow_loss'].append(avg_slow)
            history['total_loss'].append(avg_total)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Fast Loss: {avg_fast:.6f} | "
                      f"Slow Loss: {avg_slow:.6f} | "
                      f"Total: {avg_total:.6f}")

        return history


def run(exp1_dir='experiment1_separation/data', device='cpu',
        output_dir='experiment3_first_thought/results'):
    exp1_path = Path(exp1_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 3: FIRST THOUGHT vs. REASONED EXPLANATION")
    print("=" * 70)
    print()

    # ----------------------------------------------------------------
    # Collect state streams (reuse Experiment 1 executor)
    # ----------------------------------------------------------------
    print("Collecting state streams from executor...")
    trainer = DQNTrainer(env_name='CartPole-v1', hidden_dim=64)
    episodes = trainer.collect_state_stream(
        n_episodes=100,
        model_path=str(exp1_path / 'executor_primary.pt'),
    )

    # Split into train/test
    train_episodes = episodes[:80]
    test_episodes = episodes[80:]

    # Get dimensions
    sample = train_episodes[0].packets[0]
    obs_dim = sample.env_observation.shape[0]
    act_dim = sample.executor_output_logits.shape[0]
    hidden_dim = sample.executor_hidden_states[0].shape[-1]
    n_hidden_layers = len(sample.executor_hidden_states)

    # ----------------------------------------------------------------
    # Train Dual-Path Observer
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training Dual-Path Observer")
    print("=" * 70)

    dataset = StateStreamDataset(train_episodes, window_size=32, stride=8)

    dp_trainer = DualPathTrainer(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        n_hidden_layers=n_hidden_layers,
        embed_dim=128,
        lr=3e-4,
        device=device,
    )

    history = dp_trainer.train(dataset, n_epochs=50, batch_size=16, print_every=5)

    # Save model
    torch.save({
        'model_state_dict': dp_trainer.model.state_dict(),
        'config': dp_trainer.model.config,
        'training_history': history,
    }, str(output_path / 'dual_path_observer.pt'))

    # ----------------------------------------------------------------
    # Run First Thought vs. Deliberation Analysis
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("First Thought vs. Deliberation Analysis")
    print("=" * 70)

    analyzer = FirstThoughtAnalyzer(dp_trainer.model, device)
    results = analyzer.evaluate_dataset(test_episodes, max_deliberation_k=5)
    analyzer.print_results(results, max_k=5)

    # Save results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, np.bool_):
            serializable[k] = bool(v)
        elif isinstance(v, list):
            serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer))
                                else x for x in v]
        else:
            serializable[k] = v

    with open(output_path / 'first_thought_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path / 'first_thought_results.json'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 3: First Thought vs. Reasoned Explanation'
    )
    parser.add_argument('--exp1-dir', default='experiment1_separation/data')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='experiment3_first_thought/results')
    args = parser.parse_args()

    run(exp1_dir=args.exp1_dir, device=args.device, output_dir=args.output)
