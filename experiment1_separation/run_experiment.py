"""
Experiment 1: Executor-Observer Separation
==========================================

Full pipeline:
1. Train the executor (DQN on CartPole)
2. Collect state streams (the executor acts, we record everything)
3. Train the observer (predict executor's behavior from state stream)
4. Train a SECOND executor (for the "other" condition in probes)
5. Collect state streams from the second executor
6. Run consciousness indicator probes

Usage:
    python -m experiment1_separation.run_experiment

This is the foundational experiment of the Observer Hypothesis.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from .executor.trainer import DQNTrainer
from .observer.trainer import ObserverTrainer, StateStreamDataset
from .observer.model import ObserverTransformer
from .analysis.consciousness_probes import run_all_probes


def run(env_name='CartPole-v1', device='cpu', output_dir='experiment1_separation/data'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1: EXECUTOR-OBSERVER SEPARATION")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print()

    # ----------------------------------------------------------------
    # PHASE 1: Train the primary executor
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: Training Primary Executor")
    print("=" * 70)

    trainer = DQNTrainer(
        env_name=env_name,
        hidden_dim=64,
        lr=1e-3,
        epsilon_decay=3000,
        target_update_freq=100,
    )

    history = trainer.train(
        n_episodes=500,
        print_every=50,
        target_reward=475.0 if 'CartPole' in env_name else None,
        save_path=str(output_path / 'executor_primary.pt'),
    )

    # ----------------------------------------------------------------
    # PHASE 2: Collect state streams from primary executor
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: Collecting State Streams (Primary Executor)")
    print("=" * 70)

    primary_episodes = trainer.collect_state_stream(
        n_episodes=100,
        model_path=str(output_path / 'executor_primary.pt'),
    )

    # ----------------------------------------------------------------
    # PHASE 3: Train the observer
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: Training the Observer")
    print("=" * 70)

    # Determine dimensions from collected data
    sample_packet = primary_episodes[0].packets[0]
    obs_dim = sample_packet.env_observation.shape[0]
    act_dim = sample_packet.executor_output_logits.shape[0]
    hidden_dim = sample_packet.executor_hidden_states[0].shape[-1]
    n_hidden_layers = len(sample_packet.executor_hidden_states)

    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  N hidden layers: {n_hidden_layers}")

    # Create dataset
    dataset = StateStreamDataset(
        primary_episodes,
        window_size=32,
        stride=8,
    )

    # Train observer
    observer_trainer = ObserverTrainer(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        act_dim=act_dim,
        n_hidden_layers=n_hidden_layers,
        embed_dim=128,
        n_heads=4,
        n_transformer_layers=4,
        lr=3e-4,
        device=device,
    )

    observer_history = observer_trainer.train(
        dataset=dataset,
        n_epochs=50,
        batch_size=16,
        save_path=str(output_path / 'observer.pt'),
        print_every=5,
    )

    # ----------------------------------------------------------------
    # PHASE 4: Train a second executor (for "other" condition)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: Training Second Executor (Different Policy)")
    print("=" * 70)

    trainer2 = DQNTrainer(
        env_name=env_name,
        hidden_dim=64,
        lr=1e-3,
        epsilon_decay=5000,  # Different hyperparams → different policy
        target_update_freq=200,
    )

    trainer2.train(
        n_episodes=500,
        print_every=50,
        target_reward=475.0 if 'CartPole' in env_name else None,
        save_path=str(output_path / 'executor_secondary.pt'),
    )

    other_episodes = trainer2.collect_state_stream(
        n_episodes=50,
        model_path=str(output_path / 'executor_secondary.pt'),
    )

    # ----------------------------------------------------------------
    # PHASE 5: Run consciousness indicator probes
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: Consciousness Indicator Probes")
    print("=" * 70)

    # Load trained observer
    observer_model = observer_trainer.model
    observer_model.eval()

    results = run_all_probes(
        observer_model=observer_model,
        own_episodes=primary_episodes,
        other_episodes=other_episodes,
        device=device,
        save_dir=str(output_path / 'results'),
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 1: Executor-Observer Separation')
    parser.add_argument('--env', default='CartPole-v1', help='Gymnasium environment')
    parser.add_argument('--device', default='cpu', help='torch device')
    parser.add_argument('--output', default='experiment1_separation/data',
                        help='output directory')
    args = parser.parse_args()

    run(env_name=args.env, device=args.device, output_dir=args.output)
