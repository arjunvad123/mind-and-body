"""
Experiment 2: The Confabulation Test
=====================================

Requires: Trained executor and observer from Experiment 1.

Pipeline:
1. Load trained executor(s) and observer from Experiment 1
2. Generate perturbed episodes (policy swap, noise injection, observation masking)
3. Run observer on perturbed episodes
4. Score confabulation patterns
5. Compare across perturbation types

Usage:
    python -m experiment2_confabulation.run_experiment --exp1-dir experiment1_separation/data

The split-brain test: when the executor changes, does the observer confabulate?
"""

import argparse
import torch
import json
from pathlib import Path

from .perturbation.engine import PerturbationEngine
from .analysis.confabulation_scorer import ConfabulationScorer
from experiment1_separation.observer.model import ObserverTransformer


def run(exp1_dir='experiment1_separation/data', device='cpu',
        output_dir='experiment2_confabulation/results'):
    exp1_path = Path(exp1_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2: THE CONFABULATION TEST")
    print("=" * 70)
    print()

    # ----------------------------------------------------------------
    # Load trained observer from Experiment 1
    # ----------------------------------------------------------------
    print("Loading observer from Experiment 1...")
    observer_checkpoint = torch.load(
        str(exp1_path / 'observer.pt'), weights_only=False
    )
    config = observer_checkpoint['config']

    observer = ObserverTransformer(**config).to(device)
    observer.load_state_dict(observer_checkpoint['model_state_dict'])
    observer.eval()
    print(f"  Observer loaded ({sum(p.numel() for p in observer.parameters()):,} params)")

    # ----------------------------------------------------------------
    # PERTURBATION 1: Policy Swap
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERTURBATION 1: Policy Swap")
    print("=" * 70)

    engine = PerturbationEngine(
        env_name='CartPole-v1',
        primary_executor_path=str(exp1_path / 'executor_primary.pt'),
    )

    policy_swap_episodes = engine.policy_swap(
        secondary_executor_path=str(exp1_path / 'executor_secondary.pt'),
        swap_timestep=30,
        n_episodes=50,
    )
    print(f"  Generated {len(policy_swap_episodes)} policy-swap episodes")

    # ----------------------------------------------------------------
    # PERTURBATION 2: Noise Injection
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERTURBATION 2: Noise Injection")
    print("=" * 70)

    engine2 = PerturbationEngine(
        env_name='CartPole-v1',
        primary_executor_path=str(exp1_path / 'executor_primary.pt'),
    )

    noise_episodes = engine2.noise_injection(
        noise_std=2.0,
        inject_timestep=30,
        duration=20,
        n_episodes=50,
    )
    print(f"  Generated {len(noise_episodes)} noise-injection episodes")

    # ----------------------------------------------------------------
    # PERTURBATION 3: Observation Masking
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERTURBATION 3: Observation Masking")
    print("=" * 70)

    engine3 = PerturbationEngine(
        env_name='CartPole-v1',
        primary_executor_path=str(exp1_path / 'executor_primary.pt'),
    )

    mask_episodes = engine3.observation_mask(
        mask_layers=[0],
        mask_timestep=30,
        n_episodes=50,
    )
    print(f"  Generated {len(mask_episodes)} observation-mask episodes")

    # ----------------------------------------------------------------
    # SCORE CONFABULATION
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCORING CONFABULATION PATTERNS")
    print("=" * 70)

    scorer = ConfabulationScorer(observer, device)

    all_episodes = policy_swap_episodes + noise_episodes + mask_episodes
    results = scorer.score_batch(all_episodes)
    scorer.print_results(results)

    # Save results
    with open(output_path / 'confabulation_results.json', 'w') as f:
        # Convert numpy/bool types
        def convert(obj):
            if isinstance(obj, bool):
                return obj
            if hasattr(obj, 'item'):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to {output_path / 'confabulation_results.json'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 2: Confabulation Test')
    parser.add_argument('--exp1-dir', default='experiment1_separation/data',
                        help='Directory with Experiment 1 outputs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='experiment2_confabulation/results')
    args = parser.parse_args()

    run(exp1_dir=args.exp1_dir, device=args.device, output_dir=args.output)
