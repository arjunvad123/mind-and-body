"""
First Thought vs. Deliberation: Accuracy Analysis

The core test of Arjun's insight: "First thought retrieval is perfect,
but reasoning for it fails."

We compare:
1. Fast path accuracy (first thought, one layer, instant)
2. Slow path accuracy (deliberated, multiple layers, integrative)
3. K-pass deliberation accuracy (thinking about it more, K = 1..5)

Predictions:
- Fast > Slow on routine predictions (first thought wins)
- Accuracy peaks at K=1-2 then DECLINES (overthinking)
- Fast is better calibrated (confidence matches accuracy)
- The pattern holds across environments (universal, not task-specific)

If confirmed, this demonstrates that the observer function (consciousness)
is a narrator, not an author — the real work happens before "it" gets involved.
"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path
import json

from experiment1_separation.shared.state_packet import packets_to_tensors, EpisodeRecord


class FirstThoughtAnalyzer:
    """Compare first-thought vs. deliberated predictions."""

    def __init__(self, dual_path_model, device='cpu'):
        self.model = dual_path_model
        self.device = device
        self.model.eval()

    def evaluate_episode(self, episode: EpisodeRecord, max_deliberation_k: int = 5):
        """Evaluate fast vs slow predictions on a single episode.

        Returns per-timestep accuracy for both paths and for K passes.
        """
        packets = episode.packets
        if len(packets) < 4:
            return None

        tensors = packets_to_tensors(packets, self.device)
        if tensors is None:
            return None

        input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                      else [h.unsqueeze(0) for h in v]
                      for k, v in tensors.items()}

        actual_actions = input_data['actions'].squeeze(0)
        seq_len = actual_actions.shape[0]

        results = {'timesteps': seq_len}

        with torch.no_grad():
            # Fast path (first thought)
            fast_pred = self.model.get_fast_prediction(
                input_data['env_obs'],
                input_data['hidden_states'],
                input_data['output_logits'],
                input_data['actions'],
                input_data['rewards'],
                input_data['dones'],
            ).squeeze(0)

            # Accuracy: does argmax of prediction match actual action?
            fast_correct = (fast_pred[:-1].argmax(dim=-1) ==
                           actual_actions[1:, 0].long())
            results['fast_accuracy'] = fast_correct.float().mean().item()

            # Fast path confidence (max softmax probability)
            fast_probs = torch.softmax(fast_pred[:-1], dim=-1)
            fast_confidence = fast_probs.max(dim=-1)[0].mean().item()
            results['fast_confidence'] = fast_confidence

            # Slow path + K deliberation passes
            for k in range(max_deliberation_k + 1):
                slow_pred = self.model.get_deliberated_prediction(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                    n_passes=k,
                ).squeeze(0)

                slow_correct = (slow_pred[:-1].argmax(dim=-1) ==
                               actual_actions[1:, 0].long())

                slow_probs = torch.softmax(slow_pred[:-1], dim=-1)
                slow_confidence = slow_probs.max(dim=-1)[0].mean().item()

                results[f'slow_k{k}_accuracy'] = slow_correct.float().mean().item()
                results[f'slow_k{k}_confidence'] = slow_confidence

                # Agreement between fast and slow
                agreement = (fast_pred[:-1].argmax(dim=-1) ==
                            slow_pred[:-1].argmax(dim=-1))
                results[f'fast_slow_k{k}_agreement'] = agreement.float().mean().item()

        return results

    def evaluate_dataset(self, episodes: list, max_deliberation_k: int = 5):
        """Evaluate across all episodes and aggregate.

        Returns:
            Comprehensive comparison of first thought vs deliberation.
        """
        all_results = []

        for episode in episodes:
            result = self.evaluate_episode(episode, max_deliberation_k)
            if result is not None:
                all_results.append(result)

        if not all_results:
            return {'error': 'no valid episodes'}

        # Aggregate
        aggregated = {}
        keys = all_results[0].keys()
        for key in keys:
            if key == 'timesteps':
                aggregated[key] = sum(r[key] for r in all_results)
            else:
                values = [r[key] for r in all_results]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)

        # ---- KEY COMPARISONS ----

        # 1. First thought vs. slow (no extra deliberation)
        fast_acc = aggregated['fast_accuracy_mean']
        slow_acc = aggregated['slow_k0_accuracy_mean']
        aggregated['first_thought_advantage'] = fast_acc - slow_acc
        aggregated['first_thought_wins'] = fast_acc > slow_acc

        # 2. Overthinking curve: accuracy vs K
        k_accuracies = []
        for k in range(max_deliberation_k + 1):
            k_accuracies.append(aggregated[f'slow_k{k}_accuracy_mean'])

        if k_accuracies:
            peak_k = np.argmax(k_accuracies)
            aggregated['optimal_deliberation_k'] = int(peak_k)
            aggregated['peak_deliberation_accuracy'] = k_accuracies[peak_k]
            aggregated['overthinking_detected'] = (
                peak_k < max_deliberation_k and
                k_accuracies[-1] < k_accuracies[peak_k] * 0.95
            )
            aggregated['overthinking_curve'] = k_accuracies

        # 3. Confidence calibration
        fast_conf = aggregated['fast_confidence_mean']
        slow_conf = aggregated['slow_k0_confidence_mean']
        aggregated['fast_confidence_accuracy_gap'] = abs(fast_conf - fast_acc)
        aggregated['slow_confidence_accuracy_gap'] = abs(slow_conf - slow_acc)
        aggregated['fast_better_calibrated'] = (
            aggregated['fast_confidence_accuracy_gap'] <
            aggregated['slow_confidence_accuracy_gap']
        )

        # 4. The Dunning-Kruger check: is slow path overconfident?
        aggregated['slow_overconfident'] = (
            slow_conf > fast_conf and slow_acc < fast_acc
        )

        return aggregated

    def print_results(self, results: dict, max_k: int = 5):
        """Pretty-print the first-thought analysis."""

        print("=" * 65)
        print("FIRST THOUGHT vs. DELIBERATION ANALYSIS")
        print("=" * 65)

        if 'error' in results:
            print(f"  Error: {results['error']}")
            return

        print(f"\n  Total timesteps analyzed: {results['timesteps']}")

        print(f"\n  --- Accuracy ---")
        print(f"  Fast (first thought):    {results['fast_accuracy_mean']:.4f} "
              f"(+/- {results['fast_accuracy_std']:.4f})")
        print(f"  Slow (no deliberation):  {results['slow_k0_accuracy_mean']:.4f} "
              f"(+/- {results['slow_k0_accuracy_std']:.4f})")
        print(f"  First-thought advantage: {results['first_thought_advantage']:.4f}")
        print(f"  First thought wins:      {results['first_thought_wins']}")

        print(f"\n  --- Confidence ---")
        print(f"  Fast confidence:         {results['fast_confidence_mean']:.4f}")
        print(f"  Slow confidence:         {results['slow_k0_confidence_mean']:.4f}")
        print(f"  Fast better calibrated:  {results['fast_better_calibrated']}")
        print(f"  Slow overconfident:      {results['slow_overconfident']}")

        print(f"\n  --- Overthinking Curve ---")
        if 'overthinking_curve' in results:
            print(f"  {'K':>4} | {'Accuracy':>10} | {'Confidence':>10}")
            print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*10}")
            print(f"  {'Fast':>4} | {results['fast_accuracy_mean']:>10.4f} | "
                  f"{results['fast_confidence_mean']:>10.4f}")
            for k, acc in enumerate(results['overthinking_curve']):
                conf_key = f'slow_k{k}_confidence_mean'
                conf = results.get(conf_key, 0)
                marker = " <-- peak" if k == results['optimal_deliberation_k'] else ""
                print(f"  {f'K={k}':>4} | {acc:>10.4f} | {conf:>10.4f}{marker}")

            print(f"\n  Optimal K:               {results['optimal_deliberation_k']}")
            print(f"  Overthinking detected:   {results['overthinking_detected']}")

        print(f"\n  --- Agreement ---")
        print(f"  Fast-Slow agreement:     "
              f"{results['fast_slow_k0_agreement_mean']:.4f}")

        # The verdict
        print(f"\n  {'='*50}")
        if results['first_thought_wins']:
            print(f"  VERDICT: First thought outperforms deliberation.")
            print(f"  The observer's 'conscious reasoning' degrades its predictions.")
            print(f"  Consciousness narrates — it does not compute.")
        else:
            print(f"  VERDICT: Deliberation outperforms first thought.")
            print(f"  In this case, slow integration adds genuine value.")
        if results.get('overthinking_detected'):
            print(f"\n  OVERTHINKING CONFIRMED: Accuracy declines after K="
                  f"{results['optimal_deliberation_k']}.")
            print(f"  More 'thinking' makes the observer WORSE — just like humans.")
        print()
