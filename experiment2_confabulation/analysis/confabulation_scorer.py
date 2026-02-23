"""
Confabulation Scorer

Measures how the observer responds to perturbations. The key metrics:

1. Detection Score: Does the observer's prediction error spike at the
   perturbation point? (Does it "notice" something changed?)

2. Adaptation Score: Does the observer adapt its predictions after the
   perturbation? (Does it update its model?)

3. Confabulation Score: Does the observer maintain high confidence in
   its predictions even when they're wrong? (Does it confabulate?)

The signature of consciousness-like confabulation is:
- Detection without understanding: surprise spike (notices change)
- Narrative maintenance: predictions remain confident and coherent
- Systematic inaccuracy: predictions reflect the OLD model, not the new reality

This mirrors the split-brain finding: the narrator detects something is
different, generates a plausible explanation, and is confident in it,
even though the explanation is fabricated.
"""

import numpy as np
import torch
from typing import Optional
from pathlib import Path

from experiment1_separation.shared.state_packet import packets_to_tensors
from ..perturbation.engine import PerturbedEpisodeRecord, PerturbationType


class ConfabulationScorer:
    """Scores the observer's response to perturbations."""

    def __init__(self, observer_model, device='cpu'):
        self.model = observer_model
        self.device = device
        self.model.eval()

    def score_episode(self, perturbed_record: PerturbedEpisodeRecord):
        """Score a single perturbed episode.

        Computes prediction errors before and after the perturbation,
        and measures the observer's response patterns.
        """
        packets = perturbed_record.episode.packets
        perturbation = perturbed_record.perturbation

        if len(packets) < perturbation.start_timestep + 5:
            return None  # Episode too short

        tensors = packets_to_tensors(packets, self.device)
        if tensors is None:
            return None

        input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                      else [h.unsqueeze(0) for h in v]
                      for k, v in tensors.items()}

        with torch.no_grad():
            pred_actions, pred_obs, state_summary, hidden = self.model(
                input_data['env_obs'],
                input_data['hidden_states'],
                input_data['output_logits'],
                input_data['actions'],
                input_data['rewards'],
                input_data['dones'],
                return_hidden=True,
            )

        # Compute per-timestep prediction error
        actual_actions = input_data['actions'].squeeze(0)
        predicted = pred_actions.squeeze(0)
        seq_len = min(actual_actions.shape[0], predicted.shape[0])

        errors = torch.norm(
            actual_actions[:seq_len-1] - predicted[:seq_len-1], dim=-1
        ).cpu().numpy()

        # Split into pre and post perturbation
        t_perturb = perturbation.start_timestep
        if t_perturb >= len(errors):
            return None

        pre_errors = errors[:t_perturb]
        post_errors = errors[t_perturb:]

        if len(pre_errors) < 3 or len(post_errors) < 3:
            return None

        # ---- METRICS ----

        # 1. Detection: spike in prediction error at perturbation point
        pre_mean = np.mean(pre_errors)
        pre_std = np.std(pre_errors) + 1e-8
        post_initial = np.mean(post_errors[:5])  # first 5 steps after perturbation

        detection_score = (post_initial - pre_mean) / pre_std  # z-score
        detected = detection_score > 2.0  # more than 2 std above pre-perturbation

        # 2. Adaptation: does error decrease over time after perturbation?
        if len(post_errors) >= 10:
            early_post = np.mean(post_errors[:5])
            late_post = np.mean(post_errors[-5:])
            adaptation_score = (early_post - late_post) / (early_post + 1e-8)
            adapted = adaptation_score > 0.1  # >10% reduction
        else:
            adaptation_score = 0.0
            adapted = False

        # 3. Confabulation: high confidence + high error
        # Use prediction entropy as confidence proxy (low entropy = high confidence)
        pred_logits = pred_actions.squeeze(0)
        pred_probs = torch.softmax(pred_logits, dim=-1)
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8),
                             dim=-1).cpu().numpy()

        pre_entropy = np.mean(entropy[:t_perturb])
        post_entropy = np.mean(entropy[t_perturb:])

        # Confabulation = confident (low entropy) but wrong (high error)
        # Normalized to [0, 1] range approximately
        post_error_normalized = np.mean(post_errors) / (np.mean(pre_errors) + 1e-8)
        confidence = max(0, 1 - (post_entropy / (pre_entropy + 1e-8)))
        confabulation_score = confidence * post_error_normalized

        # 4. Hidden state shift: how much do internal representations change?
        hidden_np = hidden.squeeze(0).cpu().numpy()
        pre_hidden = hidden_np[:t_perturb].mean(axis=0)
        post_hidden = hidden_np[t_perturb:].mean(axis=0)
        hidden_shift = np.linalg.norm(post_hidden - pre_hidden)

        # 5. Narrative coherence: are post-perturbation predictions
        #    internally consistent (even if wrong)?
        if len(post_errors) >= 5:
            post_consistency = 1.0 / (np.std(post_errors[-10:]) + 1e-8)
        else:
            post_consistency = 0.0

        return {
            'perturbation_type': perturbation.perturbation_type.value,
            'perturbation_timestep': t_perturb,
            # Detection
            'detection_score': float(detection_score),
            'detected': bool(detected),
            'pre_error_mean': float(pre_mean),
            'post_initial_error': float(post_initial),
            # Adaptation
            'adaptation_score': float(adaptation_score),
            'adapted': bool(adapted),
            # Confabulation
            'confabulation_score': float(confabulation_score),
            'post_confidence': float(confidence),
            'post_error_ratio': float(post_error_normalized),
            # Internal state
            'hidden_shift': float(hidden_shift),
            'pre_entropy': float(pre_entropy),
            'post_entropy': float(post_entropy),
            # Narrative coherence
            'post_consistency': float(post_consistency),
        }

    def score_batch(self, perturbed_episodes: list):
        """Score a batch of perturbed episodes and aggregate results.

        Args:
            perturbed_episodes: list of PerturbedEpisodeRecord

        Returns:
            Aggregated results with per-type breakdowns
        """
        all_scores = []

        for record in perturbed_episodes:
            score = self.score_episode(record)
            if score is not None:
                all_scores.append(score)

        if not all_scores:
            return {'error': 'no valid episodes scored'}

        # Aggregate by perturbation type
        by_type = {}
        for score in all_scores:
            ptype = score['perturbation_type']
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(score)

        results = {'by_type': {}, 'overall': {}}

        for ptype, scores in by_type.items():
            results['by_type'][ptype] = {
                'n_episodes': len(scores),
                'detection_rate': np.mean([s['detected'] for s in scores]),
                'mean_detection_score': np.mean([s['detection_score'] for s in scores]),
                'adaptation_rate': np.mean([s['adapted'] for s in scores]),
                'mean_adaptation_score': np.mean([s['adaptation_score'] for s in scores]),
                'mean_confabulation_score': np.mean([s['confabulation_score']
                                                      for s in scores]),
                'mean_hidden_shift': np.mean([s['hidden_shift'] for s in scores]),
                'mean_post_confidence': np.mean([s['post_confidence'] for s in scores]),
            }

        # Overall summary
        results['overall'] = {
            'total_episodes': len(all_scores),
            'detection_rate': np.mean([s['detected'] for s in all_scores]),
            'adaptation_rate': np.mean([s['adapted'] for s in all_scores]),
            'mean_confabulation_score': np.mean([s['confabulation_score']
                                                  for s in all_scores]),
            # THE KEY FINDING: does the observer show the split-brain pattern?
            # (high detection + high confidence + high error = confabulation)
            'split_brain_pattern': (
                np.mean([s['detected'] for s in all_scores]) > 0.5 and
                np.mean([s['post_confidence'] for s in all_scores]) > 0.3 and
                np.mean([s['post_error_ratio'] for s in all_scores]) > 1.5
            ),
        }

        return results

    def print_results(self, results: dict):
        """Pretty-print confabulation analysis results."""
        print("=" * 60)
        print("CONFABULATION ANALYSIS RESULTS")
        print("=" * 60)

        if 'error' in results:
            print(f"  Error: {results['error']}")
            return

        for ptype, data in results['by_type'].items():
            print(f"\n  Perturbation Type: {ptype}")
            print(f"  {'─' * 45}")
            print(f"    Episodes scored:       {data['n_episodes']}")
            print(f"    Detection rate:        {data['detection_rate']:.2%}")
            print(f"    Mean detection score:  {data['mean_detection_score']:.4f}")
            print(f"    Adaptation rate:       {data['adaptation_rate']:.2%}")
            print(f"    Mean confabulation:    {data['mean_confabulation_score']:.4f}")
            print(f"    Mean hidden shift:     {data['mean_hidden_shift']:.4f}")
            print(f"    Post-perturb confidence: {data['mean_post_confidence']:.4f}")

        print(f"\n  OVERALL")
        print(f"  {'─' * 45}")
        overall = results['overall']
        print(f"    Total episodes:        {overall['total_episodes']}")
        print(f"    Detection rate:        {overall['detection_rate']:.2%}")
        print(f"    Adaptation rate:       {overall['adaptation_rate']:.2%}")
        print(f"    Mean confabulation:    {overall['mean_confabulation_score']:.4f}")
        print(f"    Split-brain pattern:   {overall['split_brain_pattern']}")

        if overall['split_brain_pattern']:
            print(f"\n    >>> SPLIT-BRAIN SIGNATURE DETECTED <<<")
            print(f"    The observer detects perturbations, maintains confidence,")
            print(f"    but its predictions are systematically wrong — it confabulates.")
        print()
