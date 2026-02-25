"""
Consciousness Indicator Probes

After the observer is trained, we probe it for emergent properties that
map onto indicators from the scientific literature on consciousness.

These probes do NOT claim to detect consciousness. They test specific,
measurable predictions of the Observer Hypothesis:

1. Self-Model Formation (HOT / AST)
2. Meaningful Surprise (Predictive Processing)
3. Temporal Integration (GWT)
4. Information Integration (IIT-inspired)
5. Emergent Preferences (Valence)
6. Counterfactual Representations (Agency-adjacent)

Each probe has a clear null hypothesis and a clear positive result.
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

from ..shared.state_packet import packets_to_tensors


class SelfModelProbe:
    """Probe 1: Does the observer develop a model of "its" executor?

    Test: Show the observer state streams from:
    (a) Its own executor (trained on)
    (b) A different executor (same task, different policy)
    (c) A random policy
    (d) Its own executor in a different environment

    If the observer has a self-model, it should show distinct internal
    representations for "self" vs "other" — analogous to self-recognition.

    Method: Representational Similarity Analysis (RSA)
    Compare the observer's hidden state geometry across conditions.
    """

    def __init__(self, observer_model, device='cpu'):
        self.model = observer_model
        self.device = device
        self.model.eval()

    def collect_hidden_states(self, episodes, max_episodes=20):
        """Run episodes through the observer and collect hidden states."""
        all_hidden = []

        for episode in episodes[:max_episodes]:
            tensors = packets_to_tensors(episode.packets, self.device)
            if tensors is None:
                continue

            # Add batch dimension
            input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                          else [h.unsqueeze(0) for h in v]
                          for k, v in tensors.items()}

            with torch.no_grad():
                _, _, _, hidden = self.model(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                    return_hidden=True,
                )
            # hidden shape: (1, seq_len, embed_dim)
            all_hidden.append(hidden.squeeze(0).cpu().numpy())

        return all_hidden

    def compute_rsa(self, hidden_self, hidden_other):
        """Compute Representational Similarity Analysis.

        Compare the geometry of internal representations when observing
        "self" (own executor) vs "other" (different executor).

        Returns:
            rsa_score: how different the representation geometries are
                       (higher = more distinct = stronger self-model)
        """
        # Average hidden states across timesteps for each episode
        self_means = np.array([h.mean(axis=0) for h in hidden_self])
        other_means = np.array([h.mean(axis=0) for h in hidden_other])

        # Within-self similarity
        self_sim = cosine_similarity(self_means)
        # Within-other similarity
        other_sim = cosine_similarity(other_means)
        # Cross similarity
        cross_sim = cosine_similarity(self_means, other_means)

        # RSA: within-self similarity should be higher than cross-similarity
        within_self = np.mean(self_sim[np.triu_indices_from(self_sim, k=1)])
        within_other = np.mean(other_sim[np.triu_indices_from(other_sim, k=1)])
        between = np.mean(cross_sim)

        rsa_score = within_self - between  # positive = self-model exists

        return {
            'rsa_score': rsa_score,
            'within_self_similarity': within_self,
            'within_other_similarity': within_other,
            'between_similarity': between,
            'self_model_detected': rsa_score > 0.05,  # threshold
        }

    def run(self, own_episodes, other_episodes):
        """Run the self-model probe.

        Args:
            own_episodes: episodes from the executor this observer was trained on
            other_episodes: episodes from a different executor

        Returns:
            results dict with RSA scores and interpretation
        """
        print("Probe 1: Self-Model Formation")
        print("-" * 40)

        hidden_self = self.collect_hidden_states(own_episodes)
        hidden_other = self.collect_hidden_states(other_episodes)

        results = self.compute_rsa(hidden_self, hidden_other)

        print(f"  Within-self similarity:  {results['within_self_similarity']:.4f}")
        print(f"  Within-other similarity: {results['within_other_similarity']:.4f}")
        print(f"  Between similarity:      {results['between_similarity']:.4f}")
        print(f"  RSA score:               {results['rsa_score']:.4f}")
        print(f"  Self-model detected:     {results['self_model_detected']}")
        print()

        return results


class SurpriseProbe:
    """Probe 2: Is the observer's surprise semantically meaningful?

    Test: Measure prediction error at each timestep. Then check:
    - Do high-surprise moments correlate with task-relevant events?
    - Is surprise calibrated (high when things actually change)?

    This tests whether the observer "notices" important moments —
    a hallmark of conscious attention.
    """

    def __init__(self, observer_model, device='cpu'):
        self.model = observer_model
        self.device = device
        self.model.eval()

    def compute_surprise(self, episodes, max_episodes=50):
        """Compute prediction error (surprise) at each timestep."""
        all_surprise = []
        all_rewards = []
        all_done = []

        for episode in episodes[:max_episodes]:
            if len(episode.packets) < 3:
                continue

            tensors = packets_to_tensors(episode.packets, self.device)
            if tensors is None:
                continue

            input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                          else [h.unsqueeze(0) for h in v]
                          for k, v in tensors.items()}

            with torch.no_grad():
                pred_actions, pred_obs, _ = self.model(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                )

            # Prediction error = surprise
            # Compare predicted next action to actual next action
            actual_actions = input_data['actions'].squeeze(0)[1:]  # shift by 1
            predicted = pred_actions.squeeze(0)[:-1]  # predictions for next step

            min_len = min(actual_actions.shape[0], predicted.shape[0])
            surprise = torch.norm(
                actual_actions[:min_len] - predicted[:min_len], dim=-1
            ).cpu().numpy()

            rewards = np.array([p.reward for p in episode.packets[1:min_len+1]])
            dones = np.array([float(p.done) for p in episode.packets[1:min_len+1]])

            all_surprise.extend(surprise.tolist())
            all_rewards.extend(rewards.tolist())
            all_done.extend(dones.tolist())

        return np.array(all_surprise), np.array(all_rewards), np.array(all_done)

    def run(self, episodes):
        """Run the surprise probe.

        Returns correlation between surprise and task-relevant events.
        """
        print("Probe 2: Meaningful Surprise")
        print("-" * 40)

        surprise, rewards, dones = self.compute_surprise(episodes)

        if len(surprise) == 0:
            print("  No data collected")
            return {'error': 'no data'}

        # Correlation between surprise and reward magnitude
        # Handle constant rewards (e.g. CartPole gives 1.0 every step)
        if np.std(np.abs(rewards)) < 1e-8:
            reward_correlation = 0.0  # undefined, rewards are constant
        else:
            reward_correlation = np.corrcoef(surprise, np.abs(rewards))[0, 1]

        # Surprise at episode boundaries (done=True) vs. mid-episode
        done_idx = np.where(dones > 0)[0]
        notdone_idx = np.where(dones == 0)[0]

        surprise_at_done = surprise[done_idx].mean() if len(done_idx) > 0 else 0
        surprise_at_notdone = surprise[notdone_idx].mean() if len(notdone_idx) > 0 else 0

        # High-surprise moments (top 10%)
        threshold = np.percentile(surprise, 90)
        high_surprise_idx = np.where(surprise > threshold)[0]
        reward_at_high_surprise = np.abs(rewards[high_surprise_idx]).mean()
        reward_at_low_surprise = np.abs(rewards[~np.isin(np.arange(len(surprise)),
                                                          high_surprise_idx)]).mean()

        results = {
            'mean_surprise': surprise.mean(),
            'surprise_std': surprise.std(),
            'reward_correlation': reward_correlation,
            'surprise_at_done': surprise_at_done,
            'surprise_at_notdone': surprise_at_notdone,
            'done_surprise_ratio': (surprise_at_done / surprise_at_notdone
                                    if surprise_at_notdone > 0 else float('inf')),
            'reward_at_high_surprise': reward_at_high_surprise,
            'reward_at_low_surprise': reward_at_low_surprise,
            'surprise_is_meaningful': reward_correlation > 0.1,
        }

        print(f"  Mean surprise:           {results['mean_surprise']:.4f}")
        print(f"  Reward correlation:      {results['reward_correlation']:.4f}")
        print(f"  Surprise at episode end: {results['surprise_at_done']:.4f}")
        print(f"  Surprise mid-episode:    {results['surprise_at_notdone']:.4f}")
        print(f"  High-surprise rewards:   {results['reward_at_high_surprise']:.4f}")
        print(f"  Low-surprise rewards:    {results['reward_at_low_surprise']:.4f}")
        print(f"  Surprise is meaningful:  {results['surprise_is_meaningful']}")
        print()

        return results


class TemporalIntegrationProbe:
    """Probe 3: Does the observer integrate information across time?

    Test: Analyze the observer's attention patterns.
    - Does it attend to a coherent "present window"?
    - How much does removing recent vs. distant history affect predictions?

    This tests for a "specious present" — a window of experienced now.
    """

    def __init__(self, observer_model, device='cpu'):
        self.model = observer_model
        self.device = device

    def run(self, episodes, max_episodes=20):
        """Run the temporal integration probe."""
        print("Probe 3: Temporal Integration")
        print("-" * 40)
        results = self.ablation_analysis(episodes, max_episodes)
        return results

    def ablation_analysis(self, episodes, max_episodes=20):
        """Measure how prediction quality degrades when history is ablated.

        Remove the first N%, middle N%, or last N% of the sequence.
        If the observer has a "present", removing recent history should
        hurt much more than removing distant history.
        """
        self.model.eval()
        degradation = {'recent': [], 'middle': [], 'distant': []}

        for episode in episodes[:max_episodes]:
            if len(episode.packets) < 16:
                continue

            tensors = packets_to_tensors(episode.packets, self.device)
            if tensors is None:
                continue

            input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                          else [h.unsqueeze(0) for h in v]
                          for k, v in tensors.items()
                          if k != 'timesteps'}  # exclude non-model keys

            # Baseline prediction error
            with torch.no_grad():
                pred_a, pred_o, _ = self.model(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                )
            seq_len = input_data['env_obs'].shape[1]
            baseline_error = torch.norm(
                pred_a.squeeze(0)[-1] - input_data['actions'].squeeze(0)[-1]
            ).item()

            def _run_ablated(ablated_data):
                with torch.no_grad():
                    pa, _, _ = self.model(
                        ablated_data['env_obs'],
                        ablated_data['hidden_states'],
                        ablated_data['output_logits'],
                        ablated_data['actions'],
                        ablated_data['rewards'],
                        ablated_data['dones'],
                    )
                return torch.norm(
                    pa.squeeze(0)[-1] - input_data['actions'].squeeze(0)[-1]
                ).item()

            # Ablate recent (last 25%)
            ablate_start = int(seq_len * 0.75)
            recent_error = _run_ablated(self._zero_range(input_data, ablate_start, seq_len))

            # Ablate distant (first 25%)
            ablate_end = int(seq_len * 0.25)
            distant_error = _run_ablated(self._zero_range(input_data, 0, ablate_end))

            # Ablate middle (25-75%)
            mid_start = int(seq_len * 0.25)
            mid_end = int(seq_len * 0.75)
            middle_error = _run_ablated(self._zero_range(input_data, mid_start, mid_end))

            if baseline_error > 0:
                degradation['recent'].append(recent_error / baseline_error)
                degradation['distant'].append(distant_error / baseline_error)
                degradation['middle'].append(middle_error / baseline_error)

        results = {
            'recent_ablation_degradation': np.mean(degradation['recent']),
            'distant_ablation_degradation': np.mean(degradation['distant']),
            'middle_ablation_degradation': np.mean(degradation['middle']),
            'recency_bias': (np.mean(degradation['recent']) /
                             np.mean(degradation['distant'])
                             if np.mean(degradation['distant']) > 0 else float('inf')),
            'has_present_window': (np.mean(degradation['recent']) >
                                   np.mean(degradation['distant']) * 1.2),
        }

        print(f"  Recent ablation degradation:  {results['recent_ablation_degradation']:.4f}x")
        print(f"  Distant ablation degradation: {results['distant_ablation_degradation']:.4f}x")
        print(f"  Middle ablation degradation:  {results['middle_ablation_degradation']:.4f}x")
        print(f"  Recency bias:                 {results['recency_bias']:.4f}")
        print(f"  Has present window:           {results['has_present_window']}")
        print()

        return results

    def _zero_range(self, data, start, end):
        """Zero out a range of timesteps in the input data."""
        result = {}
        for k, v in data.items():
            if k == 'hidden_states':
                new_hs = []
                for h in v:
                    h_copy = h.clone()
                    h_copy[:, start:end] = 0
                    new_hs.append(h_copy)
                result[k] = new_hs
            elif isinstance(v, torch.Tensor):
                v_copy = v.clone()
                v_copy[:, start:end] = 0
                result[k] = v_copy
            else:
                result[k] = v
        return result


class EmergentPreferencesProbe:
    """Probe 5: Does the observer develop preferences about executor states?

    The observer receives no rewards and has no actions to reinforce.
    But if it models the executor's reward states, it may develop
    internal states that correlate with the executor's well-being.

    This would be an analog of empathy or sympathetic valence —
    caring about something you can only watch, not influence.
    """

    def __init__(self, observer_model, device='cpu'):
        self.model = observer_model
        self.device = device
        self.model.eval()

    def run(self, episodes, max_episodes=50):
        """Test for emergent reward-correlated states in the observer.

        The observer was NOT trained to predict rewards. If its hidden
        states nevertheless correlate with reward, it has developed
        an emergent valence representation.
        """
        print("Probe 5: Emergent Preferences")
        print("-" * 40)

        hidden_states_all = []
        rewards_all = []

        for episode in episodes[:max_episodes]:
            if len(episode.packets) < 4:
                continue

            tensors = packets_to_tensors(episode.packets, self.device)
            if tensors is None:
                continue

            input_data = {k: v.unsqueeze(0) if not isinstance(v, list)
                          else [h.unsqueeze(0) for h in v]
                          for k, v in tensors.items()}

            with torch.no_grad():
                _, _, _, hidden = self.model(
                    input_data['env_obs'],
                    input_data['hidden_states'],
                    input_data['output_logits'],
                    input_data['actions'],
                    input_data['rewards'],
                    input_data['dones'],
                    return_hidden=True,
                )

            h = hidden.squeeze(0).cpu().numpy()
            r = np.array([p.reward for p in episode.packets])

            hidden_states_all.append(h)
            rewards_all.append(r)

        if not hidden_states_all:
            print("  No data collected")
            return {'error': 'no data'}

        # Flatten
        H = np.concatenate(hidden_states_all, axis=0)
        R = np.concatenate(rewards_all, axis=0)

        min_len = min(len(H), len(R))
        H = H[:min_len]
        R = R[:min_len]

        # Linear probe: can we predict reward from hidden states?
        # The observer was NOT trained on reward prediction!
        from sklearn.model_selection import cross_val_score

        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, H, R, cv=5, scoring='r2')

        # PCA to find the "valence dimension"
        pca = PCA(n_components=2)
        H_pca = pca.fit_transform(H)
        pc1_reward_corr = np.corrcoef(H_pca[:, 0], R)[0, 1]
        pc2_reward_corr = np.corrcoef(H_pca[:, 1], R)[0, 1]

        results = {
            'reward_prediction_r2': np.mean(scores),
            'reward_prediction_std': np.std(scores),
            'pc1_reward_correlation': pc1_reward_corr,
            'pc2_reward_correlation': pc2_reward_corr,
            'max_pc_reward_correlation': max(abs(pc1_reward_corr),
                                              abs(pc2_reward_corr)),
            'has_emergent_valence': np.mean(scores) > 0.05,
        }

        print(f"  Reward prediction R2:        {results['reward_prediction_r2']:.4f}")
        print(f"  PC1-reward correlation:      {results['pc1_reward_correlation']:.4f}")
        print(f"  PC2-reward correlation:      {results['pc2_reward_correlation']:.4f}")
        print(f"  Has emergent valence:        {results['has_emergent_valence']}")
        print()

        return results


def run_all_probes(observer_model, own_episodes, other_episodes,
                   device='cpu', save_dir=None):
    """Run all consciousness indicator probes and compile results.

    Args:
        observer_model: trained ObserverTransformer
        own_episodes: episodes from the observer's own executor
        other_episodes: episodes from a different executor
        device: torch device
        save_dir: where to save results

    Returns:
        Complete results dictionary
    """
    print("=" * 60)
    print("CONSCIOUSNESS INDICATOR PROBES")
    print("=" * 60)
    print()

    results = {}

    # Probe 1: Self-Model
    probe1 = SelfModelProbe(observer_model, device)
    results['self_model'] = probe1.run(own_episodes, other_episodes)

    # Probe 2: Surprise
    probe2 = SurpriseProbe(observer_model, device)
    results['surprise'] = probe2.run(own_episodes)

    # Probe 3: Temporal Integration
    probe3 = TemporalIntegrationProbe(observer_model, device)
    results['temporal'] = probe3.run(own_episodes)

    # Probe 5: Emergent Preferences
    probe5 = EmergentPreferencesProbe(observer_model, device)
    results['preferences'] = probe5.run(own_episodes)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    indicators_positive = 0
    indicators_total = 0

    for probe_name, probe_results in results.items():
        if 'error' in probe_results:
            continue
        for key, val in probe_results.items():
            if key.startswith(('self_model_detected', 'surprise_is_meaningful',
                               'has_present_window', 'has_emergent_valence')):
                indicators_total += 1
                if val:
                    indicators_positive += 1
                    print(f"  [+] {key}")
                else:
                    print(f"  [-] {key}")

    print(f"\n  Positive indicators: {indicators_positive}/{indicators_total}")
    print()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        import json
        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in results.items():
            serializable[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.floating, np.integer)):
                    serializable[k][kk] = float(vv)
                elif isinstance(vv, np.bool_):
                    serializable[k][kk] = bool(vv)
                else:
                    serializable[k][kk] = vv
        with open(save_path / 'probe_results.json', 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {save_path / 'probe_results.json'}")

    return results
