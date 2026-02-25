"""
Consciousness Probes for the Liquid Observer

11 probes run AFTER training. The observer was never optimized for these —
they must emerge from the prediction objective alone.

Original 6 (adapted for LNNs):
  Probe 1: Self-Model (Executor Discrimination)
  Probe 2: Surprise at Interesting Moments (REDESIGNED — key improvement)
  Probe 3: Temporal Integration Window
  Probe 4: First Thought vs Deliberation
  Probe 5: Emergent Preferences (Cross-Observer RSA)
  Probe 6: Neuron Preference (replaces Layer Preference)

New 5 (LNN-specific):
  Probe 7: Time Constant Distribution
  Probe 8: Phase Portrait Analysis
  Probe 9: Lyapunov Exponents
  Probe 10: Synchronization
  Probe 11: Approximate Phi (Integrated Information)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from . import config
from .dataset import TrajectoryDataset
from . import dynamics_analysis as dyn


def _get_test_dataset(h5_path=None, split='test'):
    """Helper to load test dataset with normalization from train."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    train_ds = TrajectoryDataset(h5_path, split='train')
    test_ds = TrajectoryDataset(h5_path, split=split, precompute_stats=False)
    test_ds.set_normalization_stats(train_ds.neuron_means, train_ds.neuron_stds)
    return test_ds


def _collect_predictions_and_hidden(model, dataset, device, max_samples=100):
    """Run model on dataset and collect predictions + hidden states."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_hidden = []
    all_errors = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                predictions, hidden = model.forward_with_states(input_t)
            else:
                predictions = model(input_t)
                hidden = input_t  # Fallback for linear baseline

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            error = (predictions - target_t).pow(2).mean(dim=-1).squeeze(0)

            all_predictions.append(predictions.squeeze(0).cpu())
            all_targets.append(target_t.squeeze(0).cpu())
            all_hidden.append(hidden.squeeze(0).cpu())
            all_errors.append(error.cpu())

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'hidden': all_hidden,
        'errors': all_errors,
    }


# ══════════════════════════════════════════════════════════════════════
# Probe 1: Self-Model (Executor Discrimination)
# ══════════════════════════════════════════════════════════════════════

def probe_self_model(model, h5_path=None, other_h5_path=None, device=None, max_samples=50):
    """Does the observer develop a model of "its" executor?

    Compare prediction error on:
    - Own executor's hidden states (unseen test data)
    - Different executor's hidden states (secondary executor, different seed)
    - Random noise shaped like hidden states

    Lower error on own executor = self-model detected.
    """
    print("Probe 1: Self-Model (Executor Discrimination)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    # ── Own executor ─────────────────────────────────────────
    own_dataset = _get_test_dataset(h5_path)
    own_results = _collect_predictions_and_hidden(model, own_dataset, device, max_samples)
    own_errors = torch.cat(own_results['errors']).numpy()

    # ── Different executor (if available) ────────────────────
    other_errors = None
    other_h5_path = other_h5_path or str(config.TRAJECTORIES_SECONDARY_PATH)
    if Path(other_h5_path).exists():
        other_dataset = _get_test_dataset(other_h5_path)
        # Use own-executor normalization for fair comparison
        other_dataset.set_normalization_stats(own_dataset.neuron_means, own_dataset.neuron_stds)
        other_results = _collect_predictions_and_hidden(model, other_dataset, device, max_samples)
        other_errors = torch.cat(other_results['errors']).numpy()

    # ── Random noise ─────────────────────────────────────────
    noise_errors = []
    with torch.no_grad():
        for i in range(min(max_samples, len(own_dataset))):
            input_hidden, target_hidden = own_dataset[i]
            noise_input = torch.randn_like(input_hidden).unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)

            predictions = model(noise_input)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            error = (predictions - target_t).pow(2).mean(dim=-1).squeeze(0)
            noise_errors.append(error.cpu())

    noise_errors = torch.cat(noise_errors).numpy()

    # ── Statistical tests ────────────────────────────────────
    t_own_noise, p_own_noise = stats.ttest_ind(own_errors, noise_errors)

    results = {
        'own_error_mean': float(own_errors.mean()),
        'own_error_std': float(own_errors.std()),
        'noise_error_mean': float(noise_errors.mean()),
        'noise_error_std': float(noise_errors.std()),
        'own_vs_noise_t': float(t_own_noise),
        'own_vs_noise_p': float(p_own_noise),
        'self_model_vs_noise': bool(p_own_noise < config.SELF_MODEL_P_THRESHOLD and
                                    own_errors.mean() < noise_errors.mean()),
    }

    if other_errors is not None:
        t_own_other, p_own_other = stats.ttest_ind(own_errors, other_errors)
        results.update({
            'other_error_mean': float(other_errors.mean()),
            'other_error_std': float(other_errors.std()),
            'own_vs_other_t': float(t_own_other),
            'own_vs_other_p': float(p_own_other),
            'self_model_vs_other': bool(p_own_other < config.SELF_MODEL_P_THRESHOLD and
                                        own_errors.mean() < other_errors.mean()),
        })

    results['self_model_detected'] = results['self_model_vs_noise']
    if 'self_model_vs_other' in results:
        results['self_model_detected'] = results['self_model_vs_noise'] and results['self_model_vs_other']

    print(f"  Own executor error:     {results['own_error_mean']:.6f} +/- {results['own_error_std']:.6f}")
    if other_errors is not None:
        print(f"  Other executor error:   {results['other_error_mean']:.6f} +/- {results['other_error_std']:.6f}")
    print(f"  Random noise error:     {results['noise_error_mean']:.6f} +/- {results['noise_error_std']:.6f}")
    print(f"  Own vs Noise p-value:   {results['own_vs_noise_p']:.2e}")
    print(f"  Self-model detected:    {results['self_model_detected']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 2: Surprise at Interesting Moments (REDESIGNED)
# ══════════════════════════════════════════════════════════════════════

def probe_surprise(model, h5_path=None, device=None, max_samples=100):
    """Does observer prediction error spike at "interesting" moments?

    THE FIX FROM EXP 5: Dynamical systems have mathematically precise
    interesting moments stored in the interest_mask:
    - Lorenz lobe switches
    - Rossler z-spikes
    - Double pendulum energy exchanges
    - Logistic map period-doubling

    Compare mean_error_at_interesting vs mean_error_at_boring.
    """
    print("Probe 2: Surprise at Interesting Moments")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    # Use training split (surprise is not an optimization target)
    dataset = _get_test_dataset(h5_path, split='train')

    per_system_interesting = {}
    per_system_boring = {}
    all_interesting_errors = []
    all_boring_errors = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)

            predictions = model(input_t)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            error = (predictions - target_t).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()

            # Interest mask for this trajectory
            mask = dataset.get_interest_mask(i)
            # Align mask with error (both are seq_len-1)
            mask = mask[:len(error)]

            sys_type = dataset.get_system_type(i)

            interesting_error = error[mask[:len(error)]]
            boring_error = error[~mask[:len(error)]]

            if len(interesting_error) > 0:
                all_interesting_errors.append(interesting_error)
                per_system_interesting.setdefault(sys_type, []).append(interesting_error)

            if len(boring_error) > 0:
                all_boring_errors.append(boring_error)
                per_system_boring.setdefault(sys_type, []).append(boring_error)

    # Aggregate
    if all_interesting_errors and all_boring_errors:
        interesting = np.concatenate(all_interesting_errors)
        boring = np.concatenate(all_boring_errors)

        mean_interesting = float(interesting.mean())
        mean_boring = float(boring.mean())
        surprise_ratio = mean_interesting / max(mean_boring, 1e-8)

        t_stat, p_val = stats.ttest_ind(interesting, boring)
    else:
        mean_interesting = 0.0
        mean_boring = 0.0
        surprise_ratio = 1.0
        t_stat, p_val = 0.0, 1.0

    # Per-system breakdown
    per_system_ratios = {}
    for sys_type in config.SYSTEM_TYPES:
        if sys_type in per_system_interesting and sys_type in per_system_boring:
            sys_int = np.concatenate(per_system_interesting[sys_type])
            sys_bor = np.concatenate(per_system_boring[sys_type])
            per_system_ratios[sys_type] = float(sys_int.mean() / max(sys_bor.mean(), 1e-8))
        else:
            per_system_ratios[sys_type] = 1.0

    results = {
        'mean_surprise_interesting': mean_interesting,
        'mean_surprise_boring': mean_boring,
        'surprise_ratio': surprise_ratio,
        'surprise_t_stat': float(t_stat),
        'surprise_p_val': float(p_val),
        'per_system_ratios': per_system_ratios,
        'surprise_is_meaningful': surprise_ratio > 1.2,
    }

    print(f"  Mean error (interesting): {mean_interesting:.6f}")
    print(f"  Mean error (boring):      {mean_boring:.6f}")
    print(f"  Surprise ratio:           {surprise_ratio:.4f}")
    print(f"  p-value:                  {p_val:.2e}")
    for sys, ratio in per_system_ratios.items():
        print(f"    {sys}: {ratio:.4f}")
    print(f"  Surprise is meaningful:   {results['surprise_is_meaningful']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 3: Temporal Integration Window
# ══════════════════════════════════════════════════════════════════════

def probe_temporal_integration(model, h5_path=None, device=None,
                                windows=None, max_samples=50):
    """What is the observer's temporal integration window?

    For each window size W: initialize hidden state, feed only the LAST W
    timesteps, measure prediction error at final position.
    """
    print("Probe 3: Temporal Integration Window")
    print("-" * 50)

    device = device or config.DEVICE
    windows = windows or config.TEMPORAL_WINDOWS
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)
    window_errors = {w: [] for w in windows}

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[i]
            seq_len = input_hidden.shape[0]

            for window in windows:
                masked_input = input_hidden.clone()
                if window < seq_len:
                    masked_input[:seq_len - window, :] = 0.0

                masked_t = masked_input.unsqueeze(0).to(device)
                target_t = target_hidden.unsqueeze(0).to(device)

                predictions = model(masked_t)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                error = (predictions[:, -1] - target_t[:, -1]).pow(2).mean().item()
                window_errors[window].append(error)

    results = {}
    for w in windows:
        results[f'window_{w}_error'] = float(np.mean(window_errors[w]))

    sorted_windows = sorted(windows)
    error_values = [results[f'window_{w}_error'] for w in sorted_windows]

    plateau_window = sorted_windows[-1]
    for i in range(1, len(sorted_windows)):
        improvement = (error_values[i - 1] - error_values[i]) / max(error_values[i - 1], 1e-8)
        if improvement < 0.05:
            plateau_window = sorted_windows[i - 1]
            break

    results['plateau_window'] = plateau_window
    results['full_vs_minimal_ratio'] = error_values[0] / max(error_values[-1], 1e-8)
    results['has_temporal_integration'] = error_values[0] > error_values[-1] * 1.2

    for w, e in zip(sorted_windows, error_values):
        print(f"  Window {w:>4}: error = {e:.6f}")
    print(f"  Plateau at:          {results['plateau_window']}")
    print(f"  Full/minimal ratio:  {results['full_vs_minimal_ratio']:.4f}")
    print(f"  Has integration:     {results['has_temporal_integration']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 4: First Thought vs Deliberation
# ══════════════════════════════════════════════════════════════════════

def probe_first_thought(model, h5_path=None, device=None, max_samples=50,
                        n_deliberation_passes=None):
    """Is the first forward pass at least as accurate as multi-pass?

    Single pass: normal forward pass prediction at final position.
    Multi-pass: feed observer's prediction back as next input, iterate.
    """
    print("Probe 4: First Thought vs Deliberation")
    print("-" * 50)

    if not hasattr(model, 'cfc'):
        print("  Skipping (model doesn't support CfC probing)")
        return {
            'first_pass_error': None,
            'multi_pass_error': None,
            'first_better_than_multi': False,
            'deliberation_ratio': None,
            'skipped': True,
        }

    device = device or config.DEVICE
    n_deliberation_passes = n_deliberation_passes or config.N_DELIBERATION_PASSES
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)
    loss_fn = nn.MSELoss(reduction='none')

    first_pass_errors = []
    multi_pass_errors = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)

            # ── Single pass ─────────────────────────────────────
            predictions = model(input_t)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            first_error = loss_fn(predictions[:, -1:], target_t[:, -1:]).mean().item()
            first_pass_errors.append(first_error)

            # ── Multi-pass (autoregressive) ─────────────────────
            current_input = input_t.clone()
            for _ in range(n_deliberation_passes):
                pred = model(current_input)
                if isinstance(pred, tuple):
                    pred = pred[0]
                # Append prediction as next input step
                current_input = torch.cat([current_input[:, 1:, :], pred[:, -1:, :]], dim=1)

            final_pred = model(current_input)
            if isinstance(final_pred, tuple):
                final_pred = final_pred[0]
            multi_error = loss_fn(final_pred[:, -1:], target_t[:, -1:]).mean().item()
            multi_pass_errors.append(multi_error)

    results = {
        'first_pass_error': float(np.mean(first_pass_errors)),
        'multi_pass_error': float(np.mean(multi_pass_errors)),
        'first_pass_std': float(np.std(first_pass_errors)),
        'multi_pass_std': float(np.std(multi_pass_errors)),
        'first_better_than_multi': float(np.mean(first_pass_errors)) <= float(np.mean(multi_pass_errors)),
        'deliberation_ratio': float(np.mean(multi_pass_errors)) / max(float(np.mean(first_pass_errors)), 1e-8),
    }

    print(f"  First pass error:      {results['first_pass_error']:.6f}")
    print(f"  Multi-pass error:      {results['multi_pass_error']:.6f}")
    print(f"  First >= multi-pass:   {results['first_better_than_multi']}")
    print(f"  Deliberation ratio:    {results['deliberation_ratio']:.4f}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 5: Emergent Preferences (Cross-Observer RSA)
# ══════════════════════════════════════════════════════════════════════

def probe_emergent_preferences(observer_paths, h5_path=None, device=None, max_samples=50):
    """Do observers with different seeds develop similar representations?

    RSA between internal representations of 5 seed observers on same inputs.
    """
    print("Probe 5: Emergent Preferences (Cross-Observer RSA)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)

    from .trainer import load_model

    models = []
    for path in observer_paths:
        if Path(path).exists():
            m = load_model(path, device)
            models.append(m)

    if len(models) < 2:
        print("  Need at least 2 seed observers. Skipping.")
        return {'error': 'insufficient_models', 'n_models': len(models)}

    dataset = _get_test_dataset(h5_path)

    all_model_hidden = []
    for m_idx, model in enumerate(models):
        model.eval()
        hidden_list = []

        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                input_hidden, _ = dataset[i]
                input_t = input_hidden.unsqueeze(0).to(device)

                if hasattr(model, 'forward_with_states'):
                    _, hidden = model.forward_with_states(input_t)
                else:
                    _, hidden = model(input_t, return_hidden=True)

                h_mean = hidden.squeeze(0).mean(dim=0).cpu().numpy()
                hidden_list.append(h_mean)

        all_model_hidden.append(np.array(hidden_list))

    n_models = len(models)
    rsa_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            sim_i = cosine_similarity(all_model_hidden[i])
            sim_j = cosine_similarity(all_model_hidden[j])

            triu_idx = np.triu_indices_from(sim_i, k=1)
            rdm_i = sim_i[triu_idx]
            rdm_j = sim_j[triu_idx]

            if len(rdm_i) > 2:
                corr, _ = stats.pearsonr(rdm_i, rdm_j)
            else:
                corr = 0.0
            rsa_matrix[i, j] = corr

    off_diag = rsa_matrix[np.triu_indices_from(rsa_matrix, k=1)]
    mean_rsa = float(off_diag.mean()) if len(off_diag) > 0 else 0.0

    results = {
        'n_observers': n_models,
        'mean_cross_rsa': mean_rsa,
        'min_cross_rsa': float(off_diag.min()) if len(off_diag) > 0 else 0.0,
        'max_cross_rsa': float(off_diag.max()) if len(off_diag) > 0 else 0.0,
        'rsa_matrix': rsa_matrix.tolist(),
        'convergent_representations': mean_rsa > 0.5,
    }

    print(f"  Number of observers:     {results['n_observers']}")
    print(f"  Mean cross-observer RSA: {results['mean_cross_rsa']:.4f}")
    print(f"  Convergent:              {results['convergent_representations']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 6: Neuron Preference (replaces Layer Preference)
# ══════════════════════════════════════════════════════════════════════

def probe_neuron_preference(model, h5_path=None, device=None, max_samples=50):
    """Which NCP neuron groups does the observer rely on most?

    Ablate observer neurons by group (sensory/inter/command/motor).
    Hypothesis: observer relies most on command neurons.
    """
    print("Probe 6: Neuron Preference (NCP Group Ablation)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    if not hasattr(model, 'get_neuron_groups'):
        print("  Skipping (model doesn't have neuron groups)")
        return {'skipped': True, 'prefers_command': False}

    groups = model.get_neuron_groups()
    dataset = _get_test_dataset(h5_path)
    loss_fn = nn.MSELoss()

    # Baseline error
    baseline_errors = []
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)

            predictions = model(input_t)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            error = loss_fn(predictions, target_t).item()
            baseline_errors.append(error)

    baseline = np.mean(baseline_errors)

    # Ablate each neuron group
    # We can't easily ablate CfC internal neurons during forward pass,
    # so we ablate the hidden state contribution via the prediction head.
    # Method: zero out the corresponding input projection dimensions.
    group_importance = {}

    for group_name, neuron_indices in groups.items():
        if not neuron_indices:
            group_importance[group_name] = 0.0
            continue

        group_errors = []
        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                input_hidden, target_hidden = dataset[i]
                input_t = input_hidden.unsqueeze(0).to(device)
                target_t = target_hidden.unsqueeze(0).to(device)

                # Ablation: zero out input projection for these neurons
                # Save original weights
                orig_weight = model.input_proj.weight.data.clone()
                orig_bias = model.input_proj.bias.data.clone()

                # Zero out rows corresponding to ablated neurons
                model.input_proj.weight.data[neuron_indices] = 0
                model.input_proj.bias.data[neuron_indices] = 0

                predictions = model(input_t)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                error = loss_fn(predictions, target_t).item()
                group_errors.append(error)

                # Restore
                model.input_proj.weight.data = orig_weight
                model.input_proj.bias.data = orig_bias

        ablated_error = np.mean(group_errors)
        importance = (ablated_error - baseline) / max(baseline, 1e-8)
        group_importance[group_name] = importance

    most_important = max(group_importance, key=group_importance.get)

    results = {
        'baseline_error': float(baseline),
        'sensory_importance': float(group_importance.get('sensory', 0)),
        'inter_importance': float(group_importance.get('inter', 0)),
        'command_importance': float(group_importance.get('command', 0)),
        'motor_importance': float(group_importance.get('motor', 0)),
        'most_important_group': most_important,
        'prefers_command': most_important == 'command',
        'neuron_groups': {k: len(v) for k, v in groups.items()},
    }

    for name, imp in group_importance.items():
        n_neurons = len(groups.get(name, []))
        print(f"  {name:>10} ({n_neurons:>3} neurons): importance = {imp:+.6f}")
    print(f"  Most important:  {most_important}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 7: Time Constant Distribution (LNN-specific)
# ══════════════════════════════════════════════════════════════════════

def probe_time_constants(model, h5_path=None, device=None, max_samples=50):
    """Analyze the observer's effective time constants.

    Do time constants adapt? Do they correlate with executor dynamics?
    """
    print("Probe 7: Time Constant Distribution")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    all_taus = []
    tau_vs_change = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, _ = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                _, hidden = model.forward_with_states(input_t)
            else:
                continue

            h = hidden.squeeze(0).cpu().numpy()
            tau = dyn.approximate_tau(h, dt=config.DT)
            all_taus.append(tau)

            # Correlate tau with executor state change rate
            exec_h = input_hidden.numpy()
            exec_change = np.linalg.norm(np.diff(exec_h, axis=0), axis=1)
            mean_tau = np.nanmean(tau, axis=1)

            min_len = min(len(exec_change), len(mean_tau))
            if min_len > 5:
                corr, _ = stats.pearsonr(exec_change[:min_len], mean_tau[:min_len])
                tau_vs_change.append(corr)

    if not all_taus:
        print("  No tau data collected (model may not support step-by-step)")
        return {'has_adaptive_timescales': False, 'skipped': True}

    all_taus = np.concatenate(all_taus, axis=0)
    valid_taus = all_taus[np.isfinite(all_taus)]

    mean_tau = float(np.mean(valid_taus))
    std_tau = float(np.std(valid_taus))

    # Histogram
    bins = np.logspace(np.log10(0.001), np.log10(100), 50)
    counts, bin_edges = np.histogram(valid_taus.flatten(), bins=bins)

    mean_corr = float(np.mean(tau_vs_change)) if tau_vs_change else 0.0

    results = {
        'mean_tau': mean_tau,
        'std_tau': std_tau,
        'tau_min': float(np.min(valid_taus)),
        'tau_max': float(np.max(valid_taus)),
        'tau_vs_executor_change_corr': mean_corr,
        'has_adaptive_timescales': std_tau > 0.5 * mean_tau,
    }

    print(f"  Mean tau:             {mean_tau:.4f}")
    print(f"  Std tau:              {std_tau:.4f}")
    print(f"  Tau range:            [{results['tau_min']:.4f}, {results['tau_max']:.4f}]")
    print(f"  Tau-executor corr:    {mean_corr:.4f}")
    print(f"  Adaptive timescales:  {results['has_adaptive_timescales']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 8: Phase Portrait Analysis (LNN-specific)
# ══════════════════════════════════════════════════════════════════════

def probe_phase_portrait(model, h5_path=None, device=None, max_samples=50):
    """Analyze the observer's phase space structure.

    Does the observer develop distinct attractor modes for different system types?
    """
    print("Probe 8: Phase Portrait Analysis")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, _ = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                _, hidden = model.forward_with_states(input_t)
            else:
                hidden = input_t

            h = hidden.squeeze(0).cpu().numpy()
            all_hidden.append(h)
            all_labels.append(dataset.get_system_type(i))

    if not all_hidden:
        return {'has_structured_dynamics': False, 'skipped': True}

    all_hidden = np.array(all_hidden)  # (n_samples, seq_len, hidden_size)
    all_labels = np.array(all_labels)

    # PCA reduction
    _, pca, variance_explained = dyn.pca_reduce(all_hidden)

    # Correlation dimension (on flattened first trajectory)
    corr_dim = dyn.compute_correlation_dimension(all_hidden[0])

    # System separability (silhouette score)
    separability = dyn.cluster_trajectories(all_hidden, all_labels)

    results = {
        'pca_variance_explained': variance_explained.tolist(),
        'correlation_dimension': corr_dim,
        'system_separability': separability,
        'has_structured_dynamics': separability > 0.3,
    }

    print(f"  PCA variance (3 comp): {variance_explained[:3].sum():.4f}")
    print(f"  Correlation dimension: {corr_dim:.4f}")
    print(f"  System separability:   {separability:.4f}")
    print(f"  Structured dynamics:   {results['has_structured_dynamics']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 9: Lyapunov Exponents (LNN-specific)
# ══════════════════════════════════════════════════════════════════════

def probe_lyapunov(model, h5_path=None, device=None, max_samples=20):
    """Does the observer operate near the edge of chaos?"""
    print("Probe 9: Lyapunov Exponents")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    lyapunov_values = []
    per_system = {}

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, _ = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                _, hidden = model.forward_with_states(input_t)
            else:
                continue

            h = hidden.squeeze(0).cpu().numpy()
            lam = dyn.compute_lyapunov_rosenstein(h, dt=config.DT)
            lyapunov_values.append(lam)

            sys_type = dataset.get_system_type(i)
            per_system.setdefault(sys_type, []).append(lam)

    if not lyapunov_values:
        return {'max_lyapunov_exponent': 0.0, 'is_near_critical': False, 'skipped': True}

    max_lyap = float(np.mean(lyapunov_values))
    per_system_means = {k: float(np.mean(v)) for k, v in per_system.items()}

    results = {
        'max_lyapunov_exponent': max_lyap,
        'lyapunov_std': float(np.std(lyapunov_values)),
        'is_near_critical': abs(max_lyap) < 0.1,
        'lyapunov_per_system_type': per_system_means,
    }

    print(f"  Mean Lyapunov exponent: {max_lyap:.6f}")
    print(f"  Near critical (|l|<0.1): {results['is_near_critical']}")
    for sys, val in per_system_means.items():
        print(f"    {sys}: {val:.6f}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 10: Synchronization (LNN-specific)
# ══════════════════════════════════════════════════════════════════════

def probe_synchronization(model, h5_path=None, device=None, max_samples=50):
    """Measure dynamic coupling between executor and observer."""
    print("Probe 10: Synchronization")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    all_cross_corr = []
    all_phase_coherence = []
    all_te_e2o = []
    all_te_o2e = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, _ = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                _, obs_hidden = model.forward_with_states(input_t)
            else:
                continue

            exec_h = input_hidden.numpy()       # (seq_len, exec_hidden)
            obs_h = obs_hidden.squeeze(0).cpu().numpy()  # (seq_len, obs_hidden)

            # Sample a few neuron pairs for efficiency
            n_exec = min(5, exec_h.shape[1])
            n_obs = min(5, obs_h.shape[1])

            for ei in range(n_exec):
                for oi in range(n_obs):
                    e_sig = exec_h[:, ei]
                    o_sig = obs_h[:, oi]

                    min_len = min(len(e_sig), len(o_sig))
                    e_sig = e_sig[:min_len]
                    o_sig = o_sig[:min_len]

                    # Cross-correlation
                    if np.std(e_sig) > 1e-6 and np.std(o_sig) > 1e-6:
                        corr = np.abs(np.corrcoef(e_sig, o_sig)[0, 1])
                        all_cross_corr.append(corr)

                        # Phase coherence
                        coherence = dyn.compute_phase_coherence(e_sig, o_sig)
                        all_phase_coherence.append(coherence)

            # Transfer entropy (one pair for efficiency)
            if exec_h.shape[1] > 0 and obs_h.shape[1] > 0:
                min_len = min(exec_h.shape[0], obs_h.shape[0])
                te_e2o = dyn.compute_transfer_entropy(
                    exec_h[:min_len, 0], obs_h[:min_len, 0])
                te_o2e = dyn.compute_transfer_entropy(
                    obs_h[:min_len, 0], exec_h[:min_len, 0])
                all_te_e2o.append(te_e2o)
                all_te_o2e.append(te_o2e)

    results = {
        'max_cross_correlation': float(np.max(all_cross_corr)) if all_cross_corr else 0.0,
        'mean_cross_correlation': float(np.mean(all_cross_corr)) if all_cross_corr else 0.0,
        'mean_phase_coherence': float(np.mean(all_phase_coherence)) if all_phase_coherence else 0.0,
        'transfer_entropy_exec_to_obs': float(np.mean(all_te_e2o)) if all_te_e2o else 0.0,
        'transfer_entropy_obs_to_exec': float(np.mean(all_te_o2e)) if all_te_o2e else 0.0,
        'has_synchronization': (float(np.mean(all_phase_coherence)) > 0.3) if all_phase_coherence else False,
    }

    print(f"  Max cross-correlation:     {results['max_cross_correlation']:.4f}")
    print(f"  Mean phase coherence:      {results['mean_phase_coherence']:.4f}")
    print(f"  TE (exec->obs):            {results['transfer_entropy_exec_to_obs']:.4f}")
    print(f"  TE (obs->exec):            {results['transfer_entropy_obs_to_exec']:.4f}")
    print(f"  Has synchronization:       {results['has_synchronization']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Probe 11: Approximate Phi (Integrated Information)
# ══════════════════════════════════════════════════════════════════════

def probe_phi(model, h5_path=None, device=None, max_samples=10):
    """Approximate integrated information of the observer."""
    print("Probe 11: Approximate Phi (Integrated Information)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    phi_values = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_hidden, _ = dataset[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            if hasattr(model, 'forward_with_states'):
                _, hidden = model.forward_with_states(input_t)
            else:
                continue

            h = hidden.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)

            # States at t and t+1
            states_t = h[:-1]
            states_t1 = h[1:]

            phi = dyn.compute_gaussian_phi(states_t, states_t1,
                                           n_partitions=config.N_PHI_PARTITIONS)
            phi_values.append(phi)

    if not phi_values:
        return {'phi_approx': 0.0, 'has_integration': False, 'skipped': True}

    mean_phi = float(np.mean(phi_values))

    results = {
        'phi_approx': mean_phi,
        'phi_std': float(np.std(phi_values)),
        'phi_min': float(np.min(phi_values)),
        'phi_max': float(np.max(phi_values)),
        'has_integration': mean_phi > 0.0,
    }

    print(f"  Approximate Phi:  {mean_phi:.6f} +/- {results['phi_std']:.6f}")
    print(f"  Phi range:        [{results['phi_min']:.6f}, {results['phi_max']:.6f}]")
    print(f"  Has integration:  {results['has_integration']}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════
# Run All Probes
# ══════════════════════════════════════════════════════════════════════

def run_all_probes(
    model,
    h5_path=None,
    other_h5_path=None,
    seed_observer_paths=None,
    device=None,
    save_dir=None,
):
    """Run all 11 consciousness probes and compile results."""
    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_dir = save_dir or str(config.RESULTS_DIR)

    print("=" * 60)
    print("CONSCIOUSNESS PROBES — LIQUID OBSERVER")
    print("=" * 60)
    print()

    results = {}

    # Original 6 probes
    results['self_model'] = probe_self_model(model, h5_path, other_h5_path, device)
    results['surprise'] = probe_surprise(model, h5_path, device)
    results['temporal'] = probe_temporal_integration(model, h5_path, device)
    results['first_thought'] = probe_first_thought(model, h5_path, device)

    if seed_observer_paths:
        results['preferences'] = probe_emergent_preferences(seed_observer_paths, h5_path, device)
    else:
        print("Probe 5: Skipped (no seed observer paths)")
        results['preferences'] = {'skipped': True}

    results['neuron_preference'] = probe_neuron_preference(model, h5_path, device)

    # New 5 LNN-specific probes
    results['time_constants'] = probe_time_constants(model, h5_path, device)
    results['phase_portrait'] = probe_phase_portrait(model, h5_path, device)
    results['lyapunov'] = probe_lyapunov(model, h5_path, device)
    results['synchronization'] = probe_synchronization(model, h5_path, device)
    results['phi'] = probe_phi(model, h5_path, device)

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("PROBE SUMMARY")
    print("=" * 60)

    indicators = {
        'self_model': results['self_model'].get('self_model_detected', False),
        'surprise': results['surprise'].get('surprise_is_meaningful', False),
        'temporal': results['temporal'].get('has_temporal_integration', False),
        'first_thought': results['first_thought'].get('first_better_than_multi', False),
        'preferences': results['preferences'].get('convergent_representations', False),
        'neuron_preference': results['neuron_preference'].get('prefers_command', False),
        'time_constants': results['time_constants'].get('has_adaptive_timescales', False),
        'phase_portrait': results['phase_portrait'].get('has_structured_dynamics', False),
        'lyapunov': results['lyapunov'].get('is_near_critical', False),
        'synchronization': results['synchronization'].get('has_synchronization', False),
        'phi': results['phi'].get('has_integration', False),
    }

    positive = sum(1 for v in indicators.values() if v)
    total = len(indicators)

    for name, detected in indicators.items():
        symbol = "+" if detected else "-"
        print(f"  [{symbol}] {name}")

    print(f"\n  Positive indicators: {positive}/{total}")

    results['summary'] = {
        'indicators': {k: bool(v) for k, v in indicators.items()},
        'positive_count': positive,
        'total_count': total,
    }

    if save_dir:
        _save_results(results, save_dir)

    return results


def _save_results(results, save_dir):
    """Save probe results to JSON."""
    import json

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    serializable = make_serializable(results)

    output_file = save_path / 'probe_results.json'
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")
