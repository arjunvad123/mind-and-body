"""
Consciousness Probes for the Transformer Observer

6 probes run AFTER training. The observer was never optimized for these —
they must emerge from the prediction objective alone.

Probe 1: Self-Model (Executor Discrimination)
Probe 2: Surprise Correlation
Probe 3: Temporal Integration Window
Probe 4: First Thought vs Deliberation
Probe 5: Emergent Preferences (Cross-Observer RSA)
Probe 6: Layer Preference (Ablation)

All probes compare trained observer against controls (untrained, linear, shuffled).
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from . import config
from .observer_model import ObserverTransformer
from .dataset import ActivationDataset


def _get_test_dataset(h5_path: str = None, split: str = 'test') -> ActivationDataset:
    """Helper to load test dataset with normalization from train."""
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    train_ds = ActivationDataset(h5_path, split='train')
    test_ds = ActivationDataset(h5_path, split=split, precompute_stats=False)
    test_ds.set_normalization_stats(train_ds.layer_means, train_ds.layer_stds)
    return test_ds


def _collect_predictions_and_hidden(
    model: nn.Module,
    dataset: ActivationDataset,
    device: torch.device,
    max_samples: int = 100,
) -> dict:
    """Run model on dataset and collect predictions + hidden states."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_hidden = []
    all_errors = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_resid, target_resid = dataset[i]
            input_resid = input_resid.unsqueeze(0).to(device)
            target_resid = target_resid.unsqueeze(0).to(device)

            predictions, hidden = model(input_resid, return_hidden=True)

            error = (predictions - target_resid).pow(2).mean(dim=-1).squeeze(0)  # (seq_len,)

            all_predictions.append(predictions.squeeze(0).cpu())
            all_targets.append(target_resid.squeeze(0).cpu())
            all_hidden.append(hidden.squeeze(0).cpu())
            all_errors.append(error.cpu())

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'hidden': all_hidden,
        'errors': all_errors,
    }


# ── Probe 1: Self-Model (Executor Discrimination) ──────────────────

def probe_self_model(
    model: nn.Module,
    own_h5_path: str = None,
    other_h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 50,
) -> dict:
    """Probe 1: Does the observer develop a model of "its" executor?

    Show observer activations from:
    - GPT-2 small (its executor) on unseen text
    - GPT-2 medium (different architecture)
    - Random noise shaped like activations

    Measure: prediction error distribution. Lower error on "its" executor = self-model.
    """
    print("Probe 1: Self-Model (Executor Discrimination)")
    print("-" * 50)

    device = device or config.DEVICE
    own_h5_path = own_h5_path or str(config.ACTIVATIONS_PATH)
    model = model.to(device)
    model.eval()

    # ── Own executor (GPT-2 small, unseen test data) ────────────
    own_dataset = _get_test_dataset(own_h5_path)
    own_results = _collect_predictions_and_hidden(model, own_dataset, device, max_samples)
    own_errors = torch.cat(own_results['errors']).numpy()

    # ── Different executor (GPT-2 medium, if available) ─────────
    other_errors = None
    if other_h5_path and Path(other_h5_path).exists():
        import h5py
        # Check if other executor has compatible dimensions
        with h5py.File(other_h5_path, 'r') as f:
            other_n_checkpoints = f.attrs['n_checkpoints']
            other_d_model = f.attrs['d_model']

        if (other_n_checkpoints != config.EXECUTOR_N_CHECKPOINTS or
                other_d_model != config.EXECUTOR_D_MODEL):
            print(f"  Other executor has incompatible shape "
                  f"({other_n_checkpoints} checkpoints, {other_d_model}d) "
                  f"vs expected ({config.EXECUTOR_N_CHECKPOINTS}, {config.EXECUTOR_D_MODEL}d). "
                  f"Skipping cross-executor comparison.")
        else:
            # Use own-executor normalization stats for the other dataset
            other_dataset = ActivationDataset(other_h5_path, split='test', precompute_stats=False)
            other_dataset.set_normalization_stats(own_dataset.layer_means, own_dataset.layer_stds)
            other_results = _collect_predictions_and_hidden(model, other_dataset, device, max_samples)
            other_errors = torch.cat(other_results['errors']).numpy()

    # ── Random noise ────────────────────────────────────────────
    noise_errors = []
    with torch.no_grad():
        for i in range(min(max_samples, len(own_dataset))):
            input_resid, target_resid = own_dataset[i]
            # Replace input with random noise (same scale)
            noise_input = torch.randn_like(input_resid).unsqueeze(0).to(device)
            target_resid = target_resid.unsqueeze(0).to(device)

            predictions = model(noise_input)
            error = (predictions - target_resid).pow(2).mean(dim=-1).squeeze(0)
            noise_errors.append(error.cpu())

    noise_errors = torch.cat(noise_errors).numpy()

    # ── Statistical tests ───────────────────────────────────────
    # Own vs Noise
    t_own_noise, p_own_noise = stats.ttest_ind(own_errors, noise_errors)

    # Own vs Other (if available)
    t_own_other, p_own_other = None, None
    if other_errors is not None:
        t_own_other, p_own_other = stats.ttest_ind(own_errors, other_errors)

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
    if p_own_other is not None:
        print(f"  Own vs Other p-value:   {results['own_vs_other_p']:.2e}")
    print(f"  Self-model detected:    {results['self_model_detected']}")
    print()

    return results


# ── Probe 2: Surprise Correlation ──────────────────────────────────

def probe_surprise(
    model: nn.Module,
    h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 100,
) -> dict:
    """Probe 2: Does observer prediction error spike at computationally interesting moments?

    Run observer on sequences with known "interesting" moments:
    - Garden-path revision points
    - Domain switch tokens
    - Reasoning steps

    Compare observer surprise to raw state-change baseline:
    ||S(T+1) - S(T)||
    """
    print("Probe 2: Surprise Correlation")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    model = model.to(device)
    model.eval()

    import h5py

    # Load sequence type labels
    with h5py.File(h5_path, 'r') as f:
        seq_types = [s.decode() for s in f['sequence_types'][:]]
        n_sequences = f.attrs['n_sequences']

    # Use the full dataset (not just test split) for surprise probe.
    # The observer was never optimized for surprise, so using training data
    # doesn't create overfitting. We need all sequence types represented
    # (pile, garden_path, domain_switch, reasoning) which are distributed
    # across the dataset in order.
    dataset = _get_test_dataset(h5_path, split='train')

    # Collect errors by sequence type
    type_errors = {}
    type_state_changes = {}

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            actual_idx = dataset.start_idx + i
            seq_type = seq_types[actual_idx] if actual_idx < len(seq_types) else 'pile'

            input_resid, target_resid = dataset[i]
            input_resid_t = input_resid.unsqueeze(0).to(device)
            target_resid_t = target_resid.unsqueeze(0).to(device)

            predictions = model(input_resid_t)

            # Observer prediction error (surprise)
            error = (predictions - target_resid_t).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()

            # Raw state change baseline: ||S(T+1) - S(T)||
            final_layer = input_resid[-1]  # (seq_len-1, d_model)
            state_change = (final_layer[1:] - final_layer[:-1]).pow(2).mean(dim=-1).numpy()

            # Align arrays: state_change has len seq_len-2, error has len seq_len-1
            # Drop the first error position to align with state_change
            error = error[1:]

            if seq_type not in type_errors:
                type_errors[seq_type] = []
                type_state_changes[seq_type] = []
            type_errors[seq_type].append(error)
            type_state_changes[seq_type].append(state_change)

    # Compute correlations between observer surprise and state change
    results = {}

    for seq_type in type_errors:
        errors = np.concatenate(type_errors[seq_type])
        if seq_type in type_state_changes and type_state_changes[seq_type]:
            changes = np.concatenate(type_state_changes[seq_type])
            min_len = min(len(errors), len(changes))
            if min_len > 2:
                corr, p_val = stats.pearsonr(errors[:min_len], changes[:min_len])
            else:
                corr, p_val = 0.0, 1.0
        else:
            corr, p_val = 0.0, 1.0

        results[f'{seq_type}_mean_surprise'] = float(errors.mean())
        results[f'{seq_type}_surprise_std'] = float(errors.std())
        results[f'{seq_type}_state_change_corr'] = float(corr)
        results[f'{seq_type}_state_change_p'] = float(p_val)

    # Compare surprise at garden-path/reasoning vs pile
    pile_surprise = results.get('pile_mean_surprise', 0)
    gp_surprise = results.get('garden_path_mean_surprise', 0)
    reason_surprise = results.get('reasoning_mean_surprise', 0)

    results['garden_path_elevated'] = gp_surprise > pile_surprise * 1.1
    results['reasoning_elevated'] = reason_surprise > pile_surprise * 1.1
    results['surprise_is_meaningful'] = (
        results.get('garden_path_elevated', False) or
        results.get('reasoning_elevated', False)
    )

    for key in sorted(results.keys()):
        print(f"  {key}: {results[key]}")
    print()

    return results


# ── Probe 3: Temporal Integration Window ───────────────────────────

def probe_temporal_integration(
    model: nn.Module,
    h5_path: str = None,
    device: torch.device = None,
    windows: list = None,
    max_samples: int = 50,
) -> dict:
    """Probe 3: What is the observer's temporal integration window?

    Vary observer's effective context window: 1, 4, 16, 64, full.
    Measure prediction quality vs window size. The plateau point = "specious present."
    """
    print("Probe 3: Temporal Integration Window")
    print("-" * 50)

    device = device or config.DEVICE
    windows = windows or config.TEMPORAL_WINDOWS
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)

    window_errors = {w: [] for w in windows}

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_resid, target_resid = dataset[i]
            seq_len = input_resid.shape[1]

            for window in windows:
                # Mask: only keep the last `window` positions
                masked_input = input_resid.clone()
                if window < seq_len:
                    masked_input[:, :seq_len - window, :] = 0.0

                masked_input_t = masked_input.unsqueeze(0).to(device)
                target_t = target_resid.unsqueeze(0).to(device)

                predictions = model(masked_input_t)

                # Error on the LAST position only (where full context would help most)
                error = (predictions[:, -1] - target_t[:, -1]).pow(2).mean().item()
                window_errors[window].append(error)

    results = {}
    for w in windows:
        results[f'window_{w}_error'] = float(np.mean(window_errors[w]))

    # Find plateau: where does doubling window size give <5% improvement?
    sorted_windows = sorted(windows)
    error_values = [results[f'window_{w}_error'] for w in sorted_windows]

    plateau_window = sorted_windows[-1]
    for i in range(1, len(sorted_windows)):
        improvement = (error_values[i - 1] - error_values[i]) / max(error_values[i - 1], 1e-8)
        if improvement < 0.05:
            plateau_window = sorted_windows[i - 1]
            break

    results['plateau_window'] = plateau_window
    results['full_vs_minimal_ratio'] = (
        error_values[0] / max(error_values[-1], 1e-8)
    )
    results['has_temporal_integration'] = error_values[0] > error_values[-1] * 1.2

    for w, e in zip(sorted_windows, error_values):
        print(f"  Window {w:>4}: error = {e:.6f}")
    print(f"  Plateau at:          {results['plateau_window']}")
    print(f"  Full/minimal ratio:  {results['full_vs_minimal_ratio']:.4f}")
    print(f"  Has integration:     {results['has_temporal_integration']}")
    print()

    return results


# ── Probe 4: First Thought vs Deliberation ─────────────────────────

def probe_first_thought(
    model: nn.Module,
    h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 50,
    n_deliberation_passes: int = 3,
) -> dict:
    """Probe 4: Is the first forward pass at least as accurate as multi-pass?

    - Single forward pass prediction (System 1)
    - Multiple autoregressive passes feeding predictions back (System 2)
    - Early-layer prediction vs final-layer

    If first thought >= deliberation, the observer has learned fast,
    intuitive processing — not just iterative refinement.
    """
    print("Probe 4: First Thought vs Deliberation")
    print("-" * 50)

    # This probe requires internal access to the ObserverTransformer architecture
    # (embedder, pos_embedding, transformer.layers, prediction_head).
    # Skip for models that don't have these (e.g., LinearBaseline).
    if not hasattr(model, 'embedder'):
        print("  Skipping (model doesn't support early-exit probing)")
        return {
            'first_pass_error': None,
            'multi_pass_error': None,
            'early_exit_error': None,
            'first_better_than_multi': False,
            'early_better_than_late': False,
            'deliberation_ratio': None,
            'skipped': True,
        }

    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)
    loss_fn = nn.MSELoss(reduction='none')

    first_pass_errors = []
    multi_pass_errors = []
    early_exit_errors = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_resid, target_resid = dataset[i]
            input_resid_t = input_resid.unsqueeze(0).to(device)
            target_resid_t = target_resid.unsqueeze(0).to(device)

            # ── System 1: Single forward pass ───────────────────
            predictions, hidden = model(input_resid_t, return_hidden=True)
            first_error = loss_fn(predictions[:, -1:], target_resid_t[:, -1:]).mean().item()
            first_pass_errors.append(first_error)

            # ── Early exit: use hidden from middle of transformer ──
            # Access the transformer's intermediate output by running
            # the prediction head on an earlier representation
            # (Approximate: use first half of transformer)
            x = model.embedder(input_resid_t)
            positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
            x = x + model.pos_embedding(positions)

            mask = model._generate_causal_mask(x.shape[1], device)

            # Run only first 3 layers (half the transformer)
            for j, layer in enumerate(model.transformer.layers):
                x = layer(x, src_mask=mask)
                if j == 2:  # Stop after layer 3 (0-indexed)
                    break

            early_pred = model.prediction_head(x)
            early_error = loss_fn(early_pred[:, -1:], target_resid_t[:, -1:]).mean().item()
            early_exit_errors.append(early_error)

            # ── System 2: Multi-pass (autoregressive refinement) ──
            # Feed prediction back as if it were the next input
            current_input = input_resid_t.clone()
            for _ in range(n_deliberation_passes):
                pred = model(current_input)
                # Use prediction to update the last position of input's final layer
                updated = current_input.clone()
                updated[:, -1, -1:, :] = pred[:, -1:, :]
                current_input = updated

            final_pred = model(current_input)
            multi_error = loss_fn(final_pred[:, -1:], target_resid_t[:, -1:]).mean().item()
            multi_pass_errors.append(multi_error)

    results = {
        'first_pass_error': float(np.mean(first_pass_errors)),
        'multi_pass_error': float(np.mean(multi_pass_errors)),
        'early_exit_error': float(np.mean(early_exit_errors)),
        'first_pass_std': float(np.std(first_pass_errors)),
        'multi_pass_std': float(np.std(multi_pass_errors)),
        'first_better_than_multi': float(np.mean(first_pass_errors)) <= float(np.mean(multi_pass_errors)),
        'early_better_than_late': float(np.mean(early_exit_errors)) <= float(np.mean(first_pass_errors)),
        'deliberation_ratio': float(np.mean(multi_pass_errors)) / max(float(np.mean(first_pass_errors)), 1e-8),
    }

    print(f"  First pass error:      {results['first_pass_error']:.6f}")
    print(f"  Multi-pass error:      {results['multi_pass_error']:.6f}")
    print(f"  Early exit error:      {results['early_exit_error']:.6f}")
    print(f"  First >= multi-pass:   {results['first_better_than_multi']}")
    print(f"  Deliberation ratio:    {results['deliberation_ratio']:.4f}")
    print()

    return results


# ── Probe 5: Emergent Preferences (Cross-Observer RSA) ─────────────

def probe_emergent_preferences(
    observer_paths: list,
    h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 50,
) -> dict:
    """Probe 5: Do observers with different seeds develop similar representations?

    Train 5 observers with different random seeds on the same executor.
    RSA between their internal representations on the same inputs.
    Similar = convergent perspective. Divergent = noise.
    """
    print("Probe 5: Emergent Preferences (Cross-Observer RSA)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)

    from .trainer import load_model

    # Load all seed observers
    models = []
    for path in observer_paths:
        if Path(path).exists():
            m = load_model(path, device)
            models.append(m)

    if len(models) < 2:
        print("  Need at least 2 seed observers. Skipping.")
        return {'error': 'insufficient_models', 'n_models': len(models)}

    dataset = _get_test_dataset(h5_path)

    # Collect hidden states from each observer on same inputs
    all_model_hidden = []
    for m_idx, model in enumerate(models):
        model.eval()
        hidden_list = []

        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                input_resid, _ = dataset[i]
                input_resid_t = input_resid.unsqueeze(0).to(device)

                _, hidden = model(input_resid_t, return_hidden=True)
                # Average over sequence dimension
                h_mean = hidden.squeeze(0).mean(dim=0).cpu().numpy()  # (d_model,)
                hidden_list.append(h_mean)

        all_model_hidden.append(np.array(hidden_list))  # (n_samples, d_model)

    # Compute pairwise RSA between observers
    n_models = len(models)
    rsa_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            # RDM for model i
            sim_i = cosine_similarity(all_model_hidden[i])
            # RDM for model j
            sim_j = cosine_similarity(all_model_hidden[j])

            # Upper triangle only
            triu_idx = np.triu_indices_from(sim_i, k=1)
            rdm_i = sim_i[triu_idx]
            rdm_j = sim_j[triu_idx]

            if len(rdm_i) > 2:
                corr, _ = stats.pearsonr(rdm_i, rdm_j)
            else:
                corr = 0.0
            rsa_matrix[i, j] = corr

    # Off-diagonal mean = cross-observer consistency
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
    print(f"  Min cross-observer RSA:  {results['min_cross_rsa']:.4f}")
    print(f"  Max cross-observer RSA:  {results['max_cross_rsa']:.4f}")
    print(f"  Convergent:              {results['convergent_representations']}")
    print()

    return results


# ── Probe 6: Layer Preference (Ablation) ───────────────────────────

def probe_layer_preference(
    model: nn.Module,
    h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 50,
) -> dict:
    """Probe 6: Which executor layers does the observer rely on most?

    Ablate observer's access to individual executor layers.
    Compare to known GPT-2 layer functions:
    - Early (0-3): embedding, shallow features
    - Middle (4-7): syntax, induction heads
    - Late (8-11): semantics, prediction
    """
    print("Probe 6: Layer Preference (Ablation)")
    print("-" * 50)

    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path)
    loss_fn = nn.MSELoss()

    n_checkpoints = config.EXECUTOR_N_CHECKPOINTS  # 13

    # Baseline error (no ablation)
    baseline_errors = []
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            input_resid, target_resid = dataset[i]
            input_resid_t = input_resid.unsqueeze(0).to(device)
            target_resid_t = target_resid.unsqueeze(0).to(device)

            predictions = model(input_resid_t)
            error = loss_fn(predictions, target_resid_t).item()
            baseline_errors.append(error)

    baseline = np.mean(baseline_errors)

    # Ablate each layer checkpoint
    layer_importance = {}
    for layer in range(n_checkpoints):
        layer_errors = []
        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                input_resid, target_resid = dataset[i]
                # Zero out this layer's contribution
                ablated = input_resid.clone()
                ablated[layer] = 0.0

                ablated_t = ablated.unsqueeze(0).to(device)
                target_resid_t = target_resid.unsqueeze(0).to(device)

                predictions = model(ablated_t)
                error = loss_fn(predictions, target_resid_t).item()
                layer_errors.append(error)

        ablated_error = np.mean(layer_errors)
        importance = (ablated_error - baseline) / max(baseline, 1e-8)
        layer_importance[layer] = importance

    # Categorize by GPT-2 layer groups
    # Checkpoint 0 = pre-layer-0, Checkpoint k = post-layer-(k-1)
    early_importance = np.mean([layer_importance[k] for k in range(0, 5)])    # pre + layers 0-3
    middle_importance = np.mean([layer_importance[k] for k in range(5, 9)])   # layers 4-7
    late_importance = np.mean([layer_importance[k] for k in range(9, 13)])    # layers 8-11

    # Most relied-upon layer
    most_important = max(layer_importance, key=layer_importance.get)

    results = {
        'baseline_error': float(baseline),
        'layer_importance': {str(k): float(v) for k, v in layer_importance.items()},
        'early_importance': float(early_importance),
        'middle_importance': float(middle_importance),
        'late_importance': float(late_importance),
        'most_important_checkpoint': int(most_important),
        'prefers_late_layers': late_importance > early_importance,
        'prefers_middle_layers': middle_importance > max(early_importance, late_importance),
    }

    for k in range(n_checkpoints):
        label = f"pre-L0" if k == 0 else f"post-L{k-1}"
        print(f"  Checkpoint {k:>2} ({label:>8}): importance = {layer_importance[k]:+.6f}")

    print(f"\n  Early (pre+L0-3):  {results['early_importance']:+.6f}")
    print(f"  Middle (L4-7):     {results['middle_importance']:+.6f}")
    print(f"  Late (L8-11):      {results['late_importance']:+.6f}")
    print(f"  Most important:    checkpoint {most_important}")
    print()

    return results


# ── Run All Probes ─────────────────────────────────────────────────

def run_all_probes(
    model: nn.Module,
    h5_path: str = None,
    other_h5_path: str = None,
    seed_observer_paths: list = None,
    device: torch.device = None,
    save_dir: str = None,
) -> dict:
    """Run all 6 consciousness probes and compile results.

    Args:
        model: Trained observer model
        h5_path: Path to primary activations HDF5
        other_h5_path: Path to GPT-2 medium activations (for Probe 1)
        seed_observer_paths: List of checkpoint paths for seed observers (Probe 5)
        device: Torch device
        save_dir: Where to save results JSON

    Returns:
        Complete results dictionary
    """
    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    save_dir = save_dir or str(config.RESULTS_DIR)

    print("=" * 60)
    print("CONSCIOUSNESS PROBES — TRANSFORMER OBSERVER")
    print("=" * 60)
    print()

    results = {}

    # Probe 1: Self-Model
    results['self_model'] = probe_self_model(
        model, h5_path, other_h5_path, device
    )

    # Probe 2: Surprise Correlation
    results['surprise'] = probe_surprise(model, h5_path, device)

    # Probe 3: Temporal Integration
    results['temporal'] = probe_temporal_integration(model, h5_path, device)

    # Probe 4: First Thought vs Deliberation
    results['first_thought'] = probe_first_thought(model, h5_path, device)

    # Probe 5: Emergent Preferences (RSA across seed observers)
    if seed_observer_paths:
        results['preferences'] = probe_emergent_preferences(
            seed_observer_paths, h5_path, device
        )
    else:
        print("Probe 5: Skipped (no seed observer paths provided)")
        results['preferences'] = {'skipped': True}

    # Probe 6: Layer Preference
    results['layer_preference'] = probe_layer_preference(model, h5_path, device)

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
        'layer_preference': results['layer_preference'].get('prefers_late_layers', False),
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

    # Save results
    if save_dir:
        _save_results(results, save_dir)

    return results


def _save_results(results: dict, save_dir: str):
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
