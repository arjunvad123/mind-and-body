"""
Experiment 6: Liquid Neural Network Observer-Executor
=====================================================

Full pipeline entry point.

A CfC observer watches a CfC executor trained on diverse dynamical systems.
Both operate in continuous time via ODEs. We probe for emergent
consciousness-like properties that leverage the shared temporal medium.

Phases:
    1. Extract — Generate dynamical systems, train executor, extract hidden states
    2. Train  — Train observer + linear baseline + shuffled control + seed observers
    3. Probe  — Run 11 consciousness probes + 4 controls + self-observation

Usage:
    # Full pipeline
    python -m experiment6_liquid_observer.run_experiment

    # Individual phases
    python -m experiment6_liquid_observer.run_experiment --extract-only
    python -m experiment6_liquid_observer.run_experiment --train-only
    python -m experiment6_liquid_observer.run_experiment --probe-only
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from . import config


def phase_extract(device=None):
    """Phase 1: Generate systems, train executor, extract hidden states."""
    from .dynamical_systems import generate_all_systems, save_system_data
    from .executor_model import (
        train_executor, extract_hidden_trajectories,
        train_secondary_executor, load_executor,
    )

    device = device or config.DEVICE

    print("=" * 70)
    print("PHASE 1: SYSTEM GENERATION & EXECUTOR TRAINING")
    print("=" * 70)

    # Step 1: Generate dynamical systems
    print("\n--- Generating Dynamical Systems ---")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not config.SYSTEM_DATA_PATH.exists():
        data = generate_all_systems()
        save_system_data(data)
    else:
        print(f"  System data already exists at {config.SYSTEM_DATA_PATH}")

    # Step 2: Train primary executor
    print("\n--- Training Primary Executor ---")
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    executor_path = str(config.CHECKPOINTS_DIR / 'executor.pt')
    if not Path(executor_path).exists():
        executor_result = train_executor(device=device)
    else:
        print(f"  Executor already trained at {executor_path}")

    # Step 3: Extract hidden state trajectories
    print("\n--- Extracting Hidden State Trajectories ---")
    if not config.TRAJECTORIES_PATH.exists():
        model = load_executor(executor_path, device)
        extract_hidden_trajectories(model, output_h5_path=str(config.TRAJECTORIES_PATH), device=device)
    else:
        print(f"  Trajectories already exist at {config.TRAJECTORIES_PATH}")

    # Step 4: Train secondary executor (different seed) for wrong-executor control
    print("\n--- Training Secondary Executor (for wrong-executor control) ---")
    secondary_path = str(config.CHECKPOINTS_DIR / 'executor_secondary.pt')
    if not Path(secondary_path).exists():
        train_secondary_executor(device=device)
    else:
        print(f"  Secondary executor already trained at {secondary_path}")

    # Step 5: Extract secondary hidden states
    print("\n--- Extracting Secondary Hidden States ---")
    if not config.TRAJECTORIES_SECONDARY_PATH.exists():
        model2 = load_executor(secondary_path, device)
        extract_hidden_trajectories(
            model2,
            output_h5_path=str(config.TRAJECTORIES_SECONDARY_PATH),
            device=device,
        )
    else:
        print(f"  Secondary trajectories already exist at {config.TRAJECTORIES_SECONDARY_PATH}")

    # Validate
    import h5py
    with h5py.File(str(config.TRAJECTORIES_PATH), 'r') as f:
        print(f"\n  Primary hidden states: {f['hidden_states'].shape}")
        print(f"  System states: {f['system_states'].shape}")

    print("\nPhase 1 complete.")


def phase_train(device=None):
    """Phase 2: Train all observers."""
    from .trainer import (
        train_observer,
        train_linear_baseline,
        train_shuffled_observer,
        train_seed_observers,
        train_scaling_observers,
    )

    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print("PHASE 2: OBSERVER TRAINING")
    print("=" * 70)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Train primary observer
    print("\n--- Primary Observer ---")
    observer_result = train_observer(device=device)

    # Train linear baseline
    print("\n--- Linear Baseline ---")
    linear_result = train_linear_baseline(device=device)

    # Train shuffled control
    print("\n--- Shuffled Observer ---")
    shuffled_result = train_shuffled_observer(device=device)

    # Train seed observers (for Probe 5 RSA)
    print("\n--- Seed Observers (for RSA) ---")
    seed_results = train_seed_observers(device=device)

    # Train scaling observers
    print("\n--- Scaling Observers ---")
    scaling_results = train_scaling_observers(device=device)

    # Save training summary
    summary = {
        'observer_final_train_loss': observer_result['history']['train_loss'][-1],
        'observer_final_val_loss': (observer_result['history']['val_loss'][-1]
                                    if observer_result['history']['val_loss'] else None),
        'linear_final_train_loss': linear_result['history']['train_loss'][-1],
        'shuffled_final_train_loss': shuffled_result['history']['train_loss'][-1],
        'n_seed_observers': len(seed_results),
        'scaling_sizes': list(scaling_results.keys()),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 60)
    print("Training Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def phase_probe(device=None):
    """Phase 3: Run consciousness probes, controls, and self-observation."""
    from .probes import run_all_probes
    from .controls import run_all_controls
    from .trainer import load_model
    from .observer_model import SelfObservingWrapper

    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print("PHASE 3: CONSCIOUSNESS PROBES")
    print("=" * 70)

    # Load trained observer
    observer_path = str(config.CHECKPOINTS_DIR / 'observer.pt')
    if not Path(observer_path).exists():
        print(f"ERROR: No trained observer found at {observer_path}")
        print("Run with --train-only first.")
        return None

    model = load_model(observer_path, device)

    # Collect seed observer paths
    seed_paths = []
    for i in range(config.N_SEED_OBSERVERS):
        p = str(config.CHECKPOINTS_DIR / f'observer_seed{i}.pt')
        if Path(p).exists():
            seed_paths.append(p)

    # Check for secondary executor trajectories
    other_h5 = (str(config.TRAJECTORIES_SECONDARY_PATH)
                if config.TRAJECTORIES_SECONDARY_PATH.exists() else None)

    # ── Run 11 probes on trained observer ──────────────────────
    probe_results = run_all_probes(
        model=model,
        h5_path=str(config.TRAJECTORIES_PATH),
        other_h5_path=other_h5,
        seed_observer_paths=seed_paths if seed_paths else None,
        device=device,
        save_dir=str(config.RESULTS_DIR),
    )

    # ── Run 4 controls ─────────────────────────────────────────
    control_results = run_all_controls(
        observer_path=observer_path,
        h5_path=str(config.TRAJECTORIES_PATH),
        device=device,
        save_dir=str(config.RESULTS_DIR),
    )

    # ── Self-Observation Experiment ────────────────────────────
    print("\n" + "=" * 60)
    print("SELF-OBSERVATION EXPERIMENT")
    print("=" * 60)

    self_obs_results = _run_self_observation(model, device)

    if self_obs_results:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.RESULTS_DIR / 'self_observation_results.json', 'w') as f:
            json.dump(_make_serializable(self_obs_results), f, indent=2)

    # ── Scaling Analysis ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    scaling_results = _run_scaling_analysis(device)

    if scaling_results:
        with open(config.RESULTS_DIR / 'scaling_results.json', 'w') as f:
            json.dump(_make_serializable(scaling_results), f, indent=2)

    # ── Final Comparison ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: OBSERVER vs CONTROLS")
    print("=" * 70)

    observer_pos = probe_results['summary']['positive_count']
    observer_total = probe_results['summary']['total_count']
    print(f"  {'Trained Observer':>25}: {observer_pos}/{observer_total} positive")

    for name, results in control_results.items():
        summary = results.get('summary', {})
        pos = summary.get('positive_count', 0)
        total = summary.get('total_count', 0)
        print(f"  {name:>25}: {pos}/{total} positive")

    # ── Interpretation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    untrained_pos = control_results.get('untrained', {}).get('summary', {}).get('positive_count', 0)
    linear_pos = control_results.get('linear', {}).get('summary', {}).get('positive_count', 0)
    shuffled_pos = control_results.get('shuffled', {}).get('summary', {}).get('positive_count', 0)
    wrong_pos = control_results.get('wrong_executor', {}).get('summary', {}).get('positive_count', 0)

    if observer_pos > untrained_pos:
        print("  [+] Observer outperforms untrained -> training matters")
    else:
        print("  [-] Observer doesn't outperform untrained -> probes may be artifacts")

    if observer_pos > linear_pos:
        print("  [+] Observer outperforms linear baseline -> temporal modeling matters")
    else:
        print("  [-] Observer doesn't outperform linear -> temporal modeling unnecessary")

    if observer_pos > shuffled_pos:
        print("  [+] Observer outperforms shuffled -> temporal dynamics matter")
    else:
        print("  [-] Observer doesn't outperform shuffled -> temporal order irrelevant")

    if observer_pos > wrong_pos:
        print("  [+] Observer outperforms wrong-executor -> executor specificity matters")
    else:
        print("  [-] Observer doesn't outperform wrong-executor -> no executor specificity")

    # Home run check
    all_positive = all(
        probe_results['summary']['indicators'].get(k, False)
        for k in ['self_model', 'surprise', 'temporal', 'first_thought',
                   'time_constants', 'phase_portrait', 'synchronization', 'phi']
    )
    all_controls_low = all(
        control_results.get(c, {}).get('summary', {}).get('positive_count', 0) <= 2
        for c in ['untrained', 'linear', 'shuffled']
    )

    if all_positive and all_controls_low:
        print("\n  *** HOME RUN: All key indicators positive, all controls negative ***")
        print("  Prediction alone produces consciousness indicators in liquid networks.")
    elif observer_pos > max(untrained_pos, linear_pos, shuffled_pos, wrong_pos):
        best_control = max(untrained_pos, linear_pos, shuffled_pos, wrong_pos)
        print(f"\n  Partial support: Observer ({observer_pos}) > best control ({best_control})")
    else:
        print("\n  No clear support for the hypothesis in this run.")

    return {'probes': probe_results, 'controls': control_results,
            'self_observation': self_obs_results, 'scaling': scaling_results}


def _run_self_observation(model, device):
    """Run the self-observation experiment."""
    from .observer_model import SelfObservingWrapper
    from .dataset import TrajectoryDataset
    from . import dynamics_analysis as dyn

    h5_path = str(config.TRAJECTORIES_PATH)
    train_ds = TrajectoryDataset(h5_path, split='train')
    test_ds = TrajectoryDataset(h5_path, split='test', precompute_stats=False)
    test_ds.set_normalization_stats(train_ds.neuron_means, train_ds.neuron_stds)

    # Create self-observing wrapper
    wrapper = SelfObservingWrapper(model, delay_steps=5).to(device)
    wrapper.eval()

    # Compare dynamics: normal observation vs self-observation
    normal_lyaps = []
    self_lyaps = []
    normal_phis = []
    self_phis = []

    max_samples = 10
    with torch.no_grad():
        for i in range(min(max_samples, len(test_ds))):
            input_hidden, _ = test_ds[i]
            input_t = input_hidden.unsqueeze(0).to(device)

            # Normal observation
            if hasattr(model, 'forward_with_states'):
                _, normal_h = model.forward_with_states(input_t)
                normal_h_np = normal_h.squeeze(0).cpu().numpy()
                normal_lyaps.append(dyn.compute_lyapunov_rosenstein(normal_h_np))
                normal_phis.append(dyn.compute_gaussian_phi(
                    normal_h_np[:-1], normal_h_np[1:], n_partitions=10))

            # Self-observation
            _, self_h = wrapper(input_t)
            self_h_np = self_h.squeeze(0).cpu().numpy()
            self_lyaps.append(dyn.compute_lyapunov_rosenstein(self_h_np))
            self_phis.append(dyn.compute_gaussian_phi(
                self_h_np[:-1], self_h_np[1:], n_partitions=10))

    results = {
        'normal_lyapunov_mean': float(np.mean(normal_lyaps)) if normal_lyaps else 0.0,
        'self_lyapunov_mean': float(np.mean(self_lyaps)) if self_lyaps else 0.0,
        'normal_phi_mean': float(np.mean(normal_phis)) if normal_phis else 0.0,
        'self_phi_mean': float(np.mean(self_phis)) if self_phis else 0.0,
        'self_observation_differs': False,
    }

    # Does self-observation change dynamics?
    if normal_lyaps and self_lyaps:
        t_stat, p_val = stats_ttest(normal_lyaps, self_lyaps)
        results['lyapunov_t_stat'] = float(t_stat)
        results['lyapunov_p_val'] = float(p_val)
        results['self_observation_differs'] = p_val < 0.05

    print(f"  Normal Lyapunov: {results['normal_lyapunov_mean']:.6f}")
    print(f"  Self Lyapunov:   {results['self_lyapunov_mean']:.6f}")
    print(f"  Normal Phi:      {results['normal_phi_mean']:.6f}")
    print(f"  Self Phi:        {results['self_phi_mean']:.6f}")
    print(f"  Differs:         {results['self_observation_differs']}")

    return results


def stats_ttest(a, b):
    """Safe t-test."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    return stats.ttest_ind(a, b)


def _run_scaling_analysis(device):
    """Run probes on each observer size."""
    from .probes import (
        probe_self_model, probe_surprise, probe_temporal_integration,
        probe_time_constants, probe_phi,
    )
    from .trainer import load_model

    results = {}
    h5_path = str(config.TRAJECTORIES_PATH)

    for size in config.OBSERVER_SIZES:
        if size == config.OBSERVER_DEFAULT_SIZE:
            path = str(config.CHECKPOINTS_DIR / 'observer.pt')
        else:
            path = str(config.CHECKPOINTS_DIR / f'observer_size{size}.pt')

        if not Path(path).exists():
            print(f"  Size {size}: checkpoint not found, skipping")
            continue

        print(f"\n--- Scaling: size={size} ---")
        model = load_model(path, device)

        size_results = {
            'self_model': probe_self_model(model, h5_path, device=device, max_samples=20),
            'surprise': probe_surprise(model, h5_path, device=device, max_samples=30),
            'temporal': probe_temporal_integration(model, h5_path, device=device, max_samples=20),
            'time_constants': probe_time_constants(model, h5_path, device=device, max_samples=20),
            'phi': probe_phi(model, h5_path, device=device, max_samples=5),
        }

        # Count positive indicators for this size
        indicators = {
            'self_model': size_results['self_model'].get('self_model_detected', False),
            'surprise': size_results['surprise'].get('surprise_is_meaningful', False),
            'temporal': size_results['temporal'].get('has_temporal_integration', False),
            'time_constants': size_results['time_constants'].get('has_adaptive_timescales', False),
            'phi': size_results['phi'].get('has_integration', False),
        }
        positive = sum(1 for v in indicators.values() if v)

        size_results['summary'] = {
            'positive_count': positive,
            'total_count': len(indicators),
            'indicators': {k: bool(v) for k, v in indicators.items()},
        }

        results[size] = size_results
        print(f"  Size {size}: {positive}/{len(indicators)} positive")

    return results


def _make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


# ── Main Entry Point ──────────────────────────────────────────────────

def run(extract=True, train=True, probe=True, device=None):
    """Run the full experiment pipeline."""
    device = device or config.DEVICE

    print("=" * 70)
    print("EXPERIMENT 6: LIQUID NEURAL NETWORK OBSERVER-EXECUTOR")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Executor: CfC {config.EXECUTOR_HIDDEN_SIZE} hidden neurons")
    print(f"Observer: CfC {config.OBSERVER_DEFAULT_SIZE} hidden neurons (default)")
    print(f"Data: {config.TOTAL_TRAJECTORIES} trajectories x {config.TRAJECTORY_LENGTH} timesteps")
    print(f"Systems: {', '.join(config.SYSTEM_TYPES)}")
    print()

    if extract:
        phase_extract(device)

    if train:
        phase_train(device)

    if probe:
        results = phase_probe(device)
    else:
        results = None

    print("\n" + "=" * 70)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 6: Liquid Neural Network Observer-Executor'
    )
    parser.add_argument('--extract-only', action='store_true',
                        help='Only generate systems and train executor')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train observers (executor must exist)')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only run probes (trained models must exist)')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (auto-detected if not set)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = config.DEVICE

    if args.extract_only:
        phase_extract(device)
    elif args.train_only:
        phase_train(device)
    elif args.probe_only:
        phase_probe(device)
    else:
        run(device=device)
