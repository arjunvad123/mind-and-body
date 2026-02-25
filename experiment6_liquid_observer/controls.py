"""
Control Baselines

4 controls essential for scientific rigor. Without these, positive probe
results could be explained by trivial factors.

1. Untrained observer — same architecture, random weights
2. Linear baseline — single matrix W predicting h(t+1) = W @ h(t)
3. Shuffled observer — trained on temporally shuffled sequences
4. Wrong-executor observer — trained on executor A, tested on executor B

Each control is run through all 11 probes for comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from . import config
from .observer_model import LiquidObserver, LinearBaseline
from .probes import (
    probe_self_model,
    probe_surprise,
    probe_temporal_integration,
    probe_first_thought,
    probe_neuron_preference,
    probe_time_constants,
    probe_phase_portrait,
    probe_lyapunov,
    probe_synchronization,
    probe_phi,
)
from .trainer import load_model


def run_control_probes(model, control_name, h5_path=None, device=None, max_samples=50):
    """Run all applicable probes on a control model.

    Args:
        model: Control model to probe
        control_name: Name of the control (for logging)
        h5_path: Path to hidden state HDF5
        device: Torch device
        max_samples: Max samples per probe

    Returns:
        Results dict with all probe outcomes
    """
    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)

    print(f"\n{'=' * 60}")
    print(f"CONTROL: {control_name}")
    print(f"{'=' * 60}\n")

    model = model.to(device)
    model.eval()

    results = {}

    # Original 6 probes (minus RSA which needs multiple models)
    results['self_model'] = probe_self_model(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['surprise'] = probe_surprise(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['temporal'] = probe_temporal_integration(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['first_thought'] = probe_first_thought(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['neuron_preference'] = probe_neuron_preference(
        model, h5_path, device=device, max_samples=max_samples
    )

    # LNN-specific probes (descriptive — run on controls too)
    results['time_constants'] = probe_time_constants(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['phase_portrait'] = probe_phase_portrait(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['lyapunov'] = probe_lyapunov(
        model, h5_path, device=device, max_samples=min(10, max_samples)
    )
    results['synchronization'] = probe_synchronization(
        model, h5_path, device=device, max_samples=max_samples
    )
    results['phi'] = probe_phi(
        model, h5_path, device=device, max_samples=min(5, max_samples)
    )

    # Summarize
    indicators = {
        'self_model': results['self_model'].get('self_model_detected', False),
        'surprise': results['surprise'].get('surprise_is_meaningful', False),
        'temporal': results['temporal'].get('has_temporal_integration', False),
        'first_thought': results['first_thought'].get('first_better_than_multi', False),
        'neuron_preference': results['neuron_preference'].get('prefers_command', False),
        'time_constants': results['time_constants'].get('has_adaptive_timescales', False),
        'phase_portrait': results['phase_portrait'].get('has_structured_dynamics', False),
        'lyapunov': results['lyapunov'].get('is_near_critical', False),
        'synchronization': results['synchronization'].get('has_synchronization', False),
        'phi': results['phi'].get('has_integration', False),
    }

    positive = sum(1 for v in indicators.values() if v)
    results['summary'] = {
        'control_name': control_name,
        'indicators': {k: bool(v) for k, v in indicators.items()},
        'positive_count': positive,
        'total_count': len(indicators),
    }

    print(f"\n  {control_name} positive indicators: {positive}/{len(indicators)}")

    return results


# ── Control 1: Untrained Observer ──────────────────────────────────

def control_untrained(h5_path=None, device=None):
    """Untrained observer — same architecture, random weights.

    If the untrained observer shows indicators, our probes are
    measuring architecture artifacts, not learned properties.
    """
    print("Creating untrained observer (random weights)...")
    model = LiquidObserver()
    return run_control_probes(model, "Untrained Observer", h5_path, device)


# ── Control 2: Linear Baseline ─────────────────────────────────────

def control_linear(checkpoint_path=None, h5_path=None, device=None):
    """Linear baseline — single matrix prediction.

    If this matches the observer, CfC temporal modeling isn't needed.
    """
    checkpoint_path = checkpoint_path or str(config.CHECKPOINTS_DIR / 'linear_baseline.pt')

    if Path(checkpoint_path).exists():
        print(f"Loading trained linear baseline from {checkpoint_path}")
        model = load_model(checkpoint_path, device)
    else:
        print("Warning: No trained linear baseline found. Using untrained.")
        model = LinearBaseline()

    return run_control_probes(model, "Linear Baseline", h5_path, device)


# ── Control 3: Shuffled Observer ───────────────────────────────────

def control_shuffled(checkpoint_path=None, h5_path=None, device=None):
    """Shuffled observer — trained on temporally shuffled sequences.

    If this matches the real observer, temporal dynamics don't matter.
    """
    checkpoint_path = checkpoint_path or str(config.CHECKPOINTS_DIR / 'shuffled_observer.pt')

    if Path(checkpoint_path).exists():
        print(f"Loading shuffled observer from {checkpoint_path}")
        model = load_model(checkpoint_path, device)
    else:
        print("Warning: No trained shuffled observer found. Using untrained.")
        model = LiquidObserver()

    return run_control_probes(model, "Shuffled Observer", h5_path, device)


# ── Control 4: Wrong-Executor Observer ─────────────────────────────

def control_wrong_executor(observer_path=None, h5_path=None, device=None):
    """Wrong-executor observer — trained on executor A, tested on executor B.

    TRUE cross-executor test: uses hidden states from a different executor
    (trained with different random seed).
    """
    observer_path = observer_path or str(config.CHECKPOINTS_DIR / 'observer.pt')

    if Path(observer_path).exists():
        print(f"Loading trained observer for wrong-executor control from {observer_path}")
        model = load_model(observer_path, device)
    else:
        print("Warning: No trained observer found. Using untrained.")
        model = LiquidObserver()

    # Use secondary executor's hidden states
    secondary_h5 = str(config.TRAJECTORIES_SECONDARY_PATH)
    if not Path(secondary_h5).exists():
        print("Warning: No secondary executor trajectories. Using primary as fallback.")
        secondary_h5 = h5_path or str(config.TRAJECTORIES_PATH)

    return run_control_probes(
        model, "Wrong-Executor (Different Seed)", secondary_h5, device
    )


# ── Run All Controls ──────────────────────────────────────────────

def run_all_controls(observer_path=None, h5_path=None, device=None, save_dir=None):
    """Run all 4 control baselines and compile results."""
    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_dir = save_dir or str(config.RESULTS_DIR)

    print("\n" + "=" * 70)
    print("RUNNING ALL CONTROL BASELINES")
    print("=" * 70)

    all_results = {}

    # Control 1: Untrained
    all_results['untrained'] = control_untrained(h5_path, device)

    # Control 2: Linear
    all_results['linear'] = control_linear(h5_path=h5_path, device=device)

    # Control 3: Shuffled
    all_results['shuffled'] = control_shuffled(h5_path=h5_path, device=device)

    # Control 4: Wrong-executor
    all_results['wrong_executor'] = control_wrong_executor(
        observer_path=observer_path, h5_path=h5_path, device=device
    )

    # ── Comparison Summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONTROL COMPARISON SUMMARY")
    print("=" * 60)

    for name, results in all_results.items():
        summary = results.get('summary', {})
        pos = summary.get('positive_count', 0)
        total = summary.get('total_count', 0)
        print(f"  {name:>25}: {pos}/{total} positive indicators")

    # Save
    if save_dir:
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

        output_file = save_path / 'control_results.json'
        with open(output_file, 'w') as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"\nControl results saved to {output_file}")

    return all_results
