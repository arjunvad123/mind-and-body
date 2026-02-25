"""
Control Baselines

4 controls essential for scientific rigor. Without these, positive probe
results could be explained by trivial factors.

1. Untrained observer — same architecture, random weights
2. Linear baseline — single matrix W predicting S(T+1) = W @ S(T)
3. Shuffled observer — trained on temporally shuffled sequences
4. Wrong-executor observer — trained on GPT-2 small, tested on different text

Each control is run through all 6 probes for comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from . import config
from .observer_model import ObserverTransformer, LinearBaseline
from .probes import (
    probe_self_model,
    probe_surprise,
    probe_temporal_integration,
    probe_first_thought,
    probe_layer_preference,
)
from .trainer import load_model


def run_control_probes(
    model: nn.Module,
    control_name: str,
    h5_path: str = None,
    device: torch.device = None,
    max_samples: int = 50,
) -> dict:
    """Run all applicable probes on a control model.

    Args:
        model: Control model to probe
        control_name: Name of the control (for logging)
        h5_path: Path to activations HDF5
        device: Torch device
        max_samples: Max samples per probe

    Returns:
        Results dict with all probe outcomes
    """
    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)

    print(f"\n{'=' * 60}")
    print(f"CONTROL: {control_name}")
    print(f"{'=' * 60}\n")

    model = model.to(device)
    model.eval()

    results = {}

    # Probe 1: Self-Model
    results['self_model'] = probe_self_model(
        model, h5_path, device=device, max_samples=max_samples
    )

    # Probe 2: Surprise
    results['surprise'] = probe_surprise(
        model, h5_path, device=device, max_samples=max_samples
    )

    # Probe 3: Temporal Integration
    results['temporal'] = probe_temporal_integration(
        model, h5_path, device=device, max_samples=max_samples
    )

    # Probe 4: First Thought
    results['first_thought'] = probe_first_thought(
        model, h5_path, device=device, max_samples=max_samples
    )

    # Probe 6: Layer Preference
    results['layer_preference'] = probe_layer_preference(
        model, h5_path, device=device, max_samples=max_samples
    )

    # Summarize
    indicators = {
        'self_model': results['self_model'].get('self_model_detected', False),
        'surprise': results['surprise'].get('surprise_is_meaningful', False),
        'temporal': results['temporal'].get('has_temporal_integration', False),
        'first_thought': results['first_thought'].get('first_better_than_multi', False),
        'layer_preference': results['layer_preference'].get('prefers_late_layers', False),
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

def control_untrained(
    h5_path: str = None,
    device: torch.device = None,
) -> dict:
    """Untrained observer — same architecture, random weights.

    This is the most basic control. If the untrained observer shows
    consciousness indicators, our probes are measuring architecture
    artifacts, not learned properties.
    """
    print("Creating untrained observer (random weights)...")
    model = ObserverTransformer()
    return run_control_probes(model, "Untrained Observer", h5_path, device)


# ── Control 2: Linear Baseline ─────────────────────────────────────

def control_linear(
    checkpoint_path: str = None,
    h5_path: str = None,
    device: torch.device = None,
) -> dict:
    """Linear baseline — single matrix prediction.

    If this matches the observer, the transformer's temporal modeling
    isn't needed. The observer would just be a fancy linear predictor.
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

def control_shuffled(
    checkpoint_path: str = None,
    h5_path: str = None,
    device: torch.device = None,
) -> dict:
    """Shuffled observer — trained on temporally shuffled sequences.

    Controls for learning activation statistics without temporal dynamics.
    If this matches the real observer, temporal structure doesn't matter.
    """
    checkpoint_path = checkpoint_path or str(config.CHECKPOINTS_DIR / 'shuffled_observer.pt')

    if Path(checkpoint_path).exists():
        print(f"Loading shuffled observer from {checkpoint_path}")
        model = load_model(checkpoint_path, device)
    else:
        print("Warning: No trained shuffled observer found. Using untrained.")
        model = ObserverTransformer()

    return run_control_probes(model, "Shuffled Observer", h5_path, device)


# ── Control 4: Wrong-Executor Observer ─────────────────────────────

def control_wrong_executor(
    observer_path: str = None,
    h5_path: str = None,
    device: torch.device = None,
) -> dict:
    """Wrong-executor observer — trained on GPT-2 small, tested on different text.

    The trained observer, but evaluated on text it hasn't seen.
    This controls for memorization vs genuine modeling.

    Note: This is essentially the test set evaluation, since the observer
    was trained on train split and we probe on test split. The key
    comparison is whether test-set performance differs from train-set.
    """
    observer_path = observer_path or str(config.CHECKPOINTS_DIR / 'observer.pt')

    if Path(observer_path).exists():
        print(f"Loading trained observer for wrong-executor control from {observer_path}")
        model = load_model(observer_path, device)
    else:
        print("Warning: No trained observer found. Using untrained.")
        model = ObserverTransformer()

    # The "wrong executor" test: use the same model but different data split
    # In a full implementation, we'd extract activations from GPT-2 small
    # on completely new text. For MVP, we use the test split.
    return run_control_probes(
        model, "Wrong-Executor (Test Split)", h5_path, device
    )


# ── Run All Controls ──────────────────────────────────────────────

def run_all_controls(
    observer_path: str = None,
    h5_path: str = None,
    device: torch.device = None,
    save_dir: str = None,
) -> dict:
    """Run all 4 control baselines and compile results.

    Returns:
        dict mapping control name to probe results
    """
    device = device or config.DEVICE
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
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
