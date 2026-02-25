"""
Experiment 5: Low-Level Transformer Observer
=============================================

Full pipeline entry point.

A small observer transformer watches GPT-2's internal activations
token-by-token, layer-by-layer, trained purely on prediction. Then
we probe for emergent consciousness-like properties.

Phases:
    1. Extract — Run GPT-2, cache activations to HDF5
    2. Train  — Train observer + linear baseline + shuffled control
    3. Probe  — Run 6 consciousness probes + 4 controls

Usage:
    # Full pipeline
    python -m experiment5_transformer_observer.run_experiment

    # Individual phases
    python -m experiment5_transformer_observer.run_experiment --extract-only
    python -m experiment5_transformer_observer.run_experiment --train-only
    python -m experiment5_transformer_observer.run_experiment --probe-only
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from . import config


def phase_extract(device: torch.device = None):
    """Phase 1: Extract GPT-2 activations to HDF5."""
    from .extract_activations import extract_activations, validate_activations

    print("=" * 70)
    print("PHASE 1: ACTIVATION EXTRACTION")
    print("=" * 70)

    # Extract primary executor (GPT-2 small)
    extract_activations(
        model_name=config.EXECUTOR_MODEL,
        output_path=str(config.ACTIVATIONS_PATH),
        device=device,
    )

    # Validate
    valid = validate_activations(
        activations_path=str(config.ACTIVATIONS_PATH),
        model_name=config.EXECUTOR_MODEL,
    )

    if not valid:
        print("WARNING: Activation validation failed!")

    # Extract GPT-2 medium activations for Probe 1 (self-model discrimination)
    print("\nExtracting GPT-2 Medium activations for self-model probe...")
    try:
        extract_activations(
            model_name=config.EXECUTOR_MODEL_MEDIUM,
            output_path=str(config.ACTIVATIONS_MEDIUM_PATH),
            n_sequences=200,  # Fewer sequences needed for comparison
            device=device,
            save_attention=False,
        )
    except Exception as e:
        print(f"Warning: Could not extract GPT-2 Medium activations: {e}")
        print("Probe 1 will run without cross-executor comparison.")


def phase_train(device: torch.device = None):
    """Phase 2: Train observer, linear baseline, and shuffled control."""
    from .trainer import (
        train_observer,
        train_linear_baseline,
        train_shuffled_observer,
        train_seed_observers,
    )

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

    # Train seed observers (for Probe 5)
    print("\n--- Seed Observers (for RSA) ---")
    seed_results = train_seed_observers(device=device)

    # Save training summary
    summary = {
        'observer_final_train_loss': observer_result['history']['train_loss'][-1],
        'observer_final_val_loss': observer_result['history']['val_loss'][-1] if observer_result['history']['val_loss'] else None,
        'linear_final_train_loss': linear_result['history']['train_loss'][-1],
        'shuffled_final_train_loss': shuffled_result['history']['train_loss'][-1],
        'n_seed_observers': len(seed_results),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 60)
    print("Training Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def phase_probe(device: torch.device = None):
    """Phase 3: Run consciousness probes and controls."""
    from .probes import run_all_probes
    from .controls import run_all_controls
    from .trainer import load_model

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

    # Check for GPT-2 medium activations
    other_h5 = str(config.ACTIVATIONS_MEDIUM_PATH) if config.ACTIVATIONS_MEDIUM_PATH.exists() else None

    # Run probes on trained observer
    probe_results = run_all_probes(
        model=model,
        h5_path=str(config.ACTIVATIONS_PATH),
        other_h5_path=other_h5,
        seed_observer_paths=seed_paths if seed_paths else None,
        device=device,
        save_dir=str(config.RESULTS_DIR),
    )

    # Run all controls
    control_results = run_all_controls(
        observer_path=observer_path,
        h5_path=str(config.ACTIVATIONS_PATH),
        device=device,
        save_dir=str(config.RESULTS_DIR),
    )

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

    if observer_pos > untrained_pos:
        print("  [+] Observer outperforms untrained → training matters")
    else:
        print("  [-] Observer doesn't outperform untrained → probes may be artifacts")

    if observer_pos > linear_pos:
        print("  [+] Observer outperforms linear baseline → temporal modeling matters")
    else:
        print("  [-] Observer doesn't outperform linear → temporal modeling unnecessary")

    if observer_pos > shuffled_pos:
        print("  [+] Observer outperforms shuffled → temporal dynamics matter")
    else:
        print("  [-] Observer doesn't outperform shuffled → temporal order irrelevant")

    # Home run check
    all_positive = all(
        probe_results['summary']['indicators'].get(k, False)
        for k in ['self_model', 'surprise', 'temporal', 'first_thought']
    )
    all_controls_low = all(
        control_results.get(c, {}).get('summary', {}).get('positive_count', 0) <= 1
        for c in ['untrained', 'linear', 'shuffled']
    )

    if all_positive and all_controls_low:
        print("\n  *** HOME RUN: All indicators positive, all controls negative ***")
        print("  Prediction alone produces consciousness indicators.")
    elif observer_pos > max(untrained_pos, linear_pos, shuffled_pos):
        print(f"\n  Partial support: Observer ({observer_pos}) > best control ({max(untrained_pos, linear_pos, shuffled_pos)})")
    else:
        print("\n  No clear support for the hypothesis in this run.")

    return {'probes': probe_results, 'controls': control_results}


def run(
    extract: bool = True,
    train: bool = True,
    probe: bool = True,
    device: torch.device = None,
):
    """Run the full experiment pipeline."""
    device = device or config.DEVICE

    print("=" * 70)
    print("EXPERIMENT 5: LOW-LEVEL TRANSFORMER OBSERVER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Executor: {config.EXECUTOR_MODEL} ({config.EXECUTOR_D_MODEL}d, {config.EXECUTOR_N_LAYERS} layers)")
    print(f"Observer: {config.OBSERVER_D_MODEL}d, {config.OBSERVER_N_LAYERS} layers, {config.OBSERVER_N_HEADS} heads")
    print(f"Data: {config.N_SEQUENCES} sequences x {config.SEQ_LEN} tokens")
    print(f"Training: {config.N_EPOCHS} epochs, batch {config.BATCH_SIZE}")
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
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 5: Low-Level Transformer Observer'
    )
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract activations')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train observer (activations must exist)')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only run probes (trained model must exist)')
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
