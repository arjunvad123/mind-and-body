# Technical Architecture

## Overview

All experiments follow a common pattern: train an **executor** on a task, then train an **observer** purely on predicting the executor's internal states, then **probe** the observer for emergent consciousness-like properties against **control** baselines.

## Shared Patterns

### Configuration (`config.py`)

Each experiment has a `config.py` that centralizes all hyperparameters, paths, and device selection. Paths use environment variables for cluster deployment (e.g., `EXP6_DATA_DIR`) with local fallbacks.

### Three-Phase Pipeline (`run_experiment.py`)

Experiments 5 and 6 follow an idempotent three-phase pipeline:

1. **Extract**: Generate data, train executor, extract hidden states to HDF5
2. **Train**: Train observer(s), baselines, seed variants for RSA
3. **Probe**: Run all probes and controls, save results as JSON

Each phase checks for existing output files before re-running, allowing restart from any point.

### Probe Design

Probes are pure evaluation functions that take a trained model and dataset, returning a dict with metrics and a boolean indicator. The standard signature:

```python
def probe_X(model, h5_path, device=None, max_samples=50) -> dict:
    # ... compute metrics ...
    return {
        'metric_name': value,
        'indicator_detected': bool,
    }
```

### Control Methodology

Four control baselines test whether observed properties are genuine:

| Control | What It Tests |
|---------|---------------|
| **Untrained** | Random-weight observer. Are properties architectural artifacts? |
| **Linear** | `h(t+1) = W @ h(t)`. Is temporal modeling needed? |
| **Shuffled** | Trained on time-scrambled data. Does temporal order matter? |
| **Wrong-Executor** | Trained on executor A, tested on executor B. Is the model executor-specific? |

## Experiment Architectures

### Experiment 1: RL

```
CartPole Environment → DQN Executor (2x64 FC) → State Packets → Transformer Observer (4L, 128d)
```

### Experiment 4: LLM

```
Coding Tasks → Claude Sonnet Executor (with tools) → Message Stream → Claude Sonnet Observer
```

### Experiment 5: Transformer on Transformer

```
Text (The Pile) → GPT-2 (12L, 768d) → 13 Residual Checkpoints → Transformer Observer (6L, 512d)
```

### Experiment 6: Liquid Neural Networks

```
8 Dynamical Systems → CfC Executor (64 hidden, NCP) → Hidden States → CfC Observer (50 hidden, NCP)
```

## Data Flow

All quantitative experiments (1, 5, 6) store extracted executor states in HDF5 format for efficient memory-mapped access during observer training. The observer never sees the raw task input -- only the executor's internal representations.

## Deployment

Experiments 5 and 6 include Kubernetes deployment configurations (`k8s/`) for GPU cluster execution:
- `pvc-*.yaml`: PersistentVolumeClaim for data persistence across pod restarts
- `job-*.yaml`: GPU job specification with resource requests and node affinity
- `deploy.sh`: Automated deployment script (create PVC, ConfigMap from code, submit job)
