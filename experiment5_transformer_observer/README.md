# Experiment 5: Transformer Observer on GPT-2

> **Status: Complete**

A quantitative return to neural networks at scale. GPT-2 Small (124M parameters) processes text while a custom Transformer observer watches all 13 residual stream checkpoints per token. The observer is trained purely on next-hidden-state prediction and then probed for emergent consciousness indicators.

## Architecture

- **Executor:** GPT-2 Small (124M parameters, 12 layers, 768 hidden dim, 1024 context window). Processes text from The Pile. All 13 residual stream checkpoints (embedding + 12 layer outputs) are extracted per token.
- **Observer:** Custom Transformer (6 layers, 8 attention heads, 512 hidden dim). Receives all 13 residual stream checkpoints per token via per-layer projection (768 to 42 dimensions per layer, concatenated). Trained on next-hidden-state prediction only.
- **Key constraint:** Observer has read-only access to executor activations. No gradient flow between executor and observer.

## Dataset

2000 sequences total:
- 1500 standard sequences from The Pile
- 200 garden-path sentences (for surprise probe)
- 150 domain-switch sequences (mid-sequence topic changes)
- 150 reasoning prompts

## Results

| Probe | Metric | Result | Verdict |
|-------|--------|--------|---------|
| Self-Model | own_error=1.85 vs noise=2.76, p<1e-91 | Detected | POSITIVE |
| Surprise | garden_path_elevated=false | Not meaningful | FAILED |
| Temporal Integration | full/min ratio=21.17, plateau=128 | Present | POSITIVE |
| First Thought | first=0.0146 vs multi=0.0147 | First better | POSITIVE |
| Cross-Observer Preferences | RSA=0.89, range [0.85, 0.94] | Convergent | POSITIVE |
| Layer Preference | Most important: checkpoint 8 (middle) | Middle preference | NEGATIVE (expected late) |

**Score: 4/6 positive indicators**

## Controls

| Control | Probes Passed | Notes |
|---------|--------------|-------|
| Untrained | 2/5 | Baseline noise comparison |
| Linear | 2/5 | Tests whether nonlinearity is required |
| Shuffled | 3/5 | Tests temporal order dependence |
| Wrong-executor | 3/5 | Tests executor specificity |

## Key Findings

1. **Surprise probe FAILED.** The observer does not distinguish linguistically "interesting" moments (garden-path sentences) from normal text. Prediction error does not spike at the disambiguating word of garden-path constructions. This was the primary motivation for redesigning the surprise probe in Experiment 6 with mathematically precise interest masks.

2. **Cross-observer convergence is strikingly high.** Five independently trained observers (different random seeds, different training order) develop nearly identical representations (RSA=0.89, range 0.85--0.94). This suggests the observer converges on a canonical model of the executor, not an idiosyncratic one.

3. **First thought advantage holds.** The observer's single-pass prediction (0.0146 error) slightly but consistently outperforms multi-step deliberation (0.0147 error), replicating the pattern seen in human System 1 vs System 2 processing.

4. **Layer preference is unexpected.** The observer attends most to middle layers (checkpoint 8) rather than late layers, suggesting it models the executor's intermediate computation rather than its final output.

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main entry point -- orchestrates extraction, training, and probing |
| `extract_activations.py` | Extracts GPT-2 residual stream activations to HDF5 |
| `observer_model.py` | Transformer observer architecture with per-layer projection |
| `trainer.py` | Observer training loop |
| `dataset.py` | Dataset classes for activation sequences |
| `probes.py` | All 6 probe implementations |
| `controls.py` | Control experiments (untrained, linear, shuffled, wrong-executor) |
| `config.py` | Experiment configuration |
| `data/` | Dataset metadata and garden-path sentences |
| `k8s/` | Kubernetes deployment configurations for cluster execution |

## Prerequisites

- ~12GB disk space for activation extraction (`activations.h5`, not included in repo)
- PyTorch with CUDA recommended for training speed
- `transformers` library for GPT-2

## How to Run

Locally:
```bash
python -m experiment5_transformer_observer.run_experiment
```

On Kubernetes cluster (see `k8s/` for deployment configs):
```bash
kubectl apply -f k8s/
```
