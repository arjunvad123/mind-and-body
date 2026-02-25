# Experiment 1: Executor-Observer Separation

> **Status: Complete**

The foundational experiment. A DQN executor learns CartPole-v1 while a Transformer observer watches its internal state stream with no write access. We probe the observer for emergent properties that map onto consciousness indicators.

## Architecture

- **Executor:** DQN with 2 hidden layers (64 units each) trained on CartPole-v1 (Gymnasium). Activations are tapped at every layer and streamed to the observer.
- **Observer:** Transformer (4 layers, 128 hidden dim) with causal attention mask. Receives state packets containing layer activations, output logits, actions taken, environment observations, rewards, and episode metadata.
- **Key constraint:** The observer has NO write access to the executor. It receives a read-only stream of state packets and is trained purely on next-state prediction (MSE loss on predicted next action and next observation). All measured properties are emergent.

## Results

| Probe | Result | Verdict |
|-------|--------|---------|
| Self-Model (RSA) | RSA=0.53, within-self=0.995, between=0.465 | POSITIVE |
| Surprise | Mean 0.974, zero variance (constant reward=1.0) | INCONCLUSIVE |
| Temporal Integration | Recency bias 1.015, no present window | NEGATIVE |
| Emergent Preferences | R²=1.0 (confounded -- reward in input) | CONFOUNDED |

## Key Finding

A self-model emerged (the observer develops distinct representations for "own" vs "other" executors, RSA=0.53), but CartPole is too simple to test the remaining probes meaningfully. Constant rewards (always 1.0 until failure) produce zero variance in the surprise signal. Trivial dynamics prevent any temporal structure from forming. The emergent preferences probe was confounded because reward was included in the observer's input stream.

This motivated the pivot to richer environments in later experiments -- first LLMs (Experiment 4), then GPT-2 language modeling (Experiment 5), and finally diverse dynamical systems (Experiment 6).

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main entry point -- trains executor, collects data, trains observer, runs probes |
| `executor/` | DQN agent and CartPole training loop |
| `observer/` | Transformer observer model and training |
| `analysis/` | Probe implementations (self-model RSA, surprise, temporal, preferences) |
| `shared/` | Shared utilities and state packet definitions |
| `data/` | Collected state stream datasets |
| `DESIGN.md` | Detailed design documentation and theoretical motivation |

## How to Run

```bash
python -m experiment1_separation.run_experiment
```

## Design Documentation

See [DESIGN.md](DESIGN.md) for the full design brainstorm, including the theoretical framework, planned embodiment gradient (CartPole through MuJoCo), all six consciousness indicator probes, and the training protocol.
