# Experiment 6: Liquid Neural Network Observer

> **Status: Complete** -- Flagship experiment

The culmination of the project. Both executor and observer are Closed-form Continuous-time (CfC) neural networks -- liquid neural networks whose dynamics are governed by ordinary differential equations. The executor predicts next states across 8 diverse dynamical systems (chaotic attractors, pendulums, oscillators). The observer watches the executor's hidden states and is probed for 11 consciousness indicators, with mathematically precise "interest masks" that fix the surprise probe failure from Experiment 5.

## Architecture

- **Executor:** CfC network (64 hidden neurons, NCP wiring with sensory/inter/command/motor neuron groups). Takes 8D padded state vectors as input, predicts next state. Trained on 8 dynamical systems simultaneously.
- **Observer:** CfC network (50 hidden neurons). Receives the executor's hidden state vector at each timestep. Trained purely on next-hidden-state prediction. No access to executor weights, gradients, or training signals.
- **Key property:** Both networks operate via ODEs, giving them continuous-time dynamics with adaptive time constants -- unlike discrete Transformers.

## Dynamical Systems

8 systems spanning a range of complexity and dimensionality, all padded to uniform 8D input:

| System | Dimensionality | Character |
|--------|---------------|-----------|
| Lorenz attractor | 3D | Chaotic, sensitive dependence |
| Rossler attractor | 3D | Chaotic, intermittent z-spikes |
| Double pendulum | 4D | Chaotic, energy exchange between arms |
| Coupled oscillators | 4D | Quasi-periodic, energy transfer |
| Van der Pol oscillator | 2D | Limit cycle, relaxation oscillations |
| Damped sine | 1D | Simple decay |
| Step function | 1D | Discontinuous jumps |
| Logistic map | 1D | Period-doubling to chaos |

## Dataset

- 800 trajectories total (100 per system)
- 500 timesteps each, dt=0.02
- All states padded to 8D for uniform input to executor

## Key Innovation: Interest Masks

The surprise probe failed in Experiment 5 because "linguistically interesting" is subjective. Here, we define interest masks as boolean arrays marking mathematically precise moments:

- **Lorenz:** lobe switches (sign changes in x-coordinate)
- **Rossler:** z-spikes (z exceeds threshold)
- **Double pendulum:** energy exchanges between arms
- **Coupled oscillators:** resonance points
- **Logistic map:** period-doubling bifurcation points

This gives a ground-truth definition of "interesting" that is objective and verifiable.

## Results

### Probes

| Probe | Metric | Value | Verdict |
|-------|--------|-------|---------|
| Self-Model | own=0.0018, noise=0.12, other_exec=22.16 | Detected | + |
| Surprise | 5.14x ratio interesting vs boring, p<1e-15 | Meaningful | + |
| Temporal Integration | 24x ratio, plateau at window 64 | Present | + |
| First Thought | first=0.000423 vs multi=0.000602 | First better | + |
| Cross-Observer Preferences | RSA=0.0005 | Not convergent | - |
| Neuron Preference (NCP Group) | Motor neurons dominate (47/50) | Motor, not command | - |
| Time Constants | mean=74, std=227, std/mean=3.06 | Adaptive | + |
| Phase Portrait | System separability=0.0 | No structure | - |
| Lyapunov Exponents | lambda=-0.33 | Stable, not critical | - |
| Synchronization | Phase coherence=0.976, cross-corr=0.927 | Synchronized | + |
| Approximate Phi (IIT) | Phi=-6.0 | Negative | - |

**Score: 6/11 positive indicators**

### Controls

| Control | Positive Indicators | Key Difference from Trained |
|---------|--------------------|-----------------------------|
| Untrained | 4/10 | No self-model, no temporal integration |
| Linear | 4/10 | No self-model via noise test |
| Shuffled | 6/10 | MATCHES trained (temporal order doesn't matter) |
| Wrong-Executor | 3/10 | No self-model, no surprise |

### Scaling Experiment

All observer sizes (10, 20, 50, 100, 200 hidden neurons) scored 4/5 on core probes. Key scaling trends:

- Surprise ratio increases with size: 1.4x (10 neurons) to 3.4x (200 neurons)
- Tau-executor correlation strengthens with size: -0.11 (10) to -0.26 (200)
- Core probe passage is size-invariant -- even 10 neurons suffice

### Self-Observation Experiment

No significant difference between normal observation and reflexive self-observation (p=0.93). The observer does not behave differently when watching itself.

## Key Findings

1. **Surprise probe FIXED.** With mathematically precise interest masks, the observer shows a 5.14x ratio of prediction error at interesting vs boring moments (p<1e-15). The observer finds Lorenz lobe switches, Rossler z-spikes, and energy exchanges genuinely surprising -- its prediction model breaks at exactly the moments that are mathematically significant.

2. **Shuffled control matches trained observer.** Feeding temporal sequences in random order produces the same probe results (6/10 positive). This means the observer builds a statistical model of the executor's state space, not a temporal narrative. It learns "what states look like" rather than "what happens next."

3. **Wrong-executor control clearly fails.** An observer trained on one executor and tested on a different executor scores only 3/10, with no self-model and no meaningful surprise. Executor specificity is real -- the observer genuinely models its particular executor.

4. **Scaling is clean and monotonic.** More neurons produce stronger surprise signals and tighter tau-executor coupling, but even minimal observers (10 neurons) pass the core probes. The phenomenon is robust, not fragile.

5. **Cross-observer convergence dropped.** Unlike Experiment 5 (RSA=0.89), CfC observers do not converge on identical representations (RSA=0.0005). Liquid networks find diverse solutions to the same observation problem.

6. **Negative IIT result.** Approximate Phi is negative (-6.0), meaning the observer's information integration does not exceed what partitioned subsystems could achieve. This is notable because it suggests consciousness-like indicators can emerge without high integrated information.

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main entry point -- full pipeline from data generation through probing |
| `dynamical_systems.py` | All 8 dynamical systems with interest mask generation |
| `executor_model.py` | CfC executor with NCP wiring |
| `observer_model.py` | CfC observer architecture |
| `trainer.py` | Training loops for executor and observer |
| `dataset.py` | Dataset generation and loading |
| `probes.py` | All 11 probe implementations |
| `controls.py` | Control experiments (untrained, linear, shuffled, wrong-executor) |
| `dynamics_analysis.py` | Phase portrait, Lyapunov, synchronization analysis |
| `config.py` | Experiment configuration |
| `results/` | Saved probe results and analysis outputs |
| `k8s/` | Kubernetes deployment configurations |

## Prerequisites

- `ncps` package (Neural Circuit Policies / CfC networks)
- PyTorch
- NumPy, SciPy

## How to Run

Locally:
```bash
python -m experiment6_liquid_observer.run_experiment
```

On Kubernetes cluster:
```bash
./k8s/deploy.sh
```
