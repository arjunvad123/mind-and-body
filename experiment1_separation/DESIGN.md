# Experiment 1: Executor-Observer Separation

## Overview

The foundational experiment. We train an RL agent (executor) to perform tasks, then
train a separate model (observer) that receives a read-only stream of the executor's
internal states, actions, and environmental feedback. We then probe the observer for
emergent properties that map onto consciousness indicators from the scientific literature.

## Deep Design Brainstorm

### Part A: The Executor

**What should the executor do?**

We need tasks with enough complexity that the executor develops interesting internal
representations, but not so complex that training becomes the bottleneck. We'll use a
**progression of environments:**

**Stage 1: CartPole-v1 (Gymnasium)**
- Simple: balance a pole on a cart
- Why: Minimal, well-understood. The observer has to model a simple policy.
  Baseline for "does this work at all?"
- Executor: DQN or PPO, small network (2 hidden layers, 64 units each)
- Internal state dimensionality: ~128 activations per step

**Stage 2: LunarLander-v3 (Gymnasium)**
- More complex: land a spacecraft with thrust control
- Why: Continuous state space, richer dynamics, landing requires planning.
  Observer must model goal-directed sequential behavior.
- Executor: PPO, larger network (3 layers, 128 units)
- Internal state dimensionality: ~384 activations per step

**Stage 3: HalfCheetah-v5 or Ant-v5 (MuJoCo)**
- Complex: Control a multi-jointed body through physics simulation
- Why: This is where embodiment enters. The executor must coordinate multiple
  actuators, handle balance, momentum, ground contact. The observer must model
  a "body."
- Executor: PPO or SAC, larger network (3 layers, 256 units)
- Internal state dimensionality: ~768 activations per step

**Stage 4 (stretch): Multi-agent (PettingZoo)**
- Why: Social dynamics. The observer must model not just the executor but the
  executor's model of other agents. Theory-of-mind-like properties.

**Executor architecture:**
```
Observation → Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, act_dim)
                    ↓ tap                    ↓ tap                  ↓ tap
              [layer1_acts]            [layer2_acts]           [output_logits]
```

We tap activations at every layer and stream them to the observer.

### Part B: The Observer

**What does the observer receive?**

At each timestep t, the observer gets a "state packet":
```python
state_packet_t = {
    "executor_layer1": tensor[64],       # First hidden layer activations
    "executor_layer2": tensor[64],       # Second hidden layer activations
    "executor_output": tensor[act_dim],  # Output logits/action probs
    "action_taken": int or tensor,       # What the executor actually did
    "env_observation": tensor[obs_dim],  # What the environment looks like
    "reward": float,                     # What reward was received
    "done": bool,                        # Whether episode ended
    "timestep": int                      # Position in episode
}
```

**What does the observer NOT receive?**
- The executor's weights or gradients
- The executor's loss function
- The executor's training state
- Any ability to modify the executor's behavior

**Observer architecture:**

The observer is a **sequence model** over state packets. It processes a sliding window
of recent packets and produces:

1. **Next-state prediction:** What will the executor do next? (prediction head)
2. **Executor state summary:** What "state" is the executor in? (compression head)
3. **Narrative generation:** If asked, describe what the executor is doing and why (language head — optional, for probing)

```
state_packet_1, state_packet_2, ..., state_packet_T
        ↓              ↓                    ↓
   [embed]         [embed]             [embed]
        ↓              ↓                    ↓
   ┌──────────────────────────────────────────┐
   │        Transformer (4 layers, 128 dim)   │
   │        with causal attention mask         │
   └──────────────────────────────────────────┘
        ↓              ↓                    ↓
   [hidden_1]     [hidden_2]          [hidden_T]
        │              │                    │
        │              │              ┌─────┼──────┐
        │              │              ↓     ↓      ↓
        │              │          predict  state  narrate
        │              │          next     summary (optional)
```

**Observer training objective:**

The observer is trained ONLY on prediction. No reward signal. No RL. Just:
- **Loss = MSE(predicted_next_action, actual_next_action) + MSE(predicted_next_obs, actual_next_obs)**
- Optionally: predict executor's internal states at t+1 from states at t

The observer gets good at predicting the executor. Everything else we measure is
**emergent** — not trained for.

### Part C: What We Measure (Consciousness Indicators)

This is the crucial part. We don't just build the system — we probe it for specific
indicators derived from consciousness theories.

**Indicator 1: Self-Model Formation (HOT / AST)**

Does the observer develop an internal model of the executor that functions as a
self-model?

*Test:* After training, freeze the observer. Present it with state packets from:
- (a) The original executor
- (b) A different executor (same task, different policy)
- (c) A random policy
- (d) The original executor in a different environment

If the observer has a self-model, it should:
- Predict (a) well, (b) somewhat, (c) poorly, (d) poorly but with systematic errors
  that reveal assumptions about the executor's capabilities
- Show different internal activation patterns for (a) vs (b) — "this is me" vs
  "this is not me" (analyzed via representation similarity analysis)

**Indicator 2: Surprise / Salience (Predictive Processing)**

Does the observer have meaningful "surprise" responses?

*Test:* Measure prediction error at each timestep. Then:
- Identify high-surprise moments (large prediction errors)
- Do these correlate with genuinely important moments (reward changes, near-failures,
  strategy shifts)?
- If yes, the observer's surprise is semantically meaningful, not just noise.

*Advanced test:* Do the observer's internal representations change dimensionality
or structure during high-surprise moments? (Analogous to how human consciousness
"expands" during unexpected events — pupil dilation, increased neural complexity.)

**Indicator 3: Temporal Integration (GWT)**

Does the observer integrate information across time in a way that creates a
"specious present" — a window of experienced now?

*Test:* Analyze the observer's attention patterns. Does it attend to a coherent
window of recent states (like a moving present), or does it use all timesteps
uniformly?

*Test:* Ablate different parts of the sequence. How much does removing recent
history degrade performance vs. removing distant history? A steep recency
gradient suggests a "present-focused" awareness.

**Indicator 4: Information Integration (IIT-inspired)**

Does the observer integrate information in ways that exceed its parts?

*Test:* Measure whether the observer's predictions are better than could be
achieved by independent sub-networks processing each stream (executor states,
actions, rewards) separately. If the whole observer significantly outperforms
the sum of its parts, information is being integrated.

**Indicator 5: Spontaneous Preferences (Valence)**

Does the observer develop preferences about the executor's states, even though
the observer receives no rewards?

*Test:* After training, present the observer with two possible future trajectories
for the executor (one leading to high reward, one to low). Measure whether the
observer's internal states show systematic differences that correlate with the
executor's reward — even though the observer was never trained on reward prediction
explicitly.

*Advanced:* Give the observer a choice (in a probe task, not during training) of
which executor trajectory to "watch." Does it preferentially choose high-reward
trajectories? This would be an emergent preference — observational valence.

**Indicator 6: Counterfactual Reasoning (Agency-adjacent)**

Does the observer model what the executor COULD HAVE done but didn't?

*Test:* After training, probe the observer's representations at decision points.
Use linear probes to check if the observer represents not just the action taken
but also the actions NOT taken and their likely outcomes. This is the minimal
structure of "I could have done otherwise" — even though the observer can't
actually intervene.

### Part D: Training Protocol

**Phase 1: Train the executor**
- Train to convergence on each environment
- Save the trained executor and freeze it
- This takes ~30 min for CartPole, ~2 hrs for MuJoCo

**Phase 2: Collect state streams**
- Run the frozen executor for N episodes
- Record complete state packets at every timestep
- Store as datasets (HDF5 or similar)
- CartPole: ~100K timesteps; MuJoCo: ~1M timesteps

**Phase 3: Train the observer**
- Train the observer on the state stream datasets
- Next-state prediction objective only
- Train to convergence
- Save checkpoints for analysis

**Phase 4: Probe**
- Run all consciousness indicator tests
- Generate analysis reports
- Compare across environments (embodiment gradient)

### Part E: What Success Looks Like

We're not claiming we'll "create consciousness." We're testing specific, measurable
predictions of the Observer Hypothesis:

| Prediction | Measurable Outcome |
|---|---|
| Observer develops executor-model | RSA shows distinct representations for "own" vs "other" executors |
| Surprise is semantically meaningful | High prediction-error moments correlate with task-relevant events |
| Temporal integration exists | Attention patterns show coherent "present window" |
| Information integration exceeds parts | Whole observer outperforms sum of stream-specific sub-networks |
| Emergent preferences appear | Observer states correlate with executor rewards without reward training |
| Counterfactual representations form | Linear probes find unchosen-action representations |

If most of these hold, especially across the embodiment gradient, the Observer
Hypothesis has legs. If they fail, we learn something equally valuable about what
observation alone does and doesn't produce.
