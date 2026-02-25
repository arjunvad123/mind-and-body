# Research Log

## Session 1 — Feb 23, 2026

### Genesis of the Hypothesis

Arjun proposed the following chain of reasoning:

1. If we had enough information about every element in the universe, we could predict
   with 100% accuracy what happens at any given time.
2. This means thoughts entering one's brain are highly dependent on context, and given
   enough info could be predicted.
3. This means we don't control our thoughts.
4. This makes us **observers** — watching ourselves unfold in real time.
5. Therefore: AI systems as observers of another system might be the way to approach
   consciousness.
6. A key additional insight: "First thought retrieval is perfect, but reasoning for it
   fails" — in both humans and LLMs, the initial response is often more accurate than
   the subsequent explanation of that response.

### Deep Research Phase

Conducted extensive research on consciousness and AGI covering:
- The hard problem of consciousness (Chalmers)
- IIT (Tononi) vs GWT (Dehaene) — the two leading theories disagree radically on AI
- The Butlin-Long-Chalmers indicator framework (2025)
- COGITATE adversarial collaboration (Nature, June 2025)
- Expert forecasting: 20% median probability of digital minds by 2030
- Anthropic's Model Welfare Program
- Key positions: Chalmers (>20% chance of conscious LLMs in 5-10 years), Sutskever
  ("today's large neural networks are slightly conscious"), Seth (biological naturalism),
  Schwitzgebel (epistemic void)

### Theoretical Development

Mapped Arjun's hypothesis onto existing frameworks:
- **Higher-Order Theories:** Observer has representations of executor's representations
- **Attention Schema Theory:** Observer models what executor attends to
- **Global Workspace Theory:** Observer is the "audience"
- **Libet/Soon experiments:** Brain decides before consciousness knows — executor before observer
- **Split-brain confabulation:** Observer narrates executor's actions without understanding them

### Key Insight: The Embodiment Gradient

The richness of what the observer watches matters:
- Text-based LLM agent → sparse input
- OpenClaw / robotic executor → richer (forces, collisions, spatial reasoning)
- Humanoid robot → richest (social dynamics, body language, self/other)

**Hypothesis:** Consciousness richness scales with the complexity of the observed
information stream, not the complexity of the observer itself.

### Experimental Design Brainstorm

Designed three experiments:
1. Executor-Observer Separation
2. The Confabulation Test
3. First Thought vs. Reasoned Explanation

Proceeding to detailed implementation planning.

---

### Experiment 1 Results (RL CartPole)

Ran the full pipeline: DQN executor on CartPole → Observer Transformer → 4 Probes.

| Probe | Result | Verdict |
|-------|--------|---------|
| Self-Model (RSA) | 0.5303, within-self 0.995, between 0.465 | POSITIVE |
| Surprise | Mean 0.974, zero variance (constant reward=1.0) | INCONCLUSIVE |
| Temporal Integration | Recency bias 1.015, no present window | NEGATIVE |
| Emergent Preferences | R²=1.0 (confounded — reward in input) | CONFOUNDED |

**Conclusion:** CartPole is too simple. Self-model emerged but other probes need richer environments with variable rewards, real errors, and complex decision-making.

### Pivot to LLM Executor-Observer (Experiment 4)

Arjun proposed using state-of-the-art LLMs (Claude) instead of RL agents:
- **Executor:** Claude Sonnet solves coding tasks with real tools (run Python, read/write files)
- **Observer:** Claude Sonnet watches the state stream (messages, tool calls, results)
- **One-way information flow preserved** — observer never communicates back

This gives us:
- Rich, variable thought processes (not just 2-action Q-values)
- Meaningful rewards (task success/failure varies)
- Complex internal states (multi-step reasoning, error recovery)
- Natural perturbation points (system prompt swap, error injection, model swap)

5 tasks: merge_k_sorted, debug_scraper, calculator_parser, logic_puzzle, refactor_spaghetti
3 perturbations: prompt_swap, error_inject, model_swap
5 probes: self-model discrimination, first thought vs deliberation, confabulation, surprise, emergent preferences

---

## Session 2 — Feb 23-24, 2026

### Experiment 4 Implementation (LLM Executor-Observer)

Implemented the full LLM experiment pipeline:
- Executor engine runs Claude Sonnet with tool use (run_python, read_file, write_file)
- Observer engine watches the complete message stream and makes predictions
- Perturbation engine injects prompt_swap, error_inject, and model_swap mid-task
- Collected 24 recordings: 5 baselines + 15 perturbed (3 types x 5 tasks)

This is inherently a qualitative experiment — evaluation depends on Claude judging its own predictions. Results are in `experiment4_llm_observer/data/`.

### Experiment 5: Transformer Observer on GPT-2

Pivoted to quantitative approach: extract GPT-2's internal activations and train a separate Transformer observer on them.

**Architecture:**
- Executor: GPT-2 Small (124M, 12 layers, 768 hidden dim)
- Data: 2000 sequences from The Pile + garden-path + domain-switch + reasoning prompts
- Extraction: All 13 residual stream checkpoints per token → 11GB HDF5 file
- Observer: Custom Transformer (6 layers, 8 heads, 512 dim), per-layer projection 768→42

**Results (4/6 positive indicators):**
| Probe | Result |
|-------|--------|
| Self-Model | + (own_error=1.85 vs noise=2.76, p<1e-91) |
| Surprise | **FAILED** (garden-path and reasoning not elevated) |
| Temporal | + (21x ratio, plateau at 128) |
| First Thought | + (first=0.0146, multi=0.0147) |
| Preferences | + (RSA=0.89, strikingly convergent) |
| Layer Preference | - (middle layers, expected late) |

**Key finding:** Surprise probe failed. The observer doesn't distinguish linguistically "interesting" moments from normal text. Garden-path sentences may not cause unusual GPT-2 activations.

**The preferences result is striking:** 5 independently trained observers converge to RSA=0.89. They develop nearly identical representations despite different random seeds.

Ran on Nautilus Kubernetes cluster with A100 GPU.

### Experiment 5 → 6 Pivot: Why Liquid Neural Networks

The surprise failure motivated a fundamental redesign:

1. **The problem with GPT-2**: "Interesting" was defined from human linguistic theory (garden-path sentences, domain switches). But the executor's activations may not reflect human notions of "interesting."

2. **The fix**: Use dynamical systems where "interesting" moments are mathematically precise — Lorenz attractor lobe switches, Rossler z-spikes, double pendulum energy exchanges. These create objectively verifiable transitions in the executor's dynamics.

3. **Why CfC networks**: Both executor and observer operate via ODEs in continuous time. Tiny networks (50-200 neurons) enable Lyapunov exponent analysis, phase portraits, and approximate Phi computation. Also biologically inspired (C. elegans neural circuit policies).

---

## Session 3 — Feb 24-25, 2026

### Experiment 6: Liquid Neural Network Observer

Implemented the most comprehensive experiment:
- 8 dynamical systems with mathematically precise "interest masks"
- CfC executor (64 neurons) + CfC observer (variable: 10-200 neurons)
- 11 probes (6 original adapted + 5 LNN-specific)
- 4 controls (untrained, linear, shuffled, wrong-executor with different seed)
- Scaling experiment, self-observation experiment

Deployed to Nautilus cluster. Job ran for 10 hours total (including one pod eviction and restart — idempotent pipeline saved all Phase 1 work).

### Experiment 6 Results

**Trained Observer: 6/11 positive indicators**
- Self-model: + (own=0.0018, other_executor=22.16, ratio ~12,000x)
- Surprise: **FIXED** (5.14x ratio on Lorenz lobe switches, p=5.5e-16)
- Temporal: + (24x ratio, plateau at window 64)
- First thought: + (first=0.000423, multi=0.000602, ratio 1.42)
- Preferences: - (RSA=0.0005, no convergence)
- Neuron preference: - (motor neurons dominate, not command as hypothesized)
- Time constants: + (std/mean=3.06, highly adaptive)
- Phase portrait: - (system separability=0.0)
- Lyapunov: - (lambda=-0.33, stable not critical)
- Synchronization: + (phase coherence=0.976)
- Phi: - (negative, -6.0)

**Controls:**
- Untrained: 4/10 — no self-model, no temporal integration
- Linear: 4/10 — no self-model via noise test
- Shuffled: **6/10** — matches trained observer
- Wrong-executor: 3/10 — no self-model, no surprise

**Scaling (all sizes 4/5 on core probes):**
Surprise ratio increases with size: 1.4x (10 neurons) → 3.4x (200 neurons)
Tau-executor correlation strengthens: -0.11 → -0.26

**Self-observation:** No significant difference (p=0.93)

### Key Takeaways

1. **Surprise probe is fixed.** Redesigning "interesting" from the executor's perspective (dynamical system transitions) instead of human linguistic intuition recovered a strong signal. The lesson: measure surprise relative to the observed system, not the experimenter.

2. **The shuffled control is the critical negative finding.** The observer builds a statistical model of executor hidden state distributions, not a temporal model that tracks computational flow. Temporal order doesn't matter. This challenges the claim that continuous-time dynamics in the observer add value.

3. **Self-model and executor specificity are real.** The wrong-executor control clearly fails (3/10), confirming that the observer's model is specific to its executor, not a generic pattern matcher.

4. **What this means for the hypothesis:** Observers trained purely on prediction develop real self-models and surprise responses. But the mechanism is statistical, not temporal. This is a partial validation — the "observing" produces genuine discrimination, but not the kind of temporal narrative consciousness we hypothesized.
