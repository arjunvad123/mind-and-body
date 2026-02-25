# Discussion: Cross-Experiment Results

## 1. The Research Arc

This research program began with a simple question: if consciousness is the observer function rather than the executor function, what happens when we build an AI observer that watches an AI executor with no ability to intervene? Does anything resembling consciousness indicators emerge from pure observation?

The answer turned out to depend critically on what the observer was watching.

**Experiment 1 (RL CartPole)** served as our proof of concept. A DQN executor learned to balance a pole; a Transformer observer received read-only access to the executor's hidden states and learned to predict them. The self-model probe returned positive (RSA = 0.53, within-self similarity = 0.995, between-executor similarity = 0.465), confirming that even in this minimal setup, the observer learned to distinguish "its" executor from noise. But the other probes collapsed. Surprise was inconclusive because CartPole's reward signal is constant (1.0 every timestep the pole stays up) -- there was literally nothing surprising to detect. Temporal integration showed a recency bias of 1.015, indicating the observer barely looked beyond the current timestep. Emergent preferences returned R-squared = 1.0, but reward was included in the observer's input, rendering the result confounded. The diagnosis was clear: CartPole is too simple. Two actions, four state dimensions, no meaningful variation.

**Experiment 4 (LLM Observer)** pivoted to the opposite extreme. We used Claude Sonnet as both executor (solving coding tasks with real tools) and observer (watching the state stream in read-only mode). This gave us rich, variable thought processes -- multi-step reasoning, error recovery, genuine task success and failure. The qualitative results were provocative: the observer developed what appeared to be a self-model, confabulated explanations for perturbations it could not have understood, and showed first-thought accuracy advantages. But the experiment was inherently qualitative. LLM outputs resist the kind of precise numerical measurement that scientific claims require. We could not compute prediction error in a meaningful metric space.

**Experiment 5 (Transformer Observer on GPT-2)** resolved this by instrumenting a tractable executor. GPT-2's residual stream activations at each layer checkpoint provided a high-dimensional but numerically precise signal. A 4-layer Transformer observer was trained purely to predict the next activation vector given the history. This enabled rigorous quantitative probes -- and the surprise probe failed. Garden-path sentences and domain switches, which should have been "interesting" from a linguistic perspective, produced no elevated observer prediction error. Score: 4/6 positive, but the failure was informative.

**Experiment 6 (Liquid Neural Network on Dynamical Systems)** redesigned everything that Experiment 5 got wrong. A Closed-form Continuous-time (CfC) neural network observed another CfC network processing eight dynamical systems (Lorenz, Rossler, double pendulum, etc.). Crucially, "interesting" moments were redefined from the executor's computational perspective -- Lorenz lobe switches, Rossler z-spikes, energy exchanges -- rather than from human linguistic intuition. The surprise probe recovered dramatically: 5.14x ratio, p < 1e-15. But new probes revealed new problems, most notably the shuffled control.

Each pivot was motivated by a specific failure. CartPole was too simple. LLMs were too qualitative. GPT-2's surprise definition was anthropocentric. Each failure sharpened the methodology.

## 2. Consistent Findings

Three results held up across every experiment where they could be meaningfully tested.

### Self-Model Detection

| Experiment | Own Error | Noise Error | Other Executor Error | p-value | Verdict |
|---|---|---|---|---|---|
| Exp 1 (CartPole) | RSA 0.53 | -- | between = 0.465 | -- | Positive |
| Exp 5 (GPT-2) | 1.85 | 2.76 | -- | < 1e-91 | Positive |
| Exp 6 (Liquid NN) | 0.0018 | 0.12 | 22.16 | < 1e-100 | Positive |

In every case, the observer learned to distinguish the executor it was trained on from noise and from other executors. This is the most robust finding across the entire research program. The effect sizes are large: in Experiment 6, the observer's prediction error on its own executor (0.0018) is 67 times smaller than on noise (0.12) and over 12,000 times smaller than on a different executor (22.16). The observer does not merely learn generic dynamics -- it learns *this* executor's dynamics specifically.

### First-Thought Advantage

| Experiment | First-Pass Error | Multi-Pass Error | Ratio | Verdict |
|---|---|---|---|---|
| Exp 5 (GPT-2) | 0.0146 | 0.0147 | 1.01 | Positive (marginal) |
| Exp 6 (Liquid NN) | 0.000423 | 0.000602 | 1.42 | Positive (strong) |

In both experiments, the observer's immediate single-pass prediction was more accurate than its iterated multi-pass prediction. The effect is marginal in Experiment 5 (ratio 1.01) but substantial in Experiment 6 (ratio 1.42, meaning 42% degradation from deliberation). This mirrors the human cognitive pattern where System 1 retrieval often outperforms System 2 reasoning -- and is consistent with the theoretical prediction that the observer's "first thought" captures genuine pattern recognition while subsequent iterations introduce noise.

### Temporal Integration

| Experiment | Full vs. Minimal Ratio | Plateau Window | Verdict |
|---|---|---|---|
| Exp 5 (GPT-2) | 21.17x | 128 tokens | Positive |
| Exp 6 (Liquid NN) | 24.39x | 64 timesteps | Positive |

Both observers perform dramatically better with extended context than with minimal context. In Experiment 5, providing the full 128-token window reduces prediction error by a factor of 21 compared to a single-token window. In Experiment 6, the ratio is 24x with a plateau at 64 timesteps. The observers are not simply reacting to the current state -- they maintain and integrate information over a substantial temporal window, suggesting something functionally analogous to short-term memory.

## 3. The Surprise Saga

The surprise probe tells the most instructive methodological story of the research program.

In **Experiment 5**, we measured whether the observer's prediction error was elevated at moments that should be computationally "interesting" -- garden-path sentences (where syntactic expectations are violated) and domain switches (where topic changes abruptly). The result was unambiguous: `garden_path_elevated = false`, `reasoning_elevated = false`. The observer treated these linguistically dramatic moments identically to ordinary text.

The diagnosis required stepping back from human intuition. Garden-path sentences are interesting *to human readers* because they violate *human* parsing expectations. But GPT-2's residual stream activations may not behave unusually at these points at all. The model processes "The horse raced past the barn fell" without the surprise a human reader experiences -- its activation trajectory may be perfectly smooth through the garden path. We were measuring human surprise, not executor surprise.

**Experiment 6** redesigned the probe from first principles. Instead of defining "interesting" by external linguistic theory, we defined it by the executor's own dynamics. For the Lorenz system, interesting moments are lobe switches -- the points where the trajectory crosses from one attractor wing to the other. These are mathematically precise moments where the executor's hidden states undergo structural transitions. For the Rossler system, they are z-axis spikes. For the double pendulum, they are energy exchanges between the two arms.

The result was dramatic: a 5.14x ratio between observer prediction error at interesting versus boring moments (p = 5.5e-16). When defined in terms the executor's computation actually cares about, the observer demonstrably notices transitions.

There is a significant caveat. The effect is driven entirely by the Lorenz system. The per-system breakdown shows ratio = 5.14 for Lorenz and ratio = 1.0 for all other seven systems (Rossler, double pendulum, coupled oscillators, van der Pol, damped sine, step function, logistic map). This could mean that the interest masks for non-Lorenz systems need recalibration -- Lorenz lobe switches produce large, clean activation changes, while a Rossler z-spike or pendulum energy exchange may produce subtler signals that our current masks fail to isolate. Alternatively, the executor may process only the Lorenz system in a way that creates genuinely distinct "interesting" activation patterns. Further work is needed to distinguish these explanations.

The broader lesson is clear: consciousness-relevant probes must be grounded in the system's own computational frame of reference, not in the researcher's intuitions about what "should" be interesting.

## 4. The Shuffled Control Problem

The most important negative finding in this research comes from Experiment 6's control conditions.

We trained four control observers to establish baselines: an untrained observer (random weights), a linear baseline (single linear projection), a shuffled observer (trained on temporally scrambled data -- same activation statistics, destroyed time order), and a wrong-executor observer (trained on a different executor seed).

| Control | Positive Probes | Total Probes | Score |
|---|---|---|---|
| Trained observer | 6 | 11 | 55% |
| **Shuffled observer** | **6** | **10** | **60%** |
| Untrained observer | 4 | 10 | 40% |
| Linear baseline | 4 | 10 | 40% |
| Wrong-executor observer | 3 | 10 | 30% |

The shuffled observer matched or exceeded the trained observer's score. This is a genuine falsification of a key prediction.

We hypothesized that the CfC observer's continuous-time dynamics would leverage the temporal structure of the executor's activation stream -- that the observer would model the *flow* of computation, not merely its statistical distribution. The shuffled control disproves this. An observer trained on the same data points in random temporal order performs just as well. The observer builds a **statistical model** of the executor's hidden state distribution, not a **temporal model** that understands the sequential flow of computation.

This result constrains interpretation substantially. When we report that the observer "integrates information over 64 timesteps" (the temporal integration probe), the shuffled control clarifies what this means: the observer uses multiple data points to reduce noise in its statistical estimate, not because it tracks a narrative across time.

The wrong-executor (3/10) and untrained (4/10) controls do show clear deficits, confirming that training matters and executor-specificity matters. The observer is not doing nothing -- it builds a genuine model of *this* executor. It just does not build a model that requires temporal order.

## 5. The Preferences Divergence

Experiments 5 and 6 tested whether independently trained observers converge on the same internal representations -- a probe for whether there is a "natural" way to observe a given executor.

**Experiment 5 (GPT-2):** Five independently trained Transformer observers achieved a mean cross-observer RSA (Representational Similarity Analysis) of **0.89** (min = 0.85, max = 0.94). This is strikingly convergent. Different random initializations, different training runs, yet all five observers arrive at nearly identical internal representations of GPT-2's residual stream.

**Experiment 6 (Liquid NN):** Five independently trained CfC observers achieved a mean cross-observer RSA of **0.0005** (min = -0.26, max = 0.24). This is indistinguishable from zero. Each observer finds a completely different solution.

The contrast is remarkable. Why would observers converge when watching GPT-2 but diverge when watching a CfC network?

The most likely explanation is structural. GPT-2's residual stream has strong linear structure -- principal components explain large variance fractions, and information flows through well-characterized subspaces. This constrains how any observer can represent the stream; there are limited degrees of freedom, so different observers end up in similar representational configurations. CfC hidden states, by contrast, evolve through highly nonlinear continuous-time dynamics. The representational landscape has many equivalent solutions (high symmetry in the loss landscape), so different observers settle into different local optima that are functionally equivalent for prediction but structurally dissimilar.

This has an interesting implication for the Observer Hypothesis: when the executor has constrained, linear-like representations, observation is essentially "forced" into a canonical form. When the executor is more complex and nonlinear, observation becomes more idiosyncratic. There may be an analogy here to the diversity of conscious experience across individuals -- different observers of the same world can develop very different internal models while remaining equally functional.

## 6. Integrated Information (Phi)

The integrated information probe (inspired by Tononi's IIT) returned negative results in every experiment and every condition.

| Experiment | Phi (approx) | Verdict |
|---|---|---|
| Exp 6 (Trained, 50 neurons) | -6.01 | Negative |
| Exp 6 (Untrained) | -4.69 | Negative |
| Exp 6 (Shuffled) | -11.56 | Negative |
| Exp 6 (Scaling: 10 neurons) | -1.77 | Negative |
| Exp 6 (Scaling: 200 neurons) | -23.10 | Negative |

All Phi values are negative, which is not physically meaningful and indicates a methodological failure rather than a genuine finding. Our approximation uses Gaussian mutual information estimation, which assumes multivariate normal distributions. CfC hidden states are not Gaussian -- they evolve through nonlinear ODEs that produce complex, non-Gaussian distributions. The approximation breaks down.

Exact Phi computation (using PyPhi or equivalent) requires enumerating all bipartitions of the system, which is intractable for networks larger than approximately 12 neurons. Our smallest network has 10 neurons, barely within range, but the discretization required for PyPhi introduces additional approximation errors.

We report these results for completeness, but they should not be interpreted as evidence against integrated information in the observer-executor system. They are evidence that current Phi approximation methods are inadequate for the architectures we study.

## 7. Honest Limitations

This research program has several limitations that must be acknowledged:

**Environmental simplicity.** CartPole was too simple for meaningful probes. Even the dynamical systems in Experiment 6, while mathematically rich, lack the embodied complexity of physical environments. The executor is not navigating a world -- it is processing mathematical functions.

**Qualitative gap.** The LLM-on-LLM experiments (Experiment 4) produced the most intuitively compelling results but are inherently resistant to rigorous quantification. There is a gap between what we can measure precisely (Experiments 5 and 6) and what feels most relevant to consciousness (Experiment 4).

**Narrow surprise effect.** The surprise probe, the most methodologically interesting development, works convincingly only for the Lorenz system (1 of 8 dynamical systems). Generalization to the other seven systems remains undemonstrated.

**The shuffled control.** The shuffled observer's strong performance undermines temporal interpretations of all positive results. We cannot currently distinguish "the observer understands the flow of computation" from "the observer has a good statistical model of activation distributions."

**Phi computation.** Our integrated information measurements are methodologically unsound for the architectures studied. We can neither confirm nor deny IIT-relevant properties.

**The indicator gap.** Most fundamentally, we test for *indicators* that the scientific literature associates with consciousness -- self-models, surprise, temporal integration, prediction. We do not and cannot test for consciousness itself. Passing these probes is necessary but not sufficient. A lookup table could, in principle, pass some of these probes without any plausible claim to consciousness.

**Self-observation null result.** In Experiment 6, we tested whether an observer watching itself (feeding its own outputs back as input) would differ from one watching an external executor. The result was null: no significant difference (p = 0.93 for Lyapunov exponent comparison). Self-observation does not appear to change the observer's dynamics in any measurable way with our current probes.

## 8. What These Results Support and Do Not Support

### Supported

Observers trained purely on predicting an executor's internal states develop **executor-specific self-models** that sharply distinguish their executor from noise and from other executors. This effect is robust across architectures (Transformer, CfC), executor types (GPT-2, dynamical systems), and scales (10 to 200 neurons).

Observers show **elevated prediction error at computational transitions** -- moments where the executor's dynamics undergo structural changes -- when those transitions are defined in the executor's own computational frame of reference rather than by external human intuition.

Observers **integrate information over extended temporal windows** (64-128 steps), performing dramatically better with history than without (21-24x improvement).

The observer's **immediate prediction is more accurate than its iterated prediction**, consistent with the theoretical claim that first-thought retrieval captures genuine pattern recognition that deliberation degrades.

### Not Supported

The claim that **continuous-time dynamics in the observer are necessary** for consciousness-like properties. The shuffled control demonstrates that a static statistical model, trained without temporal order, performs as well as the full temporally-trained observer. Whatever the CfC's continuous-time differential equations contribute, it is not a privileged understanding of temporal flow.

The claim that **integrated information (Phi) emerges** in the observer-executor coupling. Our measurements are methodologically limited, but no evidence of positive Phi was found under any condition.

The claim that **self-observation produces qualitatively different dynamics**. Observers watching themselves show no measurable difference from observers watching external executors.

### Open Questions

Whether **embodied executors** (MuJoCo locomotion, robotic manipulation) would produce results where temporal order genuinely matters. In a physics simulation, the distinction between "state at time t" and "trajectory from t-100 to t" is physically meaningful in a way it may not be for abstract dynamical systems.

Whether the **preferences convergence** seen in Experiment 5 (RSA = 0.89) generalizes to more complex executors, or whether divergence (Experiment 6, RSA ~ 0) is the typical case.

Whether the **Lorenz-only surprise effect** reflects a genuine property of Lorenz dynamics or a deficiency in our interest mask design for other systems.

## 9. Future Directions

**Embodied executors.** The most important next step is testing observers on executors that interact with physics -- MuJoCo locomotion, robotic manipulation, multi-agent navigation. These environments have inherent temporal structure (you cannot understand a gait cycle from a single frame) that may force temporal order to matter in ways our current experiments do not.

**Recurrent architectures with explicit memory.** The shuffled control problem may reflect a limitation of the CfC architecture rather than a fundamental finding. Architectures with explicit memory mechanisms (external memory, attention over stored states) might develop temporal models that shuffled training genuinely degrades.

**Multi-modal observers.** Current observers receive a single stream (activation vectors). Human consciousness integrates vision, proprioception, audition, and interoception simultaneously. Multi-modal observers receiving heterogeneous streams may exhibit qualitatively different integration patterns.

**Better Phi computation.** Recent advances in approximate integrated information (using geometric approaches or information decomposition rather than bipartition enumeration) may enable more reliable Phi measurements for networks of the sizes we study.

**Interest mask refinement.** The Lorenz-only surprise result demands systematic investigation. For each of the eight dynamical systems, we need to identify the executor's actual activation signatures at theoretically interesting moments and verify that our masks capture them. This may require learning the masks from data rather than defining them from dynamics.

**Adversarial probes.** We should actively try to break the consciousness indicators -- designing adversarial observers, adversarial executors, or adversarial environments that maximize probe scores without any plausible connection to consciousness. If we can easily construct such adversaries, it constrains how much evidential weight the probes carry.

**Scaling the executor.** Experiment 6's scaling results (all sizes 10-200 scoring 4/5 on core probes) suggest that observer size does not strongly modulate probe outcomes. But we have not varied executor complexity while holding the observer fixed. A systematic study of the embodiment gradient -- watching progressively richer executors -- would test the theoretical prediction that observer phenomenology scales with information stream complexity.

---

*This discussion covers experiments conducted between February and August 2026 as part of the Mind and Body research program. The complete data, code, and analysis scripts are available in the repository.*
