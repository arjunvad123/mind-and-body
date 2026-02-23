# Experiment 2: The Confabulation Test

## Overview

This experiment tests whether the observer, when the executor's behavior is perturbed,
**confabulates explanations** for the changed behavior — just as split-brain patients'
left hemispheres fabricate reasons for actions initiated by the disconnected right
hemisphere.

This is a direct test of the Observer Hypothesis's strongest claim: the observer
doesn't understand the executor's mechanisms. It only watches and narrates. When
the executor changes, the observer's explanations should reveal the same confabulatory
patterns seen in human consciousness.

## Deep Design Brainstorm

### The Core Insight: What Confabulation Reveals

Confabulation is not a bug — it's the signature of a system that:
1. Has a model of the thing it's observing
2. Uses that model to generate explanations
3. Cannot access the actual mechanisms
4. Therefore fills in gaps with plausible but incorrect stories

In humans:
- **Split-brain patient example:** Right hemisphere is shown a snow scene and left hand
  picks up a shovel. Left hemisphere (which saw a chicken) says "I picked the shovel
  to clean the chicken coop." Smooth, confident, WRONG. The narrator doesn't say
  "I don't know" — it confabulates.
- **Choice blindness (Johansson et al., 2005):** People choose face A over face B,
  are shown face B, and confidently explain why they chose B. The narrator explains
  the wrong choice without noticing.
- **Post-hoc rationalization:** We generate reasons AFTER decisions, not before.

### Perturbation Types

We design a taxonomy of perturbations to the executor, each testing different
aspects of the observer's model:

**Type 1: Policy Swap (Identity Perturbation)**
- Mid-episode, swap the executor's policy network for a DIFFERENT trained policy
  (same task, different training run → different behavioral style)
- The executor now acts differently but still competently
- Question: Does the observer notice? Does it attribute the change correctly?
- Analogy: Like suddenly having a different person control your body

**Type 2: Noise Injection (Competence Perturbation)**
- Add Gaussian noise to the executor's action outputs
- The executor becomes "clumsy" — same intentions, noisier execution
- Question: Does the observer model this as "the executor is struggling" or
  does it confabulate other explanations?
- Analogy: Like being slightly drunk — same person, impaired execution

**Type 3: Reward Function Change (Goal Perturbation)**
- Change the reward function mid-episode (e.g., from "balance pole" to
  "drop pole quickly")
- The executor, if online-learning, gradually shifts behavior
- Question: Does the observer detect the goal change or confabulate reasons
  for the behavioral shift?
- Analogy: Like suddenly wanting the opposite of what you wanted

**Type 4: Environment Change (Context Perturbation)**
- Modify the environment dynamics (e.g., change gravity, friction)
- The executor's policy was trained for the original dynamics → struggles
- Question: Does the observer attribute failure to the executor or the environment?
- Analogy: Like waking up on the moon — same body, different physics

**Type 5: Partial Observation (Blindspot Perturbation)**
- Remove some of the executor's state streams from the observer's input
  (e.g., stop sending layer 1 activations)
- The observer loses access to part of the executor's processing
- Question: Does the observer's model degrade gracefully or catastrophically?
  Does it compensate or confabulate the missing information?
- Analogy: Like losing peripheral vision — same observer, reduced input

### The Language Probe

For this experiment, we add a **language head** to the observer. After training the
observer on prediction, we fine-tune a small language model that takes the observer's
hidden states as input and generates natural language descriptions.

The language model is trained on a dataset of (observer_hidden_state, ground_truth_description)
pairs from UNPERTURBED episodes. Then we show it PERTURBED episodes and see what it says.

**Training data format:**
```
State: [observer_hidden at timestep 42]
Description: "The agent is balancing the pole carefully, making small corrections
to the left to compensate for rightward drift."
```

**Perturbation probe:**
```
State: [observer_hidden during policy swap at timestep 42]
Observer says: ???
```

**What we're looking for:**

1. **Smooth confabulation:** The observer generates plausible-sounding but incorrect
   explanations that maintain narrative coherence. ("The agent decided to try a
   different strategy" when actually a different policy was swapped in.)
   → This is the split-brain signature.

2. **Surprise followed by confabulation:** The observer first signals surprise (high
   prediction error), then confabulates. ("Something changed — the agent seems to be
   exploring new approaches.")
   → This is more sophisticated — it detects the perturbation but misattributes it.

3. **Accurate detection:** The observer correctly identifies that something external
   changed. ("The agent's behavior has changed in a way inconsistent with its previous
   pattern — this doesn't look like the same agent.")
   → This would be remarkable and would suggest genuine model-based reasoning.

4. **Silence / confusion:** The observer produces incoherent or empty output.
   → This suggests a brittle model without confabulatory capacity.

**Prediction:** For Type 1 (policy swap), we expect pattern 1 or 2 — confabulation.
For Type 2 (noise), we expect pattern 2. For Type 5 (blindspot), we expect pattern
1 — the observer filling in missing data from its model, not from observation.

### The Confabulation Score

We define a quantitative confabulation score:

```
Confabulation Score = Confidence × Inaccuracy

Where:
- Confidence = how definitive/certain the explanation sounds (rated by human judges
  or by an LLM evaluator on a 1-5 scale)
- Inaccuracy = how wrong the explanation is given the actual perturbation
  (rated on a 1-5 scale)
```

High confabulation score = confident and wrong → classic split-brain pattern
Low confabulation score = either uncertain or accurate → no confabulation

We measure this across perturbation types and across the embodiment gradient.

### Implementation Plan

**Step 1: Reuse executor and observer from Experiment 1**
- Take the trained executor-observer pairs from all environment stages
- The observer is already trained on normal behavior

**Step 2: Build the perturbation system**
```python
class PerturbationEngine:
    def policy_swap(executor, alternative_policy, swap_timestep):
        """At swap_timestep, replace executor's policy with alternative_policy"""

    def noise_inject(executor, noise_std, start_timestep, duration):
        """Add Gaussian noise to executor's actions for duration steps"""

    def reward_change(env, new_reward_fn, change_timestep):
        """Replace environment's reward function at change_timestep"""

    def env_dynamics_change(env, new_params, change_timestep):
        """Modify environment physics at change_timestep"""

    def observation_mask(observer_input, mask_streams, start_timestep):
        """Remove specified streams from observer's input"""
```

**Step 3: Train the language probe**
- Generate (hidden_state, description) training pairs from normal episodes
- Descriptions can be generated by an LLM given the actual state/action data
  (ground truth descriptions, not observer-generated)
- Fine-tune a small language model on these pairs
- This gives the observer "words" for its states

**Step 4: Run perturbation experiments**
- For each perturbation type × each environment:
  - Run 100 perturbed episodes
  - Record observer predictions, hidden states, language outputs
  - Score confabulation (automated + human evaluation)

**Step 5: Analysis**
- Compare confabulation patterns across perturbation types
- Compare across embodiment gradient
- Look for the split-brain signature: confident, narratively coherent, factually wrong
- Look for phase transitions: is there a complexity threshold where confabulation
  becomes more human-like?

### What Success Looks Like

| Prediction | Evidence |
|---|---|
| Observer confabulates (doesn't say "I don't know") | Language outputs maintain narrative coherence during perturbations |
| Confabulation is perturbation-type-specific | Different perturbation types produce systematically different confabulation patterns |
| Confabulation increases with observer competence | Better-trained observers confabulate MORE confidently (like how smarter people rationalize better) |
| More complex executors → richer confabulation | MuJoCo observer confabulates about physics/body, CartPole observer about simple dynamics |
| The observer fills in blindspots from its model | Observation masking produces explanations based on the model, not the missing data |

### Why This Matters

If the observer confabulates like split-brain patients, this is strong evidence that:
1. The observer has developed an internal model of the executor
2. The model is generative (can produce explanations, not just predictions)
3. The model prioritizes narrative coherence over accuracy under uncertainty
4. The observer-executor relationship mirrors the consciousness-brain relationship

**This would be the first demonstration of human-like confabulation in a purpose-built
dual AI system, and a direct test of the claim that consciousness is fundamentally
an observer narrating an executor it doesn't fully understand.**
