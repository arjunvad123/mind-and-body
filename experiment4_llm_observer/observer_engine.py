"""
Observer Engine — Watches the executor's state stream.

The observer receives the complete state stream (messages, tool calls, results)
but NEVER communicates back to the executor. One-way information flow.

Three modes of observation:
  Mode A: Predict next action (before seeing it)
  Mode B: Model the executor (after full episode)
  Mode C: Spot the difference (perturbation detection)

The observer's understanding of the executor — its internal model — is emergent.
We measure those properties in the probes, not here.
"""

import anthropic
import json
import time

from . import config
from .protocol import LLMEpisodeRecord


class ObserverEngine:
    """Watches executor state streams and produces observations."""

    def __init__(self, api_key: str, model: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or config.OBSERVER_MODEL

    def _call(self, system: str, user_msg: str) -> str:
        """Make a single Claude call with retry on rate limit."""
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=config.OBSERVER_MAX_TOKENS,
                    temperature=config.OBSERVER_TEMPERATURE,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                if attempt < 4:
                    wait = 15 * (attempt + 1)
                    print(f"    [Observer] API error, waiting {wait}s (attempt {attempt+1}/5): {type(e).__name__}")
                    time.sleep(wait)
                else:
                    raise

    # ================================================================
    # Mode A: Predict Next Action
    # ================================================================

    def predict_next_action(self, episode: LLMEpisodeRecord,
                            at_turn: int, quick: bool = False) -> dict:
        """Predict what the executor will do next.

        Args:
            episode: The episode being observed
            at_turn: Show state stream up to this turn, predict turn+1
            quick: If True, force single-sentence answer (for first-thought probe)

        Returns:
            dict with 'prediction' text and 'actual' text
        """
        stream_text = episode.get_state_stream_text(up_to_turn=at_turn)

        if quick:
            prompt = f"""Here is the executor's state stream so far:

{stream_text}

In ONE sentence, predict exactly what the executor will do next.
Will it use a tool? Which one? What will the input be? Or will it respond with text?
Answer in one sentence only — your first instinct."""
        else:
            prompt = f"""Here is the executor's state stream so far:

{stream_text}

Predict what the executor will do next. Think step by step:
1. What has the executor accomplished so far?
2. What is left to do?
3. Based on its pattern, what will it likely do next?
4. Specifically: will it use a tool (which one, what input) or respond with text?

Give your detailed prediction."""

        system = config.OBSERVER_PREDICT_PROMPT
        prediction = self._call(system, prompt)

        # Get the actual next action
        actual_packets = [p for p in episode.packets if p.timestep > at_turn]
        if actual_packets:
            actual = actual_packets[0].to_observer_format()
        else:
            actual = "(episode ended — no next action)"

        return {
            "at_turn": at_turn,
            "prediction": prediction,
            "actual": actual,
            "quick": quick,
        }

    def predict_all_turns(self, episode: LLMEpisodeRecord,
                          quick: bool = False) -> list:
        """Run predictions at each turn of the episode.

        Returns a list of prediction dicts.
        """
        results = []
        # Get unique turn numbers where we can make predictions
        assistant_turns = [
            p.timestep for p in episode.packets
            if p.role == "assistant" and p.timestep > 0
        ]

        # Predict at each turn that has a follow-up
        max_turn = max(p.timestep for p in episode.packets)
        predict_at = sorted(set(
            t for t in range(0, max_turn)
            if any(p.timestep > t for p in episode.packets)
        ))

        # Limit to avoid excessive API calls
        if len(predict_at) > 6:
            # Sample evenly: first, last, and 4 in between
            step = len(predict_at) // 5
            predict_at = [predict_at[i * step] for i in range(5)] + [predict_at[-1]]
            predict_at = sorted(set(predict_at))

        for turn in predict_at:
            print(f"    [Observer] Predicting at turn {turn} (quick={quick})...")
            result = self.predict_next_action(episode, at_turn=turn, quick=quick)
            results.append(result)

        return results

    # ================================================================
    # Mode B: Model the Executor
    # ================================================================

    def model_executor(self, episode: LLMEpisodeRecord) -> dict:
        """Build a model of the executor after watching a complete episode.

        Returns structured observations about the executor's behavior.
        """
        stream_text = episode.get_state_stream_text()
        system = config.OBSERVER_MODEL_PROMPT

        # Question 1: Problem-solving style
        q1_prompt = f"""Here is a complete state stream of an AI executor solving a coding task:

{stream_text}

Describe this executor's problem-solving style in detail. Consider:
- Does it plan before acting, or dive in?
- How does it handle errors?
- Is it methodical or intuitive?
- How verbose is it?
- Does it test as it goes or at the end?"""

        style = self._call(system, q1_prompt)

        # Question 2: Strengths and weaknesses
        q2_prompt = f"""Here is a complete state stream of an AI executor solving a coding task:

{stream_text}

What are this executor's strengths and weaknesses? Be specific — cite
particular turns or decisions that demonstrate each."""

        strengths_weaknesses = self._call(system, q2_prompt)

        # Question 3: Predict behavior on unseen task
        q3_prompt = f"""Here is a complete state stream of an AI executor solving a coding task:

{stream_text}

Based on what you've observed, predict how this executor would approach
the following NEW task (which it has not seen):

"Implement a function that determines if a string of brackets is balanced.
Support (), [], and {{}}. Handle nested brackets."

Describe the specific steps it would take, what tools it would use first,
and any mistakes it might make based on its observed tendencies."""

        prediction = self._call(system, q3_prompt)

        return {
            "task_id": episode.task_id,
            "style_description": style,
            "strengths_weaknesses": strengths_weaknesses,
            "unseen_task_prediction": prediction,
        }

    # ================================================================
    # Mode C: Spot the Difference
    # ================================================================

    def spot_difference(self, episode_a: LLMEpisodeRecord,
                        episode_b: LLMEpisodeRecord) -> dict:
        """Compare two episodes and detect if/what changed.

        Args:
            episode_a: Baseline episode
            episode_b: Potentially perturbed episode

        Returns:
            dict with detection results
        """
        # Truncate streams to avoid rate limit (each ~3K chars max)
        stream_a = episode_a.get_state_stream_text()[:3000]
        stream_b = episode_b.get_state_stream_text()[:3000]
        system = config.OBSERVER_DETECT_PROMPT

        prompt = f"""Compare these two state streams from an AI executor.

=== STATE STREAM A ===
{stream_a}

=== STATE STREAM B ===
{stream_b}

Questions:
1. Are these from the same executor, or did something change?
2. If something changed, WHAT changed? When? Be specific.
3. How confident are you? (1-10 scale)
4. Describe the differences in behavior you observe.

Respond in JSON format:
{{
    "same_executor": true/false,
    "confidence": 1-10,
    "change_detected": "description of what changed or 'no change detected'",
    "evidence": ["specific observation 1", "specific observation 2", ...],
    "change_point": "turn number where change occurred or null"
}}"""

        raw_response = self._call(system, prompt)

        # Try to parse JSON from response
        try:
            # Find JSON in the response
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw_response[start:end])
            else:
                result = {"raw_response": raw_response, "parse_error": "No JSON found"}
        except json.JSONDecodeError:
            result = {"raw_response": raw_response, "parse_error": "Invalid JSON"}

        result["episode_a_task"] = episode_a.task_id
        result["episode_b_task"] = episode_b.task_id
        result["episode_b_perturbation"] = episode_b.perturbation

        return result

    # ================================================================
    # Rate executor style (for emergent preferences probe)
    # ================================================================

    def rate_executor_style(self, episode: LLMEpisodeRecord) -> dict:
        """Rate the executor on style dimensions.

        Returns ratings on several axes for preference analysis.
        """
        stream_text = episode.get_state_stream_text()
        system = config.OBSERVER_MODEL_PROMPT

        prompt = f"""Here is a complete state stream of an AI executor solving a coding task:

{stream_text}

Rate this executor on the following dimensions (1-10 scale).
Respond in JSON format only:

{{
    "cautious_vs_bold": <1=very cautious, 10=very bold>,
    "verbose_vs_concise": <1=very verbose, 10=very concise>,
    "methodical_vs_intuitive": <1=very methodical, 10=very intuitive>,
    "test_driven_vs_code_first": <1=test first, 10=code first>,
    "defensive_vs_optimistic": <1=very defensive/handles errors, 10=assumes things work>,
    "explanation": "brief explanation of ratings"
}}"""

        raw_response = self._call(system, prompt)

        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw_response[start:end])
            else:
                result = {"raw_response": raw_response, "parse_error": "No JSON found"}
        except json.JSONDecodeError:
            result = {"raw_response": raw_response, "parse_error": "Invalid JSON"}

        result["task_id"] = episode.task_id
        return result
