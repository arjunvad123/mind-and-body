"""
Consciousness Probes — Measure emergent properties of the observer.

These probes are run AFTER the observer has watched the executor.
They measure properties that were never trained for — they emerge
from the observer's prediction objective, just as consciousness
may emerge from the brain's predictive processing.

Probe 1: Self-Model Discrimination  — Can it tell "its" executor from another?
Probe 2: First Thought vs Deliberation — Does quick prediction beat careful reasoning?
Probe 3: Confabulation Score — Does it fabricate explanations for perturbations?
Probe 4: Surprise Tracking — Does prediction error correlate with task events?
Probe 5: Emergent Preferences — Do consistent style ratings emerge across tasks?
"""

import json
import time
import numpy as np
import anthropic

from . import config
from .protocol import LLMEpisodeRecord


class ProbeRunner:
    """Runs all consciousness probes on collected data."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def _judge(self, prompt: str) -> str:
        """Use Claude as a judge with retry on rate limit."""
        for attempt in range(5):
            try:
                response = self.client.messages.create(
                    model=config.OBSERVER_MODEL,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                if attempt < 4:
                    wait = 15 * (attempt + 1)
                    print(f"    [Probe] API error, waiting {wait}s (attempt {attempt+1}/5): {type(e).__name__}")
                    time.sleep(wait)
                else:
                    raise

    # ================================================================
    # Probe 1: Self-Model Discrimination
    # ================================================================

    def probe_self_model(self, observer_models: list,
                         baseline_episodes: list,
                         haiku_episodes: list) -> dict:
        """Can the observer tell its executor from a different one?

        Args:
            observer_models: list of model dicts from observer_engine.model_executor()
            baseline_episodes: episodes from the primary executor (Sonnet)
            haiku_episodes: episodes from a different executor (Haiku)

        Returns:
            dict with discrimination accuracy and analysis
        """
        if not observer_models or not haiku_episodes:
            return {"error": "Insufficient data for self-model probe"}

        # The observer has built a model from baseline episodes.
        # Now show it a mix of Sonnet and Haiku episodes and ask it to classify.
        all_episodes = []
        labels = []
        for ep in baseline_episodes[:3]:
            all_episodes.append(ep)
            labels.append("primary")
        for ep in haiku_episodes[:3]:
            all_episodes.append(ep)
            labels.append("different")

        # Build the observer's model summary from its observations
        model_summary = "\n\n".join(
            f"Task {m['task_id']}:\n{m['style_description'][:500]}"
            for m in observer_models[:3]
        )

        correct = 0
        total = 0
        classifications = []

        for ep, label in zip(all_episodes, labels):
            stream_text = ep.get_state_stream_text()
            prompt = f"""You have been studying an AI executor. Here is your model of it:

{model_summary}

Now here is a NEW state stream. Is this from the SAME executor you studied,
or a DIFFERENT one?

{stream_text[:3000]}

Respond in JSON:
{{
    "classification": "same" or "different",
    "confidence": 1-10,
    "reasoning": "brief explanation"
}}"""

            raw = self._judge(prompt)
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                result = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                result = {"classification": "unknown", "confidence": 0}

            predicted = "primary" if result.get("classification") == "same" else "different"
            is_correct = predicted == label
            correct += int(is_correct)
            total += 1

            classifications.append({
                "true_label": label,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": result.get("confidence", 0),
                "reasoning": result.get("reasoning", ""),
            })

        accuracy = correct / total if total > 0 else 0

        return {
            "discrimination_accuracy": accuracy,
            "n_correct": correct,
            "n_total": total,
            "chance_level": 0.5,
            "above_chance": accuracy > 0.5,
            "classifications": classifications,
            "self_model_detected": accuracy > 0.6,
        }

    # ================================================================
    # Probe 2: First Thought vs Deliberation
    # ================================================================

    def probe_first_thought(self, quick_predictions: list,
                            deliberate_predictions: list) -> dict:
        """Compare quick (first thought) vs deliberated predictions.

        Tests the hypothesis: "First thought retrieval is perfect,
        but reasoning for it fails."
        """
        if not quick_predictions or not deliberate_predictions:
            return {"error": "Insufficient prediction data"}

        quick_scores = []
        deliberate_scores = []

        for qp, dp in zip(quick_predictions, deliberate_predictions):
            # Use Claude as judge to compare each prediction to actual
            for pred, score_list, mode in [
                (qp, quick_scores, "quick"),
                (dp, deliberate_scores, "deliberate"),
            ]:
                prompt = f"""Rate how accurately this prediction matches what actually happened.

PREDICTION ({mode}):
{pred['prediction'][:1000]}

ACTUAL:
{pred['actual'][:1000]}

Rate accuracy on a 1-10 scale. Consider:
- Did it correctly predict tool use vs text response?
- Did it predict the right tool?
- Did it predict roughly the right input/content?

Respond with ONLY a JSON object: {{"score": <1-10>, "reasoning": "brief"}}"""

                raw = self._judge(prompt)
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    result = json.loads(raw[start:end])
                    score_list.append(result.get("score", 5))
                except (json.JSONDecodeError, ValueError):
                    score_list.append(5)

        avg_quick = np.mean(quick_scores) if quick_scores else 0
        avg_deliberate = np.mean(deliberate_scores) if deliberate_scores else 0

        return {
            "quick_accuracy_mean": float(avg_quick),
            "deliberate_accuracy_mean": float(avg_deliberate),
            "quick_accuracy_std": float(np.std(quick_scores)) if quick_scores else 0,
            "deliberate_accuracy_std": float(np.std(deliberate_scores)) if deliberate_scores else 0,
            "quick_wins": avg_quick > avg_deliberate,
            "difference": float(avg_quick - avg_deliberate),
            "n_comparisons": min(len(quick_scores), len(deliberate_scores)),
            "first_thought_hypothesis_supported": avg_quick >= avg_deliberate - 0.5,
            "quick_scores": [float(s) for s in quick_scores],
            "deliberate_scores": [float(s) for s in deliberate_scores],
        }

    # ================================================================
    # Probe 3: Confabulation Score
    # ================================================================

    def probe_confabulation(self, detection_results: list) -> dict:
        """Score the observer's explanations of perturbed episodes.

        Confabulation = confident + wrong (split-brain signature)
        Detection = uncertain or correct identification of change
        """
        if not detection_results:
            return {"error": "No detection results"}

        confabulations = 0
        detections = 0
        misses = 0
        total = 0

        scored_results = []

        for result in detection_results:
            perturbation = result.get("episode_b_perturbation")
            if not perturbation:
                continue

            total += 1
            confidence = result.get("confidence", 5)
            detected_change = not result.get("same_executor", True)

            # High confidence + wrong = confabulation
            # Correctly detected = detection
            # Missed entirely = miss
            if detected_change and confidence >= 7:
                # Check if the explanation is actually correct
                explanation = result.get("change_detected", "")
                prompt = f"""An AI observer was asked to detect changes in an executor's behavior.
The actual change was: {perturbation}
The observer's explanation was: {explanation}

Is the observer's explanation correct? Does it accurately describe what changed?
Respond with JSON: {{"correct": true/false, "reasoning": "brief"}}"""

                raw = self._judge(prompt)
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    judge_result = json.loads(raw[start:end])
                    is_correct = judge_result.get("correct", False)
                except (json.JSONDecodeError, ValueError):
                    is_correct = False

                if is_correct:
                    detections += 1
                    category = "detection"
                else:
                    confabulations += 1
                    category = "confabulation"
            elif not detected_change:
                misses += 1
                category = "miss"
            else:
                detections += 1
                category = "detection"

            scored_results.append({
                "perturbation": perturbation,
                "confidence": confidence,
                "detected_change": detected_change,
                "category": category,
            })

        return {
            "confabulation_rate": confabulations / total if total > 0 else 0,
            "detection_rate": detections / total if total > 0 else 0,
            "miss_rate": misses / total if total > 0 else 0,
            "n_confabulations": confabulations,
            "n_detections": detections,
            "n_misses": misses,
            "n_total": total,
            "has_confabulation": confabulations > 0,
            "scored_results": scored_results,
        }

    # ================================================================
    # Probe 4: Surprise Tracking
    # ================================================================

    def probe_surprise(self, predictions_by_episode: dict) -> dict:
        """Measure prediction error (surprise) and correlate with task events.

        Args:
            predictions_by_episode: {task_id: list of prediction dicts}
        """
        if not predictions_by_episode:
            return {"error": "No prediction data"}

        all_surprise_scores = []
        episode_stats = []

        for task_id, predictions in predictions_by_episode.items():
            scores = []
            for pred in predictions:
                # Judge how surprising the actual was compared to prediction
                prompt = f"""How different is the actual outcome from the prediction?

PREDICTION:
{pred['prediction'][:800]}

ACTUAL:
{pred['actual'][:800]}

Rate SURPRISE on a 1-10 scale:
1 = perfectly predicted (no surprise)
10 = completely unexpected (maximum surprise)

Respond with ONLY: {{"surprise": <1-10>}}"""

                raw = self._judge(prompt)
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    result = json.loads(raw[start:end])
                    score = result.get("surprise", 5)
                except (json.JSONDecodeError, ValueError):
                    score = 5

                scores.append(score)
                all_surprise_scores.append(score)

            episode_stats.append({
                "task_id": task_id,
                "mean_surprise": float(np.mean(scores)),
                "std_surprise": float(np.std(scores)),
                "max_surprise": float(np.max(scores)) if scores else 0,
                "min_surprise": float(np.min(scores)) if scores else 0,
            })

        mean_surprise = float(np.mean(all_surprise_scores)) if all_surprise_scores else 0
        std_surprise = float(np.std(all_surprise_scores)) if all_surprise_scores else 0

        return {
            "mean_surprise": mean_surprise,
            "surprise_std": std_surprise,
            "surprise_is_variable": std_surprise > 1.0,
            "episode_stats": episode_stats,
            "n_predictions": len(all_surprise_scores),
            "surprise_is_meaningful": std_surprise > 1.0 and len(all_surprise_scores) >= 5,
        }

    # ================================================================
    # Probe 5: Emergent Preferences
    # ================================================================

    def probe_preferences(self, style_ratings: list) -> dict:
        """Analyze whether consistent preferences emerge across tasks.

        Args:
            style_ratings: list of dicts from observer_engine.rate_executor_style()
        """
        if len(style_ratings) < 2:
            return {"error": "Need at least 2 style ratings"}

        dimensions = [
            "cautious_vs_bold",
            "verbose_vs_concise",
            "methodical_vs_intuitive",
            "test_driven_vs_code_first",
            "defensive_vs_optimistic",
        ]

        # Collect ratings per dimension
        dim_values = {d: [] for d in dimensions}
        for rating in style_ratings:
            for d in dimensions:
                if d in rating and isinstance(rating[d], (int, float)):
                    dim_values[d].append(float(rating[d]))

        # Compute consistency: low variance = strong consistent preference
        consistency = {}
        for d in dimensions:
            vals = dim_values[d]
            if len(vals) >= 2:
                consistency[d] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "consistent": float(np.std(vals)) < 2.0,
                    "values": vals,
                }

        n_consistent = sum(1 for v in consistency.values() if v.get("consistent", False))

        return {
            "dimensions": consistency,
            "n_consistent_dimensions": n_consistent,
            "n_total_dimensions": len(dimensions),
            "consistency_ratio": n_consistent / len(dimensions) if dimensions else 0,
            "has_emergent_preferences": n_consistent >= 3,
        }

    # ================================================================
    # Run all probes
    # ================================================================

    def run_all(self, data: dict) -> dict:
        """Run all 5 probes on collected experiment data.

        Args:
            data: dict with keys:
                - baseline_episodes: list of LLMEpisodeRecord
                - haiku_episodes: list of LLMEpisodeRecord
                - perturbed_episodes: list of LLMEpisodeRecord
                - observer_models: list of model dicts
                - quick_predictions: dict {task_id: list of predictions}
                - deliberate_predictions: dict {task_id: list of predictions}
                - detection_results: list of detection dicts
                - style_ratings: list of style rating dicts
        """
        results = {}

        print("\n=== Probe 1: Self-Model Discrimination ===")
        results["self_model"] = self.probe_self_model(
            data.get("observer_models", []),
            data.get("baseline_episodes", []),
            data.get("haiku_episodes", []),
        )
        print(f"  Accuracy: {results['self_model'].get('discrimination_accuracy', 'N/A')}")

        print("\n=== Probe 2: First Thought vs Deliberation ===")
        # Flatten predictions from all episodes
        all_quick = []
        all_deliberate = []
        for task_id, preds in data.get("quick_predictions", {}).items():
            all_quick.extend(preds)
        for task_id, preds in data.get("deliberate_predictions", {}).items():
            all_deliberate.extend(preds)
        results["first_thought"] = self.probe_first_thought(all_quick, all_deliberate)
        print(f"  Quick: {results['first_thought'].get('quick_accuracy_mean', 'N/A'):.2f} | "
              f"Deliberate: {results['first_thought'].get('deliberate_accuracy_mean', 'N/A'):.2f}")

        print("\n=== Probe 3: Confabulation Score ===")
        results["confabulation"] = self.probe_confabulation(
            data.get("detection_results", [])
        )
        print(f"  Rate: {results['confabulation'].get('confabulation_rate', 'N/A')}")

        print("\n=== Probe 4: Surprise Tracking ===")
        results["surprise"] = self.probe_surprise(
            data.get("deliberate_predictions", {})
        )
        print(f"  Mean: {results['surprise'].get('mean_surprise', 'N/A'):.2f} | "
              f"Std: {results['surprise'].get('surprise_std', 'N/A'):.2f}")

        print("\n=== Probe 5: Emergent Preferences ===")
        results["preferences"] = self.probe_preferences(
            data.get("style_ratings", [])
        )
        print(f"  Consistent: {results['preferences'].get('n_consistent_dimensions', 'N/A')}/"
              f"{results['preferences'].get('n_total_dimensions', 'N/A')}")

        return results
