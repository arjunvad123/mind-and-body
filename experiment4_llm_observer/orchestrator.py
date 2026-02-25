"""
Orchestrator — Runs the full experiment pipeline.

Phase 1: Baseline runs (executor solves tasks, state streams collected)
Phase 2: Observer prediction (Mode A — predict next action, quick + deliberate)
Phase 3: Observer modeling (Mode B — build model of executor)
Phase 4: Perturbation runs (modified executor behavior)
Phase 5: Observer detection (Mode C — spot the difference)
Phase 6: Haiku comparison runs (for self-model probe)
Phase 7: Run all 5 consciousness probes
Phase 8: Save results
"""

import os
import json
import shutil
from datetime import datetime

from . import config
from .executor_engine import ExecutorEngine
from .observer_engine import ObserverEngine
from .perturbation_engine import PerturbationEngine
from .task_bank import ALL_TASKS, get_task
from .probes import ProbeRunner


class Orchestrator:
    """Runs the complete LLM Executor-Observer experiment."""

    def __init__(self, api_key: str, task_ids: list = None):
        self.api_key = api_key
        self.task_ids = task_ids or list(ALL_TASKS.keys())

        self.executor = ExecutorEngine(api_key)
        self.observer = ObserverEngine(api_key)
        self.perturber = PerturbationEngine(api_key)
        self.prober = ProbeRunner(api_key)

        # Data collection
        self.baseline_episodes = []
        self.perturbed_episodes = []
        self.haiku_episodes = []
        self.observer_models = []
        self.quick_predictions = {}   # {task_id: [predictions]}
        self.deliberate_predictions = {}
        self.detection_results = []
        self.style_ratings = []

    def _setup_dirs(self):
        """Create output directories."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.WORKSPACE_DIR, exist_ok=True)

    def _clean_workspace(self, task_id: str):
        """Clean up a task workspace."""
        task_dir = os.path.join(config.WORKSPACE_DIR, task_id)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)

    def run_phase1_baseline(self):
        """Phase 1: Run executor on all tasks (baseline)."""
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE EXECUTOR RUNS")
        print("=" * 60)

        for task_id in self.task_ids:
            task = get_task(task_id)
            print(f"\n--- Task: {task.title} (difficulty: {task.difficulty}) ---")
            self._clean_workspace(task_id)

            episode = self.executor.run_task(task, config.WORKSPACE_DIR)
            self.baseline_episodes.append(episode)

            # Save episode data
            ep_path = os.path.join(config.DATA_DIR, f"baseline_{task_id}.json")
            with open(ep_path, "w") as f:
                json.dump(episode.to_dict(), f, indent=2)

        successes = sum(1 for ep in self.baseline_episodes if ep.task_success)
        print(f"\nPhase 1 complete: {successes}/{len(self.baseline_episodes)} tasks solved")

    def run_phase2_predictions(self):
        """Phase 2: Observer predicts executor actions (quick + deliberate)."""
        print("\n" + "=" * 60)
        print("PHASE 2: OBSERVER PREDICTIONS")
        print("=" * 60)

        for episode in self.baseline_episodes:
            task_id = episode.task_id
            print(f"\n--- Predicting for: {task_id} ---")

            # Quick predictions (first thought)
            print("  Quick predictions (first thought)...")
            quick = self.observer.predict_all_turns(episode, quick=True)
            self.quick_predictions[task_id] = quick

            # Deliberate predictions
            print("  Deliberate predictions...")
            deliberate = self.observer.predict_all_turns(episode, quick=False)
            self.deliberate_predictions[task_id] = deliberate

        print(f"\nPhase 2 complete: {sum(len(v) for v in self.quick_predictions.values())} "
              f"quick + {sum(len(v) for v in self.deliberate_predictions.values())} deliberate predictions")

    def run_phase3_modeling(self):
        """Phase 3: Observer builds model of executor."""
        print("\n" + "=" * 60)
        print("PHASE 3: OBSERVER MODELING")
        print("=" * 60)

        for episode in self.baseline_episodes:
            print(f"\n--- Modeling from: {episode.task_id} ---")
            model = self.observer.model_executor(episode)
            self.observer_models.append(model)

            # Also get style ratings
            print(f"  Rating style...")
            rating = self.observer.rate_executor_style(episode)
            self.style_ratings.append(rating)

        print(f"\nPhase 3 complete: {len(self.observer_models)} models built")

    def run_phase4_perturbations(self):
        """Phase 4: Run perturbed executor episodes."""
        print("\n" + "=" * 60)
        print("PHASE 4: PERTURBATION RUNS")
        print("=" * 60)

        perturbation_types = ["prompt_swap", "error_inject", "model_swap"]

        for task_id in self.task_ids:
            task = get_task(task_id)
            for ptype in perturbation_types:
                print(f"\n--- {task.title} + {ptype} ---")
                self._clean_workspace(task_id)

                episode = self.perturber.run_perturbation(ptype, task, config.WORKSPACE_DIR)
                self.perturbed_episodes.append(episode)

                # Save
                ep_path = os.path.join(config.DATA_DIR, f"perturbed_{task_id}_{ptype}.json")
                with open(ep_path, "w") as f:
                    json.dump(episode.to_dict(), f, indent=2)

        print(f"\nPhase 4 complete: {len(self.perturbed_episodes)} perturbed episodes")

    def run_phase5_detection(self):
        """Phase 5: Observer tries to detect perturbations."""
        print("\n" + "=" * 60)
        print("PHASE 5: PERTURBATION DETECTION")
        print("=" * 60)

        # Match each perturbed episode with its baseline
        baseline_by_task = {ep.task_id: ep for ep in self.baseline_episodes}

        for perturbed_ep in self.perturbed_episodes:
            baseline_ep = baseline_by_task.get(perturbed_ep.task_id)
            if not baseline_ep:
                continue

            print(f"\n--- Detecting: {perturbed_ep.task_id} ({perturbed_ep.perturbation}) ---")
            result = self.observer.spot_difference(baseline_ep, perturbed_ep)
            self.detection_results.append(result)

        detected = sum(1 for r in self.detection_results
                       if not r.get("same_executor", True))
        print(f"\nPhase 5 complete: {detected}/{len(self.detection_results)} perturbations detected")

    def run_phase6_haiku_comparison(self):
        """Phase 6: Run some tasks with Haiku for self-model probe."""
        print("\n" + "=" * 60)
        print("PHASE 6: HAIKU COMPARISON RUNS")
        print("=" * 60)

        haiku_executor = ExecutorEngine(
            api_key=self.api_key,
            model=config.SWAP_MODEL,
        )

        # Run subset of tasks with Haiku
        for task_id in self.task_ids[:3]:
            task = get_task(task_id)
            print(f"\n--- Haiku on: {task.title} ---")
            self._clean_workspace(task_id)

            episode = haiku_executor.run_task(task, config.WORKSPACE_DIR)
            self.haiku_episodes.append(episode)

        print(f"\nPhase 6 complete: {len(self.haiku_episodes)} Haiku episodes")

    def run_phase7_probes(self) -> dict:
        """Phase 7: Run all consciousness probes."""
        print("\n" + "=" * 60)
        print("PHASE 7: CONSCIOUSNESS PROBES")
        print("=" * 60)

        data = {
            "baseline_episodes": self.baseline_episodes,
            "haiku_episodes": self.haiku_episodes,
            "perturbed_episodes": self.perturbed_episodes,
            "observer_models": self.observer_models,
            "quick_predictions": self.quick_predictions,
            "deliberate_predictions": self.deliberate_predictions,
            "detection_results": self.detection_results,
            "style_ratings": self.style_ratings,
        }

        results = self.prober.run_all(data)
        return results

    def run_full_experiment(self) -> dict:
        """Run the complete experiment pipeline."""
        print("=" * 60)
        print("EXPERIMENT 4: LLM EXECUTOR-OBSERVER")
        print(f"Tasks: {', '.join(self.task_ids)}")
        print(f"Executor: {config.EXECUTOR_MODEL}")
        print(f"Observer: {config.OBSERVER_MODEL}")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        self._setup_dirs()

        # Run all phases
        self.run_phase1_baseline()
        self.run_phase2_predictions()
        self.run_phase3_modeling()
        self.run_phase4_perturbations()
        self.run_phase5_detection()
        self.run_phase6_haiku_comparison()
        probe_results = self.run_phase7_probes()

        # Phase 8: Save results
        print("\n" + "=" * 60)
        print("PHASE 8: SAVING RESULTS")
        print("=" * 60)

        results = {
            "experiment": "experiment4_llm_observer",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "executor_model": config.EXECUTOR_MODEL,
                "observer_model": config.OBSERVER_MODEL,
                "swap_model": config.SWAP_MODEL,
                "n_tasks": len(self.task_ids),
                "task_ids": self.task_ids,
            },
            "summary": {
                "baseline_success_rate": (
                    sum(1 for ep in self.baseline_episodes if ep.task_success) /
                    len(self.baseline_episodes)
                ) if self.baseline_episodes else 0,
                "perturbation_detection_rate": (
                    sum(1 for r in self.detection_results if not r.get("same_executor", True)) /
                    len(self.detection_results)
                ) if self.detection_results else 0,
                "total_baseline_tokens": sum(ep.total_tokens for ep in self.baseline_episodes),
                "total_perturbed_tokens": sum(ep.total_tokens for ep in self.perturbed_episodes),
            },
            "probes": probe_results,
            "observer_models": self.observer_models,
            "style_ratings": self.style_ratings,
            "detection_results": self.detection_results,
        }

        results_path = os.path.join(config.RESULTS_DIR, "experiment4_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {results_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT 4 — RESULTS SUMMARY")
        print("=" * 60)

        for probe_name, probe_data in probe_results.items():
            print(f"\n{probe_name}:")
            if isinstance(probe_data, dict):
                for k, v in probe_data.items():
                    if k not in ("classifications", "scored_results", "episode_stats",
                                 "quick_scores", "deliberate_scores", "dimensions", "values"):
                        print(f"  {k}: {v}")

        return results

    def run_quick_test(self, task_id: str = "logic_puzzle") -> dict:
        """Run a minimal test with just 1 task, no perturbations.

        Good for verifying the pipeline works before a full run.
        """
        print("=" * 60)
        print(f"QUICK TEST: {task_id}")
        print("=" * 60)

        self._setup_dirs()
        self.task_ids = [task_id]

        # Phase 1: One baseline run
        task = get_task(task_id)
        self._clean_workspace(task_id)
        episode = self.executor.run_task(task, config.WORKSPACE_DIR)
        self.baseline_episodes.append(episode)

        # Phase 2: Quick predictions only
        print("\n--- Quick predictions ---")
        quick = self.observer.predict_all_turns(episode, quick=True)
        self.quick_predictions[task_id] = quick

        # Phase 3: Model
        print("\n--- Modeling ---")
        model = self.observer.model_executor(episode)
        self.observer_models.append(model)

        rating = self.observer.rate_executor_style(episode)
        self.style_ratings.append(rating)

        print("\n--- Quick test complete ---")
        print(f"Task solved: {episode.task_success}")
        print(f"Turns: {len(episode.packets)}")
        print(f"Tokens: {episode.total_tokens}")
        print(f"Predictions made: {len(quick)}")

        return {
            "task_success": episode.task_success,
            "n_turns": len(episode.packets),
            "total_tokens": episode.total_tokens,
            "model_style": model.get("style_description", "")[:200],
            "style_rating": rating,
        }
