#!/usr/bin/env python3
"""
Resume Experiment 4 from saved Phase 1 + Phase 4 data.

Loads baseline and perturbed episodes from disk, then runs:
- Phase 2: Observer predictions (quick + deliberate)
- Phase 3: Observer modeling + style ratings
- Phase 5: Perturbation detection
- Phase 6: Haiku comparison runs
- Phase 7: All 5 consciousness probes
- Phase 8: Save results
"""

import json
import os
import sys
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4_llm_observer.config import get_api_key, DATA_DIR, RESULTS_DIR, WORKSPACE_DIR
from experiment4_llm_observer.config import EXECUTOR_MODEL, OBSERVER_MODEL, SWAP_MODEL
from experiment4_llm_observer.protocol import LLMStatePacket, LLMEpisodeRecord
from experiment4_llm_observer.observer_engine import ObserverEngine
from experiment4_llm_observer.executor_engine import ExecutorEngine
from experiment4_llm_observer.task_bank import ALL_TASKS, get_task
from experiment4_llm_observer.probes import ProbeRunner


def load_episode_from_json(path: str) -> LLMEpisodeRecord:
    """Reconstruct an LLMEpisodeRecord from saved JSON."""
    with open(path) as f:
        d = json.load(f)

    episode = LLMEpisodeRecord(
        task_id=d["task_id"],
        task_success=d["task_success"],
        total_tokens=d["total_tokens"],
        executor_model=d.get("executor_model", ""),
        system_prompt_hash=d.get("system_prompt_hash", ""),
        perturbation=d.get("perturbation"),
        perturbation_turn=d.get("perturbation_turn"),
        started_at=d.get("started_at", ""),
        completed_at=d.get("completed_at", ""),
    )

    for pd in d["packets"]:
        packet = LLMStatePacket(
            timestep=pd["timestep"],
            task_id=pd["task_id"],
            role=pd["role"],
            content=pd["content"],
            tool_use=pd.get("tool_use"),
            tool_result=pd.get("tool_result"),
            token_count=pd.get("token_count", 0),
            cumulative_tokens=pd.get("cumulative_tokens", 0),
            timestamp=pd.get("timestamp", ""),
        )
        episode.packets.append(packet)

    return episode


def main():
    api_key = get_api_key()
    observer = ObserverEngine(api_key)
    prober = ProbeRunner(api_key)

    task_ids = list(ALL_TASKS.keys())

    # Load saved episodes
    print("=" * 60)
    print("LOADING SAVED DATA FROM PHASES 1 & 4")
    print("=" * 60)

    baseline_episodes = []
    for task_id in task_ids:
        path = os.path.join(DATA_DIR, f"baseline_{task_id}.json")
        if os.path.exists(path):
            ep = load_episode_from_json(path)
            baseline_episodes.append(ep)
            print(f"  Loaded baseline: {task_id} (success={ep.task_success}, {len(ep.packets)} packets)")

    perturbed_episodes = []
    for task_id in task_ids:
        for ptype in ["prompt_swap", "error_inject", "model_swap"]:
            path = os.path.join(DATA_DIR, f"perturbed_{task_id}_{ptype}.json")
            if os.path.exists(path):
                ep = load_episode_from_json(path)
                ep.perturbation = ptype  # Ensure it's set
                perturbed_episodes.append(ep)
                print(f"  Loaded perturbed: {task_id}/{ptype} (success={ep.task_success})")

    print(f"\nLoaded {len(baseline_episodes)} baselines, {len(perturbed_episodes)} perturbed")

    # ============================================================
    # Phase 2: Observer Predictions
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: OBSERVER PREDICTIONS")
    print("=" * 60)

    quick_predictions = {}
    deliberate_predictions = {}

    for episode in baseline_episodes:
        task_id = episode.task_id
        print(f"\n--- Predicting for: {task_id} ---")

        print("  Quick predictions (first thought)...")
        quick = observer.predict_all_turns(episode, quick=True)
        quick_predictions[task_id] = quick

        print("  Deliberate predictions...")
        deliberate = observer.predict_all_turns(episode, quick=False)
        deliberate_predictions[task_id] = deliberate

    print(f"\nPhase 2 complete: {sum(len(v) for v in quick_predictions.values())} "
          f"quick + {sum(len(v) for v in deliberate_predictions.values())} deliberate")

    # ============================================================
    # Phase 3: Observer Modeling
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: OBSERVER MODELING")
    print("=" * 60)

    observer_models = []
    style_ratings = []

    for episode in baseline_episodes:
        print(f"\n--- Modeling from: {episode.task_id} ---")
        model = observer.model_executor(episode)
        observer_models.append(model)

        print(f"  Rating style...")
        rating = observer.rate_executor_style(episode)
        style_ratings.append(rating)

    print(f"\nPhase 3 complete: {len(observer_models)} models built")

    # ============================================================
    # Phase 5: Perturbation Detection
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 5: PERTURBATION DETECTION")
    print("=" * 60)

    baseline_by_task = {ep.task_id: ep for ep in baseline_episodes}
    detection_results = []

    for perturbed_ep in perturbed_episodes:
        baseline_ep = baseline_by_task.get(perturbed_ep.task_id)
        if not baseline_ep:
            continue

        print(f"\n--- Detecting: {perturbed_ep.task_id} ({perturbed_ep.perturbation}) ---")
        result = observer.spot_difference(baseline_ep, perturbed_ep)
        detection_results.append(result)

    detected = sum(1 for r in detection_results if not r.get("same_executor", True))
    print(f"\nPhase 5 complete: {detected}/{len(detection_results)} perturbations detected")

    # ============================================================
    # Phase 6: Haiku Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 6: HAIKU COMPARISON RUNS")
    print("=" * 60)

    haiku_executor = ExecutorEngine(api_key=api_key, model=SWAP_MODEL)
    haiku_episodes = []
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    for task_id in task_ids[:3]:
        task = get_task(task_id)
        print(f"\n--- Haiku on: {task.title} ---")
        task_dir = os.path.join(WORKSPACE_DIR, task_id)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        episode = haiku_executor.run_task(task, WORKSPACE_DIR)
        haiku_episodes.append(episode)

    print(f"\nPhase 6 complete: {len(haiku_episodes)} Haiku episodes")

    # ============================================================
    # Phase 7: Consciousness Probes
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 7: CONSCIOUSNESS PROBES")
    print("=" * 60)

    data = {
        "baseline_episodes": baseline_episodes,
        "haiku_episodes": haiku_episodes,
        "perturbed_episodes": perturbed_episodes,
        "observer_models": observer_models,
        "quick_predictions": quick_predictions,
        "deliberate_predictions": deliberate_predictions,
        "detection_results": detection_results,
        "style_ratings": style_ratings,
    }

    probe_results = prober.run_all(data)

    # ============================================================
    # Phase 8: Save Results
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 8: SAVING RESULTS")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "experiment": "experiment4_llm_observer",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "executor_model": EXECUTOR_MODEL,
            "observer_model": OBSERVER_MODEL,
            "swap_model": SWAP_MODEL,
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
        },
        "summary": {
            "baseline_success_rate": (
                sum(1 for ep in baseline_episodes if ep.task_success) /
                len(baseline_episodes)
            ) if baseline_episodes else 0,
            "perturbation_detection_rate": (
                sum(1 for r in detection_results if not r.get("same_executor", True)) /
                len(detection_results)
            ) if detection_results else 0,
            "total_baseline_tokens": sum(ep.total_tokens for ep in baseline_episodes),
            "total_perturbed_tokens": sum(ep.total_tokens for ep in perturbed_episodes),
        },
        "probes": probe_results,
        "observer_models": observer_models,
        "style_ratings": style_ratings,
        "detection_results": detection_results,
    }

    results_path = os.path.join(RESULTS_DIR, "experiment4_results.json")
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


if __name__ == "__main__":
    main()
