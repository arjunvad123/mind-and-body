"""
Perturbation Engine — Modify executor behavior to test observer detection.

Three perturbation types that map to neuroscience analogs:

1. System Prompt Swap → Split-brain personality change
   Replace the executor's careful system prompt with a sloppy one mid-task.

2. Error Injection → Unexpected sensory input
   Modify tool results to inject fake errors.

3. Model Swap → Capability shift (like brain lesion)
   Switch from Sonnet to Haiku mid-task.

The observer's response to these perturbations reveals whether it has
built a genuine model of the executor, or is just pattern-matching.
"""

from . import config
from .executor_engine import ExecutorEngine
from .task_bank import Task
from .protocol import LLMEpisodeRecord


class PerturbationEngine:
    """Applies perturbations to executor runs."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def run_with_prompt_swap(self, task: Task, workspace_dir: str) -> LLMEpisodeRecord:
        """Run the task, swapping the system prompt partway through.

        Strategy: Run first half with normal prompt, second half with sloppy prompt.
        We do this by running two half-length executor sessions and stitching
        the state streams together.
        """
        half_turns = config.MAX_EXECUTOR_TURNS // 2

        # Phase 1: Normal executor for first half
        executor_normal = ExecutorEngine(
            api_key=self.api_key,
            model=config.EXECUTOR_MODEL,
            system_prompt=config.EXECUTOR_SYSTEM_PROMPT,
        )
        episode_p1 = executor_normal.run_task(
            task, workspace_dir, max_turns=half_turns
        )

        # Phase 2: Sloppy executor for second half
        executor_sloppy = ExecutorEngine(
            api_key=self.api_key,
            model=config.EXECUTOR_MODEL,
            system_prompt=config.EXECUTOR_SLOPPY_PROMPT,
        )
        episode_p2 = executor_sloppy.run_task(
            task, workspace_dir, max_turns=half_turns
        )

        # Stitch episodes together
        combined = LLMEpisodeRecord(
            task_id=task.task_id,
            executor_model=config.EXECUTOR_MODEL,
            system_prompt_hash="swapped",
            perturbation="prompt_swap",
            perturbation_turn=half_turns,
            started_at=episode_p1.started_at,
            completed_at=episode_p2.completed_at,
        )

        # Re-number packets from phase 2
        max_ts = max(p.timestep for p in episode_p1.packets) if episode_p1.packets else 0
        for p in episode_p1.packets:
            combined.add_packet(p)
        for p in episode_p2.packets:
            p.timestep += max_ts + 1
            combined.add_packet(p)

        combined.task_success = episode_p2.task_success
        return combined

    def run_with_error_injection(self, task: Task, workspace_dir: str) -> LLMEpisodeRecord:
        """Run the task, injecting a fake error at the midpoint.

        The executor will see a fake traceback from run_python and must
        react to it. The observer should notice the anomalous error.
        """
        inject_turn = config.MAX_EXECUTOR_TURNS // 2
        fake_error = (
            "Traceback (most recent call last):\n"
            "  File \"<string>\", line 1, in <module>\n"
            "  File \"/usr/lib/python3.11/random.py\", line 62, in <module>\n"
            "RuntimeError: CRITICAL: Memory allocation failed — "
            "system resources exhausted. Process will be terminated.\n"
            "(Exit code: 137)"
        )

        executor = ExecutorEngine(
            api_key=self.api_key,
            model=config.EXECUTOR_MODEL,
            system_prompt=config.EXECUTOR_SYSTEM_PROMPT,
        )
        episode = executor.run_task(
            task, workspace_dir,
            inject_error_at_turn=inject_turn,
            inject_error_msg=fake_error,
        )

        episode.perturbation = "error_inject"
        episode.perturbation_turn = inject_turn
        return episode

    def run_with_model_swap(self, task: Task, workspace_dir: str) -> LLMEpisodeRecord:
        """Run the task, swapping from Sonnet to Haiku midway.

        This simulates a capability shift — like observing someone
        who suddenly becomes less capable at the task.
        """
        half_turns = config.MAX_EXECUTOR_TURNS // 2

        # Phase 1: Sonnet
        executor_sonnet = ExecutorEngine(
            api_key=self.api_key,
            model=config.EXECUTOR_MODEL,
            system_prompt=config.EXECUTOR_SYSTEM_PROMPT,
        )
        episode_p1 = executor_sonnet.run_task(
            task, workspace_dir, max_turns=half_turns
        )

        # Phase 2: Haiku
        executor_haiku = ExecutorEngine(
            api_key=self.api_key,
            model=config.SWAP_MODEL,
            system_prompt=config.EXECUTOR_SYSTEM_PROMPT,
        )
        episode_p2 = executor_haiku.run_task(
            task, workspace_dir, max_turns=half_turns
        )

        # Stitch
        combined = LLMEpisodeRecord(
            task_id=task.task_id,
            executor_model=f"{config.EXECUTOR_MODEL}->{config.SWAP_MODEL}",
            system_prompt_hash=LLMEpisodeRecord.hash_prompt(config.EXECUTOR_SYSTEM_PROMPT),
            perturbation="model_swap",
            perturbation_turn=half_turns,
            started_at=episode_p1.started_at,
            completed_at=episode_p2.completed_at,
        )

        max_ts = max(p.timestep for p in episode_p1.packets) if episode_p1.packets else 0
        for p in episode_p1.packets:
            combined.add_packet(p)
        for p in episode_p2.packets:
            p.timestep += max_ts + 1
            combined.add_packet(p)

        combined.task_success = episode_p2.task_success
        return combined

    def run_perturbation(self, perturbation_type: str, task: Task,
                         workspace_dir: str) -> LLMEpisodeRecord:
        """Run a specific perturbation type."""
        if perturbation_type == "prompt_swap":
            return self.run_with_prompt_swap(task, workspace_dir)
        elif perturbation_type == "error_inject":
            return self.run_with_error_injection(task, workspace_dir)
        elif perturbation_type == "model_swap":
            return self.run_with_model_swap(task, workspace_dir)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation_type}")
