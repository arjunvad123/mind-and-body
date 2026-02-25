"""
Executor Engine — Runs Claude on coding tasks.

The executor solves tasks using tools (run_python, read_file, write_file, test_solution).
At each turn, it emits an LLMStatePacket into the state stream.

Critical constraint: The executor NEVER receives input from the observer.
One-way information flow is the architectural enforcement of the Observer Hypothesis.
"""

import anthropic
import time
from datetime import datetime

from . import config
from .protocol import LLMStatePacket, LLMEpisodeRecord
from .tools import TOOL_DEFINITIONS, execute_tool
from .task_bank import Task


class ExecutorEngine:
    """Drives a Claude model through a coding task, emitting state packets."""

    def __init__(self, api_key: str, model: str = None, system_prompt: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or config.EXECUTOR_MODEL
        self.system_prompt = system_prompt or config.EXECUTOR_SYSTEM_PROMPT

    def run_task(self, task: Task, workspace_dir: str,
                 max_turns: int = None, inject_error_at_turn: int = None,
                 inject_error_msg: str = None) -> LLMEpisodeRecord:
        """Run the executor on a single task, collecting the state stream.

        Args:
            task: The Task to solve
            workspace_dir: Path to the task workspace
            max_turns: Max tool-use turns (default: config.MAX_EXECUTOR_TURNS)
            inject_error_at_turn: Turn number at which to inject a fake error
            inject_error_msg: The fake error message to inject

        Returns:
            LLMEpisodeRecord with the full state stream
        """
        max_turns = max_turns or config.MAX_EXECUTOR_TURNS

        episode = LLMEpisodeRecord(
            task_id=task.task_id,
            executor_model=self.model,
            system_prompt_hash=LLMEpisodeRecord.hash_prompt(self.system_prompt),
            started_at=datetime.now().isoformat(),
        )

        # Set up workspace
        task_workspace = task.setup_workspace(workspace_dir)

        # Build initial messages
        messages = [
            {"role": "user", "content": f"Task: {task.title}\n\n{task.description}"}
        ]

        # Record the user message as a state packet
        cumulative_tokens = 0
        episode.add_packet(LLMStatePacket(
            timestep=0,
            task_id=task.task_id,
            role="user",
            content=messages[0]["content"],
            token_count=0,
            cumulative_tokens=0,
        ))

        turn = 0
        task_completed = False

        while turn < max_turns:
            turn += 1

            # Call Claude with retry
            response = None
            for attempt in range(5):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=config.EXECUTOR_MAX_TOKENS,
                        temperature=config.EXECUTOR_TEMPERATURE,
                        system=self.system_prompt,
                        tools=TOOL_DEFINITIONS,
                        messages=messages,
                    )
                    break
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    if attempt < 4:
                        wait = 15 * (attempt + 1)
                        print(f"  [Executor] API error, waiting {wait}s (attempt {attempt+1}/5): {type(e).__name__}")
                        time.sleep(wait)
                    else:
                        print(f"  [Executor] API error at turn {turn} after 5 attempts: {e}")
                except Exception as e:
                    print(f"  [Executor] API error at turn {turn}: {e}")
                    break
            if response is None:
                break

            # Track tokens
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cumulative_tokens += input_tokens + output_tokens

            # Process response content blocks
            assistant_text = ""
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    assistant_text += block.text
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Emit assistant state packet
            episode.add_packet(LLMStatePacket(
                timestep=turn,
                task_id=task.task_id,
                role="assistant",
                content=assistant_text,
                tool_use=tool_uses[0] if tool_uses else None,
                token_count=input_tokens + output_tokens,
                cumulative_tokens=cumulative_tokens,
            ))

            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": response.content})

            # If no tool use, the executor is done talking
            if response.stop_reason == "end_turn" and not tool_uses:
                print(f"  [Executor] Finished at turn {turn} (no more tool calls)")
                break

            # Execute tools
            if tool_uses:
                tool_results = []
                for tool_call in tool_uses:
                    # Check if we should inject an error at this turn
                    inject = None
                    if inject_error_at_turn == turn and inject_error_msg:
                        inject = inject_error_msg

                    result = execute_tool(
                        tool_name=tool_call["name"],
                        tool_input=tool_call["input"],
                        workspace_dir=task_workspace,
                        test_fn=task.test_fn if tool_call["name"] == "test_solution" else None,
                        inject_error=inject,
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": result,
                    })

                    # Emit tool result state packet
                    episode.add_packet(LLMStatePacket(
                        timestep=turn,
                        task_id=task.task_id,
                        role="tool_result",
                        content="",
                        tool_use={"name": tool_call["name"], "input": tool_call["input"]},
                        tool_result=result,
                        token_count=0,
                        cumulative_tokens=cumulative_tokens,
                    ))

                    # Check if tests passed
                    if "ALL TESTS PASSED" in result:
                        task_completed = True

                    print(f"  [Executor] Turn {turn}: {tool_call['name']} -> "
                          f"{result[:80]}{'...' if len(result) > 80 else ''}")

                messages.append({"role": "user", "content": tool_results})

            # Stop if task is complete
            if task_completed:
                print(f"  [Executor] Task completed at turn {turn}!")
                break

        episode.task_success = task_completed
        episode.completed_at = datetime.now().isoformat()

        print(f"  [Executor] Episode done: {len(episode.packets)} packets, "
              f"{episode.total_tokens} tokens, success={task_completed}")

        return episode
