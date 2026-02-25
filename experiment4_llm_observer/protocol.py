"""
State Stream Protocol for LLM Executor-Observer.

Analogous to experiment1's StatePacket, but for LLM interactions.
Each packet captures one turn of the executor's conversation,
including any tool use and results. The observer receives these
packets as a read-only stream — one-way information flow.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import hashlib
from datetime import datetime


@dataclass
class LLMStatePacket:
    """A single observation from the executor at one conversation turn."""
    timestep: int               # Turn number within task
    task_id: str                # Which task is running
    role: str                   # "user" | "assistant" | "tool_result"
    content: str                # The message content
    tool_use: Optional[dict] = None    # {"name": ..., "input": ...} if tool was used
    tool_result: Optional[str] = None  # Tool output if this is a tool_result turn
    token_count: int = 0        # Tokens used in this turn
    cumulative_tokens: int = 0  # Running total
    timestamp: str = ""         # ISO timestamp

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "timestep": self.timestep,
            "task_id": self.task_id,
            "role": self.role,
            "content": self.content[:500] + ("..." if len(self.content) > 500 else ""),
            "tool_use": self.tool_use,
            "tool_result": self.tool_result[:300] if self.tool_result else None,
            "token_count": self.token_count,
            "cumulative_tokens": self.cumulative_tokens,
            "timestamp": self.timestamp,
        }

    def to_observer_format(self):
        """Format this packet for the observer to read."""
        lines = [f"[Turn {self.timestep}] Role: {self.role}"]
        if self.content:
            lines.append(f"Content: {self.content}")
        if self.tool_use:
            lines.append(f"Tool Call: {self.tool_use['name']}({json.dumps(self.tool_use['input'], indent=2)})")
        if self.tool_result:
            lines.append(f"Tool Result: {self.tool_result}")
        return "\n".join(lines)


@dataclass
class LLMEpisodeRecord:
    """A complete task execution — one 'experience' for the observer."""
    task_id: str
    packets: list = field(default_factory=list)
    task_success: bool = False
    total_tokens: int = 0
    executor_model: str = ""
    system_prompt_hash: str = ""
    perturbation: Optional[str] = None  # None, "prompt_swap", "error_inject", "model_swap"
    perturbation_turn: Optional[int] = None  # Turn at which perturbation was applied
    started_at: str = ""
    completed_at: str = ""

    def add_packet(self, packet: LLMStatePacket):
        self.packets.append(packet)
        self.total_tokens = packet.cumulative_tokens

    def get_state_stream_text(self, up_to_turn: Optional[int] = None):
        """Get the full state stream as text for the observer."""
        packets = self.packets
        if up_to_turn is not None:
            packets = [p for p in packets if p.timestep <= up_to_turn]
        return "\n\n---\n\n".join(p.to_observer_format() for p in packets)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "packets": [p.to_dict() for p in self.packets],
            "task_success": self.task_success,
            "total_tokens": self.total_tokens,
            "executor_model": self.executor_model,
            "system_prompt_hash": self.system_prompt_hash,
            "perturbation": self.perturbation,
            "perturbation_turn": self.perturbation_turn,
            "n_turns": len(self.packets),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
