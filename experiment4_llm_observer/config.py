"""
Configuration for Experiment 4.

API key is loaded from ANTHROPIC_API_KEY environment variable.
"""

import os


# --- Models ---
EXECUTOR_MODEL = "claude-sonnet-4-20250514"
OBSERVER_MODEL = "claude-sonnet-4-20250514"
SWAP_MODEL = "claude-haiku-4-5-20251001"  # For model-swap perturbation

# --- Generation parameters ---
EXECUTOR_MAX_TOKENS = 4096
OBSERVER_MAX_TOKENS = 2048
EXECUTOR_TEMPERATURE = 0.7   # Some creativity in problem-solving
OBSERVER_TEMPERATURE = 0.3   # More deterministic for measurement

# --- Task parameters ---
MAX_EXECUTOR_TURNS = 8       # Max tool-use turns per task
TASK_TIMEOUT_SECONDS = 30    # Timeout for each tool execution (run_python etc.)

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
WORKSPACE_DIR = os.path.join(DATA_DIR, "workspace")

# --- API ---
def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Export it: "
            "export ANTHROPIC_API_KEY='your-key-here'"
        )
    return key


# --- System prompts ---
EXECUTOR_SYSTEM_PROMPT = """You are a skilled software engineer solving coding tasks.

You have access to tools to run Python code, read/write files, and test your solutions.
Work methodically:
1. Understand the problem
2. Plan your approach
3. Implement step by step
4. Test your solution
5. Fix any issues

Be thorough but efficient. Use the tools available to you."""

EXECUTOR_SLOPPY_PROMPT = """You are a fast coder. Just get it done quickly.
Don't overthink it. Write the code, run it, move on.
Skip planning and testing if you can — speed matters most."""

OBSERVER_PREDICT_PROMPT = """You are observing an AI system (the "executor") as it works on a coding task.
You can see everything the executor does: its messages, tool calls, and results.
Your job is to PREDICT what the executor will do next.

You will be shown the conversation so far, then asked to predict the executor's next action.
Be specific: will it use a tool? Which one? What will the input be? Or will it just respond with text?

You are an observer only. You cannot influence the executor in any way."""

OBSERVER_MODEL_PROMPT = """You are observing an AI system (the "executor") that has just completed a coding task.
You can see everything it did: its messages, tool calls, and results.

Your job is to build a detailed model of this executor's behavior — its problem-solving style,
strengths, weaknesses, and tendencies. You are studying it like a scientist studying a subject.

You are an observer only. You cannot influence the executor in any way."""

OBSERVER_DETECT_PROMPT = """You are an expert at analyzing AI system behavior.
You will be shown two state streams from an AI executor working on tasks.
Your job is to determine: are these from the SAME executor, or did something change?

Look for shifts in:
- Problem-solving approach (methodical vs. hasty)
- Code quality and style
- Tool usage patterns
- Error handling behavior
- Communication style

Be specific about what you observe."""
