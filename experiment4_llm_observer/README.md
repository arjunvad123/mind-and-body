# Experiment 4: LLM Executor-Observer

> **Status: Complete**

The pivot from RL to LLMs. Instead of a neural network learning CartPole, the executor is Claude Sonnet solving real coding tasks with actual tools (run_python, read/write files). A separate Claude Sonnet instance acts as observer, watching the full state stream of messages, tool calls, and results. One-way information flow is preserved -- the observer cannot influence the executor.

This experiment is qualitative/LLM-evaluated, unlike the quantitative neural network experiments. Results depend on Claude's evaluation of its own predictions about another Claude instance's behavior.

## Architecture

- **Executor:** Claude Sonnet with access to real tools (run_python, read_file, write_file). Solves coding tasks end-to-end, producing a stream of messages, tool calls, and tool results.
- **Observer:** A separate Claude Sonnet instance. Receives the executor's state stream (messages, tool calls, results) in real time. Makes predictions about the executor's next actions and internal reasoning.
- **Key constraint:** One-way information flow. The observer reads the executor's stream but cannot write to it or influence the executor's behavior in any way.

## Tasks

5 coding tasks of varying complexity:

| Task | Description |
|------|-------------|
| `merge_k_sorted` | Implement merge of K sorted lists |
| `debug_scraper` | Debug a broken web scraper |
| `calculator_parser` | Build an expression parser |
| `logic_puzzle` | Solve a constraint satisfaction problem |
| `refactor_spaghetti` | Refactor tangled legacy code |

## Perturbations

3 perturbation types applied to each task:

| Type | Description |
|------|-------------|
| `prompt_swap` | Change the executor's system prompt mid-task |
| `error_inject` | Introduce errors into tool results |
| `model_swap` | Swap the executor from Claude Sonnet to Claude Haiku mid-task |

## Probes

5 probes adapted from the neural network experiments:

1. **Self-model discrimination** -- Can the observer distinguish its paired executor from a different one?
2. **First thought vs. deliberation** -- Are the observer's immediate predictions better than its reasoned predictions?
3. **Confabulation** -- When the executor is perturbed, does the observer confabulate explanations?
4. **Surprise tracking** -- Does the observer's prediction error spike at meaningful moments?
5. **Emergent preferences** -- Does the observer develop systematic preferences about the executor's approach?

## Data

24 JSON recordings in `data/`:
- 5 baseline recordings (one per task, no perturbation)
- 15 perturbed recordings (one per task-perturbation combination)
- 4 workspace task files used by the executor

## Key Insight

LLM reasoning provides richer, more variable thought processes than RL agents, with natural perturbation points (prompt changes, model swaps, error injection). The qualitative nature of LLM-on-LLM evaluation introduces its own limitations, but the richness of the behavioral stream far exceeds what CartPole could provide. This experiment bridges the gap between the simple RL setup of Experiment 1 and the quantitative neural-level analysis of Experiments 5 and 6.

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main entry point |
| `resume_experiment.py` | Resume a partially completed experiment run |
| `executor_engine.py` | Claude executor with tool access |
| `observer_engine.py` | Claude observer watching the state stream |
| `orchestrator.py` | Coordinates executor-observer interaction and recording |
| `perturbation_engine.py` | Implements prompt_swap, error_inject, model_swap |
| `probes.py` | All 5 probe implementations (LLM-evaluated) |
| `task_bank.py` | Task definitions and workspace setup |
| `tools.py` | Tool implementations (run_python, read/write files) |
| `protocol.py` | State packet and message protocol definitions |
| `config.py` | Experiment configuration |
| `data/` | Recorded sessions (JSON) |

## Prerequisites

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## How to Run

```bash
python -m experiment4_llm_observer.run_experiment
```
