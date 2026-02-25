#!/usr/bin/env python3
"""
Entry point for Experiment 4: LLM Executor-Observer.

Usage:
    # Quick smoke test (1 task, ~$1-2)
    python -m experiment4_llm_observer.run_experiment --quick

    # Full experiment (5 tasks + perturbations + probes, ~$14-16)
    python -m experiment4_llm_observer.run_experiment --full

    # Specific tasks only
    python -m experiment4_llm_observer.run_experiment --tasks logic_puzzle,debug_scraper

Before running, set your API key:
    export ANTHROPIC_API_KEY='your-key-here'
"""

import argparse
import json
import sys
import os

# Add parent dir to path so we can import as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4_llm_observer.config import get_api_key
from experiment4_llm_observer.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: LLM Executor-Observer")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with 1 task (~$1-2)")
    parser.add_argument("--full", action="store_true",
                        help="Full experiment with all tasks and probes (~$14-16)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task IDs to run")
    parser.add_argument("--quick-task", type=str, default="logic_puzzle",
                        help="Task ID for quick test (default: logic_puzzle)")

    args = parser.parse_args()

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.quick:
        print("Running quick smoke test...")
        orchestrator = Orchestrator(api_key)
        results = orchestrator.run_quick_test(task_id=args.quick_task)
        print("\n" + json.dumps(results, indent=2, default=str))

    elif args.full:
        print("Running full experiment...")
        task_ids = args.tasks.split(",") if args.tasks else None
        orchestrator = Orchestrator(api_key, task_ids=task_ids)
        results = orchestrator.run_full_experiment()

    elif args.tasks:
        print(f"Running selected tasks: {args.tasks}")
        task_ids = args.tasks.split(",")
        orchestrator = Orchestrator(api_key, task_ids=task_ids)
        results = orchestrator.run_full_experiment()

    else:
        parser.print_help()
        print("\nExample:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  python -m experiment4_llm_observer.run_experiment --quick")


if __name__ == "__main__":
    main()
