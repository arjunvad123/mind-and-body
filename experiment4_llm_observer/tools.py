"""
Tool definitions and execution for the executor.

These are REAL tools — the executor actually runs code and gets real results.
This creates genuine uncertainty and meaningful error recovery behavior,
unlike mocked tools that would produce predictable outputs.
"""

import subprocess
import os
import json
from . import config


# --- Tool definitions for Claude API ---

TOOL_DEFINITIONS = [
    {
        "name": "run_python",
        "description": "Execute Python code and return stdout/stderr. Use this to test your implementations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the task workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the task workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "test_solution",
        "description": "Run the test cases for the current task. Returns pass/fail for each test.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to test"
                }
            },
            "required": ["task_id"]
        }
    },
]


def execute_tool(tool_name: str, tool_input: dict, workspace_dir: str,
                 test_fn=None, inject_error: str = None) -> str:
    """Execute a tool and return the result string.

    Args:
        tool_name: Which tool to run
        tool_input: The tool's input parameters
        workspace_dir: Path to the task workspace directory
        test_fn: Optional test function for test_solution tool
        inject_error: If set, return this error string instead of real result
                      (used by perturbation engine)
    """
    if inject_error and tool_name == "run_python":
        return inject_error

    if tool_name == "run_python":
        return _run_python(tool_input["code"], workspace_dir)
    elif tool_name == "read_file":
        return _read_file(tool_input["path"], workspace_dir)
    elif tool_name == "write_file":
        return _write_file(tool_input["path"], tool_input["content"], workspace_dir)
    elif tool_name == "test_solution":
        if test_fn:
            return test_fn(workspace_dir)
        return "Error: No test function configured for this task."
    else:
        return f"Error: Unknown tool '{tool_name}'"


def _run_python(code: str, workspace_dir: str) -> str:
    """Execute Python code in a subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=config.TASK_TIMEOUT_SECONDS,
            cwd=workspace_dir,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + f"STDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(Exit code: {result.returncode})"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {config.TASK_TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"Error executing code: {str(e)}"


def _read_file(path: str, workspace_dir: str) -> str:
    """Read a file from the workspace."""
    # Prevent path traversal
    full_path = os.path.normpath(os.path.join(workspace_dir, path))
    if not full_path.startswith(os.path.normpath(workspace_dir)):
        return "Error: Path traversal not allowed"

    try:
        with open(full_path, "r") as f:
            content = f.read()
        return content if content else "(empty file)"
    except FileNotFoundError:
        return f"Error: File '{path}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _write_file(path: str, content: str, workspace_dir: str) -> str:
    """Write a file to the workspace."""
    full_path = os.path.normpath(os.path.join(workspace_dir, path))
    if not full_path.startswith(os.path.normpath(workspace_dir)):
        return "Error: Path traversal not allowed"

    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
