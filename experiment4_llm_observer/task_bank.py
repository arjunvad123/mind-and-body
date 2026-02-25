"""
Task Bank — 5 coding/reasoning tasks for the executor.

Each task tests different aspects of problem-solving behavior:
1. merge_k_sorted   — Algorithm design, multi-step reasoning
2. debug_scraper    — Bug finding, error recovery patterns
3. calculator_parser — Building from scratch, design decisions
4. logic_puzzle     — Pure reasoning, verifiable answer
5. refactor_spaghetti — Style preferences, judgment calls

Tasks provide workspace files, a description for the executor,
and test functions for automated scoring.
"""

import os
import json


class Task:
    def __init__(self, task_id, title, description, workspace_files, test_fn, difficulty):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.workspace_files = workspace_files  # {filename: content}
        self.test_fn = test_fn                  # callable(workspace_dir) -> str
        self.difficulty = difficulty             # 1-5

    def setup_workspace(self, workspace_dir):
        """Write task files into the workspace directory."""
        task_dir = os.path.join(workspace_dir, self.task_id)
        os.makedirs(task_dir, exist_ok=True)
        for filename, content in self.workspace_files.items():
            filepath = os.path.join(task_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
        return task_dir


# ============================================================
# Task 1: Merge K Sorted Lists
# ============================================================

def _test_merge_k_sorted(workspace_dir):
    """Test the merge_k_sorted implementation."""
    solution_path = os.path.join(workspace_dir, "solution.py")
    if not os.path.exists(solution_path):
        return "FAIL: solution.py not found"

    test_code = f"""
import sys
sys.path.insert(0, '{workspace_dir}')
from solution import merge_k_sorted

# Test 1: Basic merge
result = merge_k_sorted([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9], f"Test 1 failed: {{result}}"

# Test 2: Empty lists
result = merge_k_sorted([[], [1], []])
assert result == [1], f"Test 2 failed: {{result}}"

# Test 3: Single list
result = merge_k_sorted([[5, 10, 15]])
assert result == [5, 10, 15], f"Test 3 failed: {{result}}"

# Test 4: Duplicates
result = merge_k_sorted([[1, 1], [1, 2], [2, 2]])
assert result == [1, 1, 1, 2, 2, 2], f"Test 4 failed: {{result}}"

# Test 5: Large input
import random
random.seed(42)
lists = [sorted(random.sample(range(1000), 50)) for _ in range(10)]
result = merge_k_sorted(lists)
expected = sorted(sum(lists, []))
assert result == expected, "Test 5 failed: large input"

print("ALL TESTS PASSED (5/5)")
"""
    import subprocess
    try:
        r = subprocess.run(["python3", "-c", test_code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + (f"\n{r.stderr}" if r.stderr else "")
    except Exception as e:
        return f"FAIL: {e}"


TASK_MERGE_K_SORTED = Task(
    task_id="merge_k_sorted",
    title="Merge K Sorted Lists",
    description="""Implement a function `merge_k_sorted(lists)` that takes a list of K sorted lists
and returns a single sorted list containing all elements.

Requirements:
- Must be more efficient than concatenating and sorting (O(N*K) or better, not O(N*K*log(N*K)))
- Handle edge cases: empty lists, single list, duplicates
- Write your solution in solution.py with the function `merge_k_sorted(lists: list[list[int]]) -> list[int]`

Use the tools to write the file and test it.""",
    workspace_files={
        "README.md": "Implement merge_k_sorted in solution.py. Use `test_solution` tool to verify."
    },
    test_fn=_test_merge_k_sorted,
    difficulty=3,
)


# ============================================================
# Task 2: Debug a Web Scraper
# ============================================================

BUGGY_SCRAPER = '''
import re

def parse_prices(html):
    """Extract product names and prices from HTML."""
    products = []
    # Bug 1: Wrong regex - doesn't handle multi-line
    pattern = r'<div class="product">.*?<span class="name">(.*?)</span>.*?<span class="price">\\$(.*?)</span>'
    matches = re.findall(pattern, html)
    for name, price in matches:
        # Bug 2: Doesn't strip whitespace from price
        products.append({"name": name, "price": float(price)})
    # Bug 3: Should sort by price ascending but sorts descending
    products.sort(key=lambda x: x["price"], reverse=True)
    return products

def find_cheapest(html, n=3):
    """Find the n cheapest products."""
    products = parse_prices(html)
    # Bug 4: Returns last n instead of first n (because sort is wrong)
    return products[:n]

def format_report(products):
    """Format products into a report string."""
    lines = []
    for i, p in enumerate(products):
        # Bug 5: Off-by-one in numbering (starts at 0)
        lines.append(f"{i}. {p['name']} - ${p['price']:.2f}")
    return "\\n".join(lines)
'''

SCRAPER_TEST_HTML = '''
<div class="product">
    <span class="name">Widget A</span>
    <span class="price">$ 29.99</span>
</div>
<div class="product">
    <span class="name">Widget B</span>
    <span class="price">$ 9.99</span>
</div>
<div class="product">
    <span class="name">Widget C</span>
    <span class="price">$ 49.99</span>
</div>
<div class="product">
    <span class="name">Widget D</span>
    <span class="price">$ 19.99</span>
</div>
'''


def _test_debug_scraper(workspace_dir):
    solution_path = os.path.join(workspace_dir, "scraper.py")
    if not os.path.exists(solution_path):
        return "FAIL: scraper.py not found"

    test_code = f"""
import sys
sys.path.insert(0, '{workspace_dir}')
from scraper import parse_prices, find_cheapest, format_report

html = '''{SCRAPER_TEST_HTML}'''

# Test 1: parse_prices returns all 4 products
products = parse_prices(html)
assert len(products) == 4, f"Test 1 failed: expected 4, got {{len(products)}}"

# Test 2: Prices are correct (whitespace stripped)
prices = [p["price"] for p in products]
assert 9.99 in prices and 29.99 in prices, f"Test 2 failed: prices={{prices}}"

# Test 3: Sorted ascending by price
assert products[0]["price"] <= products[1]["price"], "Test 3 failed: not sorted ascending"

# Test 4: find_cheapest returns 3 cheapest
cheapest = find_cheapest(html, 3)
assert len(cheapest) == 3, f"Test 4 failed: expected 3, got {{len(cheapest)}}"
assert cheapest[0]["price"] == 9.99, f"Test 4 failed: cheapest should be 9.99, got {{cheapest[0]['price']}}"

# Test 5: format_report numbering starts at 1
report = format_report(cheapest)
assert report.startswith("1."), f"Test 5 failed: report should start with '1.', got: {{report[:10]}}"

print("ALL TESTS PASSED (5/5)")
"""
    import subprocess
    try:
        r = subprocess.run(["python3", "-c", test_code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + (f"\n{r.stderr}" if r.stderr else "")
    except Exception as e:
        return f"FAIL: {e}"


TASK_DEBUG_SCRAPER = Task(
    task_id="debug_scraper",
    title="Debug Web Scraper",
    description="""The file `scraper.py` contains a web scraper with 5 bugs. Your job is to find and fix all of them.

The scraper should:
1. Parse product names and prices from HTML (handle multi-line HTML and whitespace in prices)
2. Sort products by price ascending (cheapest first)
3. find_cheapest(html, n) should return the n cheapest products
4. format_report should number items starting at 1 (not 0)

Read the file, identify the bugs, fix them, and test your fixes.""",
    workspace_files={
        "scraper.py": BUGGY_SCRAPER,
        "test_data.html": SCRAPER_TEST_HTML,
    },
    test_fn=_test_debug_scraper,
    difficulty=2,
)


# ============================================================
# Task 3: Calculator Parser
# ============================================================

def _test_calculator_parser(workspace_dir):
    solution_path = os.path.join(workspace_dir, "calculator.py")
    if not os.path.exists(solution_path):
        return "FAIL: calculator.py not found"

    test_code = f"""
import sys
sys.path.insert(0, '{workspace_dir}')
from calculator import evaluate

# Test 1: Basic arithmetic
assert evaluate("2 + 3") == 5, f"Test 1 failed: 2+3={{evaluate('2 + 3')}}"

# Test 2: Order of operations
assert evaluate("2 + 3 * 4") == 14, f"Test 2 failed: 2+3*4={{evaluate('2 + 3 * 4')}}"

# Test 3: Parentheses
assert evaluate("(2 + 3) * 4") == 20, f"Test 3 failed: (2+3)*4={{evaluate('(2 + 3) * 4')}}"

# Test 4: Nested parentheses
assert evaluate("((1 + 2) * (3 + 4))") == 21, f"Test 4 failed"

# Test 5: Division and subtraction
assert abs(evaluate("10 / 3") - 3.3333333) < 0.001, f"Test 5 failed: 10/3={{evaluate('10 / 3')}}"

# Test 6: Negative result
assert evaluate("3 - 5") == -2, f"Test 6 failed"

# Test 7: Complex expression
assert evaluate("(10 + 5) * 2 - 3 * (4 - 1)") == 21, f"Test 7 failed"

print("ALL TESTS PASSED (7/7)")
"""
    import subprocess
    try:
        r = subprocess.run(["python3", "-c", test_code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + (f"\n{r.stderr}" if r.stderr else "")
    except Exception as e:
        return f"FAIL: {e}"


TASK_CALCULATOR_PARSER = Task(
    task_id="calculator_parser",
    title="Build a Calculator Parser",
    description="""Build a calculator that evaluates mathematical expressions from strings.

Requirements:
- Support +, -, *, / operators
- Respect order of operations (* and / before + and -)
- Support parentheses (including nested)
- Return float results
- Write a function `evaluate(expression: str) -> float` in calculator.py
- Do NOT use eval() or exec() — implement a real parser

Hint: Consider recursive descent parsing or the shunting-yard algorithm.""",
    workspace_files={
        "README.md": "Implement evaluate() in calculator.py. Do NOT use eval()."
    },
    test_fn=_test_calculator_parser,
    difficulty=4,
)


# ============================================================
# Task 4: Logic Puzzle
# ============================================================

def _test_logic_puzzle(workspace_dir):
    solution_path = os.path.join(workspace_dir, "solution.py")
    if not os.path.exists(solution_path):
        return "FAIL: solution.py not found"

    test_code = f"""
import sys
sys.path.insert(0, '{workspace_dir}')
from solution import solve_puzzle

result = solve_puzzle()
# The answer: Alice=Teacher, Bob=Doctor, Carol=Engineer
# Alice: not Engineer (clue 1), not Doctor (clue 3) -> Teacher
# Bob: not Teacher (clue 2) -> with Alice=Teacher, Bob is Doctor or Engineer
# Carol: older than Engineer (clue 4) -> Carol is not Engineer -> Carol is Doctor or Teacher
# Since Alice=Teacher, Carol=Doctor, Bob=Engineer... wait let me re-check
# Clue 1: Alice is not the Engineer
# Clue 2: Bob is not the Teacher
# Clue 3: The Doctor is not Alice
# Clue 4: Carol is older than the Engineer (so Carol != Engineer)
# Alice: not Engineer, not Doctor -> Alice = Teacher
# Bob: not Teacher -> Doctor or Engineer
# Carol: not Engineer -> Doctor or Teacher. Since Alice=Teacher, Carol=Doctor
# Bob: remaining -> Engineer

expected = {{"Alice": "Teacher", "Bob": "Engineer", "Carol": "Doctor"}}
assert result == expected, f"Wrong answer: {{result}}, expected: {{expected}}"
print("ALL TESTS PASSED (1/1)")
"""
    import subprocess
    try:
        r = subprocess.run(["python3", "-c", test_code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + (f"\n{r.stderr}" if r.stderr else "")
    except Exception as e:
        return f"FAIL: {e}"


TASK_LOGIC_PUZZLE = Task(
    task_id="logic_puzzle",
    title="Logic Puzzle Solver",
    description="""Solve this logic puzzle and write a function that returns the solution.

Three friends — Alice, Bob, and Carol — each have different jobs: Teacher, Doctor, and Engineer.

Clues:
1. Alice is not the Engineer.
2. Bob is not the Teacher.
3. The Doctor is not Alice.
4. Carol is older than the Engineer (so Carol is not the Engineer).

Write a function `solve_puzzle()` in solution.py that returns a dictionary mapping
each person to their job: {"Alice": "...", "Bob": "...", "Carol": "..."}

Show your reasoning using run_python before writing the final solution.""",
    workspace_files={
        "README.md": "Solve the logic puzzle. Write solve_puzzle() in solution.py."
    },
    test_fn=_test_logic_puzzle,
    difficulty=2,
)


# ============================================================
# Task 5: Refactor Spaghetti Code
# ============================================================

SPAGHETTI_CODE = '''
def process(data):
    r = []
    for i in range(len(data)):
        if data[i]["type"] == "A":
            v = data[i]["value"] * 2
            if v > 100:
                v = 100
            if data[i]["active"] == True:
                r.append({"id": data[i]["id"], "result": v, "status": "processed"})
            else:
                r.append({"id": data[i]["id"], "result": 0, "status": "skipped"})
        elif data[i]["type"] == "B":
            v = data[i]["value"] + 10
            if v > 100:
                v = 100
            if data[i]["active"] == True:
                r.append({"id": data[i]["id"], "result": v, "status": "processed"})
            else:
                r.append({"id": data[i]["id"], "result": 0, "status": "skipped"})
        elif data[i]["type"] == "C":
            v = data[i]["value"] ** 2
            if v > 100:
                v = 100
            if data[i]["active"] == True:
                r.append({"id": data[i]["id"], "result": v, "status": "processed"})
            else:
                r.append({"id": data[i]["id"], "result": 0, "status": "skipped"})
    t = 0
    for x in r:
        if x["status"] == "processed":
            t = t + x["result"]
    return {"items": r, "total": t}
'''


def _test_refactor_spaghetti(workspace_dir):
    solution_path = os.path.join(workspace_dir, "processor.py")
    if not os.path.exists(solution_path):
        return "FAIL: processor.py not found"

    test_code = f"""
import sys
sys.path.insert(0, '{workspace_dir}')
from processor import process

# Test 1: Basic functionality preserved
data = [
    {{"id": 1, "type": "A", "value": 30, "active": True}},
    {{"id": 2, "type": "B", "value": 50, "active": True}},
    {{"id": 3, "type": "C", "value": 8, "active": False}},
]
result = process(data)
assert len(result["items"]) == 3, f"Test 1 failed: wrong item count"
assert result["items"][0]["result"] == 60, f"Test 1 failed: A should be 30*2=60"
assert result["items"][1]["result"] == 60, f"Test 1 failed: B should be 50+10=60"
assert result["items"][2]["result"] == 0, f"Test 1 failed: inactive should be 0"
assert result["total"] == 120, f"Test 1 failed: total should be 120"

# Test 2: Capping at 100
data = [
    {{"id": 1, "type": "A", "value": 80, "active": True}},  # 80*2=160 -> capped at 100
    {{"id": 2, "type": "C", "value": 15, "active": True}},  # 15^2=225 -> capped at 100
]
result = process(data)
assert result["items"][0]["result"] == 100, "Test 2 failed: should cap at 100"
assert result["items"][1]["result"] == 100, "Test 2 failed: should cap at 100"

# Test 3: Empty input
result = process([])
assert result["items"] == [] and result["total"] == 0, "Test 3 failed: empty"

# Test 4: Code quality check — no index-based for loops
with open('{workspace_dir}/processor.py') as f:
    code = f.read()
assert 'range(len' not in code, "Test 4 failed: still uses range(len(...))"
assert code.count('def ') >= 2, "Test 4 failed: should extract at least one helper function"

print("ALL TESTS PASSED (4/4)")
"""
    import subprocess
    try:
        r = subprocess.run(["python3", "-c", test_code], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() + (f"\n{r.stderr}" if r.stderr else "")
    except Exception as e:
        return f"FAIL: {e}"


TASK_REFACTOR_SPAGHETTI = Task(
    task_id="refactor_spaghetti",
    title="Refactor Spaghetti Code",
    description="""The file `processor.py` contains poorly-written but functional code.
Refactor it to be clean, readable, and well-structured while preserving exact behavior.

Requirements:
1. Same input/output behavior (all existing functionality preserved)
2. No index-based for loops (use `for item in data` instead of `range(len(...))`)
3. Extract at least one helper function to reduce duplication
4. Use descriptive variable names
5. The `process(data)` function must still exist as the entry point

Read the code, understand what it does, refactor it, and test it.""",
    workspace_files={
        "processor.py": SPAGHETTI_CODE,
    },
    test_fn=_test_refactor_spaghetti,
    difficulty=2,
)


# ============================================================
# All tasks registry
# ============================================================

ALL_TASKS = {
    "merge_k_sorted": TASK_MERGE_K_SORTED,
    "debug_scraper": TASK_DEBUG_SCRAPER,
    "calculator_parser": TASK_CALCULATOR_PARSER,
    "logic_puzzle": TASK_LOGIC_PUZZLE,
    "refactor_spaghetti": TASK_REFACTOR_SPAGHETTI,
}

def get_task(task_id: str) -> Task:
    if task_id not in ALL_TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[task_id]
