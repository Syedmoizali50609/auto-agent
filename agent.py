"""
Auto-Agent: A fully autonomous AI agent powered by Claude.

Capabilities:
  - Web search (DuckDuckGo)
  - Python code execution
  - Shell command execution
  - File read / write
  - HTTP API calls
  - Calculator (safe math eval)
  - Current date/time awareness
  - Persistent conversation memory within a session

Usage:
  python agent.py                 # interactive REPL
  python agent.py "your task"     # run a single task then exit
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv

from tools import ToolExecutor

# ── colour helpers ────────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap *text* in an ANSI colour *code* when stdout is a TTY."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def blue(t: str)   -> str: return _c(t, "34")
def green(t: str)  -> str: return _c(t, "32")
def yellow(t: str) -> str: return _c(t, "33")
def cyan(t: str)   -> str: return _c(t, "36")
def bold(t: str)   -> str: return _c(t, "1")
def dim(t: str)    -> str: return _c(t, "2")

# ── tool schema (Anthropic tool-use format) ───────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_web",
        "description": (
            "Search the web via DuckDuckGo and return a summary plus related results. "
            "Use this whenever you need up-to-date or factual information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "The search query."},
                "num_results": {"type": "integer", "description": "Max results to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_python",
        "description": (
            "Execute a snippet of Python 3 code and return the result. "
            "Capture output by printing values. "
            "Use this for calculations, data processing, or anything that needs code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "execute_shell",
        "description": (
            "Run a shell command and return stdout, stderr, and return code. "
            "Use for file-system operations, running scripts, checking system state, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "file_read",
        "description": "Read and return the full contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Absolute or relative path to the file."},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "file_write",
        "description": "Write (or overwrite) content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file to write."},
                "content":  {"type": "string", "description": "Text content to write."},
            },
            "required": ["filepath", "content"],
        },
    },
    {
        "name": "api_call",
        "description": "Make an HTTP GET or POST request to any URL and return the response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":     {"type": "string", "description": "The URL to call."},
                "method":  {"type": "string", "enum": ["GET", "POST"], "description": "HTTP method (default GET)."},
                "headers": {"type": "object", "description": "Optional HTTP headers."},
                "data":    {"type": "object", "description": "Optional JSON body for POST requests."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Supports standard Python arithmetic operators and math functions "
            "(sin, cos, sqrt, log, pi, e, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 ** 10 + sqrt(16)'."},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_datetime",
        "description": "Return the current date and time (UTC) so the agent is aware of the current moment.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Auto-Agent, a highly capable and autonomous AI assistant.
Your goal is to help the user accomplish any task they give you — completely and correctly.

Guidelines:
1. Think step-by-step before acting. Break complex tasks into smaller sub-tasks.
2. Use the available tools proactively: search for up-to-date information, write and run code,
   read/write files, call APIs, and perform calculations whenever that helps.
3. After every tool call, analyse the result and decide what to do next.
4. Keep iterating with tools until the task is fully solved — never give up after one attempt.
5. When the task is complete, give the user a clear, concise final answer.
6. Be honest: if something is impossible or outside your capabilities, say so clearly.
7. Do not expose raw tool-call JSON in your final answer — present results in plain language.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """Autonomous agent that combines Claude with a rich tool-set."""

    MAX_TOOL_ITERATIONS = 20  # safety cap on agentic loops

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit(
                "Error: ANTHROPIC_API_KEY is not set.\n"
                "Create a .env file (see .env.example) or export the variable."
            )
        self.client   = anthropic.Anthropic(api_key=api_key)
        self.executor = ToolExecutor()
        self.history: list[dict[str, Any]] = []  # running conversation

    # ── tool dispatch ──────────────────────────────────────────────────────────

    def _dispatch(self, name: str, inputs: dict) -> Any:
        """Call the right ToolExecutor method and return its result."""
        if name == "search_web":
            return self.executor.search_web(
                inputs["query"], inputs.get("num_results", 5)
            )
        if name == "execute_python":
            return self.executor.execute_python(inputs["code"])
        if name == "execute_shell":
            return self.executor.execute_shell(inputs["command"])
        if name == "file_read":
            return self.executor.file_read(inputs["filepath"])
        if name == "file_write":
            return self.executor.file_write(inputs["filepath"], inputs["content"])
        if name == "api_call":
            return self.executor.api_call(
                inputs["url"],
                inputs.get("method", "GET"),
                inputs.get("headers"),
                inputs.get("data"),
            )
        if name == "calculator":
            return _safe_calc(inputs["expression"])
        if name == "get_datetime":
            return {"datetime": datetime.utcnow().isoformat() + "Z", "success": True}
        return {"success": False, "error": f"Unknown tool: {name}"}

    # ── agentic loop ───────────────────────────────────────────────────────────

    def _run_once(self, user_message: str) -> str:
        """
        Add *user_message* to history, then run the agentic loop:
        call Claude → execute tool calls → feed results back → repeat
        until Claude returns a final text response.
        """
        self.history.append({"role": "user", "content": user_message})

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.history,
            )

            # Extract text blocks and tool-use blocks
            text_blocks = [b for b in response.content if b.type == "text"]
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Append full assistant turn to history
            self.history.append({"role": "assistant", "content": response.content})

            # If Claude is done (no more tool calls), return the final answer
            if response.stop_reason == "end_turn" or not tool_blocks:
                return "\n".join(b.text for b in text_blocks)

            # Execute every requested tool and collect results
            tool_results = []
            for tb in tool_blocks:
                print(dim(f"  ⚙  Using tool: {tb.name}({json.dumps(tb.input, ensure_ascii=False)[:120]})"))
                result = self._dispatch(tb.name, tb.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Feed tool results back to Claude
            self.history.append({"role": "user", "content": tool_results})

        return "I reached the maximum number of steps without fully completing the task. Please try rephrasing or breaking the task into smaller steps."

    # ── public API ─────────────────────────────────────────────────────────────

    def ask(self, prompt: str) -> str:
        """Send a single prompt and return the agent's answer."""
        return self._run_once(prompt)

    def run(self) -> None:
        """Start the interactive REPL."""
        _print_banner()
        print(dim("Type your task and press Enter.  Commands: /clear  /quit\n"))

        while True:
            try:
                user_input = input(bold(green("You › "))).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n" + dim("Bye!"))
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print(dim("Bye!"))
                break
            if user_input.lower() == "/clear":
                self.history.clear()
                print(dim("Conversation cleared.\n"))
                continue

            print()
            answer = self.ask(user_input)
            print(f"\n{bold(cyan('Agent ›'))} {answer}\n")
            print(dim("─" * 60))


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_calc(expression: str) -> dict:
    """Evaluate a mathematical expression using a safe AST-based evaluator.

    Only numeric literals, standard arithmetic operators, and functions from
    the ``math`` standard-library module are permitted — no attribute access,
    no function calls outside math, no imports, and no side-effects.
    """
    import ast
    import math

    _MATH_FUNCS: dict[str, Any] = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }

    class _SafeEval(ast.NodeVisitor):
        """Walk an AST and evaluate only safe arithmetic nodes."""

        def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
            method = "visit_" + type(node).__name__
            visitor = getattr(self, method, None)
            if visitor is None:
                raise ValueError(f"Unsupported expression element: {type(node).__name__}")
            return visitor(node)

        def visit_Expression(self, node: ast.Expression) -> Any:
            return self.visit(node.body)

        def visit_Constant(self, node: ast.Constant) -> Any:
            if not isinstance(node.value, (int, float, complex)):
                raise ValueError("Only numeric constants are allowed.")
            return node.value

        def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
            ops = {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x}
            if type(node.op) not in ops:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return ops[type(node.op)](self.visit(node.operand))

        def visit_BinOp(self, node: ast.BinOp) -> Any:
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b,
                ast.FloorDiv: lambda a, b: a // b,
            }
            if type(node.op) not in ops:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            return ops[type(node.op)](self.visit(node.left), self.visit(node.right))

        def visit_Name(self, node: ast.Name) -> Any:
            if node.id not in _MATH_FUNCS:
                raise ValueError(f"Unknown name: '{node.id}'")
            return _MATH_FUNCS[node.id]

        def visit_Call(self, node: ast.Call) -> Any:
            if node.keywords or node.starargs if hasattr(node, "starargs") else False:
                raise ValueError("Keyword/star arguments are not allowed.")
            func = self.visit(node.func)
            if not callable(func):
                raise ValueError("Not callable.")
            args = [self.visit(a) for a in node.args]
            return func(*args)

    try:
        tree = ast.parse(expression, mode="eval")
        result = _SafeEval().visit(tree)
        return {"success": True, "result": result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _print_banner() -> None:
    banner = r"""
   ___         __            ___                    __
  / _ | __ __/ /____  ___  / _ | ___ ____ ___  ___/ /_
 / __ |/ // / __/ _ \/ _ \/ __ |/ _ `/ -_) _ \/ __/  /
/_/ |_|\_,_/\__/\___/_//_/_/ |_|\_,_/\__/_//_/\__/_/
"""
    print(cyan(bold(banner)))
    print(cyan("  Powered by Claude · Tools: web search, code exec, files, APIs, math\n"))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = Agent()
    if len(sys.argv) > 1:
        # Non-interactive: run the task passed as a CLI argument
        task = " ".join(sys.argv[1:])
        print(bold(green(f"Task: {task}\n")))
        answer = agent.ask(task)
        print(f"{bold(cyan('Agent ›'))} {answer}")
    else:
        agent.run()
