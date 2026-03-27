# auto-agent

A fully autonomous AI agent powered by [Claude](https://www.anthropic.com) that can tackle any task by combining language understanding with real-world tools.

## Features

| Tool | Description |
|------|-------------|
| 🔍 `search_web` | DuckDuckGo web search — no API key required |
| 🐍 `execute_python` | Run Python code and capture output |
| 🖥 `execute_shell` | Run shell commands |
| 📄 `file_read` / `file_write` | Read and write files |
| 🌐 `api_call` | HTTP GET / POST to any URL |
| 🔢 `calculator` | Safe maths expression evaluator |
| 🕐 `get_datetime` | Current UTC date & time |

The agent runs in an **agentic loop** — it keeps calling tools, observing results, and iterating until the task is complete.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

## Usage

### Interactive REPL

```bash
python agent.py
```

Type any task and press **Enter**. Use `/clear` to reset conversation history, `/quit` to exit.

### Single task (non-interactive)

```bash
python agent.py "Summarise the latest news about Python 3.13"
```

## Configuration

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) |
