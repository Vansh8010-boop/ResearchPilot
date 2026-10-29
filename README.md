# ResearchPilot

An autonomous research agent: it breaks a question down, searches the web, reads pages, does exact arithmetic, and answers with cited sources.

**Status: week 1.** A ReAct loop (think → call tools → observe) with three tools, a hard step limit, a tool cache and per-run JSON traces. Plan-and-execute, the benchmark, hardening, LLM-as-judge evaluation and the FastAPI/Streamlit/Docker layer come in later weeks.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
uv run rp "What is the population of Japan divided by the area of Texas in square km?"
uv run rp --effort low --max-steps 5 "Who won the 2024 Nobel Prize in Physics, and for what?"
uv run rp --no-cache "..."   # bypass the tool cache
```

Each run prints the tool calls as they happen, then the final answer and the path of its trace in `traces/`.

## Layout

```
src/research_pilot/
  agent/react.py      ReAct loop: step limit, forced final answer, refusal handling
  tools/
    __init__.py       tool schemas sent to the model + Toolbox (validation, dispatch, cache)
    search.py         web_search via Tavily
    fetch.py          fetch_page: httpx + trafilatura, 8k-char cap
    calculator.py     safe AST evaluator (no eval())
    cache.py          disk cache for search/fetch results
  tracing.py          per-run JSON trace (LLM calls, tool calls, tokens, latency)
  config.py           settings from env / .env
  cli.py              `rp` command
tests/                offline tests (fake LLM client, mocked HTTP)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RP_MODEL` | `claude-opus-5-5` | model the agent runs on |
| `RP_EFFORT` | `medium` | reasoning effort: low / medium / high / xhigh / max |
| `RP_MAX_STEPS` | `8` | tool-using turns before the agent is forced to answer |
| `RP_MAX_TOKENS` | `16000` | max output tokens per model call |

## Tests

```bash
uv run pytest
```
