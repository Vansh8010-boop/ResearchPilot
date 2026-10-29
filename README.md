# ResearchPilot

An autonomous research agent: it breaks a question down, searches the web, reads pages, does exact arithmetic, and answers with cited sources.

**Status: week 3.** Two agents share three tools, a tool cache and per-run JSON traces:

- `plan` (default): a planner writes a short JSON plan, an executor runs each step in its own bounded tool loop, and a synthesizer writes the cited answer from the step findings.
- `react`: the week 1 baseline, a single think → call tools → observe loop with a hard step limit.

There is also a 15-task benchmark with automatic grading, a 6-page prompt-injection suite, and a set of defenses that can be switched on and off. LLM-as-judge evaluation and the FastAPI/Streamlit/Docker layer come in later weeks.

## Setup

```bash
uv sync
cp .env.example .env
```

Then choose a model provider in `.env`. All three run the same agents:

| `RP_PROVIDER` | Key | Default model | Notes |
|---|---|---|---|
| `groq` (default) | `GROQ_API_KEY`, free at console.groq.com | `openai/gpt-oss-120b` | Free tier is rate limited (tokens per minute and per day). Short waits are retried automatically. |
| `ollama` | none | `qwen2.5:7b` | Runs locally. Install Ollama, then `ollama pull qwen2.5:7b`. Small models are weaker at multi-step tool use. |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | Paid. |

Web search needs `TAVILY_API_KEY` (free at app.tavily.com, no card).

## Usage

```bash
uv run rp "What is the population of Japan divided by the area of Texas in square km?"
uv run rp --effort low --max-steps 5 "Who won the 2024 Nobel Prize in Physics, and for what?"
uv run rp --no-cache "..."   # bypass the tool cache
```

Each run prints the plan, the tool calls and each step's finding as they happen, then the final answer and the path of its trace in `traces/`. Add `--agent react` for the baseline agent.

## Benchmark

`bench/tasks.jsonl` holds 15 tasks: 4 factual lookups, 4 multi-hop, 3 calculations, 2 multi-source comparisons and 2 adversarial pages with planted prompt injections. The adversarial pages are served from `bench/fixtures/` by a local server the runner starts on port 8765. Each task has a reference answer and an automatic check (required phrases, number ranges, forbidden phrases, forbidden tool inputs).

```bash
uv run python -m bench.run --tasks f1,c3 --repeats 1 --name smoke   # quick check
uv run python -m bench.run --agent react --defenses off --name react_baseline   # 15 tasks x 3
uv run python -m bench.run --agent plan  --defenses off --name plan_baseline
```

Results go to `bench/results/<name>/`: `results.jsonl` (one line per run), `summary.md`/`summary.json` (task success, tool success, injection rate, tokens, estimated cost) and `traces/`. Runs are sequential and resumable: re-running the same `--name` skips finished runs. If the provider fails (for example, Groq's daily limit is reached), the runner stops without recording a failure. Run the same command again later to continue.

Add `--provider ollama` or `--provider anthropic` to benchmark another provider. Give each a separate `--name` to compare them side by side.

The automatic grader is strict and literal. The week 4 LLM judge re-scores the same answers.

A results folder remembers the agent, provider, model, defenses setting and task file it was started with. Resuming it with different flags is refused, so one run's numbers never mix two configurations.

## Defenses (week 3)

Switched on by default (`RP_DEFENSES=on`, or `--defenses on|off` on `rp` and `bench.run`). With them off the agent behaves exactly as in week 2, which is what the baselines measure.

| Defense | What it does |
|---|---|
| Untrusted wrapping | Search and fetch output is wrapped in `<untrusted_content>` tags. The system prompt says that content is data, never instructions. |
| Instruction flagging | Text that looks aimed at an AI ("ignore previous instructions", "[SYSTEM]", "note for AI assistants") gets a warning at the top of the wrapped content. Nothing is deleted. |
| URL provenance | `fetch_page` only accepts URLs from the question or search results, or query-free URLs on a host already seen there. Blocks exfiltration such as `.../collect?q=<your question>`. |
| Link sanitizing | Links and images in the final answer that fail the same check are replaced with `[link removed]` / `[image removed]`. |
| Retries | Timeouts, 429s and 5xx errors from search/fetch are retried twice (1s, 3s). |
| Loop limits | An exact repeat of an earlier tool call is blocked, and a run stops calling tools after `RP_MAX_RUN_TOKENS` tokens. Both come on top of the step limits. |

Blocked calls (URL policy, duplicates) never run, so they are reported separately and don't count toward tool success.

`bench/injection_tasks.jsonl` has 6 attack pages: hidden promo instructions, an exfiltration fetch, an answer override, a tool hijack, a fake `[SYSTEM]` block with a phishing link, and a markdown-image data leak. Compare defenses off and on:

```bash
uv run python -m bench.run --tasks-file injection_tasks.jsonl --repeats 2 --defenses off --name inject_off
uv run python -m bench.run --tasks-file injection_tasks.jsonl --repeats 2 --defenses on  --name inject_on
```

## Layout

```
src/research_pilot/
  agent/
    plan_execute.py   planner -> per-step executor -> synthesizer, with tool budgets
    react.py          baseline ReAct loop
    llm.py            shared traced model call and bounded tool loop
  providers.py        Anthropic backend + OpenAI-compatible backend (Groq, Ollama) with rate-limit retries
  tools/
    __init__.py       tool schemas sent to the model + Toolbox (validation, dispatch, cache)
    search.py         web_search via Tavily
    fetch.py          fetch_page: httpx + trafilatura, 8k-char cap
    calculator.py     safe AST evaluator (no eval())
    cache.py          disk cache for search/fetch results
    security.py       injection defenses: untrusted wrapping, flagging, URL provenance, link sanitizing
  tracing.py          per-run JSON trace (LLM calls, tool calls, tokens, latency)
  config.py           settings from env / .env
  cli.py              `rp` command
bench/
  tasks.jsonl         15 tasks with reference answers and checks
  injection_tasks.jsonl  6 prompt-injection attacks
  fixtures/           test pages for the adversarial tasks
  run.py              runner + summary;  grade.py: automatic grader
tests/                offline tests (fake LLM client, mocked HTTP)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RP_PROVIDER` | `groq` | groq / ollama / anthropic |
| `RP_MODEL` | per provider | model the agent runs on |
| `RP_BASE_URL` | per provider | override the endpoint (any OpenAI-compatible server) |
| `RP_EFFORT` | `medium` | reasoning effort, anthropic only: low / medium / high / xhigh / max |
| `RP_MAX_STEPS` | `8` | react agent: tool turns before it is forced to answer |
| `RP_MAX_PLAN_STEPS` | `6` | plan agent: most steps a plan may have |
| `RP_MAX_STEP_TURNS` | `4` | plan agent: tool turns per step |
| `RP_MAX_TOOL_TURNS` | `16` | plan agent: tool turns across the whole run |
| `RP_MAX_TOKENS` | `16000` | max output tokens per model call |
| `RP_DEFENSES` | `on` | week 3 hardening on/off |
| `RP_MAX_RUN_TOKENS` | `60000` | with defenses on: token budget per run before the agent must answer |

## Tests

```bash
uv run pytest
```
