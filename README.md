# Agent

A lightweight ReAct (Reasoning + Acting) agent loop supporting Anthropic and
OpenAI-compatible APIs. Runs anywhere — as a standalone CLI, a Unix pipe, or a
handler forked by the [Actor Mesh]

## Quick Start

```sh
# Install
cargo build --release

# Run with a task
./target/release/agent "List files in /tmp and summarize them"

# Or pipe input
echo "What's the largest file in /etc?" | ./target/release/agent

# With model override
LLM_MODEL=claude-sonnet-4-20250514 ./target/release/agent "Review this code for bugs"
```

## How It Works

```
task → [think] → tool call → [act: run_shell, read_image, read_pdf, spawn_agent] → result → [think] → ... → answer
```

The agent loops until the LLM returns a text response (no more tool calls) or the
iteration limit is reached (default 50). Each iteration:

1. Sends the full conversation history + system prompt to the LLM
2. If the LLM returns tool calls, executes them and feeds results back
3. If the LLM returns text, the loop exits with that answer

## Built-in Tools

| Tool | Description |
|------|-------------|
| `run_shell` | Execute a shell command. Returns stdout + stderr. |
| `read_image` | Read a PNG/JPEG using vision AI. |
| `read_pdf` | Extract information from a PDF using AI. |
| `spawn_agent` | Delegate a self-contained subtask to an isolated sub-agent (top-level only). |

## Sub-agents

A top-level run can hand a focused subtask to a **sub-agent** via `spawn_agent`.
The sub-agent runs its own tool loop with **fresh, isolated context** and reports
back a structured result — the caller never has to carry the sub-agent's
intermediate steps in its own context. Sub-agents **cannot delegate further**
(recursion is capped at one level).

```jsonc
// spawn_agent arguments
{
  "task": "Find and summarize every TODO in ./src",  // required
  "checks": ["cite exact file:line for each"],        // optional: requirements the result must satisfy
  "persistence": "ephemeral",                          // "ephemeral" (default) | "durable"
  "max_iter": 25                                        // optional iteration budget
}
```

The result is a small JSON object: `{ "status": "success | partial | failure |
blocked", "output": "...", "steps_taken": N, ... }`.

- **Ephemeral** (default) sub-agents keep no state of their own; if interrupted
  they simply re-run.
- **Durable** sub-agents persist their progress (when the parent is itself a
  `--thread` run) so a long subtask can resume after a restart instead of
  starting over.

Each run — the top-level loop and every sub-agent — gets its own independent
tool-call budget (no shared pool), and a fan-out cap limits how many sub-agents a
run may spawn (see [Configuration](#configuration)).

## Skills (Runtime Injection)

Place Markdown files in `$AGENT_DIR/skills/` (default `/var/actor/.agent/skills/`).
They are automatically injected into the system prompt at startup — no recompile
needed.

```
/var/actor/.agent/
├── system.md          ← base system prompt (optional, loaded from $AGENT_DIR)
└── skills/
    ├── 01-domain.md   ← domain knowledge
    └── 02-rules.md    ← behavioural rules
```

## Provider Detection

The provider is inferred automatically from `LLM_URL`:

| URL contains | Provider |
|---|---|
| `/v1/chat/completions` | OpenAI-compatible |
| anything else | Anthropic Messages API (default) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://llm.aispec-system.svc.cluster.local/anthropic/v1/messages` | LLM endpoint (in-cluster proxy) |
| `LLM_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Model ID |
| `LLM_API_KEY` | — | API key (omit if auth is handled by proxy) |
| `AGENT_DIR` | `/var/actor/.agent` | Path to system prompt + skills directory |
| `AGENT_TOOL_BUDGET` | `200` | Max tool calls per run (the top-level run and each sub-agent get their own) |
| `AGENT_MAX_FANOUT` | `8` | Max sub-agents a single run may delegate |

## Threads

Persist multi-turn conversations across restarts:

```sh
agent --thread my-session "Start a code review of src/main.rs"
agent --thread my-session "Now check for SQL injection risks"
```

Threads are stored as JSONL in `/tmp/agent_thread_<id>.jsonl`. Durable sub-agents
spawned within a thread track their status in `/tmp/agent_registry_<id>.jsonl` so
in-flight work can be resumed on the next start.

## Usage as Actor-Mesh Handler

The binary doubles as a handler for the [actor-mesh]
runtime — just set `ACTOR_HANDLER`:

```sh
ACTOR_ID=ai-agent \
ACTOR_TOPIC=ai_task \
ACTOR_RESULT_TOPIC=ai_result \
ACTOR_HANDLER=./target/release/agent \
ACTOR_LMDB_PATH=/var/actor/agent \
./bin/actor &
```

The agent reads the task payload from stdin and writes the result with a topic
override (`ai_result\n<response>`) to stdout.

## Testing

Unit tests are hermetic — no network or keys, deterministic via a scripted client
and temp dirs. They cover the loop, sub-agent delegation, budget/fan-out, and
durable crash-resume:

```sh
cargo test
```

For a live smoke test against a real endpoint (delegation + durable threads),
export your endpoint and key, then run the script:

```sh
export LLM_URL=...          # Bearer-authenticated proxy/gateway
export LLM_API_KEY=...      # omit if the proxy handles auth
scripts/smoke.sh
```

It builds the release binary, runs a task that delegates to a sub-agent (watch for
`[agent d1]` in the logs), and exercises a durable sub-agent inside a `--thread`
session. It also prints the commands to verify crash-resume by hand.

## License

Apache 2.0 — see [LICENSE](LICENSE).
