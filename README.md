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
task → [think] → tool call → [act: run_shell, read_image, read_pdf] → result → [think] → ... → answer
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
| `anthropic` | Anthropic Messages API |
| anything else | OpenAI-compatible (`/v1/chat/completions`) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://llm.aispec-system.svc.cluster.local/anthropic/v1/messages` | LLM endpoint (in-cluster proxy) |
| `LLM_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Model ID |
| `LLM_API_KEY` | — | API key (omit if auth is handled by proxy) |
| `AGENT_DIR` | `/var/actor/.agent` | Path to system prompt + skills directory |

## Threads

Persist multi-turn conversations across restarts:

```sh
agent --thread my-session "Start a code review of src/main.rs"
agent --thread my-session "Now check for SQL injection risks"
```

Threads are stored as JSONL in `/tmp/agent_thread_<id>.jsonl`.

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

## License

Apache 2.0 — see [LICENSE](LICENSE).
