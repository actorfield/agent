#!/usr/bin/env sh
# Live smoke test for the agent loop and sub-agent delegation.
#
# Requires a reachable LLM endpoint. Export before running:
#   LLM_URL      endpoint (Bearer-authenticated proxy or gateway)
#   LLM_API_KEY  key (omit only if the proxy handles auth)
#   LLM_MODEL    optional model id override
#
# Usage:
#   scripts/smoke.sh
#
# Exits non-zero if the binary fails to run. Delegation is model-driven, so the
# script reports whether a sub-agent actually ran rather than hard-failing on it.

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/agent"
WORK="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

note()  { printf '\n=== %s ===\n' "$1"; }
ok()    { printf 'PASS: %s\n' "$1"; PASS=$((PASS + 1)); }
bad()   { printf 'FAIL: %s\n' "$1"; FAIL=$((FAIL + 1)); }

if [ -z "${LLM_URL:-}" ]; then
    echo "LLM_URL is not set — export your endpoint (and LLM_API_KEY) first." >&2
    exit 2
fi

note "Build"
cargo build --release --manifest-path "$ROOT/Cargo.toml" >/dev/null
[ -x "$BIN" ] || { echo "binary not found at $BIN" >&2; exit 2; }

# ── 1. Delegation: task that asks for a sub-agent ───────────────────────────────
note "1. Delegation"
TASK='You MUST use the spawn_agent tool to delegate this: in a sub-agent, run a shell command that prints the current kernel name (uname -s), then report the result back.'
if "$BIN" "$TASK" >"$WORK/out1.txt" 2>"$WORK/err1.txt"; then
    ok "top-level run completed"
else
    bad "top-level run exited non-zero (see $WORK/err1.txt)"
fi
if grep -q '\[agent d1\]' "$WORK/err1.txt"; then
    ok "a sub-agent ran (saw [agent d1] in logs)"
else
    echo "NOTE: no sub-agent ran — the model chose not to delegate this time."
fi
head -c 400 "$WORK/out1.txt"; echo

# ── 2. Durable delegation within a thread ───────────────────────────────────────
note "2. Durable sub-agent in a thread"
TID="smoke-$$"
export AGENT_TOOL_BUDGET="${AGENT_TOOL_BUDGET:-200}"
# The agent writes state under std::env::temp_dir(), which honors $TMPDIR
# (e.g. nix-shell). Check the same location, not a hardcoded /tmp.
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
THREAD_FILE="$TMP/agent_thread_${TID}.jsonl"
REGISTRY_FILE="$TMP/agent_registry_${TID}.jsonl"
DTASK='Use spawn_agent with persistence "durable" to delegate: list the files in /etc and count them, then report the count.'
if "$BIN" --thread "$TID" "$DTASK" >"$WORK/out2.txt" 2>"$WORK/err2.txt"; then
    ok "durable thread run completed"
else
    bad "durable thread run exited non-zero (see $WORK/err2.txt)"
fi
[ -f "$THREAD_FILE" ] && ok "conversation persisted" || bad "no thread file written"
if [ -f "$REGISTRY_FILE" ]; then
    ok "durable registry written"
    echo "  To test crash-resume manually:"
    echo "    $BIN --thread $TID '<a long durable task>'   # then SIGKILL mid-run"
    echo "    $BIN --thread $TID '<same task>'             # should resume from the registry"
else
    echo "NOTE: no registry file — the model did not delegate a durable sub-agent."
fi

# Clean up this run's thread/registry artifacts.
rm -f "$THREAD_FILE" "$REGISTRY_FILE"

note "Summary"
printf 'passed: %s   failed: %s\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
