//! The single generic loop. It calls the model, runs any tool calls, and asks
//! the injected `Policy` when to stop and how to label the result. A top-level
//! run and a delegated sub-run are the same function with different config.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::job::{Effort, FailureKind, Job, JobResult, Status};
use crate::llm::LlmClient;
use crate::policy::{Ending, Policy, Progress};
use crate::provider::{Provider, ToolResult};
use crate::thread::Paths;
use crate::{registry, spawn, thread, tools};

/// Iteration budget for a delegated sub-run when the caller does not set one.
pub const DEFAULT_SUB_MAX_ITER: usize = 25;
/// Largest tool output kept inline before spilling to disk.
const MAX_INLINE_CHARS: usize = 16_000;

/// Shared services, plus this run's own tool-call budget. Budgets are per-run and
/// independent — a sub-agent never draws from its parent's pool.
pub struct Ctx<'a> {
    pub client: &'a dyn LlmClient,
    pub provider: &'a dyn Provider,
    pub paths: Paths,
    pub model: &'a str,
    /// Extended-reasoning effort for this run's model calls. `Effort::None` sends
    /// no reasoning param at all — opt-in, matches prior behavior.
    pub effort: Effort,
    /// Tool-call budget for the current run only. `Arc<AtomicUsize>` (not
    /// `Rc<Cell<_>>`) so `Ctx` can be shared (`&Ctx`, requires `Sync`) across the
    /// scoped threads spawn_agent uses to run concurrent sub-agents — even though
    /// each child gets its own fresh budget (never actually shared), the field's
    /// type still has to satisfy `Sync` for `Ctx` as a whole to cross that boundary.
    pub budget: Arc<AtomicUsize>,
    /// Starting tool-call budget handed to each delegated sub-agent (its own).
    pub sub_budget: usize,
    /// Number of sub-agents delegated so far by the top-level run. Genuinely
    /// shared (same Arc cloned into every descendant), so mutated via a
    /// compare-exchange loop wherever it's checked-then-incremented, never a
    /// plain read-then-write — see spawn::handle.
    pub fanout: Arc<AtomicUsize>,
    pub max_fanout: usize,
    /// Sequential spawn-order counter for THIS run's own direct children, used
    /// only to build their log labels (e.g. this run is "d1", its 3rd spawned
    /// child is "d1-2") — distinct from `fanout`, which enforces the fan-out
    /// cap; this is purely cosmetic numbering, reset fresh per run (each
    /// spawned child gets its own fresh counter for ITS children, though
    /// today's policy never lets a sub-agent spawn further).
    pub spawn_index: Arc<AtomicUsize>,
}

/// Per-run configuration: what to do, how to steer, and where in the tree.
pub struct RunConfig {
    pub job: Job,
    pub policy: Policy,
    pub depth: usize,
    /// Log label for this run, e.g. "d0" for the top-level run or "d1-2" for
    /// the 3rd (0-indexed) sub-agent spawned directly by that top-level run.
    /// Purely cosmetic — `depth` (not this) drives actual behavior (the
    /// reconcile gate, tool-set shaping, delegation refusal below depth 0).
    /// Distinct from `depth` because multiple sub-agents share the same
    /// depth but must be told apart in logs — previously every sub-agent
    /// printed as "d1", making it impossible to tell which log lines
    /// belonged to which of several concurrently- or sequentially-run
    /// sub-agents without cross-referencing AGENT_RUN_DIR paths by hand.
    pub label: String,
    /// Conversation id for persistence; `None` means no own persistence.
    pub thread_id: Option<String>,
}

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn render_task(job: &Job) -> String {
    let mut s = job.task.clone();
    if !job.checks.is_empty() {
        s.push_str("\n\nRequirements:\n");
        for c in &job.checks {
            s.push_str(&format!("- {c}\n"));
        }
    }
    s
}

/// Run a job to conclusion and return its structured result.
pub fn run(ctx: &Ctx, cfg: RunConfig) -> JobResult {
    let RunConfig {
        job,
        policy,
        depth,
        label,
        thread_id,
    } = cfg;

    let system = build_system();
    // Every run — top-level or a nested sub-agent — learns its own scratchpad/
    // spill directory this way, since sub-agents are recursive in-process calls
    // (not new OS processes), so a single env var can't hold a distinct value
    // per sub-agent the way it could for the top-level run alone.
    // Load prior conversation (and, at the top level, resume any durable work that
    // was in flight when a previous process died).
    let mut history: Vec<Value> = match &thread_id {
        Some(tid) => {
            let mut hist = thread::load_thread(&ctx.paths);
            if !hist.is_empty() {
                eprintln!(
                    "[agent {label}] resuming {tid} ({} prior messages)",
                    hist.len()
                );
            }
            if depth == 0 {
                reconcile(ctx, &policy, tid, &mut hist);
            }
            hist
        }
        None => vec![],
    };
    // Everything loaded is already on disk; everything appended below is not.
    // Taken before the repair so the synthesized tool results get persisted
    // too -- if they were treated as already-saved, the next resume would load
    // the same dangling call again and fail the same way.
    let persisted_base = history.len();

    // Close out a round the previous run stopped in the middle of.
    //
    // `ask_user` breaks the loop BEFORE running any tool, so the assistant turn
    // it stopped on is persisted with tool calls and no results -- and BOTH
    // providers reject a conversation in that shape. Without this, every answer
    // to a question would fail the request outright: the feature would look
    // fine right up to the moment someone replied.
    //
    // Anything the model batched alongside the question is answered too. Those
    // tools deliberately did not run, and saying so is what stops the model
    // from carrying on as though they had.
    let pending = ctx.provider.pending_tool_calls(&json!(&history[..]));
    let answered_question = pending
        .iter()
        .any(|(_, name)| name == crate::tools::ASK_USER);
    let task_text = if answered_question {
        // The answer itself rides in the tool result below, where the model
        // expects a reply to its question -- repeating it here would read as
        // being asked and answered twice.
        format!(
            "[run resumed: {}]\nYour run directory: {}",
            now(),
            ctx.paths.dir().display()
        )
    } else {
        format!(
            "[run started: {}]\nYour run directory: {}\n{}",
            now(),
            ctx.paths.dir().display(),
            render_task(&job)
        )
    };
    let text_block =
        json!({"type": "text", "text": task_text, "cache_control": {"type": "ephemeral"}});

    if pending.is_empty() {
        history.push(json!({ "role": "user", "content": [text_block] }));
    } else {
        // Deliberately NOT run through `redact_results`, unlike the other two
        // ToolResult sites. Both contents here are safe by construction: one is
        // a compile-time constant, the other is `job.task`, which the caller
        // redacts once when the Job is built. Sending it through the door again
        // would re-scan already-redacted text on every resume, against a
        // detector that serves one request at a time.
        //
        // The invariant this rests on: `job.task` is redacted at construction.
        // If that ever stops being true, this site starts leaking silently, so
        // it belongs in the same change as any alteration to how the task is
        // built.
        let results: Vec<ToolResult> = pending
            .iter()
            .map(|(id, name)| ToolResult {
                tool_use_id: id.clone(),
                content: if name == crate::tools::ASK_USER {
                    render_task(&job)
                } else {
                    tools::NOT_RUN_WHILE_ASKING.to_string()
                },
            })
            .collect();
        let mut tool_msgs = ctx.provider.wrap_tool_results(results);
        // Anthropic returns the results as a single `user` message. Appending
        // the framing as a second `user` message would put two of them back to
        // back, so it folds into the same one. OpenAI returns `tool` messages,
        // which a following `user` message is the correct shape for.
        let fold = tool_msgs.last().is_some_and(|m| m["role"] == "user");
        if fold {
            if let Some(content) = tool_msgs.last_mut().unwrap()["content"].as_array_mut() {
                content.push(text_block);
            }
            history.extend(tool_msgs);
        } else {
            history.extend(tool_msgs);
            history.push(json!({ "role": "user", "content": [text_block] }));
        }
    }
    let mut messages = json!(history);
    let mut persisted_len = persisted_base;

    // Tool set: the policy may swap the builtin definitions for its own;
    // spawn_agent is appended by the loop either way (delegation is loop
    // infrastructure — an override can't grant it to sub-agents or drop it
    // from the top level).
    let base_defs = match policy.tool_defs {
        Some(f) => {
            let mut defs = f();
            if depth == 0 {
                if let Some(arr) = defs.as_array_mut() {
                    arr.push(tools::spawn_agent_def());
                }
            }
            defs
        }
        None => tools::base_tool_defs(depth),
    };
    let tool_set = ctx.provider.shape_tools(&base_defs);

    let mut steps_taken = 0usize;
    let mut last_text = String::new();
    let mut iter = 0usize;
    let ending;

    loop {
        // Check for external cancellation signal (written by cancel-agent handler).
        if ctx.paths.dir().join("cancel").exists() {
            eprintln!("[agent {label}] cancelled by signal");
            ending = Ending::Cancelled;
            break;
        }
        let prog = Progress {
            iter,
            max_iter: job.max_iter,
            budget_remaining: ctx.budget.load(Ordering::Relaxed),
            steps_taken,
            last_text: &last_text,
            checks: &job.checks,
        };
        if !(policy.should_continue)(&prog) {
            ending = if iter >= job.max_iter {
                Ending::IterExhausted
            } else {
                Ending::BudgetExhausted
            };
            break;
        }

        eprintln!("[agent {label}] iter {}", iter + 1);
        let resp = match ctx.client.call(
            ctx.provider,
            ctx.model,
            &mut messages,
            &system,
            &tool_set,
            ctx.effort,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[agent {label}] llm error: {e}");
                // A context overflow is a ceiling, not a fault: retrying the
                // same thread cannot help, but a fresh or compacted one can.
                // Reported as a generic failure it reads as a bug and the one
                // action that helps goes unsuggested.
                ending = if crate::llm::is_context_limit_error(&e.to_string()) {
                    Ending::ContextExhausted
                } else {
                    Ending::Failed
                };
                break;
            }
        };
        let parsed = match ctx.provider.parse_response(resp) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[agent {label}] parse error: {e}");
                ending = Ending::Failed;
                break;
            }
        };
        if !parsed.text_parts.is_empty() {
            last_text = parsed.text_parts.join("\n");
        }
        if thread_id.is_some() {
            if let Some(u) = &parsed.usage {
                thread::append_usage(&ctx.paths, u.input_tokens, u.output_tokens);
            }
        }
        messages.as_array_mut().unwrap().push(parsed.assistant_msg);

        let had_tool_calls = !parsed.tool_calls.is_empty();
        if (policy.is_done)(had_tool_calls) {
            if thread_id.is_some() {
                let all = messages.as_array().unwrap();
                thread::append_thread(&ctx.paths, &all[persisted_len..]);
            }
            ending = Ending::Stopped;
            break;
        }

        // ask_user is a terminus, not a tool call: there is no result to feed
        // back, because the answer is the user's next message. Intercepted
        // BEFORE any tool runs so a turn that asks a question does not also
        // execute whatever else the model batched alongside it -- those side
        // effects would land while the run was supposedly waiting, and the
        // user would be answering a question about work that had already
        // moved on.
        if let Some(tc) = parsed
            .tool_calls
            .iter()
            .find(|tc| tc.name == crate::tools::ASK_USER)
        {
            if let Some(q) = crate::tools::ask_user_question(&tc.input) {
                last_text = q;
            }
            if thread_id.is_some() {
                let all = messages.as_array().unwrap();
                thread::append_thread(&ctx.paths, &all[persisted_len..]);
            }
            ending = Ending::AwaitingInput;
            break;
        }

        // spawn_agent calls run concurrently with each other (each gets its own
        // fresh context and budget, and its own nested Paths, so nothing about
        // running them on separate threads is unsafe by construction) — every
        // other tool call still runs sequentially, as before. Splitting the
        // batch this way, rather than parallelizing everything, keeps run_shell/
        // read_image/read_pdf (which have no reason to run concurrently here)
        // simple while cutting the wall-clock cost of a turn that delegates to
        // several independent sub-agents at once.
        let (spawn_calls, other_calls): (Vec<_>, Vec<_>) = parsed
            .tool_calls
            .iter()
            .partition(|tc| tc.name == "spawn_agent");

        let mut results: Vec<ToolResult> = vec![];
        for tc in &other_calls {
            let (content, took_step) =
                run_one_tool(ctx, &policy, depth, &label, thread_id.as_deref(), tc);
            if took_step {
                steps_taken += 1;
            }
            results.push(ToolResult {
                tool_use_id: tc.id.clone(),
                content,
            });
        }

        if !spawn_calls.is_empty() {
            let tid_ref = thread_id.as_deref();
            let label_ref = label.as_str();
            let outcomes: Vec<(String, String, bool)> = std::thread::scope(|scope| {
                let handles: Vec<_> = spawn_calls
                    .iter()
                    .map(|tc| {
                        let tc = *tc;
                        let policy = &policy;
                        scope.spawn(move || {
                            let (content, took_step) =
                                run_one_tool(ctx, policy, depth, label_ref, tid_ref, tc);
                            (tc.id.clone(), content, took_step)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            (
                                "unknown".to_string(),
                                "[spawn_agent thread panicked]".to_string(),
                                false,
                            )
                        })
                    })
                    .collect()
            });
            for (id, content, took_step) in outcomes {
                if took_step {
                    steps_taken += 1;
                }
                results.push(ToolResult {
                    tool_use_id: id,
                    content,
                });
            }
        }

        let results = redact_results(&policy, results, &label);
        for m in ctx.provider.wrap_tool_results(results) {
            messages.as_array_mut().unwrap().push(m);
        }
        if thread_id.is_some() {
            let all = messages.as_array().unwrap();
            thread::append_thread(&ctx.paths, &all[persisted_len..]);
            persisted_len = all.len();
        }
        iter += 1;
    }

    // Label the outcome. An unmet requirement overrides an otherwise-clean stop.
    let prog = Progress {
        iter,
        max_iter: job.max_iter,
        budget_remaining: ctx.budget.load(Ordering::Relaxed),
        steps_taken,
        last_text: &last_text,
        checks: &job.checks,
    };
    let issues = (policy.check)(&prog);
    let (status, failure) = if !issues.is_empty() {
        (Status::Failure, Some(FailureKind::CheckFailed))
    } else {
        (policy.classify)(ending, &prog)
    };

    let output = match status {
        Status::Success | Status::Partial => Some(last_text),
        // A run awaiting an answer classifies as Blocked, but unlike a genuine
        // block its last text IS the payload -- the question the user has to
        // answer. Dropping it here would stop the run and show the user
        // nothing to respond to, which is the one outcome HITL must not have.
        Status::Blocked if ending == Ending::AwaitingInput => Some(last_text),
        Status::Failure | Status::Blocked => None,
    };
    JobResult {
        id: job.id,
        status,
        output,
        failure,
        steps_taken,
        issues,
        ending: Some(ending),
    }
}

/// Redact tool output before it becomes part of the conversation.
///
/// This is the single door for content the model has never seen. Tool results
/// and sub-agent results are the only ways such content reaches the message
/// array, so redacting here cannot be forgotten by a tool added later -- which
/// is the difference between a guarantee and a convention each new tool has to
/// remember to keep.
///
/// Failure withholds the output rather than passing it through. A caller
/// cannot distinguish "nothing needed redacting" from "redacting broke", so
/// failing open would be indistinguishable from working.
fn redact_results(policy: &Policy, mut results: Vec<ToolResult>, label: &str) -> Vec<ToolResult> {
    let Some(redact) = policy.redact else {
        return results;
    };
    for r in results.iter_mut() {
        if let Err(e) = redact(&mut r.content) {
            eprintln!("[agent {label}] redaction failed, withholding tool result: {e}");
            r.content = "[redaction error: tool output withheld]".to_string();
        }
    }
    results
}

/// Check-and-decrement the run's tool-call budget, then dispatch if any remains.
/// Returns `(content, took_step)`. The check-then-decrement is a compare-exchange
/// loop rather than a plain read-then-write because `budget` is an `AtomicUsize`
/// that can now be raced by concurrently-running spawn_agent calls in the same
/// turn — a naive get/set pair could let two callers both observe budget > 0 and
/// both decrement, undercounting the exhaustion point.
fn run_one_tool(
    ctx: &Ctx,
    policy: &Policy,
    depth: usize,
    label: &str,
    thread_id: Option<&str>,
    tc: &crate::provider::ToolCall,
) -> (String, bool) {
    loop {
        let cur = ctx.budget.load(Ordering::Relaxed);
        if cur == 0 {
            return ("[budget exhausted — could not run tool]".to_string(), false);
        }
        if ctx
            .budget
            .compare_exchange(cur, cur - 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return (
                dispatch_tool(ctx, policy, depth, label, thread_id, tc),
                true,
            );
        }
        // Lost the race to a concurrent caller — reload and retry.
    }
}

/// Dispatch one tool call to its implementation, returning the tool result text.
fn dispatch_tool(
    ctx: &Ctx,
    policy: &Policy,
    depth: usize,
    label: &str,
    thread_id: Option<&str>,
    tc: &crate::provider::ToolCall,
) -> String {
    // A policy dispatcher gets first refusal on every tool call except
    // spawn_agent (delegation stays with the loop). None falls through.
    if tc.name != "spawn_agent" {
        if let Some(f) = policy.dispatch {
            if let Some(out) = f(tc) {
                eprintln!("[agent {label}] {} (policy dispatch)", tc.name);
                return finalize(&ctx.paths, &tc.id, out);
            }
        }
    }
    match tc.name.as_str() {
        "run_shell" => {
            let cmd = tc.input["command"].as_str().unwrap_or("");
            eprintln!("[agent {label}] run_shell: {cmd}");
            finalize(&ctx.paths, &tc.id, tools::run_shell(cmd))
        }
        "read_image" => {
            let path = tc.input["path"].as_str().unwrap_or("");
            eprintln!("[agent {label}] read_image: {path}");
            let q = tc.input["question"]
                .as_str()
                .unwrap_or("Extract all text and data from this image verbatim.");
            finalize(&ctx.paths, &tc.id, tools::read_image(path, q))
        }
        "read_pdf" => {
            let path = tc.input["path"].as_str().unwrap_or("");
            eprintln!("[agent {label}] read_pdf: {path}");
            let q = tc.input["question"]
                .as_str()
                .unwrap_or("Extract all text and data from this PDF verbatim.");
            finalize(&ctx.paths, &tc.id, tools::read_pdf(path, q))
        }
        "spawn_agent" => {
            if !policy.may_delegate || depth >= 1 {
                let r = JobResult::blocked(&crate::job::new_id(), FailureKind::ToolUnavailable, 0);
                return r.to_json();
            }
            spawn::handle(ctx, policy, depth, label, thread_id, &tc.input).to_json()
        }
        other => format!("unknown tool: {other}"),
    }
}

/// Stamp a tool result with the time, spilling oversized output to disk.
fn finalize(paths: &Paths, id: &str, raw: String) -> String {
    let ts = now();
    if raw.len() > MAX_INLINE_CHARS {
        let out_path = paths.spill_path(id);
        match std::fs::write(&out_path, &raw) {
            Ok(_) => format!(
                "[{ts}] Output too large ({} chars) — full content saved to {}\n\
                 Query it with grep/head/sed/awk rather than reading the whole file.\n\
                 Example: grep -n 'keyword' {} | head -30",
                raw.len(),
                out_path.display(),
                out_path.display()
            ),
            Err(e) => format!(
                "[{ts}] Output too large ({} chars) and could not save to disk: {e}",
                raw.len()
            ),
        }
    } else {
        format!("[{ts}]\n{raw}")
    }
}

/// Resume durable jobs that were in flight when a previous process exited, and
/// fold each result back into the conversation. Idempotent: safe to run on every
/// start, and skips any job whose result is already committed to the conversation.
fn reconcile(ctx: &Ctx, policy: &Policy, parent_tid: &str, history: &mut Vec<Value>) {
    let records = registry::load(&ctx.paths);
    if records.is_empty() {
        return;
    }
    let done = registry::results_map(&records);
    for job in registry::issued_jobs(&records) {
        let synth_id = format!("toolu_{}", job.id);
        let convo = json!(&history[..]);
        if ctx.provider.has_tool_result(&convo, &synth_id) {
            continue; // already committed
        }
        // Get the result: from the registry, or by resuming the child now.
        let result = match done.get(&job.id) {
            Some(r) => r.clone(),
            None => {
                let child_index = ctx.spawn_index.fetch_add(1, Ordering::Relaxed);
                let child_label = format!("d1-{child_index}");
                eprintln!("[agent {child_label}] resuming in-flight job {}", job.id);
                let child_tid = thread::child_thread_id(parent_tid, &job.id);
                // Reconstruct the exact directory spawn::handle would have used for
                // this child — for_child is a pure function of parent dir + job id,
                // so this reproduces it deterministically without persisting it
                // separately. Match spawn::handle's effort (job's own, not the
                // parent run's) — budget/fanout reuse the parent ctx here as before
                // this change.
                let resumed_ctx = Ctx {
                    client: ctx.client,
                    provider: ctx.provider,
                    paths: Paths::for_child(&ctx.paths, &job.id),
                    model: ctx.model,
                    effort: job.effort,
                    budget: ctx.budget.clone(),
                    sub_budget: ctx.sub_budget,
                    fanout: ctx.fanout.clone(),
                    max_fanout: ctx.max_fanout,
                    spawn_index: Arc::new(AtomicUsize::new(0)),
                };
                let r = run(
                    &resumed_ctx,
                    RunConfig {
                        job: job.clone(),
                        policy: crate::policy::sub_policy_of(policy),
                        depth: 1,
                        label: child_label,
                        thread_id: Some(child_tid),
                    },
                );
                registry::append_result(&ctx.paths, &r);
                r
            }
        };
        // Synthesize the round and append it to conversation + persist it.
        let assistant =
            ctx.provider
                .tool_call_message(&synth_id, "spawn_agent", &spawn::job_to_input(&job));
        // The same door as the tool-result path: a sub-agent's output enters
        // the parent's conversation here and is persisted below, so it cannot
        // be the one route that skips redaction.
        let tool_msgs = ctx.provider.wrap_tool_results(redact_results(
            policy,
            vec![ToolResult {
                tool_use_id: synth_id,
                content: result.to_json(),
            }],
            parent_tid,
        ));
        let mut committed = vec![assistant];
        committed.extend(tool_msgs);
        history.extend(committed.iter().cloned());
        thread::append_thread(&ctx.paths, &committed);
    }
}

// ── Prompt assembly ─────────────────────────────────────────────────────────────

fn agent_dir() -> String {
    std::env::var("AGENT_DIR").unwrap_or_else(|_| "/var/actor/.agent".to_string())
}

fn build_system() -> String {
    format!("{}{}", system_prompt(), inject_skills())
}

fn system_prompt() -> String {
    let dir = agent_dir();
    let path = format!("{dir}/system.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| {
            format!("You are a task-executing ai-agent. On your first action, run: cat {dir}/skills/0-claude.md")
        })
        .replace("{AGENT_DIR}", &dir)
}

fn inject_skills() -> String {
    let dir = agent_dir();
    let skills_dir = format!("{dir}/skills");
    let mut out = String::new();
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "md").unwrap_or(false))
            .collect();
        paths.sort();
        for path in paths {
            if let Ok(content) = std::fs::read_to_string(&path) {
                out.push_str(&format!(
                    "\n\n---\n## SKILL — {}\n\n{}",
                    path.display(),
                    content
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Persistence;
    use crate::policy::{root_policy, sub_policy};
    use crate::provider::{Anthropic, OpenAI};
    use std::sync::Mutex;

    /// A scripted client: returns queued responses in order. `Mutex`, not
    /// `RefCell`, because `LlmClient` now requires `Send + Sync` (so `Ctx` can
    /// cross the scoped threads spawn_agent uses) — this is a compile-time
    /// requirement of the trait bound, not a sign these tests run concurrently.
    struct ScriptedClient {
        responses: Mutex<Vec<Value>>,
    }
    impl ScriptedClient {
        fn new(responses: Vec<Value>) -> ScriptedClient {
            ScriptedClient {
                responses: Mutex::new(responses),
            }
        }
    }
    impl LlmClient for ScriptedClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            _msgs: &mut Value,
            _s: &str,
            _t: &Value,
            _effort: Effort,
        ) -> Result<Value, String> {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                Err("no scripted response".into())
            } else {
                Ok(q.remove(0))
            }
        }
    }

    fn text_turn(t: &str) -> Value {
        json!({"content":[{"type":"text","text":t}]})
    }
    fn shell_turn(id: &str, cmd: &str) -> Value {
        json!({"content":[{"type":"tool_use","id":id,"name":"run_shell","input":{"command":cmd}}]})
    }

    fn ctx<'a>(
        client: &'a dyn LlmClient,
        provider: &'a dyn Provider,
        paths: Paths,
        budget: usize,
    ) -> Ctx<'a> {
        Ctx {
            client,
            provider,
            paths,
            model: "m",
            effort: Effort::None,
            budget: Arc::new(AtomicUsize::new(budget)),
            sub_budget: budget,
            fanout: Arc::new(AtomicUsize::new(0)),
            max_fanout: 8,
            spawn_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn job(max_iter: usize) -> Job {
        Job::new("do it".into(), vec![], Persistence::Ephemeral, max_iter)
    }

    fn ask_turn(id: &str, q: &str) -> Value {
        json!({"content":[{"type":"tool_use","id":id,"name":"ask_user","input":{"question":q}}]})
    }

    /// ask_user must END the run, not be executed like any other tool.
    #[test]
    fn ask_user_stops_the_run_awaiting_input() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        // A second turn is scripted deliberately: if the loop kept going, it
        // would consume this and the assertions below would see "kept going"
        // instead of the question.
        let client = ScriptedClient::new(vec![
            ask_turn("t1", "Which bucket should I write to?"),
            text_turn("kept going"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.ending, Some(Ending::AwaitingInput));
        assert_eq!(r.status, Status::Blocked);
    }

    /// The question is the payload. Blocked normally discards output, so this
    /// is the case that would silently leave the user nothing to answer.
    #[test]
    fn ask_user_returns_the_question_as_output() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![ask_turn("t1", "Prod or staging?")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.output.as_deref(), Some("Prod or staging?"));
    }

    /// A turn that asks AND batches other calls must not run them: those side
    /// effects would land while the run was supposedly waiting, and the user
    /// would answer a question about work that had already moved on.
    #[test]
    fn ask_user_suppresses_tools_batched_in_the_same_turn() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let marker = dir.path().join("side-effect");
        let both = json!({"content":[
            {"type":"tool_use","id":"a","name":"run_shell",
             "input":{"command": format!("touch {}", marker.display())}},
            {"type":"tool_use","id":"b","name":"ask_user",
             "input":{"question":"Continue?"}}
        ]});
        let client = ScriptedClient::new(vec![both]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.ending, Some(Ending::AwaitingInput));
        assert!(!marker.exists(), "batched tool ran despite ask_user");
        assert_eq!(r.steps_taken, 0, "asking is not a step");
    }

    /// Records the conversation as sent, so a test can assert on the request the
    /// API would really have received rather than on the saved transcript.
    struct RecordingClient {
        responses: Mutex<Vec<Value>>,
        seen: Mutex<Vec<Value>>,
    }
    impl RecordingClient {
        fn new(responses: Vec<Value>) -> RecordingClient {
            RecordingClient {
                responses: Mutex::new(responses),
                seen: Mutex::new(vec![]),
            }
        }
        fn first_request(&self) -> Value {
            self.seen
                .lock()
                .unwrap()
                .first()
                .cloned()
                .expect("no request was made")
        }
    }
    impl LlmClient for RecordingClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            msgs: &mut Value,
            _s: &str,
            _t: &Value,
            _effort: Effort,
        ) -> Result<Value, String> {
            self.seen.lock().unwrap().push(msgs.clone());
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                Err("no scripted response".into())
            } else {
                Ok(q.remove(0))
            }
        }
    }

    /// Every tool call in `msgs` that no result answers. This is the invariant
    /// both providers enforce, and the one a paused question breaks.
    fn dangling_calls(p: &dyn Provider, msgs: &Value) -> Vec<String> {
        msgs.as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "assistant")
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter(|b| b["type"] == "tool_use")
            .filter_map(|b| b["id"].as_str())
            .filter(|id| !p.has_tool_result(msgs, id))
            .map(str::to_string)
            .collect()
    }

    fn resume(paths: &Paths, client: &dyn LlmClient, provider: &dyn Provider, answer: &str) {
        let c = ctx(client, provider, paths.clone(), 100);
        let _ = run(
            &c,
            RunConfig {
                job: Job::new(answer.into(), vec![], Persistence::Ephemeral, 5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".to_string()),
            },
        );
    }

    fn pause_on_question(paths: &Paths, provider: &dyn Provider, turn: Value) {
        let client = ScriptedClient::new(vec![turn]);
        let c = ctx(&client, provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".to_string()),
            },
        );
        assert_eq!(r.ending, Some(Ending::AwaitingInput));
    }

    /// Answering a question must produce a conversation the API will accept.
    ///
    /// The pause persists an assistant turn whose tool call has no result, and
    /// both providers reject that outright — so without the repair the FIRST
    /// reply to any question fails the request. The feature looks fine until
    /// someone actually answers.
    #[test]
    fn answering_a_question_sends_no_dangling_tool_call() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        pause_on_question(&paths, &provider, ask_turn("t1", "Which bucket?"));

        // The pause really does leave a dangling call on disk. Without this the
        // test could pass while exercising nothing.
        let saved = json!(thread::load_thread(&paths));
        assert_eq!(
            dangling_calls(&provider, &saved),
            vec!["t1".to_string()],
            "expected the paused question to be left unanswered on disk"
        );

        let client = RecordingClient::new(vec![text_turn("done")]);
        resume(&paths, &client, &provider, "the prod bucket");

        let sent = client.first_request();
        assert!(
            dangling_calls(&provider, &sent).is_empty(),
            "sent a tool call with no result: {sent}"
        );
        let answer = sent.to_string();
        assert!(
            answer.contains("the prod bucket"),
            "the person's answer never reached the model: {sent}"
        );
    }

    /// Anthropic returns tool results as a `user` message, and the run framing
    /// is also a `user` message. Emitted separately they would be two user
    /// turns back to back, which the API rejects for a different reason than
    /// the one this repair exists to fix.
    #[test]
    fn answering_a_question_does_not_emit_two_user_turns() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        pause_on_question(&paths, &provider, ask_turn("t1", "Which bucket?"));
        let client = RecordingClient::new(vec![text_turn("done")]);
        resume(&paths, &client, &provider, "prod");

        let sent = client.first_request();
        let roles: Vec<&str> = sent
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap_or(""))
            .collect();
        assert!(
            !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
            "consecutive user turns: {roles:?}"
        );
    }

    /// A tool the model batched alongside the question never ran. On resume it
    /// still needs a result, and that result has to SAY it did not run —
    /// otherwise the model carries on as though the side effect happened.
    #[test]
    fn tools_batched_with_a_question_resume_as_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        pause_on_question(
            &paths,
            &provider,
            json!({"content":[
                {"type":"tool_use","id":"t1","name":"ask_user","input":{"question":"which?"}},
                {"type":"tool_use","id":"t2","name":"run_shell","input":{"command":"rm -rf /data"}}
            ]}),
        );

        let client = RecordingClient::new(vec![text_turn("done")]);
        resume(&paths, &client, &provider, "prod");

        let sent = client.first_request();
        assert!(
            dangling_calls(&provider, &sent).is_empty(),
            "batched call left unanswered: {sent}"
        );
        assert!(
            provider.has_tool_result(&sent, "t2"),
            "the batched shell call got no result: {sent}"
        );
        assert!(
            sent.to_string().contains("not run"),
            "the batched call was not reported as skipped: {sent}"
        );
    }

    /// The repair must survive being resumed twice. The synthesized results are
    /// only correct if they were persisted; if they were treated as already on
    /// disk, the next resume would reload the same dangling call and fail the
    /// same way.
    #[test]
    fn a_repaired_question_stays_repaired_on_the_next_resume() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        pause_on_question(&paths, &provider, ask_turn("t1", "Which bucket?"));
        let first = RecordingClient::new(vec![text_turn("done")]);
        resume(&paths, &first, &provider, "prod");

        let second = RecordingClient::new(vec![text_turn("done again")]);
        resume(&paths, &second, &provider, "and now deploy");

        let sent = second.first_request();
        assert!(
            dangling_calls(&provider, &sent).is_empty(),
            "the repair did not persist: {sent}"
        );
    }

    /// The same repair over the OpenAI wire shape, which is what production
    /// actually runs. Its tool calls live in a `tool_calls` array rather than
    /// content blocks, and its results are separate `tool` messages, so the
    /// Anthropic tests above prove nothing about it.
    ///
    /// The paused turn is copied from a real thread on the cluster: an
    /// assistant message with `tool_calls` and no `tool` reply after it.
    #[test]
    fn openai_answering_a_question_sends_no_dangling_tool_call() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = OpenAI;

        let ask = json!({"choices":[{"message":{
            "role": "assistant",
            "content": "Let me demonstrate it right now:",
            "tool_calls": [{
                "id": "call_00_dNwvS4CQ",
                "index": 0,
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "arguments": "{\"question\": \"HITL test — does this reach you?\"}"
                }
            }]
        }}]});

        let client = ScriptedClient::new(vec![ask]);
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".to_string()),
            },
        );
        assert_eq!(r.ending, Some(Ending::AwaitingInput));

        // The dangling call is real on disk in the OpenAI shape too.
        let saved = json!(thread::load_thread(&paths));
        assert!(
            !provider.pending_tool_calls(&saved).is_empty(),
            "expected an unanswered tool_call: {saved}"
        );

        let client = RecordingClient::new(vec![
            json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]}),
        ]);
        resume(&paths, &client, &provider, "yes it reached me");

        let sent = client.first_request();
        // OpenAI answers with `tool` messages, so the Anthropic-shaped
        // dangling_calls helper does not apply -- check the id directly.
        assert!(
            provider.has_tool_result(&sent, "call_00_dNwvS4CQ"),
            "the question was never answered: {sent}"
        );
        assert!(
            provider.pending_tool_calls(&sent).is_empty(),
            "still ends on an unanswered call: {sent}"
        );
        assert!(
            sent.to_string().contains("yes it reached me"),
            "the person's answer never reached the model: {sent}"
        );
    }

    /// An ordinary resume — no question pending — must be untouched.
    #[test]
    fn an_ordinary_resume_still_carries_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        let first = ScriptedClient::new(vec![text_turn("hello")]);
        let c = ctx(&first, &provider, paths.clone(), 100);
        let _ = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".to_string()),
            },
        );

        let client = RecordingClient::new(vec![text_turn("ok")]);
        resume(&paths, &client, &provider, "next thing please");

        let sent = client.first_request().to_string();
        assert!(
            sent.contains("next thing please"),
            "the new task never reached the model"
        );
        assert!(
            sent.contains("run started"),
            "a resume with no pending question should still frame as a start"
        );
    }

    #[test]
    fn single_turn_no_tools_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![text_turn("all done")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.output.as_deref(), Some("all done"));
        assert_eq!(r.steps_taken, 0);
    }

    #[test]
    fn text_only_turn_persists_task_and_reply_with_thread() {
        // No tool calls, but a thread id is set: the terminal branch must persist
        // the task message and the assistant reply (persisted_len still at start).
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![text_turn("done, no tools")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".into()),
            },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.output.as_deref(), Some("done, no tools"));
        assert_eq!(r.steps_taken, 0);

        let saved = thread::load_thread(&paths);
        assert_eq!(saved.len(), 2, "task message + assistant reply persisted");
        assert_eq!(saved[0]["role"], "user");
        assert_eq!(saved[1]["role"], "assistant");
    }

    #[test]
    fn runs_a_tool_then_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("t1", "printf hi"), text_turn("got hi")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.steps_taken, 1);
        assert_eq!(c.budget.load(Ordering::Relaxed), 99); // one tool call consumed
    }

    #[test]
    fn two_spawn_agent_calls_in_one_turn_run_concurrently_and_fanout_is_exact() {
        // The orchestrator issues two spawn_agent tool_use blocks in a single
        // assistant turn; each child gets its own thread (agent_loop::run splits
        // spawn_agent calls out of the sequential dispatch loop and runs them via
        // std::thread::scope). Both must complete, and the shared fanout counter
        // must land at exactly 2 — not under- or over-counted by the concurrency.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            json!({"content": [
                {"type":"tool_use","id":"s1","name":"spawn_agent","input":{"task":"metadata+headings"}},
                {"type":"tool_use","id":"s2","name":"spawn_agent","input":{"task":"body paragraphs"}}
            ]}),
            // Each child consumes one of these from the shared, mutex-backed
            // queue — order between the two concurrently-running children is
            // non-deterministic, so the test only asserts aggregate outcomes.
            text_turn("child done"),
            text_turn("child done"),
            text_turn("parent wraps up"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(c.fanout.load(Ordering::Relaxed), 2);
        assert_eq!(r.steps_taken, 2);
    }

    #[test]
    fn iter_exhaustion_yields_partial() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        // Always asks for another shell call; never stops.
        let client = ScriptedClient::new(vec![
            shell_turn("a", "true"),
            shell_turn("b", "true"),
            shell_turn("c", "true"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(2),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Partial);
        assert_eq!(r.failure, Some(FailureKind::BudgetExceeded));
    }

    #[test]
    fn budget_exhaustion_yields_partial() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("a", "true"), shell_turn("b", "true")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 1); // only one tool call allowed
        let r = run(
            &c,
            RunConfig {
                job: job(10),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Partial);
        assert_eq!(c.budget.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn llm_error_yields_failure() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]); // errors immediately
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: None,
            },
        );
        assert_eq!(r.status, Status::Failure);
        assert!(r.output.is_none());
    }

    #[test]
    fn conversation_persists_atomically_across_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("t1", "printf hi"), text_turn("done")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let _ = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("persist".into()),
            },
        );
        let saved = thread::load_thread(&paths);
        // Every assistant tool_use has its tool_result present: no dangling call.
        let has_toolu = saved.iter().any(|m| {
            m["content"]
                .as_array()
                .is_some_and(|b| b.iter().any(|x| x["type"] == "tool_use"))
        });
        assert!(has_toolu);
        assert!(provider.has_tool_result(&json!(saved), "t1"));
    }

    #[test]
    fn sub_policy_blocks_delegation() {
        // At depth 1 the delegation attempt is refused with a typed blocked result.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let out = dispatch_tool(
            &c,
            &sub_policy(),
            1,
            "d1-0",
            None,
            &crate::provider::ToolCall {
                id: "x".into(),
                name: "spawn_agent".into(),
                input: json!({"task":"nested"}),
            },
        );
        assert!(out.contains("\"status\":\"blocked\""));
    }

    #[test]
    fn durable_resume_reconciles_into_conversation_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let provider = Anthropic;

        // A durable job was issued but never completed (process died mid-run):
        // registry has the issue line, no result; conversation has no tool_result.
        let pending = Job::new("long subtask".into(), vec![], Persistence::Durable, 10);
        registry::append_issued(&paths, &pending);
        let synth_id = format!("toolu_{}", pending.id);

        // First start: reconcile resumes the child (1st response), then the
        // top-level loop finishes (2nd response).
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"child recovered"}]}),
            json!({"content":[{"type":"text","text":"parent done"}]}),
        ]);
        let c = ctx(&client, &provider, paths.clone(), 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("main".into()),
            },
        );
        assert_eq!(r.status, Status::Success);

        // Conversation now has a valid tool_use + tool_result pair for the job.
        let convo = json!(thread::load_thread(&paths));
        assert!(
            provider.has_tool_result(&convo, &synth_id),
            "result not committed"
        );
        // Registry closed out: nothing left in flight.
        let recs = registry::load(&paths);
        assert!(registry::in_flight(&recs).is_empty());

        // Second start with the same thread: reconcile must be a no-op (result
        // already in the conversation), so only the parent turn is consumed.
        let client2 = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"parent again"}]}),
        ]);
        let c2 = ctx(&client2, &provider, paths.clone(), 100);
        let r2 = run(
            &c2,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("main".into()),
            },
        );
        assert_eq!(r2.status, Status::Success);
        // No duplicate result appended by the idempotent second pass.
        let result_lines = registry::load(&paths)
            .iter()
            .filter(|rec| matches!(rec, registry::Record::Result { .. }))
            .count();
        assert_eq!(result_lines, 1);
    }

    // ── policy tool overrides ────────────────────────────────────────────────

    /// A client that records the tool set it was handed, then ends the run.
    struct ToolRecordingClient {
        seen: Mutex<Option<Value>>,
    }
    impl LlmClient for ToolRecordingClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            _msgs: &mut Value,
            _s: &str,
            t: &Value,
            _effort: Effort,
        ) -> Result<Value, String> {
            *self.seen.lock().unwrap() = Some(t.clone());
            Ok(json!({"content":[{"type":"text","text":"done"}]}))
        }
    }

    fn custom_defs() -> Value {
        json!([{
            "name": "custom_echo",
            "description": "echo",
            "input_schema": {"type":"object","properties":{},"required":[]}
        }])
    }

    /// Records the messages array on every call, so a test can see what the
    /// conversation actually contains after a tool result was pushed.
    struct MsgRecordingClient {
        responses: Mutex<Vec<Value>>,
        last_msgs: Mutex<Option<Value>>,
    }
    impl LlmClient for MsgRecordingClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            msgs: &mut Value,
            _s: &str,
            _t: &Value,
            _effort: Effort,
        ) -> Result<Value, String> {
            *self.last_msgs.lock().unwrap() = Some(msgs.clone());
            let mut r = self.responses.lock().unwrap();
            if r.is_empty() {
                Ok(json!({"content":[{"type":"text","text":"done"}]}))
            } else {
                Ok(r.remove(0))
            }
        }
    }

    fn shell_call_then_stop() -> Vec<Value> {
        vec![json!({"content":[{
            "type":"tool_use","id":"t1","name":"run_shell",
            "input":{"command":"echo hi"}
        }]})]
    }

    fn redact_secrets(s: &mut String) -> Result<(), String> {
        *s = s.replace("SECRET", "[REDACTED]");
        Ok(())
    }

    fn leak_secret(_tc: &crate::provider::ToolCall) -> Option<String> {
        Some("value is SECRET".to_string())
    }

    #[test]
    fn tool_results_are_redacted_before_entering_the_conversation() {
        // The guarantee: a dispatcher that returns sensitive output cannot put
        // it into the history, because the loop redacts at the door. This is
        // what makes a tool added later unable to leak by forgetting to mask.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = MsgRecordingClient {
            responses: Mutex::new(shell_call_then_stop()),
            last_msgs: Mutex::new(None),
        };
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths, 100);
        let policy = Policy {
            dispatch: Some(leak_secret),
            redact: Some(redact_secrets),
            ..root_policy()
        };
        run(&c, RunConfig {
            job: job(5),
            policy,
            depth: 0,
            label: "d0".to_string(),
            thread_id: None,
        });
        let msgs = client.last_msgs.lock().unwrap().take().unwrap();
        let dump = serde_json::to_string(&msgs).unwrap();
        assert!(dump.contains("[REDACTED]"), "redaction did not run: {dump}");
        assert!(!dump.contains("SECRET"), "raw value reached the conversation: {dump}");
    }

    #[test]
    fn a_failing_redactor_withholds_the_tool_result() {
        // Fail closed. A caller cannot tell "nothing needed redacting" from
        // "redacting broke", so passing the original through on error would be
        // indistinguishable from success.
        fn always_fails(_s: &mut String) -> Result<(), String> {
            Err("detector unavailable".to_string())
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = MsgRecordingClient {
            responses: Mutex::new(shell_call_then_stop()),
            last_msgs: Mutex::new(None),
        };
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths, 100);
        let policy = Policy {
            dispatch: Some(leak_secret),
            redact: Some(always_fails),
            ..root_policy()
        };
        run(&c, RunConfig {
            job: job(5),
            policy,
            depth: 0,
            label: "d0".to_string(),
            thread_id: None,
        });
        let msgs = client.last_msgs.lock().unwrap().take().unwrap();
        let dump = serde_json::to_string(&msgs).unwrap();
        assert!(!dump.contains("SECRET"), "raw value survived a failed redaction: {dump}");
        assert!(dump.contains("withheld"), "expected a withholding marker: {dump}");
    }

    #[test]
    fn tool_defs_override_replaces_builtins_and_keeps_spawn_agent_at_root() {
        for (depth, wants_spawn) in [(0usize, true), (1usize, false)] {
            let dir = tempfile::tempdir().unwrap();
            let paths = Paths::for_root_under(dir.path().to_path_buf());
            let client = ToolRecordingClient {
                seen: Mutex::new(None),
            };
            let provider = Anthropic;
            let c = ctx(&client, &provider, paths, 100);
            let policy = Policy {
                tool_defs: Some(custom_defs),
                ..root_policy()
            };
            run(
                &c,
                RunConfig {
                    job: job(5),
                    policy,
                    depth,
                    label: "d0".to_string(),
                    thread_id: None,
                },
            );
            let seen = client.seen.lock().unwrap().take().unwrap();
            let names: Vec<&str> = seen
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["name"].as_str().unwrap())
                .collect();
            assert!(names.contains(&"custom_echo"), "custom tool offered");
            assert!(!names.contains(&"run_shell"), "builtins replaced");
            assert_eq!(
                names.contains(&"spawn_agent"),
                wants_spawn,
                "spawn_agent appended only at depth 0 (depth {depth})"
            );
        }
    }

    fn intercept_run_shell(tc: &crate::provider::ToolCall) -> Option<String> {
        (tc.name == "run_shell").then(|| "INTERCEPTED".to_string())
    }

    fn never_intercept(_tc: &crate::provider::ToolCall) -> Option<String> {
        None
    }

    #[test]
    fn policy_dispatch_intercepts_before_builtin() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        // The command's OUTPUT ("x-zzz") differs from the command TEXT, so its
        // absence proves the builtin never executed (the command text itself
        // legitimately appears in the conversation as the tool_use input).
        let client = ScriptedClient::new(vec![
            shell_turn("t1", "printf x-%s zzz"),
            text_turn("done"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let policy = Policy {
            dispatch: Some(intercept_run_shell),
            ..root_policy()
        };
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy,
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".into()),
            },
        );
        assert_eq!(r.status, Status::Success);
        let hist = json!(thread::load_thread(&paths)).to_string();
        assert!(hist.contains("INTERCEPTED"), "dispatcher output in convo");
        assert!(
            !hist.contains("x-zzz"),
            "builtin run_shell must not have executed"
        );
    }

    #[test]
    fn policy_dispatch_none_falls_through_to_builtin() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            shell_turn("t1", "printf builtin-ran"),
            text_turn("done"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, paths.clone(), 100);
        let policy = Policy {
            dispatch: Some(never_intercept),
            ..root_policy()
        };
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy,
                depth: 0,
                label: "d0".to_string(),
                thread_id: Some("t".into()),
            },
        );
        assert_eq!(r.status, Status::Success);
        let hist = json!(thread::load_thread(&paths)).to_string();
        assert!(hist.contains("builtin-ran"), "fell through to builtin");
    }
}
