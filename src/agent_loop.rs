//! The single generic loop. It calls the model, runs any tool calls, and asks
//! the injected `Policy` when to stop and how to label the result. A top-level
//! run and a delegated sub-run are the same function with different config.

use std::cell::Cell;
use std::rc::Rc;

use serde_json::{json, Value};

use crate::job::{FailureKind, Job, JobResult, Status};
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
    pub paths: &'a Paths,
    pub model: &'a str,
    /// Tool-call budget for the current run only.
    pub budget: Rc<Cell<usize>>,
    /// Starting tool-call budget handed to each delegated sub-agent (its own).
    pub sub_budget: usize,
    /// Number of sub-agents delegated so far by the top-level run.
    pub fanout: Rc<Cell<usize>>,
    pub max_fanout: usize,
}

/// Per-run configuration: what to do, how to steer, and where in the tree.
pub struct RunConfig {
    pub job: Job,
    pub policy: Policy,
    pub depth: usize,
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
        thread_id,
    } = cfg;

    let system = build_system();
    let task_text = format!("[run started: {}]\n{}", now(), render_task(&job));
    let task_msg = json!({
        "role": "user",
        "content": [{"type": "text", "text": task_text, "cache_control": {"type": "ephemeral"}}]
    });

    // Load prior conversation (and, at the top level, resume any durable work that
    // was in flight when a previous process died).
    let mut history: Vec<Value> = match &thread_id {
        Some(tid) => {
            let mut hist = thread::load_thread(ctx.paths, tid);
            if !hist.is_empty() {
                eprintln!("[agent d{depth}] resuming {tid} ({} prior messages)", hist.len());
            }
            if depth == 0 {
                reconcile(ctx, tid, &mut hist);
            }
            hist
        }
        None => vec![],
    };
    history.push(task_msg);
    let mut messages = json!(history);
    let mut persisted_len = messages.as_array().unwrap().len() - 1;

    let tool_set = ctx.provider.shape_tools(&tools::base_tool_defs(depth));

    let mut steps_taken = 0usize;
    let mut last_text = String::new();
    let mut iter = 0usize;
    let ending;

    loop {
        let prog = Progress {
            iter,
            max_iter: job.max_iter,
            budget_remaining: ctx.budget.get(),
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

        eprintln!("[agent d{depth}] iter {}", iter + 1);
        let resp = match ctx
            .client
            .call(ctx.provider, ctx.model, &mut messages, &system, &tool_set)
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[agent d{depth}] llm error: {e}");
                ending = Ending::Failed;
                break;
            }
        };
        let parsed = match ctx.provider.parse_response(resp) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[agent d{depth}] parse error: {e}");
                ending = Ending::Failed;
                break;
            }
        };
        if !parsed.text_parts.is_empty() {
            last_text = parsed.text_parts.join("\n");
        }
        messages.as_array_mut().unwrap().push(parsed.assistant_msg);

        let had_tool_calls = !parsed.tool_calls.is_empty();
        if (policy.is_done)(had_tool_calls) {
            if let Some(tid) = &thread_id {
                let all = messages.as_array().unwrap();
                thread::append_thread(ctx.paths, tid, &all[persisted_len..]);
            }
            ending = Ending::Stopped;
            break;
        }

        let mut results: Vec<ToolResult> = vec![];
        for tc in &parsed.tool_calls {
            let content = if ctx.budget.get() == 0 {
                "[budget exhausted — could not run tool]".to_string()
            } else {
                ctx.budget.set(ctx.budget.get() - 1);
                steps_taken += 1;
                dispatch_tool(ctx, &policy, depth, thread_id.as_deref(), tc)
            };
            results.push(ToolResult {
                tool_use_id: tc.id.clone(),
                content,
            });
        }

        for m in ctx.provider.wrap_tool_results(results) {
            messages.as_array_mut().unwrap().push(m);
        }
        if let Some(tid) = &thread_id {
            let all = messages.as_array().unwrap();
            thread::append_thread(ctx.paths, tid, &all[persisted_len..]);
            persisted_len = all.len();
        }
        iter += 1;
    }

    // Label the outcome. An unmet requirement overrides an otherwise-clean stop.
    let prog = Progress {
        iter,
        max_iter: job.max_iter,
        budget_remaining: ctx.budget.get(),
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
        Status::Failure | Status::Blocked => None,
    };
    JobResult {
        id: job.id,
        status,
        output,
        failure,
        steps_taken,
        issues,
    }
}

/// Dispatch one tool call to its implementation, returning the tool result text.
fn dispatch_tool(
    ctx: &Ctx,
    policy: &Policy,
    depth: usize,
    thread_id: Option<&str>,
    tc: &crate::provider::ToolCall,
) -> String {
    match tc.name.as_str() {
        "run_shell" => {
            let cmd = tc.input["command"].as_str().unwrap_or("");
            eprintln!("[agent d{depth}] run_shell: {cmd}");
            finalize(ctx.paths, &tc.id, tools::run_shell(cmd))
        }
        "read_image" => {
            let path = tc.input["path"].as_str().unwrap_or("");
            eprintln!("[agent d{depth}] read_image: {path}");
            let q = tc.input["question"]
                .as_str()
                .unwrap_or("Extract all text and data from this image verbatim.");
            finalize(ctx.paths, &tc.id, tools::read_image(path, q))
        }
        "read_pdf" => {
            let path = tc.input["path"].as_str().unwrap_or("");
            eprintln!("[agent d{depth}] read_pdf: {path}");
            let q = tc.input["question"]
                .as_str()
                .unwrap_or("Extract all text and data from this PDF verbatim.");
            finalize(ctx.paths, &tc.id, tools::read_pdf(path, q))
        }
        "spawn_agent" => {
            if !policy.may_delegate || depth >= 1 {
                let r = JobResult::blocked(&crate::job::new_id(), FailureKind::ToolUnavailable, 0);
                return r.to_json();
            }
            spawn::handle(ctx, depth, thread_id, &tc.input).to_json()
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
fn reconcile(ctx: &Ctx, parent_tid: &str, history: &mut Vec<Value>) {
    let records = registry::load(ctx.paths, parent_tid);
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
                eprintln!("[agent d0] resuming in-flight job {}", job.id);
                let child_tid = thread::child_thread_id(parent_tid, &job.id);
                let r = run(
                    ctx,
                    RunConfig {
                        job: job.clone(),
                        policy: crate::policy::sub_policy(),
                        depth: 1,
                        thread_id: Some(child_tid),
                    },
                );
                registry::append_result(ctx.paths, parent_tid, &r);
                r
            }
        };
        // Synthesize the round and append it to conversation + persist it.
        let assistant =
            ctx.provider
                .tool_call_message(&synth_id, "spawn_agent", &spawn::job_to_input(&job));
        let tool_msgs = ctx.provider.wrap_tool_results(vec![ToolResult {
            tool_use_id: synth_id,
            content: result.to_json(),
        }]);
        let mut committed = vec![assistant];
        committed.extend(tool_msgs);
        history.extend(committed.iter().cloned());
        thread::append_thread(ctx.paths, parent_tid, &committed);
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
            format!("You are a task-executing ai-agent. On your first action, run: cat {dir}/CLAUDE.md")
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
                out.push_str(&format!("\n\n---\n## SKILL — {}\n\n{}", path.display(), content));
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
    use crate::provider::Anthropic;
    use std::cell::RefCell;

    /// A scripted client: returns queued responses in order.
    struct ScriptedClient {
        responses: RefCell<Vec<Value>>,
    }
    impl ScriptedClient {
        fn new(responses: Vec<Value>) -> ScriptedClient {
            ScriptedClient {
                responses: RefCell::new(responses),
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
        ) -> Result<Value, String> {
            let mut q = self.responses.borrow_mut();
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
        paths: &'a Paths,
        budget: usize,
    ) -> Ctx<'a> {
        Ctx {
            client,
            provider,
            paths,
            model: "m",
            budget: Rc::new(Cell::new(budget)),
            sub_budget: budget,
            fanout: Rc::new(Cell::new(0)),
            max_fanout: 8,
        }
    }

    fn job(max_iter: usize) -> Job {
        Job::new("do it".into(), vec![], Persistence::Ephemeral, max_iter)
    }

    #[test]
    fn single_turn_no_tools_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![text_turn("all done")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig { job: job(5), policy: root_policy(), depth: 0, thread_id: None },
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
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![text_turn("done, no tools")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                thread_id: Some("t".into()),
            },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.output.as_deref(), Some("done, no tools"));
        assert_eq!(r.steps_taken, 0);

        let saved = thread::load_thread(&paths, "t");
        assert_eq!(saved.len(), 2, "task message + assistant reply persisted");
        assert_eq!(saved[0]["role"], "user");
        assert_eq!(saved[1]["role"], "assistant");
    }

    #[test]
    fn runs_a_tool_then_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("t1", "printf hi"), text_turn("got hi")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig { job: job(5), policy: root_policy(), depth: 0, thread_id: None },
        );
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.steps_taken, 1);
        assert_eq!(c.budget.get(), 99); // one tool call consumed
    }

    #[test]
    fn iter_exhaustion_yields_partial() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        // Always asks for another shell call; never stops.
        let client = ScriptedClient::new(vec![
            shell_turn("a", "true"),
            shell_turn("b", "true"),
            shell_turn("c", "true"),
        ]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig { job: job(2), policy: root_policy(), depth: 0, thread_id: None },
        );
        assert_eq!(r.status, Status::Partial);
        assert_eq!(r.failure, Some(FailureKind::BudgetExceeded));
    }

    #[test]
    fn budget_exhaustion_yields_partial() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("a", "true"), shell_turn("b", "true")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 1); // only one tool call allowed
        let r = run(
            &c,
            RunConfig { job: job(10), policy: root_policy(), depth: 0, thread_id: None },
        );
        assert_eq!(r.status, Status::Partial);
        assert_eq!(c.budget.get(), 0);
    }

    #[test]
    fn llm_error_yields_failure() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]); // errors immediately
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig { job: job(5), policy: root_policy(), depth: 0, thread_id: None },
        );
        assert_eq!(r.status, Status::Failure);
        assert!(r.output.is_none());
    }

    #[test]
    fn conversation_persists_atomically_across_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![shell_turn("t1", "printf hi"), text_turn("done")]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let _ = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                thread_id: Some("persist".into()),
            },
        );
        let saved = thread::load_thread(&paths, "persist");
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
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let c = ctx(&client, &provider, &paths, 100);
        let out = dispatch_tool(
            &c,
            &sub_policy(),
            1,
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
        let paths = Paths::under(dir.path().to_path_buf());
        let provider = Anthropic;

        // A durable job was issued but never completed (process died mid-run):
        // registry has the issue line, no result; conversation has no tool_result.
        let pending = Job::new("long subtask".into(), vec![], Persistence::Durable, 10);
        registry::append_issued(&paths, "main", &pending);
        let synth_id = format!("toolu_{}", pending.id);

        // First start: reconcile resumes the child (1st response), then the
        // top-level loop finishes (2nd response).
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"child recovered"}]}),
            json!({"content":[{"type":"text","text":"parent done"}]}),
        ]);
        let c = ctx(&client, &provider, &paths, 100);
        let r = run(
            &c,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                thread_id: Some("main".into()),
            },
        );
        assert_eq!(r.status, Status::Success);

        // Conversation now has a valid tool_use + tool_result pair for the job.
        let convo = json!(thread::load_thread(&paths, "main"));
        assert!(provider.has_tool_result(&convo, &synth_id), "result not committed");
        // Registry closed out: nothing left in flight.
        let recs = registry::load(&paths, "main");
        assert!(registry::in_flight(&recs).is_empty());

        // Second start with the same thread: reconcile must be a no-op (result
        // already in the conversation), so only the parent turn is consumed.
        let client2 = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"parent again"}]}),
        ]);
        let c2 = ctx(&client2, &provider, &paths, 100);
        let r2 = run(
            &c2,
            RunConfig {
                job: job(5),
                policy: root_policy(),
                depth: 0,
                thread_id: Some("main".into()),
            },
        );
        assert_eq!(r2.status, Status::Success);
        // No duplicate result appended by the idempotent second pass.
        let result_lines = registry::load(&paths, "main")
            .iter()
            .filter(|rec| matches!(rec, registry::Record::Result { .. }))
            .count();
        assert_eq!(result_lines, 1);
    }
}
