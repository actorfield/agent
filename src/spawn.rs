//! Delegation: hand a self-contained subtask to an isolated sub-agent that runs
//! its own loop and reports back a structured result. The sub-agent gets a fresh
//! context (that is the whole point — the caller does not carry its intermediate
//! steps) and cannot delegate further.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::agent_loop::{run, Ctx, RunConfig, DEFAULT_SUB_MAX_ITER};
use crate::job::{Effort, FailureKind, Job, JobResult, Persistence};
use crate::policy::sub_policy;
use crate::thread::Paths;
use crate::{registry, thread};

/// Build a job from the tool arguments and run it as a sub-agent.
pub fn handle(
    ctx: &Ctx,
    parent_depth: usize,
    parent_label: &str,
    parent_tid: Option<&str>,
    input: &Value,
) -> JobResult {
    let task = input["task"].as_str().unwrap_or("").trim().to_string();
    if task.is_empty() {
        return JobResult::blocked(&crate::job::new_id(), FailureKind::AmbiguousRequest, 0);
    }

    // Fan-out cap: a single top-level run may only delegate so many sub-agents.
    // Concurrent spawn_agent calls (agent_loop::run now dispatches them on
    // separate threads) can race here, so this is a compare-exchange loop, not
    // a plain read-then-write — the latter could let two callers both observe
    // room under the cap and both admit, overshooting max_fanout.
    loop {
        let f = ctx.fanout.load(Ordering::Relaxed);
        if f >= ctx.max_fanout {
            let id = crate::job::new_id();
            return JobResult::partial(&id, "fan-out limit reached".into(), FailureKind::BudgetExceeded, 0);
        }
        if ctx
            .fanout
            .compare_exchange(f, f + 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            break;
        }
        // Lost the race to a concurrent sibling — reload and retry.
    }

    let checks = input["checks"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let persistence = Persistence::parse(input["persistence"].as_str());
    let max_iter = input["max_iter"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_SUB_MAX_ITER);
    let mut job = Job::new(task, checks, persistence, max_iter);

    // Effort is per-run: a sub-agent can be given its own level via the tool
    // call; when omitted it inherits the parent's, so a high-effort planner's
    // delegated execution steps don't silently drop back to no reasoning. Stored
    // on the job (not just passed to Ctx) so a durable resume reconstructs the
    // same level via job_to_input rather than silently reverting to None.
    let effort = input
        .get("effort")
        .and_then(|v| v.as_str())
        .map(|s| Effort::parse(Some(s)))
        .unwrap_or(ctx.effort);
    job.effort = effort;

    // Durable only takes effect when the parent itself is persisted, since resume
    // is driven from the parent's registry.
    let durable = persistence == Persistence::Durable && parent_tid.is_some();
    let child_tid = if durable {
        Some(thread::child_thread_id(parent_tid.unwrap(), &job.id))
    } else {
        None
    };

    if durable {
        registry::append_issued(&ctx.paths, &job);
    }

    // The sub-agent runs with its own fresh budget — independent of the caller's —
    // and its own directory, nested under the parent's, so its scratchpad and any
    // other filesystem state it keeps can never collide with the parent's or a
    // sibling's. Its log label is "{parent_label}-{index}" where index is this
    // parent's own sequential spawn counter — e.g. the first sub-agent a top-level
    // "d0" run spawns is "d0-0", the second is "d0-1", regardless of whether they
    // run sequentially or concurrently (see agent_loop.rs's thread::scope batch).
    let child_index = ctx.spawn_index.fetch_add(1, Ordering::Relaxed);
    let child_label = format!("{parent_label}-{child_index}");
    let child_ctx = Ctx {
        client: ctx.client,
        provider: ctx.provider,
        paths: Paths::for_child(&ctx.paths, &job.id),
        model: ctx.model,
        effort,
        budget: Arc::new(AtomicUsize::new(ctx.sub_budget)),
        sub_budget: ctx.sub_budget,
        fanout: ctx.fanout.clone(),
        max_fanout: ctx.max_fanout,
        spawn_index: Arc::new(AtomicUsize::new(0)),
    };
    let result = run(
        &child_ctx,
        RunConfig {
            job: job.clone(),
            policy: sub_policy(),
            depth: parent_depth + 1,
            label: child_label,
            thread_id: child_tid,
        },
    );

    if durable {
        registry::append_result(&ctx.paths, &result);
    }

    result
}

/// Reconstruct the tool arguments that would produce this job — used to rebuild a
/// delegation round when resuming persisted work.
pub fn job_to_input(job: &Job) -> Value {
    json!({
        "task": job.task,
        "checks": job.checks,
        "persistence": match job.persistence {
            Persistence::Ephemeral => "ephemeral",
            Persistence::Durable => "durable",
        },
        "effort": match job.effort {
            Effort::None => "none",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        },
        "max_iter": job.max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Status;
    use crate::llm::LlmClient;
    use crate::provider::{Anthropic, Provider};
    use std::sync::Mutex;

    /// `Mutex`, not `RefCell` — `LlmClient` now requires `Send + Sync` (so `Ctx`
    /// can cross the scoped threads spawn_agent uses for concurrent sub-agents).
    /// This is a compile-time requirement of the trait bound; most tests below
    /// still call it from a single thread.
    struct ScriptedClient {
        responses: Mutex<Vec<Value>>,
        seen_effort: Mutex<Vec<Effort>>,
        seen_msgs: Mutex<Vec<Value>>,
    }
    impl ScriptedClient {
        fn new(responses: Vec<Value>) -> ScriptedClient {
            ScriptedClient {
                responses: Mutex::new(responses),
                seen_effort: Mutex::new(vec![]),
                seen_msgs: Mutex::new(vec![]),
            }
        }
    }
    impl LlmClient for ScriptedClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            msgs: &mut Value,
            _s: &str,
            _t: &Value,
            effort: Effort,
        ) -> Result<Value, String> {
            self.seen_effort.lock().unwrap().push(effort);
            self.seen_msgs.lock().unwrap().push(msgs.clone());
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                Err("no scripted response".into())
            } else {
                Ok(q.remove(0))
            }
        }
    }

    fn mk_ctx<'a>(
        client: &'a dyn LlmClient,
        provider: &'a dyn Provider,
        paths: Paths,
        budget: usize,
        max_fanout: usize,
        fanout: Arc<AtomicUsize>,
    ) -> Ctx<'a> {
        Ctx {
            client,
            provider,
            paths,
            model: "m",
            effort: Effort::None,
            budget: Arc::new(AtomicUsize::new(budget)),
            sub_budget: budget,
            fanout,
            max_fanout,
            spawn_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[test]
    fn empty_task_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, paths, 100, 8, Arc::new(AtomicUsize::new(0)));
        let r = handle(&c, 0, "d0", None, &json!({"task": "  "}));
        assert_eq!(r.status, Status::Blocked);
    }

    #[test]
    fn fanout_cap_returns_partial_without_running() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let fanout = Arc::new(AtomicUsize::new(2));
        let c = mk_ctx(&client, &provider, paths, 100, 2, fanout);
        let r = handle(&c, 0, "d0", None, &json!({"task": "do"}));
        assert_eq!(r.status, Status::Partial);
        assert_eq!(r.failure, Some(FailureKind::BudgetExceeded));
    }

    #[test]
    fn concurrent_spawn_calls_do_not_overshoot_fanout_cap() {
        // max_fanout = 1: two threads race to call handle() at the same time; only
        // one may be admitted, the other must be capped, and the final counter
        // must land at exactly 1 — never 2, which a check-then-set race (instead
        // of the compare-exchange loop in handle()) could produce.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"a"}]}),
            json!({"content":[{"type":"text","text":"b"}]}),
        ]);
        let provider = Anthropic;
        let fanout = Arc::new(AtomicUsize::new(0));
        let c = mk_ctx(&client, &provider, paths, 100, 1, fanout.clone());
        let (r1, r2) = std::thread::scope(|scope| {
            let h1 = scope.spawn(|| handle(&c, 0, "d0", None, &json!({"task": "x"})));
            let h2 = scope.spawn(|| handle(&c, 0, "d0", None, &json!({"task": "y"})));
            (h1.join().unwrap(), h2.join().unwrap())
        });
        let successes = [&r1, &r2].iter().filter(|r| r.status == Status::Success).count();
        let capped = [&r1, &r2]
            .iter()
            .filter(|r| r.failure == Some(FailureKind::BudgetExceeded))
            .count();
        assert_eq!(successes, 1);
        assert_eq!(capped, 1);
        assert_eq!(fanout.load(Ordering::Relaxed), 1); // never overshoots the cap
    }

    #[test]
    fn ephemeral_sub_runs_and_returns_result() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"sub done"}]})]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, paths.clone(), 100, 8, Arc::new(AtomicUsize::new(0)));
        let r = handle(&c, 0, "d0", None, &json!({"task": "compute"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.output.as_deref(), Some("sub done"));
        // No registry writes for an ephemeral sub.
        assert!(registry::load(&paths).is_empty());
    }

    #[test]
    fn successive_spawns_get_distinct_sequential_spawn_index() {
        // Log labels are built as "{parent_label}-{spawn_index}" — this test
        // pins down the actual invariant (ctx.spawn_index incrementing once per
        // handle() call from the same parent Ctx) rather than the display
        // string itself, since the label is eprintln!-only and not part of
        // JobResult. Previously every sub-agent printed as the same "d1" label
        // regardless of how many ran, making concurrent/sequential sub-agents
        // indistinguishable in a log without cross-referencing AGENT_RUN_DIR.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"text","text":"first"}]}),
            json!({"content":[{"type":"text","text":"second"}]}),
            json!({"content":[{"type":"text","text":"third"}]}),
        ]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, paths, 100, 8, Arc::new(AtomicUsize::new(0)));

        assert_eq!(c.spawn_index.load(Ordering::Relaxed), 0);
        handle(&c, 0, "d0", None, &json!({"task": "one"}));
        assert_eq!(c.spawn_index.load(Ordering::Relaxed), 1);
        handle(&c, 0, "d0", None, &json!({"task": "two"}));
        assert_eq!(c.spawn_index.load(Ordering::Relaxed), 2);
        handle(&c, 0, "d0", None, &json!({"task": "three"}));
        assert_eq!(c.spawn_index.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn sub_agent_task_text_is_unmodified_and_gets_own_nested_paths() {
        // Isolation between a sub-agent and its parent/siblings is now structural
        // (each gets its own directory nested under the parent's), not prompt-based.
        // spawn.rs must not rewrite or annotate the caller's task text at all.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let parent_dir = paths.dir().to_path_buf();
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, paths.clone(), 100, 8, Arc::new(AtomicUsize::new(0)));
        let r = handle(&c, 0, "d0", None, &json!({"task": "fill section 2.0"}));
        assert_eq!(r.status, Status::Success);

        // The task text sent to the model is the caller's task, verbatim — no
        // Rust-level scratchpad/discovery prose injected.
        let sent = client.seen_msgs.lock().unwrap();
        let task_text = sent[0][0]["content"][0]["text"].as_str().unwrap();
        assert!(task_text.contains("fill section 2.0"));
        assert!(!task_text.contains("SCRATCHPAD"));
        assert!(!task_text.contains("discovery.txt"));

        // Structural isolation: the child's own directory is nested under, and
        // distinct from, the parent's — this is what actually prevents
        // scratchpad/discovery clashes now, not prose.
        let child_paths = Paths::for_child(&paths, &r.id);
        assert!(child_paths.dir().starts_with(&parent_dir));
        assert_ne!(child_paths.dir(), parent_dir);
    }

    #[test]
    fn durable_sub_records_issued_and_result() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"durable done"}]})]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, paths.clone(), 100, 8, Arc::new(AtomicUsize::new(0)));
        let r = handle(&c, 0, "d0", Some("main"), &json!({"task":"long","persistence":"durable"}));
        assert_eq!(r.status, Status::Success);
        let recs = registry::load(&paths);
        assert_eq!(recs.len(), 2); // issued + result
        assert!(registry::in_flight(&recs).is_empty());
    }

    #[test]
    fn sub_budget_is_independent_of_parent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"tool_use","id":"s1","name":"run_shell","input":{"command":"true"}}]}),
            json!({"content":[{"type":"text","text":"sub done"}]}),
        ]);
        let provider = Anthropic;
        // Parent's own budget is exhausted, but the sub-agent gets its own allowance
        // and runs to completion — no tree-wide pool couples them.
        let c = Ctx {
            client: &client,
            provider: &provider,
            paths,
            model: "m",
            effort: Effort::None,
            budget: Arc::new(AtomicUsize::new(0)),
            sub_budget: 50,
            fanout: Arc::new(AtomicUsize::new(0)),
            max_fanout: 8,
            spawn_index: Arc::new(AtomicUsize::new(0)),
        };
        let r = handle(&c, 0, "d0", None, &json!({"task": "work"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.steps_taken, 1); // sub spent its own budget, not the parent's
        assert_eq!(c.budget.load(Ordering::Relaxed), 0); // parent counter untouched
    }

    #[test]
    fn job_to_input_round_trips_fields() {
        let mut job = Job::new("t".into(), vec!["c".into()], Persistence::Durable, 7);
        job.effort = Effort::High;
        let v = job_to_input(&job);
        assert_eq!(v["task"], "t");
        assert_eq!(v["persistence"], "durable");
        assert_eq!(v["effort"], "high");
        assert_eq!(v["max_iter"], 7);
    }

    #[test]
    fn effort_defaults_to_parent_when_omitted() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let mut c = mk_ctx(&client, &provider, paths, 100, 8, Arc::new(AtomicUsize::new(0)));
        c.effort = Effort::High;
        let r = handle(&c, 0, "d0", None, &json!({"task": "compute"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(client.seen_effort.lock().unwrap().as_slice(), &[Effort::High]);
    }

    #[test]
    fn effort_override_wins_over_parent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let mut c = mk_ctx(&client, &provider, paths, 100, 8, Arc::new(AtomicUsize::new(0)));
        c.effort = Effort::High;
        let r = handle(&c, 0, "d0", None, &json!({"task": "compute", "effort": "low"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(client.seen_effort.lock().unwrap().as_slice(), &[Effort::Low]);
    }

    #[test]
    fn effort_none_override_disables_reasoning_despite_parent_high() {
        // "none" is now a reachable schema value (tools.rs) — a caller must be able
        // to explicitly force no reasoning on a sub-agent even when it inherits a
        // high-effort parent, not just omit the field to inherit.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let mut c = mk_ctx(&client, &provider, paths, 100, 8, Arc::new(AtomicUsize::new(0)));
        c.effort = Effort::High;
        let r = handle(&c, 0, "d0", None, &json!({"task": "compute", "effort": "none"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(client.seen_effort.lock().unwrap().as_slice(), &[Effort::None]);
    }
}
