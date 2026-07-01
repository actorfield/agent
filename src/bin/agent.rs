// ai-agent actor handler — a tool loop against Anthropic or OpenAI-compatible APIs.
//
// Usage (forked by the runtime per task tuple):
//   payload bytes written to stdin by the runtime
//
// Standalone usage:
//   agent [--thread <id>] 'task string' [model] [max_iter]
//   agent [--thread <id>] < /tmp/task.txt
//
// Provider is inferred from LLM_URL (see agent::provider).

use std::cell::Cell;
use std::io::Read;
use std::rc::Rc;

use agent::agent_loop::{run, Ctx, RunConfig};
use agent::job::{Job, Persistence};
use agent::llm::{llm_model, UreqClient};
use agent::policy::root_policy;
use agent::provider::detect_provider;
use agent::thread::Paths;

const DEFAULT_MAX_ITER: usize = 50;
const DEFAULT_TOOL_BUDGET: usize = 200;
const DEFAULT_MAX_FANOUT: usize = 8;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --thread <id> from anywhere in args.
    let thread_flag_pos = args.iter().position(|a| a == "--thread");
    let thread_id = thread_flag_pos.and_then(|i| args.get(i + 1)).cloned();
    let skip: std::collections::HashSet<usize> = thread_flag_pos
        .map(|i| [i, i + 1].into_iter().collect())
        .unwrap_or_default();
    let filtered: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, a)| a)
        .collect();

    let (task, model, max_iter) = if filtered.len() > 1 {
        let task = filtered[1].clone();
        let model = filtered
            .get(2)
            .map(|s| s.to_string())
            .unwrap_or_else(llm_model);
        let max_iter = filtered
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_ITER);
        (task, model, max_iter)
    } else {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).ok();
        (buf.trim().to_string(), llm_model(), DEFAULT_MAX_ITER)
    };

    if task.is_empty() {
        eprintln!("usage: agent [--thread <id>] 'task' [model] [max_iter]");
        std::process::exit(1);
    }

    let url = agent::llm::llm_url();
    let provider = detect_provider(&url);
    let client = UreqClient;
    let paths = Paths::system();

    // Per-run tool-call budget: the top-level run and each sub-agent each get their
    // own fresh allowance of this size — no tree-wide pool.
    let tool_budget = env_usize("AGENT_TOOL_BUDGET", DEFAULT_TOOL_BUDGET);
    let ctx = Ctx {
        client: &client,
        provider: provider.as_ref(),
        paths: &paths,
        model: &model,
        budget: Rc::new(Cell::new(tool_budget)),
        sub_budget: tool_budget,
        fanout: Rc::new(Cell::new(0)),
        max_fanout: env_usize("AGENT_MAX_FANOUT", DEFAULT_MAX_FANOUT),
    };

    let job = Job::new(task, Vec::new(), Persistence::Ephemeral, max_iter);
    let result = run(
        &ctx,
        RunConfig {
            job,
            policy: root_policy(),
            depth: 0,
            thread_id,
        },
    );

    // Emit result topic override then the result text — the runtime uses the first
    // line as the publish topic when it matches `topic_name\n`.
    match result.output {
        Some(text) if matches!(result.status, agent::job::Status::Success | agent::job::Status::Partial) => {
            println!("ai_result");
            println!("{text}");
        }
        _ => {
            eprintln!("run ended with status {:?}: {:?}", result.status, result.failure);
            std::process::exit(1);
        }
    }
}
