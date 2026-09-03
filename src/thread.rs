//! Conversation persistence. A run may be resumed: prior messages are reloaded,
//! new ones appended. Paths are injected so tests use temp dirs.
//!
//! Each run gets its own directory, keyed by the job's id (always present, whether
//! or not the run is durable). A delegated sub-agent's directory nests under its
//! parent's (`<parent>/sub/<child_job_id>/`), so the filesystem mirrors the
//! delegation tree instead of correlating flat files by filename convention.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Filesystem location for one run's persisted state — its own directory.
#[derive(Clone, Debug)]
pub struct Paths {
    dir: PathBuf,
}

impl Paths {
    /// Root Paths for a fresh run, under `base/<job_id>`. Lazy: no I/O here.
    pub fn for_root(base: &Path, job_id: &str) -> Paths {
        Paths { dir: base.join(job_id) }
    }

    /// Derive a child run's Paths, nested under the parent's own directory.
    pub fn for_child(parent: &Paths, child_job_id: &str) -> Paths {
        Paths { dir: parent.dir.join("sub").join(child_job_id) }
    }

    /// Production default root directory for top-level runs.
    pub fn default_base() -> PathBuf {
        std::env::temp_dir().join("agent_runs")
    }

    /// Test/tooling helper: root Paths under an arbitrary base with a fresh job id.
    pub fn for_root_under(base: PathBuf) -> Paths {
        Paths::for_root(&base, &crate::job::new_id())
    }

    /// The run's own directory. Ensures it exists on first access — every other
    /// path-building method goes through this, so callers never need to create
    /// directories themselves (including the LLM's own `run_shell` calls against
    /// `scratchpad_path()`, which Rust never writes directly).
    pub fn dir(&self) -> &Path {
        std::fs::create_dir_all(&self.dir).ok();
        &self.dir
    }

    pub fn thread_path(&self) -> PathBuf {
        self.dir().join("thread.jsonl")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.dir().join("registry.jsonl")
    }

    pub fn spill_path(&self, id: &str) -> PathBuf {
        let d = self.dir().join("spill");
        std::fs::create_dir_all(&d).ok();
        d.join(format!("tool_result_{id}.txt"))
    }

    pub fn scratchpad_path(&self) -> PathBuf {
        self.dir().join("scratchpad.md")
    }
}

/// Deterministic child-run id for a delegated job, stable across restarts.
pub fn child_thread_id(parent: &str, job_id: &str) -> String {
    format!("{parent}__sub_{job_id}")
}

pub fn load_thread(paths: &Paths) -> Vec<Value> {
    let path = paths.thread_path();
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    contents
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        // Marker lines (see append_usage, append_ending) are NOT conversation
        // turns — every caller here feeds the result straight back to the LLM
        // as message history on resume. A marker has no "role", so it wouldn't
        // match the shape either provider expects; excluding it here (once,
        // centrally) means no caller has to remember to filter it out itself.
        //
        // Keyed on the ABSENCE of "role" rather than on a list of known kinds.
        // Matching `kind == "usage"` meant every marker kind added later
        // silently leaked into history, shaped like nothing the provider
        // accepts, and the first symptom would be a malformed-request error on
        // resume rather than anything pointing here.
        .filter(|v| v.get("role").is_some())
        .collect()
}

/// Append a token-usage marker as its own thread.jsonl line — NOT part of
/// the conversation `messages` array, so it's never sent back to the LLM.
/// Distinguished from a real turn by `"kind":"usage"` (real turns have a
/// "role" instead); load_thread filters these out before reconstructing
/// history. The frontend reads them directly from the same file to show a
/// context-usage indicator.
pub fn append_usage(paths: &Paths, input_tokens: u64, output_tokens: u64) {
    use serde_json::json;
    append_thread(
        paths,
        &[json!({
            "kind": "usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })],
    );
}

/// Append an end-of-run marker as its own thread.jsonl line, same convention
/// as `append_usage`: no "role", so `load_thread` never feeds it back to the
/// LLM, and the frontend can read it straight from the file.
///
/// Exists because a run that stops early is otherwise indistinguishable from
/// one that finished. `Ending::IterExhausted` is already classified inside
/// the loop and then discarded at the edge, so the caller prints a partial
/// answer exactly like a complete one -- the user sees a confident-looking
/// reply with no hint that the agent simply ran out of turns mid-task, and no
/// way to ask it to continue.
/// Coarse disposition of a run: the single dimension a reader switches on.
///
/// `status` grew ad hoc into success/partial/interrupted/cancelled while
/// `reason` separately carried iter_exhausted/pod_restart -- the same concept
/// split across two fields, with no way at all to say "paused". Consumers had
/// to infer that from a combination, and each inferred it slightly differently.
///
/// Three values, because there are three things a reader does about a run:
/// nothing (done), answer or continue it (paused), investigate (failed).
/// Waiting for a person, hitting the turn cap and being cancelled are all
/// PAUSED -- none is a fault, and colouring them like failures trains people
/// to ignore failures.
///
/// `status` is still written alongside for the older consumers; `outcome` is
/// additive and authoritative.
pub fn outcome_for(status: &str, reason: Option<&str>) -> &'static str {
    match (status, reason) {
        ("success", _) => "done",
        // Ceilings, not faults: the work stands and the user can act on it.
        (_, Some("context_exhausted")) => "paused",
        ("partial", _) | ("blocked", Some("awaiting_input")) | ("cancelled", _) => "paused",
        (_, Some("awaiting_input")) => "paused",
        _ => "failed",
    }
}

pub fn append_ending(paths: &Paths, status: &str, reason: Option<&str>, iter: usize, max_iter: usize) {
    append_ending_detailed(paths, status, reason, None, iter, max_iter)
}

/// As `append_ending`, plus what the provider actually said.
///
/// `reason` is a fixed vocabulary -- "llm_error" covers an exhausted balance,
/// a rejected key, a model the endpoint does not serve and an unreachable
/// endpoint alike, and each is fixed somewhere different. `detail` carries the
/// one sentence that tells them apart, so a reader is told what to DO rather
/// than shown a code. Omitted from the JSON when absent, so a successful
/// ending is unchanged and older readers see exactly the fields they did.
pub fn append_ending_detailed(
    paths: &Paths,
    status: &str,
    reason: Option<&str>,
    detail: Option<&str>,
    iter: usize,
    max_iter: usize,
) {
    use serde_json::json;
    let mut marker = json!({
        "kind": "ending",
        "outcome": outcome_for(status, reason),
        "status": status,
        "reason": reason,
        "iter": iter,
        "max_iter": max_iter,
    });
    if let Some(d) = detail {
        // Clipped: this is a human-readable hint, not a log. A provider that
        // answers an error with a page of HTML must not push the rest of the
        // thread out of the reader's way.
        let clipped: String = d.chars().take(300).collect();
        marker["detail"] = json!(clipped);
    }
    append_thread(paths, &[marker]);
}

pub fn append_thread(paths: &Paths, new_messages: &[Value]) {
    use std::io::Write;
    let path = paths.thread_path();
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[agent] thread write error: {e}");
            return;
        }
    };
    for msg in new_messages {
        if let Ok(line) = serde_json::to_string(msg) {
            let _ = writeln!(file, "{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn append_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        let msgs = vec![json!({"role":"user","content":"a"}), json!({"role":"assistant","content":"b"})];
        append_thread(&paths, &msgs);
        let loaded = load_thread(&paths);
        assert_eq!(loaded, msgs);
    }

    #[test]
    fn append_is_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        // Real turns, not bare objects: load_thread returns conversation
        // history and now filters on the presence of "role", so a role-less
        // object is a marker by definition and is excluded.
        append_thread(&paths, &[json!({"role":"user","n":1})]);
        append_thread(&paths, &[json!({"role":"assistant","n":2})]);
        let loaded = load_thread(&paths);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1]["n"], 2);
    }

    /// Markers must never reach the LLM: load_thread reconstructs message
    /// history, and a role-less line matches no shape either provider accepts.
    #[test]
    fn markers_are_excluded_from_history() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_thread(&paths, &[json!({"role":"user","content":"a"})]);
        append_usage(&paths, 10, 20);
        append_ending(&paths, "partial", Some("iter_exhausted"), 50, 50);
        let loaded = load_thread(&paths);
        assert_eq!(loaded.len(), 1, "only the real turn survives");
        assert_eq!(loaded[0]["role"], "user");
    }

    #[test]
    fn missing_thread_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        assert!(load_thread(&paths).is_empty());
    }

    #[test]
    fn usage_marker_is_excluded_from_load_thread() {
        // The bug this guards against: load_thread's result is fed straight
        // back to the LLM as conversation history on resume. A usage marker
        // has no "role" field, so if it weren't filtered here it would
        // reach the API as a malformed message on the next turn.
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_thread(&paths, &[json!({"role": "user", "content": "hi"})]);
        append_usage(&paths, 100, 20);
        append_thread(&paths, &[json!({"role": "assistant", "content": "hey"})]);
        let loaded = load_thread(&paths);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.iter().all(|m| m.get("kind").is_none()));
    }

    #[test]
    fn a_failed_ending_carries_what_the_provider_said() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_ending_detailed(
            &paths,
            "failure",
            Some("llm_error"),
            Some("HTTP 402: Insufficient Balance"),
            0,
            250,
        );
        let line = std::fs::read_to_string(paths.thread_path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["reason"], "llm_error");
        // The whole point: a reader can tell this from a rejected key.
        assert_eq!(v["detail"], "HTTP 402: Insufficient Balance");
    }

    #[test]
    fn an_ending_without_detail_is_written_exactly_as_before() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_ending(&paths, "success", None, 3, 250);
        let line = std::fs::read_to_string(paths.thread_path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        // Absent, not null: an older reader must see the fields it always saw.
        assert!(v.get("detail").is_none());
        assert_eq!(v["outcome"], "done");
    }

    #[test]
    fn a_pathological_provider_body_cannot_flood_the_thread() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_ending_detailed(&paths, "failure", Some("llm_error"), Some(&"x".repeat(5000)), 0, 9);
        let line = std::fs::read_to_string(paths.thread_path()).unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["detail"].as_str().unwrap().len(), 300);
    }

    #[test]
    fn outcome_collapses_status_and_reason_to_one_dimension() {
        // The three things a reader does about a run: nothing, continue it,
        // investigate it.
        assert_eq!(outcome_for("success", None), "done");
        assert_eq!(outcome_for("partial", Some("iter_exhausted")), "paused");
        assert_eq!(outcome_for("blocked", Some("awaiting_input")), "paused");
        assert_eq!(outcome_for("cancelled", None), "paused");
        assert_eq!(outcome_for("failure", Some("llm_error")), "failed");
        assert_eq!(outcome_for("interrupted", Some("pod_restart")), "failed");
    }

    #[test]
    fn awaiting_input_is_paused_whatever_status_it_arrives_with() {
        // The HITL terminus classifies to Blocked today, but that mapping is
        // policy-owned and injectable. The reason is the durable signal, so a
        // policy change must not silently turn "waiting for you" into a
        // failure the user is never asked to answer.
        for st in ["blocked", "partial", "failure", "weird"] {
            assert_eq!(outcome_for(st, Some("awaiting_input")), "paused", "status {st}");
        }
    }

    #[test]
    fn context_exhaustion_is_paused_not_failed() {
        // Nothing is broken and retrying the same thread cannot help, but a
        // fresh or compacted one can -- that is a pause with an action, not a
        // failure to investigate.
        assert_eq!(outcome_for("partial", Some("context_exhausted")), "paused");
        assert_eq!(outcome_for("failure", Some("context_exhausted")), "paused");
    }

    #[test]
    fn an_unknown_status_reads_as_failed_not_done() {
        // Fail safe: a status this function has never heard of must never be
        // reported as a clean finish.
        assert_eq!(outcome_for("bananas", None), "failed");
        assert_eq!(outcome_for("", None), "failed");
    }

    #[test]
    fn append_usage_writes_a_kind_usage_line() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        append_usage(&paths, 42, 7);
        let contents = std::fs::read_to_string(paths.thread_path()).unwrap();
        let line: Value = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
        assert_eq!(line["kind"], "usage");
        assert_eq!(line["input_tokens"], 42);
        assert_eq!(line["output_tokens"], 7);
    }

    #[test]
    fn child_thread_id_is_stable_and_namespaced() {
        assert_eq!(child_thread_id("main", "abc"), "main__sub_abc");
    }

    #[test]
    fn paths_are_distinct_per_kind() {
        let dir = tempfile::tempdir().unwrap();
        let p = Paths::for_root_under(dir.path().to_path_buf());
        assert_ne!(p.thread_path(), p.registry_path());
        assert!(p.spill_path("x").to_string_lossy().contains("tool_result_x"));
    }

    #[test]
    fn child_paths_nest_under_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let parent = Paths::for_root_under(dir.path().to_path_buf());
        let child = Paths::for_child(&parent, "child-job-1");
        assert!(child.dir().starts_with(parent.dir()));
        assert_ne!(child.dir(), parent.dir());
    }
}
