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
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
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
        append_thread(&paths, &[json!({"n":1})]);
        append_thread(&paths, &[json!({"n":2})]);
        let loaded = load_thread(&paths);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1]["n"], 2);
    }

    #[test]
    fn missing_thread_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_root_under(dir.path().to_path_buf());
        assert!(load_thread(&paths).is_empty());
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
