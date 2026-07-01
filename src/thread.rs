//! Conversation persistence. A run may be resumed by id: prior messages are
//! reloaded, new ones appended. Paths are injected so tests use temp dirs.

use std::path::PathBuf;

use serde_json::Value;

/// Filesystem locations for persisted state. Defaults to the system temp dir;
/// tests override with a temp dir.
#[derive(Clone)]
pub struct Paths {
    pub thread_dir: PathBuf,
    pub registry_dir: PathBuf,
    pub spill_dir: PathBuf,
}

impl Paths {
    /// Production default: everything under the system temp dir.
    pub fn system() -> Paths {
        let tmp = std::env::temp_dir();
        Paths {
            thread_dir: tmp.clone(),
            registry_dir: tmp.clone(),
            spill_dir: tmp,
        }
    }

    /// All state under a single base dir (used by tests).
    pub fn under(base: PathBuf) -> Paths {
        Paths {
            thread_dir: base.clone(),
            registry_dir: base.clone(),
            spill_dir: base,
        }
    }

    pub fn thread_path(&self, thread_id: &str) -> PathBuf {
        self.thread_dir.join(format!("agent_thread_{thread_id}.jsonl"))
    }

    pub fn registry_path(&self, thread_id: &str) -> PathBuf {
        self.registry_dir
            .join(format!("agent_registry_{thread_id}.jsonl"))
    }

    pub fn spill_path(&self, id: &str) -> PathBuf {
        self.spill_dir.join(format!("tool_result_{id}.txt"))
    }
}

/// Deterministic child-run id for a delegated job, stable across restarts.
pub fn child_thread_id(parent: &str, job_id: &str) -> String {
    format!("{parent}__sub_{job_id}")
}

pub fn load_thread(paths: &Paths, thread_id: &str) -> Vec<Value> {
    let path = paths.thread_path(thread_id);
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    contents
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub fn append_thread(paths: &Paths, thread_id: &str, new_messages: &[Value]) {
    use std::io::Write;
    let path = paths.thread_path(thread_id);
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
        let paths = Paths::under(dir.path().to_path_buf());
        let msgs = vec![json!({"role":"user","content":"a"}), json!({"role":"assistant","content":"b"})];
        append_thread(&paths, "t1", &msgs);
        let loaded = load_thread(&paths, "t1");
        assert_eq!(loaded, msgs);
    }

    #[test]
    fn append_is_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        append_thread(&paths, "t", &[json!({"n":1})]);
        append_thread(&paths, "t", &[json!({"n":2})]);
        let loaded = load_thread(&paths, "t");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1]["n"], 2);
    }

    #[test]
    fn missing_thread_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        assert!(load_thread(&paths, "nope").is_empty());
    }

    #[test]
    fn child_thread_id_is_stable_and_namespaced() {
        assert_eq!(child_thread_id("main", "abc"), "main__sub_abc");
    }

    #[test]
    fn paths_are_distinct_per_kind() {
        let dir = tempfile::tempdir().unwrap();
        let p = Paths::under(dir.path().to_path_buf());
        assert!(p.thread_path("x").to_string_lossy().contains("agent_thread_x"));
        assert!(p.registry_path("x").to_string_lossy().contains("agent_registry_x"));
        assert!(p.spill_path("x").to_string_lossy().contains("tool_result_x"));
    }
}
