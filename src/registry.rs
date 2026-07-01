//! Durable job registry. When a run delegates a durable sub-agent, the job is
//! recorded here before it starts and its result recorded when it finishes. On a
//! restart this lets the top-level run tell which delegated jobs were still in
//! flight, resume them, and fold their results back into the conversation.
//!
//! This is a separate persistence layer from the conversation itself, so the
//! conversation on disk is always well formed (never a tool call without its
//! result). The fold/idempotency logic here is pure and unit tested; the
//! orchestration that re-runs in-flight jobs lives in the loop.

use std::collections::HashMap;
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::job::{Job, JobResult};
use crate::thread::Paths;

/// One append-only registry line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Record {
    Issued { job: Job },
    Result { id: String, result: JobResult },
}

pub fn append_issued(paths: &Paths, parent_thread_id: &str, job: &Job) {
    append(paths, parent_thread_id, &Record::Issued { job: job.clone() });
}

pub fn append_result(paths: &Paths, parent_thread_id: &str, result: &JobResult) {
    append(
        paths,
        parent_thread_id,
        &Record::Result {
            id: result.id.clone(),
            result: result.clone(),
        },
    );
}

fn append(paths: &Paths, parent_thread_id: &str, record: &Record) {
    let path = paths.registry_path(parent_thread_id);
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[agent] registry write error: {e}");
            return;
        }
    };
    if let Ok(line) = serde_json::to_string(record) {
        let _ = writeln!(file, "{line}");
    }
}

pub fn load(paths: &Paths, parent_thread_id: &str) -> Vec<Record> {
    let path = paths.registry_path(parent_thread_id);
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    contents
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Every job that was issued, in issue order (deduplicated by id).
pub fn issued_jobs(records: &[Record]) -> Vec<Job> {
    let mut seen = std::collections::HashSet::new();
    let mut out = vec![];
    for r in records {
        if let Record::Issued { job } = r {
            if seen.insert(job.id.clone()) {
                out.push(job.clone());
            }
        }
    }
    out
}

/// Map of job id -> its recorded result (last write wins).
pub fn results_map(records: &[Record]) -> HashMap<String, JobResult> {
    let mut m = HashMap::new();
    for r in records {
        if let Record::Result { id, result } = r {
            m.insert(id.clone(), result.clone());
        }
    }
    m
}

/// Issued jobs that have no result yet — the ones to resume.
pub fn in_flight(records: &[Record]) -> Vec<Job> {
    let done = results_map(records);
    issued_jobs(records)
        .into_iter()
        .filter(|j| !done.contains_key(&j.id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{FailureKind, Persistence};

    fn job(task: &str) -> Job {
        Job::new(task.into(), vec![], Persistence::Durable, 10)
    }

    #[test]
    fn append_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let j = job("a");
        append_issued(&paths, "main", &j);
        append_result(&paths, "main", &JobResult::success(&j.id, "ok".into(), 2));
        let recs = load(&paths, "main");
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn in_flight_excludes_completed() {
        let a = job("a");
        let b = job("b");
        let records = vec![
            Record::Issued { job: a.clone() },
            Record::Issued { job: b.clone() },
            Record::Result {
                id: a.id.clone(),
                result: JobResult::success(&a.id, "done".into(), 1),
            },
        ];
        let pending = in_flight(&records);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, b.id);
    }

    #[test]
    fn issued_jobs_dedupe_and_preserve_order() {
        let a = job("a");
        let records = vec![
            Record::Issued { job: a.clone() },
            Record::Issued { job: a.clone() }, // duplicate issue line
        ];
        assert_eq!(issued_jobs(&records).len(), 1);
    }

    #[test]
    fn results_map_last_write_wins() {
        let a = job("a");
        let records = vec![
            Record::Result {
                id: a.id.clone(),
                result: JobResult::partial(&a.id, "half".into(), FailureKind::BudgetExceeded, 1),
            },
            Record::Result {
                id: a.id.clone(),
                result: JobResult::success(&a.id, "full".into(), 3),
            },
        ];
        let m = results_map(&records);
        assert_eq!(m[&a.id].status, crate::job::Status::Success);
    }

    #[test]
    fn missing_registry_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        assert!(load(&paths, "none").is_empty());
    }
}
