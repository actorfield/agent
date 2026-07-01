//! Work handed to a sub-agent and the structured result it returns.
//!
//! A `Job` is a self-contained unit of work the top-level loop can delegate to a
//! nested loop running in its own isolated context. The nested loop reports back a
//! typed `JobResult` instead of free-form prose, so the caller can react to the
//! outcome programmatically.

use serde::{Deserialize, Serialize};

/// Generate a fresh, time-ordered job id (UUIDv7).
pub fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// How long a sub-agent's own state should live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Persistence {
    /// No own persistence; re-run from scratch if interrupted.
    Ephemeral,
    /// Persist progress so it can resume after a restart.
    Durable,
}

impl Persistence {
    /// Parse a tool argument value; defaults to `Ephemeral`.
    pub fn parse(s: Option<&str>) -> Persistence {
        match s {
            Some("durable") => Persistence::Durable,
            _ => Persistence::Ephemeral,
        }
    }
}

/// One high-level step describing intent (WHAT, not HOW).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_hint: Option<String>,
}

/// A unit of delegated work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub task: String,
    #[serde(default)]
    pub steps: Vec<Step>,
    /// Requirements the result must satisfy.
    #[serde(default)]
    pub checks: Vec<String>,
    #[serde(default = "default_persistence")]
    pub persistence: Persistence,
    pub max_iter: usize,
}

fn default_persistence() -> Persistence {
    Persistence::Ephemeral
}

impl Job {
    /// Build a job with a fresh id.
    pub fn new(task: String, checks: Vec<String>, persistence: Persistence, max_iter: usize) -> Job {
        Job {
            id: new_id(),
            task,
            steps: Vec::new(),
            checks,
            persistence,
            max_iter,
        }
    }
}

/// Terminal disposition of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Success,
    Partial,
    Failure,
    Blocked,
}

/// Why a job did not fully succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    PreconditionUnmet,
    ToolUnavailable,
    CheckFailed,
    BudgetExceeded,
    RetrievalFailed,
    AmbiguousRequest,
}

/// Structured outcome returned to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub id: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureKind>,
    pub steps_taken: usize,
    /// Unsatisfied requirements; empty on success.
    #[serde(default)]
    pub issues: Vec<String>,
}

impl JobResult {
    pub fn success(id: &str, output: String, steps_taken: usize) -> JobResult {
        JobResult {
            id: id.to_string(),
            status: Status::Success,
            output: Some(output),
            failure: None,
            steps_taken,
            issues: Vec::new(),
        }
    }

    pub fn partial(id: &str, output: String, failure: FailureKind, steps_taken: usize) -> JobResult {
        JobResult {
            id: id.to_string(),
            status: Status::Partial,
            output: Some(output),
            failure: Some(failure),
            steps_taken,
            issues: Vec::new(),
        }
    }

    pub fn failure(id: &str, failure: FailureKind, steps_taken: usize) -> JobResult {
        JobResult {
            id: id.to_string(),
            status: Status::Failure,
            output: None,
            failure: Some(failure),
            steps_taken,
            issues: Vec::new(),
        }
    }

    pub fn blocked(id: &str, failure: FailureKind, steps_taken: usize) -> JobResult {
        JobResult {
            id: id.to_string(),
            status: Status::Blocked,
            output: None,
            failure: Some(failure),
            steps_taken,
            issues: Vec::new(),
        }
    }

    /// Serialize to the JSON string a tool result carries back to the caller.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"id\":\"{}\",\"status\":\"failure\",\"steps_taken\":0}}",
                self.id
            )
        })
    }

    /// True when the result honors the invariants a well-formed outcome must hold.
    pub fn is_consistent(&self) -> bool {
        match self.status {
            Status::Success => {
                self.failure.is_none() && self.issues.is_empty() && self.output.is_some()
            }
            Status::Partial => self.output.is_some() && self.failure.is_some(),
            Status::Failure => self.failure.is_some() && self.output.is_none(),
            Status::Blocked => self.failure.is_some() && self.output.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_time_ordered() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        // UUIDv7 is time-ordered: a issued before b sorts before b.
        assert!(a < b);
    }

    #[test]
    fn persistence_parses_with_ephemeral_default() {
        assert_eq!(Persistence::parse(Some("durable")), Persistence::Durable);
        assert_eq!(Persistence::parse(Some("ephemeral")), Persistence::Ephemeral);
        assert_eq!(Persistence::parse(None), Persistence::Ephemeral);
        assert_eq!(Persistence::parse(Some("garbage")), Persistence::Ephemeral);
    }

    #[test]
    fn job_round_trips_through_json() {
        let job = Job::new(
            "do the thing".into(),
            vec!["must be idempotent".into()],
            Persistence::Durable,
            25,
        );
        let s = serde_json::to_string(&job).unwrap();
        let back: Job = serde_json::from_str(&s).unwrap();
        assert_eq!(back.task, job.task);
        assert_eq!(back.checks, job.checks);
        assert_eq!(back.persistence, Persistence::Durable);
        assert_eq!(back.max_iter, 25);
    }

    #[test]
    fn result_constructors_are_consistent() {
        assert!(JobResult::success("i", "out".into(), 3).is_consistent());
        assert!(JobResult::partial("i", "half".into(), FailureKind::BudgetExceeded, 5).is_consistent());
        assert!(JobResult::failure("i", FailureKind::ToolUnavailable, 1).is_consistent());
        assert!(JobResult::blocked("i", FailureKind::PreconditionUnmet, 0).is_consistent());
    }

    #[test]
    fn inconsistent_results_are_flagged() {
        // Success must carry output and no failure.
        let bad = JobResult {
            id: "i".into(),
            status: Status::Success,
            output: None,
            failure: Some(FailureKind::CheckFailed),
            steps_taken: 1,
            issues: vec!["nope".into()],
        };
        assert!(!bad.is_consistent());
    }

    #[test]
    fn result_json_uses_lowercase_status() {
        let r = JobResult::success("i", "out".into(), 1);
        let s = r.to_json();
        assert!(s.contains("\"status\":\"success\""));
    }
}
