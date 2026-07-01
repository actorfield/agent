//! Injected control for the loop. The loop itself contains no role-specific
//! branching: it consults a `Policy` bundle of small pure functions to decide
//! whether to keep going, whether it has finished, how to label the outcome, and
//! whether delegation is permitted. A top-level run and a sub-agent run use the
//! same loop with different bundles.

use crate::job::{FailureKind, Status};

/// A read-only view of the loop's progress, passed to the control functions.
pub struct Progress<'a> {
    pub iter: usize,
    pub max_iter: usize,
    pub budget_remaining: usize,
    pub steps_taken: usize,
    pub last_text: &'a str,
    pub checks: &'a [String],
}

/// How a run concluded — the loop reports one of these and the policy turns it
/// into a status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// Model produced no tool calls: natural completion.
    Stopped,
    /// Reached the per-run iteration cap.
    IterExhausted,
    /// Reached this run's tool-call budget cap.
    BudgetExhausted,
    /// Transport or model error.
    Failed,
    /// Could not proceed (tool/precondition unavailable).
    Blocked,
}

/// The injectable control bundle.
pub struct Policy {
    /// Whether this run may hand work to a sub-agent.
    pub may_delegate: bool,
    /// Keep looping? (iteration + budget guard)
    pub should_continue: fn(&Progress) -> bool,
    /// Has the run reached a natural end this turn?
    pub is_done: fn(had_tool_calls: bool) -> bool,
    /// Label a conclusion.
    pub classify: fn(Ending, &Progress) -> (Status, Option<FailureKind>),
    /// Requirements not satisfied by the run (empty when all hold).
    pub check: fn(&Progress) -> Vec<String>,
}

fn default_should_continue(p: &Progress) -> bool {
    p.iter < p.max_iter && p.budget_remaining > 0
}

fn default_is_done(had_tool_calls: bool) -> bool {
    !had_tool_calls
}

fn default_classify(end: Ending, _p: &Progress) -> (Status, Option<FailureKind>) {
    match end {
        Ending::Stopped => (Status::Success, None),
        Ending::IterExhausted => (Status::Partial, Some(FailureKind::BudgetExceeded)),
        Ending::BudgetExhausted => (Status::Partial, Some(FailureKind::BudgetExceeded)),
        Ending::Failed => (Status::Failure, Some(FailureKind::RetrievalFailed)),
        Ending::Blocked => (Status::Blocked, Some(FailureKind::ToolUnavailable)),
    }
}

fn no_issues(_p: &Progress) -> Vec<String> {
    // Structural runs report no unmet requirements; a richer check can be injected.
    Vec::new()
}

/// Bundle for a top-level run: may delegate.
pub fn root_policy() -> Policy {
    Policy {
        may_delegate: true,
        should_continue: default_should_continue,
        is_done: default_is_done,
        classify: default_classify,
        check: no_issues,
    }
}

/// Bundle for a sub-agent run: identical, but may not delegate.
pub fn sub_policy() -> Policy {
    Policy {
        may_delegate: false,
        ..root_policy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(iter: usize, max_iter: usize, budget: usize) -> Progress<'static> {
        Progress {
            iter,
            max_iter,
            budget_remaining: budget,
            steps_taken: iter,
            last_text: "",
            checks: &[],
        }
    }

    #[test]
    fn should_continue_respects_iter_and_budget() {
        let p = root_policy();
        assert!((p.should_continue)(&progress(0, 5, 10)));
        assert!(!(p.should_continue)(&progress(5, 5, 10))); // iter cap
        assert!(!(p.should_continue)(&progress(1, 5, 0))); // budget cap
    }

    #[test]
    fn is_done_when_no_tool_calls() {
        let p = root_policy();
        assert!((p.is_done)(false));
        assert!(!(p.is_done)(true));
    }

    #[test]
    fn classify_covers_every_ending() {
        let p = root_policy();
        let pr = progress(1, 5, 5);
        assert_eq!((p.classify)(Ending::Stopped, &pr), (Status::Success, None));
        assert_eq!(
            (p.classify)(Ending::IterExhausted, &pr),
            (Status::Partial, Some(FailureKind::BudgetExceeded))
        );
        assert_eq!(
            (p.classify)(Ending::BudgetExhausted, &pr),
            (Status::Partial, Some(FailureKind::BudgetExceeded))
        );
        assert_eq!(
            (p.classify)(Ending::Failed, &pr),
            (Status::Failure, Some(FailureKind::RetrievalFailed))
        );
        assert_eq!(
            (p.classify)(Ending::Blocked, &pr),
            (Status::Blocked, Some(FailureKind::ToolUnavailable))
        );
    }

    #[test]
    fn check_reports_no_issues_by_default() {
        let p = root_policy();
        assert!((p.check)(&progress(1, 5, 5)).is_empty());
    }

    #[test]
    fn root_and_sub_differ_only_in_delegation() {
        let root = root_policy();
        let sub = sub_policy();
        assert!(root.may_delegate);
        assert!(!sub.may_delegate);
        // Same control behaviour otherwise.
        let pr = progress(2, 5, 3);
        assert_eq!((root.should_continue)(&pr), (sub.should_continue)(&pr));
        assert_eq!(
            (root.classify)(Ending::Stopped, &pr),
            (sub.classify)(Ending::Stopped, &pr)
        );
    }
}
