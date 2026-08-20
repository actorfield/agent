//! Injected control for the loop. The loop itself contains no role-specific
//! branching: it consults a `Policy` bundle of small pure functions to decide
//! whether to keep going, whether it has finished, how to label the outcome, and
//! whether delegation is permitted. A top-level run and a sub-agent run use the
//! same loop with different bundles.

use serde::{Deserialize, Serialize};
use crate::job::{FailureKind, Status};
use crate::provider::ToolCall;
use serde_json::Value;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    /// Cancelled by external signal (cancel file).
    Cancelled,
    /// The conversation outgrew the model's context window.
    ///
    /// Distinct from Failed: nothing is broken and nothing will be fixed by
    /// retrying the same thread -- the run needs a fresh thread or a compacted
    /// history. Reported as a generic failure it reads as a bug, and the one
    /// action that actually helps is not suggested.
    ContextExhausted,
    /// The model asked the user a question and is waiting for the answer.
    ///
    /// A terminus, not a failure: the run stopped because it needs a human,
    /// which is the correct behaviour when a task is genuinely ambiguous. Kept
    /// distinct from Blocked (a tool or precondition is missing, which the user
    /// cannot resolve by replying) and from Stopped (the model believes it is
    /// finished) so a caller can tell "answer me" from "I'm done" -- they look
    /// identical in the transcript otherwise, and the run that is waiting is
    /// the one that must not be reported as complete.
    AwaitingInput,
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
    /// Optional replacement for the builtin tool definitions. When set, the
    /// loop offers this array instead of the builtins; `spawn_agent` is still
    /// appended by the loop itself at depth 0 (delegation stays structural,
    /// not something an override can grant or revoke).
    pub tool_defs: Option<fn() -> Value>,
    /// Optional dispatcher consulted BEFORE the builtin match for each tool
    /// call. `Some(result)` means the call was handled and `result` is the
    /// tool output; `None` falls through to the builtin implementations.
    /// `spawn_agent` is never routed here — delegation is loop infrastructure.
    pub dispatch: Option<fn(&ToolCall) -> Option<String>>,
    /// Optional redaction applied to every tool result BEFORE it enters the
    /// conversation, and therefore before it is sent, persisted to
    /// `thread.jsonl`, or synced anywhere.
    ///
    /// Takes the raw output string, deliberately -- NOT the wrapped message.
    /// By the time a result has been through `wrap_tool_results` it carries
    /// provider-specific structure (Anthropic nests the text under a
    /// `tool_result` block, OpenAI does not), so a redactor operating there
    /// would need to know both shapes, and one recursing the whole value would
    /// rewrite `tool_use_id` and break the conversation. Here it is just text.
    ///
    /// This is a door, not a convention: tool results are the only route by
    /// which content the model has never seen reaches the message array, so a
    /// hook here cannot be forgotten by a tool added later. A dispatcher arm
    /// that redacts is a promise each new arm has to remember to keep; this is
    /// the loop keeping it on their behalf.
    ///
    /// Returning `Err` withholds the result rather than passing it through —
    /// the caller cannot distinguish "nothing to redact" from "redaction
    /// failed", so failing open here would be indistinguishable from working.
    pub redact: Option<fn(&mut String) -> Result<(), String>>,
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
        Ending::Cancelled => (Status::Partial, Some(FailureKind::BudgetExceeded)),
        // Blocked, not Partial: the work is not incomplete through any fault of
        // the run, it is suspended pending an answer. AmbiguousRequest is the
        // existing kind that means exactly "needs the user to disambiguate".
        Ending::AwaitingInput => (Status::Blocked, Some(FailureKind::AmbiguousRequest)),
        // Partial, like the other ceilings: the work done so far stands, and
        // the run stopped because it hit a limit rather than because anything
        // went wrong.
        Ending::ContextExhausted => (Status::Partial, Some(FailureKind::BudgetExceeded)),
    }
}

fn no_issues(_p: &Progress) -> Vec<String> {
    // Structural runs report no unmet requirements; a richer check can be injected.
    Vec::new()
}

/// Bundle for a top-level run: may delegate, builtin tools, builtin dispatch.
pub fn root_policy() -> Policy {
    Policy {
        may_delegate: true,
        should_continue: default_should_continue,
        is_done: default_is_done,
        classify: default_classify,
        check: no_issues,
        tool_defs: None,
        dispatch: None,
        redact: None,
    }
}

/// Bundle for a sub-agent run: identical, but may not delegate.
pub fn sub_policy() -> Policy {
    Policy {
        may_delegate: false,
        ..root_policy()
    }
}

/// Sub-agent bundle derived from a specific parent: may not delegate, and
/// inherits the parent's tool overrides — a custom tool set or dispatcher
/// applies to the whole run tree, not just the top level.
pub fn sub_policy_of(parent: &Policy) -> Policy {
    Policy {
        tool_defs: parent.tool_defs,
        dispatch: parent.dispatch,
        redact: parent.redact,
        ..sub_policy()
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
    fn sub_policy_of_inherits_tool_overrides() {
        fn defs() -> Value {
            serde_json::json!([])
        }
        fn disp(_tc: &ToolCall) -> Option<String> {
            None
        }
        let parent = Policy {
            tool_defs: Some(defs),
            dispatch: Some(disp),
            ..root_policy()
        };
        let child = sub_policy_of(&parent);
        assert!(!child.may_delegate);
        assert!(child.tool_defs.is_some(), "tool_defs inherited");
        assert!(child.dispatch.is_some(), "dispatch inherited");
        // A hook-free parent produces a hook-free child.
        let plain = sub_policy_of(&root_policy());
        assert!(plain.tool_defs.is_none() && plain.dispatch.is_none());
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
