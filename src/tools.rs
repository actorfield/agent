//! Tool definitions and implementations available to the loop.
//!
//! The tool set is depth-aware: only a top-level loop (depth 0) is offered the
//! `spawn_agent` tool, so a sub-agent (depth 1) structurally cannot delegate.

use base64::Engine;
use serde_json::{json, Value};

use crate::llm;

/// Neutral tool schema. `spawn_agent` is included only at the top level.
pub fn base_tool_defs(depth: usize) -> Value {
    let mut tools = vec![
        json!({
            "name": "run_shell",
            "description": "Run a shell command in Alpine Linux. Returns stdout+stderr.",
            "input_schema": {
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }
        }),
        json!({
            "name": "read_image",
            "description": "Read a JPEG or PNG image and extract information using vision AI. \
Processes the entire image in a single API call — ask for everything needed in one question.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path":     { "type": "string", "description": "Absolute path to the image file." },
                    "question": { "type": "string", "description": "What to extract or look for." }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "read_pdf",
            "description": "Read a PDF file and extract information using AI. \
Submits the entire PDF in a single API call (limit: 100 pages / 32 MB) — ask for everything needed in one comprehensive question.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path":     { "type": "string", "description": "Absolute path to the PDF file." },
                    "question": { "type": "string", "description": "What to extract or look for." }
                },
                "required": ["path"]
            }
        }),
    ];

    if depth == 0 {
        tools.push(spawn_agent_def());
    }

    json!(tools)
}

/// The delegation tool. A top-level loop can hand a self-contained subtask to an
/// isolated sub-agent that runs its own loop and returns a structured result.
fn spawn_agent_def() -> Value {
    json!({
        "name": "spawn_agent",
        "description": "Delegate a self-contained subtask to an isolated sub-agent. It runs its own \
tool loop with fresh context and returns a structured result. It cannot delegate further. Use it for \
focused subtasks whose intermediate steps you do not need to keep in your own context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task":   { "type": "string", "description": "The subtask to accomplish." },
                "checks": { "type": "array", "items": { "type": "string" },
                            "description": "Requirements the result must satisfy." },
                "persistence": { "type": "string", "enum": ["ephemeral", "durable"],
                                 "description": "Whether the sub-agent should persist progress to resume after a restart. Defaults to ephemeral." },
                "max_iter": { "type": "integer", "description": "Iteration budget for the sub-agent." }
            },
            "required": ["task"]
        }
    })
}

/// True iff a tool name is one the loop dispatches directly (not delegation).
pub fn is_builtin(name: &str) -> bool {
    matches!(name, "run_shell" | "read_image" | "read_pdf")
}

// ── Implementations ─────────────────────────────────────────────────────────────

pub fn run_shell(command: &str) -> String {
    match std::process::Command::new("sh").arg("-c").arg(command).output() {
        Ok(out) => {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&out.stderr));
            if s.trim().is_empty() {
                "(no output)".into()
            } else {
                s
            }
        }
        Err(e) => format!("Error: {e}"),
    }
}

pub fn file_to_b64(path: &str) -> Result<(String, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Error: could not read {path}: {e}"))?;
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        _ => "image/jpeg",
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok((mime.to_string(), b64))
}

pub fn read_image(path: &str, question: &str) -> String {
    let (mime, b64) = match file_to_b64(path) {
        Ok(v) => v,
        Err(e) => return e,
    };
    llm::single_call(
        json!({"type":"image","source":{"type":"base64","media_type":mime,"data":b64}}),
        question,
    )
}

pub fn read_pdf(path: &str, question: &str) -> String {
    let (_, b64) = match file_to_b64(path) {
        Ok(v) => v,
        Err(e) => return e,
    };
    llm::single_call(
        json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":b64}}),
        question,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(defs: &Value) -> Vec<String> {
        defs.as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn spawn_agent_only_offered_at_top_level() {
        assert!(tool_names(&base_tool_defs(0)).contains(&"spawn_agent".to_string()));
        assert!(!tool_names(&base_tool_defs(1)).contains(&"spawn_agent".to_string()));
    }

    #[test]
    fn builtin_tools_present_at_every_depth() {
        for depth in [0usize, 1] {
            let names = tool_names(&base_tool_defs(depth));
            for t in ["run_shell", "read_image", "read_pdf"] {
                assert!(names.contains(&t.to_string()), "missing {t} at depth {depth}");
            }
        }
    }

    #[test]
    fn is_builtin_excludes_delegation() {
        assert!(is_builtin("run_shell"));
        assert!(!is_builtin("spawn_agent"));
        assert!(!is_builtin("unknown"));
    }

    #[test]
    fn run_shell_captures_output() {
        assert_eq!(run_shell("printf hi").trim(), "hi");
    }

    #[test]
    fn run_shell_reports_empty_as_no_output() {
        assert_eq!(run_shell("true"), "(no output)");
    }

    #[test]
    fn run_shell_captures_stderr() {
        assert!(run_shell("printf oops 1>&2").contains("oops"));
    }

    #[test]
    fn file_to_b64_detects_mime() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.png");
        std::fs::write(&p, b"\x89PNG").unwrap();
        let (mime, _) = file_to_b64(p.to_str().unwrap()).unwrap();
        assert_eq!(mime, "image/png");
    }

    #[test]
    fn read_image_missing_file_returns_error_without_network() {
        let out = read_image("/no/such/file.png", "what");
        assert!(out.starts_with("Error:"));
    }
}
