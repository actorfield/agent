// AISpec ai-agent actor handler — ReAct loop against Anthropic Messages API
//
// Usage (forked by actor-mesh C runtime per ai_task tuple):
//   payload bytes written to stdin by runtime
//
// Standalone usage:
//   agent 'task string' [model] [max_iter]
//   agent < /tmp/task.txt

use base64::Engine;
use serde_json::{json, Value};
use std::io::Read;

const DEFAULT_MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
const DEFAULT_MAX_ITER: usize = 50;
const DEFAULT_LLM_URL: &str = "http://llm.aispec-system.svc.cluster.local/anthropic/v1/messages";

fn llm_url() -> String {
    std::env::var("LLM_URL").unwrap_or_else(|_| DEFAULT_LLM_URL.to_string())
}

fn llm_model() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

fn llm_request(url: &str) -> ureq::Request {
    let req = ureq::post(url).set("Content-Type", "application/json");
    if let Ok(key) = std::env::var("LLM_API_KEY") {
        req.set("Authorization", &format!("Bearer {key}"))
    } else {
        req
    }
}

fn agent_dir() -> String {
    std::env::var("AGENT_DIR").unwrap_or_else(|_| "/var/actor/.agent".to_string())
}

fn system_prompt() -> String {
    let dir = agent_dir();
    let path = format!("{dir}/system.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| format!("You are a task-executing ai-agent. On your first action, run: cat {dir}/CLAUDE.md"))
        .replace("{AGENT_DIR}", &dir)
}

fn inject_skill(_task: &str) -> String {
    let dir = agent_dir();
    let skills_dir = format!("{dir}/skills");
    let mut out = String::new();
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        let mut paths: Vec<_> = entries.flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "md").unwrap_or(false))
            .collect();
        paths.sort();
        for path in paths {
            if let Ok(content) = std::fs::read_to_string(&path) {
                out.push_str(&format!("\n\n---\n## SKILL — {}\n\n{}", path.display(), content));
            }
        }
    }
    out
}

// ── Tool definitions ──────────────────────────────────────────────────────────

fn tool_defs() -> Value {
    json!([
        {
            "name": "run_shell",
            "description": "Run a shell command in Alpine Linux. Returns stdout+stderr.",
            "input_schema": {
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }
        },
        {
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
        },
        {
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
        }
    ])
}

// ── Tool implementations ──────────────────────────────────────────────────────

fn run_shell(command: &str) -> String {
    match std::process::Command::new("sh").arg("-c").arg(command).output() {
        Ok(out) => {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&out.stderr));
            if s.trim().is_empty() { "(no output)".into() } else { s }
        }
        Err(e) => format!("Error: {e}"),
    }
}

fn file_to_b64(path: &str) -> Result<(String, String), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Error: could not read {path}: {e}"))?;
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        _     => "image/jpeg",
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok((mime.to_string(), b64))
}

fn llm_single_call(model: &str, content_block: Value, question: &str) -> String {
    let req = json!({
        "model": model,
        "max_tokens": 4096,
        "messages": [{ "role": "user", "content": [content_block, {"type":"text","text":question}] }]
    });
    let mut last_err = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        match llm_request(&llm_url()).send_json(&req) {
            Ok(resp) => match resp.into_json::<Value>() {
                Ok(v)  => return v["content"][0]["text"].as_str().unwrap_or("No response").to_string(),
                Err(e) => return format!("Parse error: {e}"),
            },
            Err(ureq::Error::Status(502, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                last_err = format!("HTTP error: 502 {}", &body[..body.len().min(300)]);
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("NetworkError") || msg.contains("Connection") {
                    last_err = format!("HTTP error: {msg}");
                } else {
                    return format!("HTTP error: {msg}");
                }
            }
        }
    }
    last_err
}

fn read_image(model: &str, path: &str, question: &str) -> String {
    let (mime, b64) = match file_to_b64(path) { Ok(v) => v, Err(e) => return e };
    llm_single_call(model,
        json!({"type":"image","source":{"type":"base64","media_type":mime,"data":b64}}),
        question)
}

fn read_pdf(model: &str, path: &str, question: &str) -> String {
    let (_, b64) = match file_to_b64(path) { Ok(v) => v, Err(e) => return e };
    llm_single_call(model,
        json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":b64}}),
        question)
}

// ── LLM ReAct call ────────────────────────────────────────────────────────────

fn llm_call(model: &str, messages: &Value, full_system: &str) -> Result<Value, String> {
    let mut msgs = messages.clone();
    let mut last_err = String::new();

    // Tools are constant across attempts — build once with cache_control
    let tools = {
        let mut t = tool_defs();
        if let Some(arr) = t.as_array_mut() {
            if let Some(last) = arr.last_mut() {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        t
    };

    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        // Stamp cache_control on a fresh snapshot so trimming doesn't corrupt later attempts
        let mut req_msgs = msgs.clone();
        if let Some(arr) = req_msgs.as_array_mut() {
            // Skip tool_result messages (role:user with tool_result content) — same fix as orchestrator
            if let Some(last_human) = arr.iter_mut().rev().find(|m| {
                m["role"] == "user"
                    && m["content"].as_array()
                        .and_then(|a| a.first())
                        .and_then(|b| b.get("type"))
                        .and_then(|t| t.as_str())
                        != Some("tool_result")
            }) {
                if let Some(content) = last_human["content"].as_array_mut() {
                    if let Some(last_block) = content.last_mut() {
                        last_block["cache_control"] = json!({"type": "ephemeral"});
                    }
                }
            }
        }
        let req = json!({
            "model":      model,
            "max_tokens": 16000,
            "system":     [{"type": "text", "text": full_system, "cache_control": {"type": "ephemeral"}}],
            "tools":      tools.clone(),
            "messages":   req_msgs,
        });
        match llm_request(&llm_url()).send_json(&req) {
            Ok(resp) => return resp.into_json::<Value>().map_err(|e| format!("Parse error: {e}")),
            Err(ureq::Error::Status(400, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                if body.contains("prompt is too long") {
                    // Context too large — find the largest tool_result in the last user message
                    // and replace its content with a notice, then retry.
                    eprintln!("[agent] context too large — trimming last tool result and retrying");
                    if let Some(arr) = msgs.as_array_mut() {
                        if let Some(last) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                            if let Some(content) = last["content"].as_array_mut() {
                                // Find the largest tool_result block and shrink it
                                let largest = content.iter_mut()
                                    .filter(|b| b["type"] == "tool_result")
                                    .max_by_key(|b| b["content"].as_str().map(|s| s.len()).unwrap_or(0));
                                if let Some(block) = largest {
                                    block["content"] = json!("[output removed — too large for context. The data was saved to disk; query it there.]");
                                }
                            }
                        }
                    }
                    last_err = "context trimmed after prompt-too-long".into();
                    continue;
                }
                return Err(format!("HTTP 400: {}", &body[..body.len().min(300)]));
            }
            Err(ureq::Error::Status(502, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                last_err = format!("HTTP 502: {}", &body[..body.len().min(300)]);
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                return Err(format!("HTTP {code}: {}", &body[..body.len().min(300)]));
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("NetworkError") || msg.contains("Connection") {
                    last_err = format!("HTTP error: {msg}");
                } else {
                    return Err(format!("HTTP error: {msg}"));
                }
            }
        }
    }
    Err(last_err)
}

// ── Thread persistence ────────────────────────────────────────────────────────

fn thread_path(thread_id: &str) -> String {
    format!("/tmp/agent_thread_{}.jsonl", thread_id)
}

fn load_thread(thread_id: &str) -> Vec<Value> {
    let path = thread_path(thread_id);
    let Ok(contents) = std::fs::read_to_string(&path) else { return vec![] };
    contents.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

fn append_thread(thread_id: &str, new_messages: &[Value]) {
    use std::io::Write;
    let path = thread_path(thread_id);
    let mut file = match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(e) => { eprintln!("[agent] thread write error: {e}"); return; }
    };
    for msg in new_messages {
        if let Ok(line) = serde_json::to_string(msg) {
            let _ = writeln!(file, "{line}");
        }
    }
}

// ── ReAct loop ────────────────────────────────────────────────────────────────

fn run_task(model: &str, task: &str, max_iter: usize, thread_id: Option<&str>) -> Result<String, String> {
    // Build system prompt once — identical string every iteration = stable cache key
    let skill_ctx = inject_skill(task);
    let full_system = format!("{}{}", system_prompt(), skill_ctx);
    // 4th cache breakpoint — task never changes within a run
    let task_msg = json!({
        "role": "user",
        "content": [{"type": "text", "text": task, "cache_control": {"type": "ephemeral"}}]
    });
    // Load prior thread history if thread_id given, then append the new task
    let mut messages = if let Some(tid) = thread_id {
        let mut history = load_thread(tid);
        if !history.is_empty() {
            eprintln!("[agent] resuming thread {} ({} prior messages)", tid, history.len());
        }
        history.push(task_msg);
        json!(history)
    } else {
        json!([task_msg])
    };
    // Track how many messages are already on disk — only append new ones each round-trip
    let mut persisted_len = messages.as_array().map(|a| a.len()).unwrap_or(0) - 1;

    for iter in 0..max_iter {
        eprintln!("[agent] iter {}", iter + 1);
        let resp = match llm_call(model, &messages, &full_system) {
            Ok(v)  => v,
            Err(e) => return Err(format!("llm error on iteration {}: {}", iter + 1, e)),
        };

        let content = match resp["content"].as_array() {
            Some(c) => c.clone(),
            None    => return Err("no content in response".into()),
        };

        let mut text_parts: Vec<String> = vec![];
        let mut tool_calls: Vec<Value>  = vec![];
        for block in &content {
            match block["type"].as_str() {
                Some("text")     => { if let Some(t) = block["text"].as_str() { text_parts.push(t.to_string()); } }
                Some("tool_use") => { tool_calls.push(block.clone()); }
                _ => {}
            }
        }

        messages.as_array_mut().unwrap()
            .push(json!({ "role": "assistant", "content": content }));

        if tool_calls.is_empty() {
            eprintln!("[agent] done after {} iter(s)", iter + 1);
            if let Some(tid) = thread_id {
                let all = messages.as_array().unwrap();
                append_thread(tid, &all[persisted_len..]);
            }
            return Ok(text_parts.join("\n"));
        }

        let mut tool_results: Vec<Value> = vec![];
        for tc in &tool_calls {
            let name  = tc["name"].as_str().unwrap_or("");
            let input = &tc["input"];
            let id    = tc["id"].as_str().unwrap_or("");

            match name {
                "run_shell" => eprintln!("[agent] run_shell: {}", input["command"].as_str().unwrap_or("")),
                "read_image" => eprintln!("[agent] read_image: {}", input["path"].as_str().unwrap_or("")),
                "read_pdf"   => eprintln!("[agent] read_pdf: {}", input["path"].as_str().unwrap_or("")),
                _ => eprintln!("[agent] tool: {name}"),
            }

            let raw = match name {
                "run_shell"  => run_shell(input["command"].as_str().unwrap_or("")),
                "read_image" => read_image(
                    model,
                    input["path"].as_str().unwrap_or(""),
                    input["question"].as_str().unwrap_or("Extract all text and data from this image verbatim."),
                ),
                "read_pdf"    => read_pdf(
                    model,
                    input["path"].as_str().unwrap_or(""),
                    input["question"].as_str().unwrap_or("Extract all text and data from this PDF verbatim."),
                ),
                _ => format!("unknown tool: {name}"),
            };

            // If output exceeds threshold, write full content to disk and send agent a pointer.
            const MAX_INLINE_CHARS: usize = 16_000;
            let result = if raw.len() > MAX_INLINE_CHARS {
                let out_path = format!("/tmp/tool_result_{}.txt", id);
                match std::fs::write(&out_path, &raw) {
                    Ok(_) => format!(
                        "Output too large ({} chars) — full content saved to {out_path}\n\
                         Query it with grep/head/sed/awk rather than reading the whole file.\n\
                         Example: grep -n 'keyword' {out_path} | head -30",
                        raw.len()
                    ),
                    Err(e) => format!("Output too large ({} chars) and could not save to disk: {e}", raw.len()),
                }
            } else {
                raw
            };

            tool_results.push(json!({
                "type":        "tool_result",
                "tool_use_id": id,
                "content":     result,
            }));
        }

        messages.as_array_mut().unwrap()
            .push(json!({ "role": "user", "content": tool_results }));

        if let Some(tid) = thread_id {
            let all = messages.as_array().unwrap();
            append_thread(tid, &all[persisted_len..]);
            persisted_len = all.len();
        }
    }

    Err(format!("Max iterations ({max_iter}) reached"))
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --thread <id> from anywhere in args, filter by position not value
    let thread_flag_pos = args.iter().position(|a| a == "--thread");
    let thread_id = thread_flag_pos.and_then(|i| args.get(i + 1)).cloned();
    let skip: std::collections::HashSet<usize> = thread_flag_pos
        .map(|i| [i, i + 1].into_iter().collect())
        .unwrap_or_default();
    let filtered: Vec<&String> = args.iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, a)| a)
        .collect();

    // CLI mode:   agent [--thread <id>] 'task' [model] [max_iter]
    // Stdin mode: agent [--thread <id>] < task.txt
    let (task, model, max_iter) = if filtered.len() > 1 {
        let task     = filtered[1].clone();
        let model    = filtered.get(2).map(|s| s.to_string()).unwrap_or_else(|| llm_model());
        let max_iter = filtered.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MAX_ITER);
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

    match run_task(&model, &task, max_iter, thread_id.as_deref()) {
        Ok(result) => {
            // Emit result topic override then result — actor-mesh runtime uses first line
            // as the publish topic if it matches `topic_name\n` pattern.
            println!("ai_result");
            println!("{result}");
        }
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}
