// AISpec ai-agent actor handler — ReAct loop against Anthropic or OpenAI-compatible APIs
//
// Usage (forked by actor-mesh C runtime per ai_task tuple):
//   payload bytes written to stdin by runtime
//
// Standalone usage:
//   agent 'task string' [model] [max_iter]
//   agent < /tmp/task.txt
//
// Provider is inferred from LLM_URL:
//   contains /v1/chat/completions  → OpenAI-compatible
//   anything else                  → Anthropic (default)

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

// ── Provider abstraction ──────────────────────────────────────────────────────

struct ParsedResponse {
    text_parts: Vec<String>,
    tool_calls: Vec<ToolCall>,
    /// Raw assistant message to push into history (provider-shaped)
    assistant_msg: Value,
}

struct ToolCall {
    id:    String,
    name:  String,
    input: Value,
}

struct ToolResult {
    tool_use_id: String,
    content:     String,
}

trait Provider {
    fn tool_defs(&self) -> Value;
    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value;
    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String>;
    /// Returns one or more messages to append to history for the given tool results.
    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value>;
    /// Provider-specific context-trim: shrink the largest tool result in history.
    fn trim_last_tool_result(&self, messages: &mut Value);
}

// ── Anthropic provider ────────────────────────────────────────────────────────

struct Anthropic;

impl Provider for Anthropic {
    fn tool_defs(&self) -> Value {
        let mut tools = base_tool_defs();
        // cache_control on the last tool — stable cache breakpoint
        if let Some(arr) = tools.as_array_mut() {
            if let Some(last) = arr.last_mut() {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        tools
    }

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value {
        // Stamp cache_control on the last content block of the last message so the
        // growing conversation tail (assistant turns + tool_results) is cached
        // incrementally: each turn the previous tail becomes a cache hit and the
        // breakpoint moves forward. The task message keeps its own permanent
        // breakpoint as a stable anchor; together with the system and tools
        // breakpoints that holds us at Anthropic's 4-breakpoint maximum. Stamping
        // happens on this per-request clone only, so it never accumulates on the
        // persisted `messages` array.
        let mut req_msgs = messages.clone();
        if let Some(last_msg) = req_msgs.as_array_mut().and_then(|a| a.last_mut()) {
            if let Some(last_block) = last_msg["content"]
                .as_array_mut()
                .and_then(|c| c.last_mut())
            {
                last_block["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        json!({
            "model":      model,
            "max_tokens": 16000,
            "system":     [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools":      tools,
            "messages":   req_msgs,
        })
    }

    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String> {
        let content = resp["content"].as_array()
            .ok_or("no content in response")?
            .clone();

        let mut text_parts = vec![];
        let mut tool_calls = vec![];
        for block in &content {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(t) = block["text"].as_str() { text_parts.push(t.to_string()); }
                }
                Some("tool_use") => {
                    tool_calls.push(ToolCall {
                        id:    block["id"].as_str().unwrap_or("").to_string(),
                        name:  block["name"].as_str().unwrap_or("").to_string(),
                        input: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }

        Ok(ParsedResponse {
            text_parts,
            tool_calls,
            assistant_msg: json!({ "role": "assistant", "content": content }),
        })
    }

    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value> {
        let blocks: Vec<Value> = results.into_iter().map(|r| json!({
            "type":        "tool_result",
            "tool_use_id": r.tool_use_id,
            "content":     r.content,
        })).collect();
        vec![json!({ "role": "user", "content": blocks })]
    }

    fn trim_last_tool_result(&self, messages: &mut Value) {
        if let Some(arr) = messages.as_array_mut() {
            if let Some(last) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                if let Some(content) = last["content"].as_array_mut() {
                    let largest = content.iter_mut()
                        .filter(|b| b["type"] == "tool_result")
                        .max_by_key(|b| b["content"].as_str().map(|s| s.len()).unwrap_or(0));
                    if let Some(block) = largest {
                        block["content"] = json!("[output removed — too large for context. The data was saved to disk; query it there.]");
                    }
                }
            }
        }
    }
}

// ── OpenAI provider ───────────────────────────────────────────────────────────

struct OpenAI;

impl Provider for OpenAI {
    fn tool_defs(&self) -> Value {
        let base = base_tool_defs();
        let arr = base.as_array().unwrap();
        let tools: Vec<Value> = arr.iter().map(|t| json!({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            }
        })).collect();
        json!(tools)
    }

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value {
        // Prepend system as first message
        let mut all_msgs = vec![json!({"role": "system", "content": system})];
        if let Some(arr) = messages.as_array() {
            all_msgs.extend(arr.iter().cloned());
        }
        json!({
            "model":      model,
            "max_tokens": 16000,
            "tools":      tools,
            "messages":   all_msgs,
        })
    }

    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String> {
        let msg = resp["choices"][0]["message"].clone();
        if msg.is_null() {
            return Err("no choices[0].message in response".into());
        }

        let mut text_parts = vec![];
        if let Some(t) = msg["content"].as_str() {
            if !t.is_empty() { text_parts.push(t.to_string()); }
        }

        let mut tool_calls = vec![];
        if let Some(tcs) = msg["tool_calls"].as_array() {
            for tc in tcs {
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                tool_calls.push(ToolCall {
                    id:    tc["id"].as_str().unwrap_or("").to_string(),
                    name:  tc["function"]["name"].as_str().unwrap_or("").to_string(),
                    input,
                });
            }
        }

        // Reconstruct assistant message in OpenAI history format
        let assistant_msg = json!({ "role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"] });

        Ok(ParsedResponse { text_parts, tool_calls, assistant_msg })
    }

    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value> {
        // Each tool result is its own message in OpenAI format
        results.into_iter().map(|r| json!({
            "role":         "tool",
            "tool_call_id": r.tool_use_id,
            "content":      r.content,
        })).collect()
    }

    fn trim_last_tool_result(&self, messages: &mut Value) {
        if let Some(arr) = messages.as_array_mut() {
            // Find the largest role:tool message and shrink its content
            let largest = arr.iter_mut()
                .filter(|m| m["role"] == "tool")
                .max_by_key(|m| m["content"].as_str().map(|s| s.len()).unwrap_or(0));
            if let Some(msg) = largest {
                msg["content"] = json!("[output removed — too large for context. The data was saved to disk; query it there.]");
            }
        }
    }
}

fn detect_provider(url: &str) -> Box<dyn Provider> {
    if url.contains("anthropic") {
        Box::new(Anthropic)
    } else {
        Box::new(OpenAI)
    }
}

// ── Shared tool definitions (provider-neutral) ────────────────────────────────

fn base_tool_defs() -> Value {
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

fn llm_single_call(content_block: Value, question: &str) -> String {
    let req = json!({
        "model": llm_model(),
        "max_tokens": 4096,
        "messages": [{ "role": "user", "content": [content_block, {"type":"text","text":question}] }]
    });
    let url = llm_url();
    let mut last_err = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        match llm_request(&url).send_json(&req) {
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

fn read_image(path: &str, question: &str) -> String {
    let (mime, b64) = match file_to_b64(path) { Ok(v) => v, Err(e) => return e };
    llm_single_call(
        json!({"type":"image","source":{"type":"base64","media_type":mime,"data":b64}}),
        question)
}

fn read_pdf(path: &str, question: &str) -> String {
    let (_, b64) = match file_to_b64(path) { Ok(v) => v, Err(e) => return e };
    llm_single_call(
        json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":b64}}),
        question)
}

// ── LLM ReAct call ────────────────────────────────────────────────────────────

fn llm_call(provider: &dyn Provider, model: &str, messages: &mut Value, full_system: &str) -> Result<Value, String> {
    let tools = provider.tool_defs();
    let url = llm_url();
    let mut last_err = String::new();

    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let req = provider.build_request(model, full_system, &tools, messages);
        match llm_request(&url).send_json(&req) {
            Ok(resp) => return resp.into_json::<Value>().map_err(|e| format!("Parse error: {e}")),
            Err(ureq::Error::Status(400, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                if body.contains("prompt is too long") {
                    eprintln!("[agent] context too large — trimming last tool result and retrying");
                    provider.trim_last_tool_result(messages);
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
    let url = llm_url();
    let provider = detect_provider(&url);

    let skill_ctx = inject_skill(task);
    let full_system = format!("{}{}", system_prompt(), skill_ctx);

    let run_ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let task_with_ts = format!("[run started: {run_ts}]\n{task}");
    let task_msg = json!({
        "role": "user",
        "content": [{"type": "text", "text": task_with_ts, "cache_control": {"type": "ephemeral"}}]
    });

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
    let mut persisted_len = messages.as_array().map(|a| a.len()).unwrap_or(0) - 1;

    for iter in 0..max_iter {
        eprintln!("[agent] iter {}", iter + 1);
        let resp = match llm_call(provider.as_ref(), model, &mut messages, &full_system) {
            Ok(v)  => v,
            Err(e) => return Err(format!("llm error on iteration {}: {}", iter + 1, e)),
        };

        let parsed = provider.parse_response(resp)?;

        messages.as_array_mut().unwrap().push(parsed.assistant_msg);

        if parsed.tool_calls.is_empty() {
            eprintln!("[agent] done after {} iter(s)", iter + 1);
            if let Some(tid) = thread_id {
                let all = messages.as_array().unwrap();
                append_thread(tid, &all[persisted_len..]);
            }
            return Ok(parsed.text_parts.join("\n"));
        }

        let mut tool_results: Vec<ToolResult> = vec![];
        for tc in &parsed.tool_calls {
            match tc.name.as_str() {
                "run_shell"  => eprintln!("[agent] run_shell: {}", tc.input["command"].as_str().unwrap_or("")),
                "read_image" => eprintln!("[agent] read_image: {}", tc.input["path"].as_str().unwrap_or("")),
                "read_pdf"   => eprintln!("[agent] read_pdf: {}", tc.input["path"].as_str().unwrap_or("")),
                name         => eprintln!("[agent] tool: {name}"),
            }

            let raw = match tc.name.as_str() {
                "run_shell"  => run_shell(tc.input["command"].as_str().unwrap_or("")),
                "read_image" => read_image(
                    tc.input["path"].as_str().unwrap_or(""),
                    tc.input["question"].as_str().unwrap_or("Extract all text and data from this image verbatim."),
                ),
                "read_pdf"   => read_pdf(
                    tc.input["path"].as_str().unwrap_or(""),
                    tc.input["question"].as_str().unwrap_or("Extract all text and data from this PDF verbatim."),
                ),
                name => format!("unknown tool: {name}"),
            };

            let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

            const MAX_INLINE_CHARS: usize = 16_000;
            let content = if raw.len() > MAX_INLINE_CHARS {
                let out_path = format!("/tmp/tool_result_{}.txt", tc.id);
                match std::fs::write(&out_path, &raw) {
                    Ok(_) => format!(
                        "[{ts}] Output too large ({} chars) — full content saved to {out_path}\n\
                         Query it with grep/head/sed/awk rather than reading the whole file.\n\
                         Example: grep -n 'keyword' {out_path} | head -30",
                        raw.len()
                    ),
                    Err(e) => format!("[{ts}] Output too large ({} chars) and could not save to disk: {e}", raw.len()),
                }
            } else {
                format!("[{ts}]\n{raw}")
            };

            tool_results.push(ToolResult { tool_use_id: tc.id.clone(), content });
        }

        for msg in provider.wrap_tool_results(tool_results) {
            messages.as_array_mut().unwrap().push(msg);
        }

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
            println!("ai_result");
            println!("{result}");
        }
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}
