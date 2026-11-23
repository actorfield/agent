//! LLM transport. The loop depends on the `LlmClient` trait, not on `ureq`
//! directly, so tests can drive it with scripted responses. The real client's
//! retry/trim decisions are factored into a pure `decide` function that is unit
//! tested without any network.

use serde_json::{json, Value};

use crate::job::Effort;
use crate::provider::Provider;

const DEFAULT_MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
// Fallback only — the endpoint is normally injected via LLM_URL. Defaults to the
// canonical Anthropic Messages API (provider is inferred from the URL).
const DEFAULT_LLM_URL: &str = "https://api.anthropic.com/v1/messages";

pub fn llm_url() -> String {
    std::env::var("LLM_URL").unwrap_or_else(|_| DEFAULT_LLM_URL.to_string())
}

/// True when an LLM error is the context window overflowing, rather than a
/// transport or auth fault.
///
/// Matched on the message because providers disagree on how they say it and
/// none of them gives a machine-readable code every time: Anthropic sends
/// "prompt is too long", OpenAI "maximum context length"/"context_length_
/// exceeded", others "too many tokens". A miss just means the run reports a
/// generic failure -- the status quo -- so this errs towards not matching
/// rather than mislabelling an unrelated error as a context overflow.
pub fn is_context_limit_error(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    [
        "context length",
        "context_length_exceeded",
        "maximum context",
        "prompt is too long",
        "too many tokens",
        "reduce the length",
    ]
    .iter()
    .any(|needle| m.contains(needle))
}

pub fn llm_model() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

pub fn llm_request(url: &str) -> ureq::Request {
    authorized(ureq::post(url))
}

fn authorized(req: ureq::Request) -> ureq::Request {
    let req = req.set("Content-Type", "application/json");
    if let Ok(key) = std::env::var("LLM_API_KEY") {
        req.set("Authorization", &format!("Bearer {key}"))
    } else {
        req
    }
}

/// How long a streamed reply may go silent before it counts as hung
/// (`LLM_STREAM_IDLE_SECS`, default 180). Silence, not length: a long answer
/// keeps sending, so only a stalled one is cut off.
fn stream_idle() -> std::time::Duration {
    let secs = std::env::var("LLM_STREAM_IDLE_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(180);
    std::time::Duration::from_secs(secs)
}

/// A request whose reply is read as a stream, each read waiting at most
/// `stream_idle`.
fn llm_stream_request(url: &str) -> ureq::Request {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    let agent = AGENT.get_or_init(|| ureq::AgentBuilder::new().timeout_read(stream_idle()).build());
    authorized(agent.post(url))
}

/// A streamed OpenAI-compatible reply put back into the shape a plain one has
/// (`choices[0].message`, `usage`), so `parse_response` reads it unchanged:
/// text and reasoning concatenated, each tool call's id, name and argument
/// pieces joined by its index, usage from the final chunk.
fn assemble_openai_stream(lines: impl Iterator<Item = std::io::Result<String>>) -> Result<Value, String> {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let (mut calls, mut finish, mut usage, mut any): (Vec<Value>, Value, Value, bool) = (vec![], Value::Null, Value::Null, false);
    let append = |v: &mut Value, more: &str| *v = json!(format!("{}{more}", v.as_str().unwrap_or_default()));
    for line in lines {
        let line = line.map_err(|e| match e.kind() {
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => format!("stream stalled: nothing for {}s", stream_idle().as_secs()),
            _ => format!("stream broken: {e}"),
        })?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
        if data == "[DONE]" {
            break;
        }
        if data.is_empty() {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).map_err(|e| format!("Parse error in stream: {e}"))?;
        if !chunk["error"].is_null() {
            return Err(format!("stream error: {}", clip(&chunk["error"].to_string())));
        }
        any = true;
        if !chunk["usage"].is_null() {
            usage = chunk["usage"].clone();
        }
        let Some(choice) = chunk["choices"].get(0) else { continue };
        if !choice["finish_reason"].is_null() {
            finish = choice["finish_reason"].clone();
        }
        let delta = &choice["delta"];
        content.push_str(delta["content"].as_str().unwrap_or_default());
        reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let i = tc["index"].as_u64().map_or(calls.len().saturating_sub(1), |i| i as usize);
            while calls.len() <= i {
                calls.push(json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}));
            }
            let call = &mut calls[i];
            if let Some(id) = tc["id"].as_str() {
                call["id"] = json!(id);
            }
            append(&mut call["function"]["name"], tc["function"]["name"].as_str().unwrap_or_default());
            append(&mut call["function"]["arguments"], tc["function"]["arguments"].as_str().unwrap_or_default());
        }
    }
    if !any {
        return Err("stream broken: it ended before any reply".into());
    }
    let mut message = json!({"role": "assistant", "content": content});
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    Ok(json!({"choices": [{"message": message, "finish_reason": finish}], "usage": usage}))
}

/// One call to the model: send `messages`, return the raw response.
/// `Send + Sync` so a `Ctx` holding `&dyn LlmClient` can be shared across the
/// scoped threads spawn_agent uses to run concurrent sub-agents.
pub trait LlmClient: Send + Sync {
    fn call(
        &self,
        provider: &dyn Provider,
        model: &str,
        messages: &mut Value,
        system: &str,
        tools: &Value,
        effort: Effort,
    ) -> Result<Value, String>;
}

/// Outcome of a single HTTP attempt (network abstracted so `decide` is pure).
#[derive(Debug)]
enum Attempt {
    Ok(Value),
    Status(u16, String),
    Network(String),
}

/// What the retry loop should do next given one attempt's outcome.
#[derive(Debug, PartialEq)]
enum Next {
    Done, // response ready (carried separately)
    Retry(String),
    TrimRetry(String),
    Fail(String),
}

fn decide(attempt: &Attempt) -> Next {
    match attempt {
        Attempt::Ok(_) => Next::Done,
        Attempt::Status(400, body) => {
            if body.contains("prompt is too long") {
                Next::TrimRetry("context trimmed after prompt-too-long".into())
            } else {
                Next::Fail(format!("HTTP 400: {}", clip(body)))
            }
        }
        Attempt::Status(429, body) => Next::Retry(format!("HTTP 429: {}", clip(body))),
        Attempt::Status(code, body) if (500..600).contains(code) => {
            Next::Retry(format!("HTTP {code}: {}", clip(body)))
        }
        Attempt::Status(code, body) => Next::Fail(format!("HTTP {code}: {}", clip(body))),
        Attempt::Network(msg) => {
            if msg.contains("NetworkError") || msg.contains("Connection") || msg.starts_with("stream stalled") || msg.starts_with("stream broken") {
                Next::Retry(format!("HTTP error: {msg}"))
            } else {
                Next::Fail(format!("HTTP error: {msg}"))
            }
        }
    }
}

fn clip(body: &str) -> String {
    body[..body.len().min(300)].to_string()
}

/// The real, network-backed client.
pub struct UreqClient;

impl LlmClient for UreqClient {
    fn call(
        &self,
        provider: &dyn Provider,
        model: &str,
        messages: &mut Value,
        system: &str,
        tools: &Value,
        effort: Effort,
    ) -> Result<Value, String> {
        let url = llm_url();
        let mut last_err = String::new();
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            let mut req = provider.build_request(model, system, tools, messages, effort);
            // Streamed where the provider can: a reply that takes minutes to
            // write keeps arriving, so only silence counts as hung. Whole, a
            // long one outlived the gateway's limit and was retried from
            // scratch.
            let streamed = provider.streams();
            if streamed {
                req["stream"] = json!(true);
                req["stream_options"] = json!({"include_usage": true});
            }
            let sent = if streamed { llm_stream_request(&url).send_json(&req) } else { llm_request(&url).send_json(&req) };
            let outcome = match sent {
                Ok(resp) if resp.content_type() == "text/event-stream" => {
                    use std::io::BufRead;
                    match assemble_openai_stream(std::io::BufReader::new(resp.into_reader()).lines()) {
                        Ok(v) => Attempt::Ok(v),
                        Err(e) if e.starts_with("Parse error") => return Err(e),
                        Err(e) => Attempt::Network(e),
                    }
                }
                Ok(resp) => match resp.into_json::<Value>() {
                    Ok(v) => Attempt::Ok(v),
                    Err(e) => return Err(format!("Parse error: {e}")),
                },
                Err(ureq::Error::Status(code, resp)) => {
                    Attempt::Status(code, resp.into_string().unwrap_or_default())
                }
                Err(e) => Attempt::Network(e.to_string()),
            };
            match decide(&outcome) {
                Next::Done => {
                    if let Attempt::Ok(v) = outcome {
                        return Ok(v);
                    }
                }
                Next::TrimRetry(msg) => {
                    eprintln!("[agent] context too large — trimming last tool result and retrying");
                    provider.trim_last_tool_result(messages);
                    last_err = msg;
                }
                Next::Retry(msg) => last_err = msg,
                Next::Fail(msg) => return Err(msg),
            }
        }
        Err(last_err)
    }
}

/// A single, stateless model call used by `read_image` / `read_pdf` to extract
/// information from a document block.
pub fn single_call(content_block: Value, question: &str) -> String {
    let req = json!({
        "model": llm_model(),
        "max_tokens": 4096,
        "messages": [{ "role": "user", "content": [content_block, {"type":"text","text":question}] }]
    });
    let url = llm_url();
    let provider = crate::provider::detect_provider(&url);
    let mut last_err = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        match llm_request(&url).send_json(&req) {
            Ok(resp) => match resp.into_json::<Value>() {
                // Parsed by the same provider that shaped the block. Reading
                // Anthropic's `content[0].text` off an OpenAI response yielded
                // "No response" on a perfectly good 200.
                Ok(v) => {
                    return provider
                        .single_text(&v)
                        .unwrap_or_else(|| "No response".to_string())
                }
                Err(e) => return format!("Parse error: {e}"),
            },
            Err(ureq::Error::Status(502, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                last_err = format!("HTTP error: 502 {}", clip(&body));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_response_is_done() {
        assert_eq!(decide(&Attempt::Ok(json!({}))), Next::Done);
    }

    #[test]
    fn prompt_too_long_triggers_trim_retry() {
        let a = Attempt::Status(400, "error: prompt is too long for the window".into());
        assert!(matches!(decide(&a), Next::TrimRetry(_)));
    }

    #[test]
    fn other_400_fails() {
        let a = Attempt::Status(400, "bad request".into());
        assert!(matches!(decide(&a), Next::Fail(_)));
    }

    #[test]
    fn server_5xx_retries() {
        for code in [500, 502, 503, 504, 599] {
            let a = Attempt::Status(code, "server error".into());
            assert!(matches!(decide(&a), Next::Retry(_)), "expected retry for {code}");
        }
    }

    #[test]
    fn rate_limit_429_retries() {
        let a = Attempt::Status(429, "rate limited".into());
        assert!(matches!(decide(&a), Next::Retry(_)));
    }

    #[test]
    fn other_status_fails() {
        assert!(matches!(decide(&Attempt::Status(403, "no".into())), Next::Fail(_)));
        assert!(matches!(decide(&Attempt::Status(404, "no".into())), Next::Fail(_)));
    }

    #[test]
    fn transient_network_retries_but_hard_errors_fail() {
        assert!(matches!(
            decide(&Attempt::Network("Connection refused".into())),
            Next::Retry(_)
        ));
        assert!(matches!(
            decide(&Attempt::Network("bad certificate".into())),
            Next::Fail(_)
        ));
    }

    #[test]
    fn a_streamed_reply_is_rebuilt_into_the_plain_shape() {
        let sse = [
            r#"data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"Think"}}]}"#,
            "",
            r#"data: {"choices":[{"delta":{"reasoning_content":"ing.","content":"Hello "}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"there","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"run_shell","arguments":"{\"comm"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"ls\"}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"run_shell","arguments":"{}"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":9,"prompt_cache_hit_tokens":64}}"#,
            "data: [DONE]",
        ];
        let v = assemble_openai_stream(sse.iter().map(|l| Ok(l.to_string()))).unwrap();
        let m = &v["choices"][0]["message"];
        assert_eq!(m["content"], "Hello there");
        assert_eq!(m["reasoning_content"], "Thinking.");
        assert_eq!(m["tool_calls"][0]["id"], "call_1");
        assert_eq!(m["tool_calls"][0]["function"]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(m["tool_calls"][1]["function"]["name"], "run_shell");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        // parse_response reads it as it reads a plain reply, usage included.
        let parsed = crate::provider::OpenAI.parse_response(v).unwrap();
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].input["command"], "ls");
        let u = parsed.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens, u.cached_tokens), (100, 9, 64));
    }

    #[test]
    fn a_stalled_or_empty_stream_is_retried_and_a_provider_error_is_not() {
        let stalled = vec![
            Ok(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#.to_string()),
            Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out")),
        ];
        let e = assemble_openai_stream(stalled.into_iter()).unwrap_err();
        assert!(e.starts_with("stream stalled"), "{e}");
        assert!(matches!(decide(&Attempt::Network(e)), Next::Retry(_)));
        let e = assemble_openai_stream(vec![Ok("data: [DONE]".to_string())].into_iter()).unwrap_err();
        assert!(matches!(decide(&Attempt::Network(e)), Next::Retry(_)));
        let err = vec![Ok(r#"data: {"error":{"message":"content filtered"}}"#.to_string())];
        let e = assemble_openai_stream(err.into_iter()).unwrap_err();
        assert!(matches!(decide(&Attempt::Network(e)), Next::Fail(_)));
    }

    #[test]
    fn clip_bounds_body_length() {
        let long = "x".repeat(1000);
        assert_eq!(clip(&long).len(), 300);
    }
}

#[cfg(test)]
mod context_limit_tests {
    use super::is_context_limit_error;

    #[test]
    fn recognises_each_provider_wording() {
        // No provider gives a reliable machine-readable code, so this matches
        // on the message. These are the actual phrasings seen in the wild.
        for m in [
            "prompt is too long: 210000 tokens > 200000 maximum",
            "This model's maximum context length is 128000 tokens",
            "context_length_exceeded",
            "Please reduce the length of the messages",
            "too many tokens in request",
        ] {
            assert!(is_context_limit_error(m), "missed: {m}");
        }
    }

    #[test]
    fn does_not_claim_unrelated_errors() {
        // A miss costs a generic failure label (the status quo); a false match
        // would tell the user to shorten a conversation over an auth problem.
        for m in [
            "connection reset by peer",
            "401 Unauthorized: invalid api key",
            "429 rate limit exceeded",
            "upstream timed out",
            "",
        ] {
            assert!(!is_context_limit_error(m), "false positive: {m}");
        }
    }

    #[test]
    fn is_case_insensitive() {
        assert!(is_context_limit_error("PROMPT IS TOO LONG"));
    }
}
