//! LLM transport. The loop depends on the `LlmClient` trait, not on `ureq`
//! directly, so tests can drive it with scripted responses. The real client's
//! retry/trim decisions are factored into a pure `decide` function that is unit
//! tested without any network.

use serde_json::{json, Value};

use crate::provider::Provider;

const DEFAULT_MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
const DEFAULT_LLM_URL: &str = "http://llm.aispec-system.svc.cluster.local/anthropic/v1/messages";

pub fn llm_url() -> String {
    std::env::var("LLM_URL").unwrap_or_else(|_| DEFAULT_LLM_URL.to_string())
}

pub fn llm_model() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

pub fn llm_request(url: &str) -> ureq::Request {
    let req = ureq::post(url).set("Content-Type", "application/json");
    if let Ok(key) = std::env::var("LLM_API_KEY") {
        req.set("Authorization", &format!("Bearer {key}"))
    } else {
        req
    }
}

/// One call to the model: send `messages`, return the raw response.
pub trait LlmClient {
    fn call(
        &self,
        provider: &dyn Provider,
        model: &str,
        messages: &mut Value,
        system: &str,
        tools: &Value,
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
        Attempt::Status(502, body) => Next::Retry(format!("HTTP 502: {}", clip(body))),
        Attempt::Status(code, body) => Next::Fail(format!("HTTP {code}: {}", clip(body))),
        Attempt::Network(msg) => {
            if msg.contains("NetworkError") || msg.contains("Connection") {
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
    ) -> Result<Value, String> {
        let url = llm_url();
        let mut last_err = String::new();
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            let req = provider.build_request(model, system, tools, messages);
            let outcome = match llm_request(&url).send_json(&req) {
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
    let mut last_err = String::new();
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        match llm_request(&url).send_json(&req) {
            Ok(resp) => match resp.into_json::<Value>() {
                Ok(v) => return v["content"][0]["text"].as_str().unwrap_or("No response").to_string(),
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
    fn server_502_retries() {
        let a = Attempt::Status(502, "bad gateway".into());
        assert!(matches!(decide(&a), Next::Retry(_)));
    }

    #[test]
    fn other_status_fails() {
        assert!(matches!(decide(&Attempt::Status(403, "no".into())), Next::Fail(_)));
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
    fn clip_bounds_body_length() {
        let long = "x".repeat(1000);
        assert_eq!(clip(&long).len(), 300);
    }
}
