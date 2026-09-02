//! Provider adapters: translate a neutral message/tool history into a concrete
//! request for either an Anthropic-style or OpenAI-compatible endpoint, and parse
//! the response back into a neutral shape.
//!
//! Provider is inferred from the endpoint URL:
//!   contains `/v1/chat/completions`  -> OpenAI-compatible
//!   anything else                    -> Anthropic (default)

use serde_json::{json, Value};

use crate::job::Effort;

/// A model turn parsed into a neutral shape.
pub struct ParsedResponse {
    pub text_parts: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Raw assistant message to push into history (provider-shaped).
    pub assistant_msg: Value,
    /// Token usage for this call, if the API reported it. Anthropic and
    /// OpenAI-compatible APIs both return a top-level `usage` object on
    /// every response (not just the final turn) — normalized here into one
    /// neutral shape since the two providers use different field names
    /// (input_tokens/output_tokens vs. prompt_tokens/completion_tokens).
    pub usage: Option<Usage>,
}

pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
}

/// `Send + Sync` so a `Ctx` holding `&dyn Provider` can be shared across the
/// scoped threads spawn_agent uses to run concurrent sub-agents.
pub trait Provider: Send + Sync {
    /// Shape the neutral tool list into the provider's request format.
    fn shape_tools(&self, base: &Value) -> Value;
    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value, effort: Effort) -> Value;
    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String>;
    /// One or more messages to append to history for the given tool results.
    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value>;
    /// Shrink the largest tool result in history to recover from an over-long prompt.
    fn trim_last_tool_result(&self, messages: &mut Value);
    /// Build an assistant message that issues a single tool call. Used to rebuild a
    /// round when resuming persisted work.
    fn tool_call_message(&self, id: &str, name: &str, input: &Value) -> Value;
    /// True iff `messages` already contains a tool result answering `tool_use_id`.
    fn has_tool_result(&self, messages: &Value, tool_use_id: &str) -> bool;
    /// Tool calls issued by the FINAL assistant message that no tool result
    /// answers, as (id, name) in the order the model issued them.
    ///
    /// This is how a run that stopped mid-round is detected on resume. Both
    /// providers reject a conversation containing a tool call with no result,
    /// so a paused `ask_user` -- which deliberately breaks the loop before any
    /// tool runs -- leaves history that cannot be sent back as-is.
    fn pending_tool_calls(&self, messages: &Value) -> Vec<(String, String)>;

    /// Content block wrapping an image for a one-shot `llm::single_call`
    /// (`read_image`). Providers disagree on this the same way they disagree
    /// about everything else here, and single_call used to hardcode the
    /// Anthropic shape -- so against an OpenAI-compatible endpoint every image
    /// read failed at the parser with "cannot unmarshal JSON data as string or
    /// array of content parts", no matter which model was picked.
    fn image_block(&self, mime: &str, b64: &str) -> Value;

    /// Content block wrapping a PDF, or `None` where the provider's chat API
    /// has no document part. `None` is a real answer, not a gap to paper over:
    /// sending an unsupported block produces a 400 the caller cannot act on,
    /// whereas "this endpoint cannot read PDFs" is something it can.
    fn pdf_block(&self, b64: &str) -> Option<Value>;

    /// Assistant text from a one-shot response. `parse_response` is the
    /// loop's path and carries tool calls, history and usage with it; this is
    /// the same extraction with none of that attached.
    fn single_text(&self, resp: &Value) -> Option<String>;
}

/// Pick a provider from the endpoint URL.
pub fn detect_provider(url: &str) -> Box<dyn Provider> {
    if url.contains("/v1/chat/completions") {
        Box::new(OpenAI)
    } else {
        Box::new(Anthropic)
    }
}

// ── Anthropic ──────────────────────────────────────────────────────────────────

/// Model families confirmed to support adaptive thinking (`thinking.effort`).
/// Anthropic 400s on `type: "adaptive"` for anything else — see
/// https://github.com/anomalyco/opencode/issues/17876 — so this must stay an
/// allowlist, not a denylist.
const ADAPTIVE_THINKING_MODELS: &[&str] = &["opus-4-8", "sonnet-5", "haiku-4-6", "claude-5", "mythos"];

fn is_adaptive_capable(model: &str) -> bool {
    ADAPTIVE_THINKING_MODELS.iter().any(|m| model.contains(m))
}

/// Legacy fixed-budget thinking, for models that predate adaptive thinking.
/// 1024 is Anthropic's documented minimum; values chosen conservatively below
/// our 16000 max_tokens ceiling.
fn budget_tokens(effort: Effort) -> u32 {
    match effort {
        Effort::Low => 1024,
        Effort::Medium => 4096,
        Effort::High => 10000,
        Effort::None => 0,
    }
}

fn effort_str(effort: Effort) -> Option<&'static str> {
    match effort {
        Effort::Low => Some("low"),
        Effort::Medium => Some("medium"),
        Effort::High => Some("high"),
        Effort::None => None,
    }
}

fn anthropic_thinking(model: &str, effort: Effort) -> Option<Value> {
    let level = effort_str(effort)?;
    if is_adaptive_capable(model) {
        Some(json!({"type": "adaptive", "effort": level}))
    } else {
        Some(json!({"type": "enabled", "budget_tokens": budget_tokens(effort)}))
    }
}

pub struct Anthropic;

impl Provider for Anthropic {
    fn shape_tools(&self, base: &Value) -> Value {
        let mut tools = base.clone();
        // Stable cache breakpoint on the last tool.
        if let Some(arr) = tools.as_array_mut() {
            if let Some(last) = arr.last_mut() {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        tools
    }

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value, effort: Effort) -> Value {
        // Stamp cache_control on the last content block of the last message so the
        // growing conversation tail is cached incrementally: each turn the previous
        // tail becomes a cache hit and the breakpoint moves forward. The task message
        // keeps its own permanent breakpoint; together with the system and tools
        // breakpoints this holds at the four-breakpoint maximum. Stamping happens on
        // this per-request clone only, so it never accumulates on the stored history.
        let mut req_msgs = messages.clone();
        if let Some(last_msg) = req_msgs.as_array_mut().and_then(|a| a.last_mut()) {
            if let Some(last_block) = last_msg["content"].as_array_mut().and_then(|c| c.last_mut()) {
                last_block["cache_control"] = json!({"type": "ephemeral"});
            }
        }
        let mut req = json!({
            "model":      model,
            "max_tokens": 16000,
            "system":     [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools":      tools,
            "messages":   req_msgs,
        });
        // Opt-in only: Effort::None omits `thinking` entirely, preserving prior
        // behavior. Adaptive thinking (`type: "adaptive"`) 400s on models that
        // predate it, so we gate on model name and fall back to the legacy
        // fixed-budget form for everything else.
        if let Some(thinking) = anthropic_thinking(model, effort) {
            req["thinking"] = thinking;
        }
        req
    }

    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String> {
        let content = resp["content"]
            .as_array()
            .ok_or("no content in response")?
            .clone();

        let mut text_parts = vec![];
        let mut tool_calls = vec![];
        for block in &content {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(t) = block["text"].as_str() {
                        text_parts.push(t.to_string());
                    }
                }
                Some("tool_use") => {
                    tool_calls.push(ToolCall {
                        id: block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        input: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }

        let usage = (!resp["usage"].is_null()).then(|| Usage {
            input_tokens: resp["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: resp["usage"]["output_tokens"].as_u64().unwrap_or(0),
        });

        Ok(ParsedResponse {
            text_parts,
            tool_calls,
            assistant_msg: json!({ "role": "assistant", "content": content }),
            usage,
        })
    }

    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value> {
        let blocks: Vec<Value> = results
            .into_iter()
            .map(|r| {
                json!({
                    "type":        "tool_result",
                    "tool_use_id": r.tool_use_id,
                    "content":     r.content,
                })
            })
            .collect();
        vec![json!({ "role": "user", "content": blocks })]
    }

    fn trim_last_tool_result(&self, messages: &mut Value) {
        if let Some(arr) = messages.as_array_mut() {
            if let Some(last) = arr.iter_mut().rev().find(|m| m["role"] == "user") {
                if let Some(content) = last["content"].as_array_mut() {
                    let largest = content
                        .iter_mut()
                        .filter(|b| b["type"] == "tool_result")
                        .max_by_key(|b| b["content"].as_str().map(|s| s.len()).unwrap_or(0));
                    if let Some(block) = largest {
                        block["content"] = json!("[output removed — too large for context. The data was saved to disk; query it there.]");
                    }
                }
            }
        }
    }

    fn tool_call_message(&self, id: &str, name: &str, input: &Value) -> Value {
        json!({
            "role": "assistant",
            "content": [{ "type": "tool_use", "id": id, "name": name, "input": input }]
        })
    }

    fn has_tool_result(&self, messages: &Value, tool_use_id: &str) -> bool {
        messages.as_array().is_some_and(|msgs| {
            msgs.iter().any(|m| {
                m["content"].as_array().is_some_and(|blocks| {
                    blocks.iter().any(|b| {
                        b["type"] == "tool_result" && b["tool_use_id"] == tool_use_id
                    })
                })
            })
        })
    }

    fn pending_tool_calls(&self, messages: &Value) -> Vec<(String, String)> {
        let Some(last) = messages.as_array().and_then(|m| m.last()) else {
            return vec![];
        };
        if last["role"] != "assistant" {
            return vec![];
        }
        last["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b["type"] == "tool_use")
                    .filter_map(|b| {
                        Some((b["id"].as_str()?.to_string(), b["name"].as_str()?.to_string()))
                    })
                    .filter(|(id, _)| !self.has_tool_result(messages, id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn image_block(&self, mime: &str, b64: &str) -> Value {
        json!({"type":"image","source":{"type":"base64","media_type":mime,"data":b64}})
    }

    fn pdf_block(&self, b64: &str) -> Option<Value> {
        Some(json!({
            "type":"document",
            "source":{"type":"base64","media_type":"application/pdf","data":b64}
        }))
    }

    fn single_text(&self, resp: &Value) -> Option<String> {
        resp["content"][0]["text"].as_str().map(str::to_string)
    }
}

// ── OpenAI-compatible ───────────────────────────────────────────────────────────

pub struct OpenAI;

impl Provider for OpenAI {
    fn shape_tools(&self, base: &Value) -> Value {
        let arr = match base.as_array() {
            Some(a) => a,
            None => return json!([]),
        };
        let tools: Vec<Value> = arr
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name":        t["name"],
                        "description": t["description"],
                        "parameters":  t["input_schema"],
                    }
                })
            })
            .collect();
        json!(tools)
    }

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value, effort: Effort) -> Value {
        let mut all_msgs = vec![json!({"role": "system", "content": system})];
        if let Some(arr) = messages.as_array() {
            all_msgs.extend(arr.iter().cloned());
        }
        let mut req = json!({
            "model":      model,
            "max_tokens": 16000,
            "tools":      tools,
            "messages":   all_msgs,
        });
        // DeepSeek's thinking mode: {"type":"enabled","reasoning_effort":"high"|"max"}.
        // DeepSeek itself coerces low/medium to high and xhigh to max, so we do the
        // same rather than sending values it would silently remap anyway.
        if effort != Effort::None {
            let reasoning_effort = if effort == Effort::High { "max" } else { "high" };
            req["thinking"] = json!({"type": "enabled", "reasoning_effort": reasoning_effort});
        }
        req
    }

    fn parse_response(&self, resp: Value) -> Result<ParsedResponse, String> {
        let msg = resp["choices"][0]["message"].clone();
        if msg.is_null() {
            return Err("no choices[0].message in response".into());
        }

        let mut text_parts = vec![];
        if let Some(t) = msg["content"].as_str() {
            if !t.is_empty() {
                text_parts.push(t.to_string());
            }
        }

        let mut tool_calls = vec![];
        if let Some(tcs) = msg["tool_calls"].as_array() {
            for tc in tcs {
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or("").to_string(),
                    name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                    input,
                });
            }
        }

        // reasoning_content (DeepSeek thinking mode) must be preserved verbatim in
        // tool-call history or the next request 400s — carry it through untouched
        // when present; omitted entirely when absent (thinking disabled/unset).
        let mut assistant_msg =
            json!({ "role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"] });
        if !msg["reasoning_content"].is_null() {
            assistant_msg["reasoning_content"] = msg["reasoning_content"].clone();
        }

        let usage = (!resp["usage"].is_null()).then(|| Usage {
            input_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: resp["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        });

        Ok(ParsedResponse {
            text_parts,
            tool_calls,
            assistant_msg,
            usage,
        })
    }

    fn wrap_tool_results(&self, results: Vec<ToolResult>) -> Vec<Value> {
        results
            .into_iter()
            .map(|r| {
                json!({
                    "role":         "tool",
                    "tool_call_id": r.tool_use_id,
                    "content":      r.content,
                })
            })
            .collect()
    }

    fn trim_last_tool_result(&self, messages: &mut Value) {
        if let Some(arr) = messages.as_array_mut() {
            let largest = arr
                .iter_mut()
                .filter(|m| m["role"] == "tool")
                .max_by_key(|m| m["content"].as_str().map(|s| s.len()).unwrap_or(0));
            if let Some(msg) = largest {
                msg["content"] = json!("[output removed — too large for context. The data was saved to disk; query it there.]");
            }
        }
    }

    fn tool_call_message(&self, id: &str, name: &str, input: &Value) -> Value {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": input.to_string() }
            }]
        })
    }

    fn has_tool_result(&self, messages: &Value, tool_use_id: &str) -> bool {
        messages.as_array().is_some_and(|msgs| {
            msgs.iter()
                .any(|m| m["role"] == "tool" && m["tool_call_id"] == tool_use_id)
        })
    }

    fn pending_tool_calls(&self, messages: &Value) -> Vec<(String, String)> {
        let Some(last) = messages.as_array().and_then(|m| m.last()) else {
            return vec![];
        };
        if last["role"] != "assistant" {
            return vec![];
        }
        last["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|c| {
                        Some((
                            c["id"].as_str()?.to_string(),
                            c["function"]["name"].as_str()?.to_string(),
                        ))
                    })
                    .filter(|(id, _)| !self.has_tool_result(messages, id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn image_block(&self, mime: &str, b64: &str) -> Value {
        // A data: URI, not a nested source object -- the OpenAI content-part
        // schema has no `source`, which is what the endpoint rejects outright.
        json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{b64}")}})
    }

    fn pdf_block(&self, _b64: &str) -> Option<Value> {
        // Chat-completions content parts are text and images only. Callers
        // report this rather than sending something that cannot work.
        None
    }

    fn single_text(&self, resp: &Value) -> Option<String> {
        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B64: &str = "aGVsbG8=";

    #[test]
    fn image_block_follows_the_endpoint() {
        // The bug this exists for: an Anthropic-shaped block sent to an
        // OpenAI-compatible endpoint is rejected by its parser before any
        // model sees it, so no choice of model could make read_image work.
        let oai = OpenAI.image_block("image/png", B64);
        assert_eq!(oai["type"], "image_url");
        assert_eq!(
            oai["image_url"]["url"],
            format!("data:image/png;base64,{B64}")
        );
        assert!(oai["source"].is_null(), "OpenAI parts have no `source`");

        let anth = Anthropic.image_block("image/png", B64);
        assert_eq!(anth["type"], "image");
        assert_eq!(anth["source"]["media_type"], "image/png");
        assert_eq!(anth["source"]["data"], B64);
    }

    #[test]
    fn pdf_is_offered_only_where_it_is_accepted() {
        assert!(Anthropic.pdf_block(B64).is_some());
        // Not a gap: chat-completions content parts are text and images only,
        // and saying so beats a 400 the caller cannot act on.
        assert!(OpenAI.pdf_block(B64).is_none());
    }

    #[test]
    fn single_text_reads_each_provider_own_response() {
        let anth = json!({"content":[{"type":"text","text":"from anthropic"}]});
        assert_eq!(Anthropic.single_text(&anth).as_deref(), Some("from anthropic"));

        let oai = json!({"choices":[{"message":{"role":"assistant","content":"from openai"}}]});
        assert_eq!(OpenAI.single_text(&oai).as_deref(), Some("from openai"));

        // Crossed over, each returns None rather than a wrong answer -- which
        // is what turned a good 200 into "No response" before.
        assert!(Anthropic.single_text(&oai).is_none());
        assert!(OpenAI.single_text(&anth).is_none());
    }

    #[test]
    fn detect_provider_picks_the_shape_that_endpoint_accepts() {
        let gateway = "http://aispec-llm-gateway.svc/v1/chat/completions";
        assert_eq!(detect_provider(gateway).image_block("image/png", B64)["type"], "image_url");
        assert_eq!(
            detect_provider("https://api.anthropic.com/v1/messages").image_block("image/png", B64)["type"],
            "image"
        );
    }

    fn count_cache_control(v: &Value) -> usize {
        // Recursively count cache_control markers in a request.
        match v {
            Value::Object(m) => {
                let here = if m.contains_key("cache_control") { 1 } else { 0 };
                here + m.values().map(count_cache_control).sum::<usize>()
            }
            Value::Array(a) => a.iter().map(count_cache_control).sum(),
            _ => 0,
        }
    }

    fn sample_messages() -> Value {
        json!([
            {"role":"user","content":[{"type":"text","text":"task","cache_control":{"type":"ephemeral"}}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"run_shell","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"out"}]}
        ])
    }

    #[test]
    fn anthropic_moves_breakpoint_to_last_block_and_stays_within_limit() {
        let base = json!([{"name":"a","description":"d","input_schema":{}}]);
        let tools = Anthropic.shape_tools(&base);
        let req = Anthropic.build_request("m", "sys", &tools, &sample_messages(), Effort::None);

        // Last message's last block is stamped.
        let msgs = req["messages"].as_array().unwrap();
        let last = msgs.last().unwrap();
        let last_block = last["content"].as_array().unwrap().last().unwrap();
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");

        // system + tools + task + moving tail = at most 4 breakpoints.
        assert!(count_cache_control(&req) <= 4, "too many cache breakpoints");
    }

    #[test]
    fn anthropic_stamp_does_not_mutate_source_messages() {
        let msgs = sample_messages();
        let base = json!([{"name":"a","description":"d","input_schema":{}}]);
        let tools = Anthropic.shape_tools(&base);
        let _ = Anthropic.build_request("m", "sys", &tools, &msgs, Effort::None);
        // The tool_result block in the original still has no cache_control.
        let tr = &msgs[2]["content"][0];
        assert!(tr.get("cache_control").is_none());
    }

    #[test]
    fn openai_request_has_no_cache_control_and_prepends_system() {
        let base = json!([{"name":"a","description":"d","input_schema":{"type":"object"}}]);
        let tools = OpenAI.shape_tools(&base);
        let msgs = json!([{"role":"user","content":"hi"}]);
        let req = OpenAI.build_request("m", "sys", &tools, &msgs, Effort::None);
        assert_eq!(count_cache_control(&req), 0);
        assert_eq!(req["messages"][0]["role"], "system");
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "a");
    }

    #[test]
    fn effort_none_omits_thinking_on_both_providers() {
        let msgs = json!([]);
        let anthropic_req = Anthropic.build_request("claude-opus-4-8", "sys", &json!([]), &msgs, Effort::None);
        assert!(anthropic_req.get("thinking").is_none());
        let openai_req = OpenAI.build_request("deepseek-v4-pro", "sys", &json!([]), &msgs, Effort::None);
        assert!(openai_req.get("thinking").is_none());
    }

    #[test]
    fn anthropic_adaptive_thinking_for_capable_model() {
        let msgs = json!([]);
        let req = Anthropic.build_request("claude-opus-4-8", "sys", &json!([]), &msgs, Effort::High);
        assert_eq!(req["thinking"]["type"], "adaptive");
        assert_eq!(req["thinking"]["effort"], "high");
    }

    #[test]
    fn anthropic_legacy_budget_for_older_model() {
        let msgs = json!([]);
        let req = Anthropic.build_request(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "sys",
            &json!([]),
            &msgs,
            Effort::Low,
        );
        assert_eq!(req["thinking"]["type"], "enabled");
        assert_eq!(req["thinking"]["budget_tokens"], 1024);
        assert!(req["thinking"].get("effort").is_none());
    }

    #[test]
    fn deepseek_reasoning_effort_maps_low_and_medium_to_high_and_high_to_max() {
        let msgs = json!([]);
        for (effort, expected) in [(Effort::Low, "high"), (Effort::Medium, "high"), (Effort::High, "max")] {
            let req = OpenAI.build_request("deepseek-v4-pro", "sys", &json!([]), &msgs, effort);
            assert_eq!(req["thinking"]["type"], "enabled");
            assert_eq!(req["thinking"]["reasoning_effort"], expected, "effort={effort:?}");
        }
    }

    #[test]
    fn openai_parse_response_carries_reasoning_content_into_assistant_msg() {
        let resp = json!({"choices":[{"message":{
            "content": "",
            "reasoning_content": "step by step...",
            "tool_calls": [{"id":"c1","function":{"name":"run_shell","arguments":"{}"}}]
        }}]});
        let parsed = OpenAI.parse_response(resp).unwrap();
        assert_eq!(parsed.assistant_msg["reasoning_content"], "step by step...");
    }

    #[test]
    fn openai_parse_response_omits_reasoning_content_when_absent() {
        let resp = json!({"choices":[{"message":{"content": "hi", "tool_calls": null}}]});
        let parsed = OpenAI.parse_response(resp).unwrap();
        assert!(parsed.assistant_msg.get("reasoning_content").is_none());
    }

    #[test]
    fn anthropic_parses_text_tool_and_mixed() {
        let resp = json!({"content":[
            {"type":"text","text":"thinking"},
            {"type":"tool_use","id":"x","name":"run_shell","input":{"command":"ls"}}
        ]});
        let parsed = Anthropic.parse_response(resp).unwrap();
        assert_eq!(parsed.text_parts, vec!["thinking".to_string()]);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "run_shell");
    }

    #[test]
    fn anthropic_rejects_malformed_response() {
        assert!(Anthropic.parse_response(json!({"nope": true})).is_err());
    }

    #[test]
    fn anthropic_parses_usage() {
        let resp = json!({
            "content": [{"type":"text","text":"hi"}],
            "usage": {"input_tokens": 123, "output_tokens": 45}
        });
        let parsed = Anthropic.parse_response(resp).unwrap();
        let usage = parsed.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, 123);
        assert_eq!(usage.output_tokens, 45);
    }

    #[test]
    fn anthropic_usage_absent_when_response_has_none() {
        let resp = json!({"content": [{"type":"text","text":"hi"}]});
        let parsed = Anthropic.parse_response(resp).unwrap();
        assert!(parsed.usage.is_none());
    }

    #[test]
    fn openai_parses_tool_calls_from_arguments_string() {
        let resp = json!({"choices":[{"message":{
            "content": "",
            "tool_calls": [{"id":"c1","function":{"name":"run_shell","arguments":"{\"command\":\"ls\"}"}}]
        }}]});
        let parsed = OpenAI.parse_response(resp).unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].input["command"], "ls");
    }

    #[test]
    fn openai_parses_usage() {
        let resp = json!({
            "choices":[{"message":{"content":"hi"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230}
        });
        let parsed = OpenAI.parse_response(resp).unwrap();
        let usage = parsed.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(usage.output_tokens, 30);
    }

    #[test]
    fn openai_usage_absent_when_response_has_none() {
        let resp = json!({"choices":[{"message":{"content":"hi"}}]});
        let parsed = OpenAI.parse_response(resp).unwrap();
        assert!(parsed.usage.is_none());
    }

    #[test]
    fn trim_shrinks_largest_block() {
        let mut msgs = json!([
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content":"short"},
                {"type":"tool_result","tool_use_id":"b","content":"a very long output that dominates the message and should be the one shrunk first"}
            ]}
        ]);
        Anthropic.trim_last_tool_result(&mut msgs);
        assert!(msgs[0]["content"][1]["content"].as_str().unwrap().contains("output removed"));
        assert_eq!(msgs[0]["content"][0]["content"], "short");
    }

    #[test]
    fn anthropic_synthesizes_and_detects_tool_result() {
        let msg = Anthropic.tool_call_message("toolu_1", "spawn_agent", &json!({"task":"x"}));
        assert_eq!(msg["content"][0]["type"], "tool_use");
        assert_eq!(msg["content"][0]["id"], "toolu_1");

        let convo = json!([
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"spawn_agent","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"done"}]}
        ]);
        assert!(Anthropic.has_tool_result(&convo, "toolu_1"));
        assert!(!Anthropic.has_tool_result(&convo, "toolu_2"));
    }

    #[test]
    fn openai_synthesizes_and_detects_tool_result() {
        let msg = OpenAI.tool_call_message("c1", "spawn_agent", &json!({"task":"x"}));
        assert_eq!(msg["tool_calls"][0]["id"], "c1");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "spawn_agent");

        let convo = json!([
            {"role":"tool","tool_call_id":"c1","content":"done"}
        ]);
        assert!(OpenAI.has_tool_result(&convo, "c1"));
        assert!(!OpenAI.has_tool_result(&convo, "c2"));
    }

    #[test]
    fn trim_is_bounded_the_placeholder_becomes_largest() {
        // Documents the shallow-trim limit: after one trim the placeholder is the
        // longest block, so a second trim re-selects it and the small block is left.
        let mut msgs = json!([
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content":"short"},
                {"type":"tool_result","tool_use_id":"b","content":"a very long output that dominates the message"}
            ]}
        ]);
        Anthropic.trim_last_tool_result(&mut msgs);
        Anthropic.trim_last_tool_result(&mut msgs);
        assert_eq!(msgs[0]["content"][0]["content"], "short");
    }
}
