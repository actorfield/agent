//! Provider adapters: translate a neutral message/tool history into a concrete
//! request for either an Anthropic-style or OpenAI-compatible endpoint, and parse
//! the response back into a neutral shape.
//!
//! Provider is inferred from the endpoint URL:
//!   contains `/v1/chat/completions`  -> OpenAI-compatible
//!   anything else                    -> Anthropic (default)

use serde_json::{json, Value};

/// A model turn parsed into a neutral shape.
pub struct ParsedResponse {
    pub text_parts: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Raw assistant message to push into history (provider-shaped).
    pub assistant_msg: Value,
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

pub trait Provider {
    /// Shape the neutral tool list into the provider's request format.
    fn shape_tools(&self, base: &Value) -> Value;
    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value;
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

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value {
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
        json!({
            "model":      model,
            "max_tokens": 16000,
            "system":     [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools":      tools,
            "messages":   req_msgs,
        })
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

        Ok(ParsedResponse {
            text_parts,
            tool_calls,
            assistant_msg: json!({ "role": "assistant", "content": content }),
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

    fn build_request(&self, model: &str, system: &str, tools: &Value, messages: &Value) -> Value {
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

        let assistant_msg =
            json!({ "role": "assistant", "content": msg["content"], "tool_calls": msg["tool_calls"] });

        Ok(ParsedResponse {
            text_parts,
            tool_calls,
            assistant_msg,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let req = Anthropic.build_request("m", "sys", &tools, &sample_messages());

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
        let _ = Anthropic.build_request("m", "sys", &tools, &msgs);
        // The tool_result block in the original still has no cache_control.
        let tr = &msgs[2]["content"][0];
        assert!(tr.get("cache_control").is_none());
    }

    #[test]
    fn openai_request_has_no_cache_control_and_prepends_system() {
        let base = json!([{"name":"a","description":"d","input_schema":{"type":"object"}}]);
        let tools = OpenAI.shape_tools(&base);
        let msgs = json!([{"role":"user","content":"hi"}]);
        let req = OpenAI.build_request("m", "sys", &tools, &msgs);
        assert_eq!(count_cache_control(&req), 0);
        assert_eq!(req["messages"][0]["role"], "system");
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "a");
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
