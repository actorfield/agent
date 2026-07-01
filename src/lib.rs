use serde::{Deserialize, Serialize};

pub mod agent_loop;
pub mod job;
pub mod llm;
pub mod policy;
pub mod provider;
pub mod registry;
pub mod spawn;
pub mod thread;
pub mod tools;

#[derive(Serialize, Deserialize)]
pub struct ShellExecPayload {
    pub command: String,
}

#[derive(Serialize, Deserialize)]
pub struct ShellResultPayload {
    pub ok: bool,
    pub output: String,
}
