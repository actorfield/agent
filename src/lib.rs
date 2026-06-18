use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ShellExecPayload {
    pub command: String,
}

#[derive(Serialize, Deserialize)]
pub struct ShellResultPayload {
    pub ok: bool,
    pub output: String,
}
