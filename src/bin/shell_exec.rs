// shell-exec actor handler — forked by actor-mesh C runtime per shell_exec tuple
//
// Contract (actor-mesh handler):
//   stdin:  raw payload bytes (JSON) written by runtime after stripping 256-byte header
//   stdout: optional topic line `shell_result\n` then result JSON
//   exit:   0 on success, non-zero triggers runtime retry

use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Deserialize)]
struct ShellExecPayload {
    command: String,
}

#[derive(Serialize)]
struct ShellResult {
    ok: bool,
    output: String,
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap_or(0);

    let payload: ShellExecPayload = match serde_json::from_str(input.trim()) {
        Ok(p) => p,
        Err(e) => {
            let result = ShellResult { ok: false, output: format!("invalid payload: {e}") };
            println!("shell_result");
            println!("{}", serde_json::to_string(&result).unwrap());
            std::process::exit(1);
        }
    };

    let output = match std::process::Command::new("sh")
        .arg("-c")
        .arg(&payload.command)
        .output()
    {
        Ok(out) => {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&out.stderr));
            let ok = out.status.success();
            ShellResult { ok, output: if s.trim().is_empty() { "(no output)".into() } else { s } }
        }
        Err(e) => ShellResult { ok: false, output: format!("exec error: {e}") },
    };

    // Emit topic override on first line so actor-mesh routes result correctly
    println!("shell_result");
    println!("{}", serde_json::to_string(&output).unwrap());
}
