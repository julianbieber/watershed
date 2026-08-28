//! Sends commands to a running editor and prints what it says back.
//!
//! ```text
//! watershed-ctl <command> [args...]
//! watershed-ctl run <scenario file>
//! ```
//!
//! The socket comes from `WATERSHED_CONTROL`, the same variable the editor was started
//! with. One command per connection: connect, write a line, read a line of JSON.
//!
//! **Blocking is the feature.** The editor holds each reply until the command has
//! really happened, so `solve-water` returns when the water is solved and `capture`
//! when the PNG is on disk. There is nothing to poll and nothing to sleep on.
//!
//! A scenario is these same commands in a file, one per line, `#` to the end of a line
//! being a comment. It is replayed here rather than in the editor, which keeps the
//! protocol at one command per connection and keeps a script format out of the editor.
//! Replay stops at the first failure — a scenario is a sequence, and carrying on past a
//! step that did not happen would report on a document that never existed.
//!
//! The exit status carries the verdict as well as the printed JSON, so a caller can
//! branch on it without parsing.

use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::UnixStream,
    process::ExitCode,
};

const SOCKET_ENV: &str = "WATERSHED_CONTROL";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: watershed-ctl <command> [args...]");
        eprintln!("       watershed-ctl run <scenario file>");
        eprintln!("socket comes from ${SOCKET_ENV}");
        return ExitCode::FAILURE;
    }

    let Ok(socket) = std::env::var(SOCKET_ENV) else {
        eprintln!("${SOCKET_ENV} is not set; start the editor with it to open a socket");
        return ExitCode::FAILURE;
    };

    if args[0] == "run" {
        let Some(path) = args.get(1) else {
            eprintln!("run needs a scenario file");
            return ExitCode::FAILURE;
        };
        return scenario(&socket, path);
    }

    match send(&socket, &args.join(" ")) {
        Ok(reply) => {
            print!("{reply}");
            verdict(&reply)
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn scenario(socket: &str, path: &str) -> ExitCode {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut replies: Vec<String> = Vec::new();
    let mut failed = false;

    for line in text.lines() {
        let command = line.split('#').next().unwrap_or("").trim();
        if command.is_empty() {
            continue;
        }
        eprintln!("> {command}");
        match send(socket, command) {
            Ok(reply) => {
                let ok = reply.contains("\"ok\":true");
                replies.push(entry(command, reply.trim()));
                if !ok {
                    failed = true;
                    break;
                }
            }
            Err(error) => {
                replies.push(entry(
                    command,
                    &format!("{{\"ok\":false,\"error\":{error:?}}}"),
                ));
                failed = true;
                break;
            }
        }
    }

    println!(
        "{{\"scenario\":{:?},\"ok\":{},\"steps\":[{}]}}",
        path,
        !failed,
        replies.join(",")
    );
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn send(socket: &str, command: &str) -> Result<String, String> {
    let mut stream = UnixStream::connect(socket)
        .map_err(|error| format!("no editor listening on {socket}: {error}"))?;

    writeln!(stream, "{command}").map_err(|error| format!("could not send: {error}"))?;

    let mut reply = String::new();
    BufReader::new(&stream)
        .read_line(&mut reply)
        .map_err(|error| format!("no reply: {error}"))?;
    Ok(reply)
}

fn entry(command: &str, reply: &str) -> String {
    format!("{{\"step\":{command:?},\"reply\":{reply}}}")
}

fn verdict(reply: &str) -> ExitCode {
    if reply.contains("\"ok\":true") {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
