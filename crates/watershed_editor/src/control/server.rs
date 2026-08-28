//! The socket the editor is driven through: accepting a caller, reading its command,
//! and holding the reply until the command has finished happening.
//!
//! One client at a time, and one command per connection. Commands are a sequence, and
//! a second caller interleaving its own would make "what state is the editor in"
//! unanswerable; a caller that connects while a command is in flight waits in the
//! accept queue.

use std::{
    io::{ErrorKind, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
};

use bevy::{diagnostic::FrameCount, input::InputSystems, prelude::*};
use serde_json::{Value, json};

use super::command::{Command, Poll};

/// Names the Unix socket to listen on. Unset, the plugin does nothing at all.
pub(super) const SOCKET_ENV: &str = "WATERSHED_CONTROL";

/// Binds the socket named by `WATERSHED_CONTROL` and installs the system that serves
/// it. Does nothing if the variable is unset, and logs rather than failing if the
/// socket cannot be bound — the editor runs either way.
pub(super) fn build(app: &mut App) {
    let Ok(path) = std::env::var(SOCKET_ENV) else {
        return;
    };
    let path = PathBuf::from(path);

    match ControlServer::bind(path.clone()) {
        Ok(server) => {
            info!("control socket listening on {}", path.display());
            app.insert_resource(server);
            app.add_systems(PreUpdate, serve.before(InputSystems));
        }
        Err(error) => error!("control socket {} unavailable: {error}", path.display()),
    }
}

#[derive(Resource)]
struct ControlServer {
    listener: UnixListener,
    path: PathBuf,
    client: Option<Client>,
}

struct Client {
    stream: UnixStream,
    input: Vec<u8>,
    pending: Option<Pending>,
}

struct Pending {
    command: Command,
    started_at: u32,
}

impl ControlServer {
    fn bind(path: PathBuf) -> std::io::Result<Self> {
        if UnixStream::connect(&path).is_ok() {
            return Err(std::io::Error::new(
                ErrorKind::AddrInUse,
                "another instance is already listening",
            ));
        }
        if path.exists() {
            std::fs::remove_file(&path)?;
        }

        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            path,
            client: None,
        })
    }

    fn accept(&mut self) {
        if self.client.is_some() {
            return;
        }
        match self.listener.accept() {
            Ok((stream, _)) => {
                if let Err(error) = stream.set_nonblocking(true) {
                    warn!("control client rejected: {error}");
                    return;
                }
                self.client = Some(Client {
                    stream,
                    input: Vec::new(),
                    pending: None,
                });
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => warn!("control accept failed: {error}"),
        }
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn serve(world: &mut World) {
    world.resource_scope(|world, mut server: Mut<ControlServer>| {
        server.accept();

        let Some(mut client) = server.client.take() else {
            return;
        };

        let frame = world.resource::<FrameCount>().0;

        if client.pending.is_none() {
            match client.read_command() {
                Ok(Some(line)) => match Command::parse(&line) {
                    Ok(command) => {
                        client.pending = Some(Pending {
                            command,
                            started_at: frame,
                        });
                    }
                    Err(message) => {
                        client.reply(&failed(&message));
                        return;
                    }
                },
                Ok(None) => {
                    server.client = Some(client);
                    return;
                }
                Err(_) => return,
            }
        }

        let pending = client
            .pending
            .as_mut()
            .expect("a command was just parsed into place");
        let elapsed = frame.saturating_sub(pending.started_at);

        match pending.command.poll(world, elapsed) {
            Poll::Running => server.client = Some(client),
            Poll::Done(fields) => {
                let reply = succeeded(pending.command.verb(), elapsed, fields);
                client.reply(&reply);
            }
            Poll::Failed(message) => client.reply(&failed(&message)),
        }
    });
}

impl Client {
    fn read_command(&mut self) -> std::io::Result<Option<String>> {
        let mut buffer = [0u8; 512];
        loop {
            match self.stream.read(&mut buffer) {
                Ok(0) => return Err(std::io::Error::from(ErrorKind::UnexpectedEof)),
                Ok(read) => self.input.extend_from_slice(&buffer[..read]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) => return Err(error),
            }
        }

        let Some(end) = self.input.iter().position(|&byte| byte == b'\n') else {
            return Ok(None);
        };
        let line = String::from_utf8_lossy(&self.input[..end])
            .trim()
            .to_owned();
        self.input.drain(..=end);
        Ok(Some(line))
    }

    fn reply(&mut self, value: &Value) {
        if let Err(error) = writeln!(self.stream, "{value}") {
            warn!("control reply failed: {error}");
        }
    }
}

fn succeeded(verb: &str, frames: u32, mut fields: Value) -> Value {
    let object = fields
        .as_object_mut()
        .expect("a command's fields must be a JSON object");
    object.insert("ok".into(), json!(true));
    object.insert("command".into(), json!(verb));
    object.insert("frames".into(), json!(frames));
    fields
}

fn failed(message: &str) -> Value {
    json!({ "ok": false, "error": message })
}
