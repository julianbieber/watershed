//! The socket, and the one system that pumps it.

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

pub(super) fn build(app: &mut App) {
    let Ok(path) = std::env::var(SOCKET_ENV) else {
        return;
    };
    let path = PathBuf::from(path);

    match ControlServer::bind(path.clone()) {
        Ok(server) => {
            info!("control socket listening on {}", path.display());
            app.insert_resource(server);
            // Before `InputSystems`, because `keyboard_input_system` drains
            // `MessageReader<KeyboardInput>` into `ButtonInput` there: an injected key has
            // to be written ahead of it or it lands a frame late.
            app.add_systems(PreUpdate, serve.before(InputSystems));
        }
        Err(error) => error!("control socket {} unavailable: {error}", path.display()),
    }
}

/// The listener, plus at most one client.
///
/// One at a time on purpose: commands are a sequence, and a second client interleaving its
/// own would make "what state is the editor in" unanswerable. A caller that connects while
/// a command is in flight simply waits in the accept queue.
#[derive(Resource)]
struct ControlServer {
    listener: UnixListener,
    path: PathBuf,
    client: Option<Client>,
}

struct Client {
    stream: UnixStream,
    /// Bytes seen so far. A client's line can arrive split across frames, so the command is
    /// only parsed once its newline turns up.
    input: Vec<u8>,
    pending: Option<Pending>,
}

/// A command that has been read but has not finished happening yet.
struct Pending {
    command: Command,
    started_at: u32,
}

impl ControlServer {
    fn bind(path: PathBuf) -> std::io::Result<Self> {
        // A socket file left behind by a crash is not a running editor, but `bind` cannot
        // tell the difference and refuses either way. Connecting is the only way to ask: if
        // someone answers, this is a real instance and we must not steal its path.
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

/// The socket outlives nothing: when the resource drops, the path goes with it, so a clean
/// exit never leaves a file the next run has to reason about.
impl Drop for ControlServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Exclusive because an observation reads a dozen unrelated resources, and threading all of
/// them through a system signature is the thing that would make adding one expensive.
fn serve(world: &mut World) {
    world.resource_scope(|world, mut server: Mut<ControlServer>| {
        server.accept();

        // Taken out rather than borrowed: finishing a command drops the client, and that is
        // far easier to express when the client is not borrowed from the resource.
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
                // Still arriving, or the client hung up before saying anything.
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
    /// Reads until a newline turns up. `Ok(None)` means the line is still arriving.
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

    /// One command per connection, so writing the reply is also the goodbye — the client is
    /// dropped by the caller straight afterwards, which closes the stream.
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
