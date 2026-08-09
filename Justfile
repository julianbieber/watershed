# Use bash strict mode
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Shared env (same as CI)
RUSTFLAGS_BASE := "-Zshare-generics=y -Zthreads=0"
RUSTDOCFLAGS_BASE := "-Zshare-generics=y -Zthreads=0"
WASM_TARGET := "wasm32-unknown-unknown"
CONTROL_SOCKET := "/tmp/watershed-control.sock"

# Default: list recipes
default:
    @just --list

# Install system libraries used by CI (Ubuntu/Debian)
# TODO(jb-comment): why clang is in this list — which recipe needs it and for which target.
deps:
	@sudo apt-get update
	@sudo apt-get install --no-install-recommends -y libasound2-dev libudev-dev libwayland-dev clang

# Format check
fmt:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo fmt --all -- --check

# Docs check
docs:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo doc --locked --workspace --profile ci --all-features --document-private-items --no-deps

# Clippy lints
clippy:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo clippy --locked --workspace --all-targets --profile ci --all-features

# Bevy lints (requires bevy_lint on PATH)
bevy-lints:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	bevy_lint --locked --package watershed_editor --all-targets --profile ci --all-features

# Install Bevy linter via the Bevy CLI installer, then ensure bevy_lint exists
bevy-lint-install:
	@bevy lint install
	@command -v bevy_lint >/dev/null 2>&1 || { echo "bevy_lint not on PATH; ensure installer completed."; exit 1; }

# Tests
test:
	cargo test --locked --workspace

# The library is the member that has to build for the web; the editor is native only.
check-web:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo check --locked --package watershed --profile ci --target {{WASM_TARGET}}

# Run everything in CI order
all: fmt docs clippy bevy-lints test check-web

# Clean
clean:
	@cargo clean

# Run the editor
run:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo run --package watershed_editor

# Start the editor with its control socket open. Release, because a scenario solves water
# over a whole document and a debug solve is minutes rather than seconds.
drive-start:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	WATERSHED_CONTROL="{{CONTROL_SOCKET}}" \
	cargo run --release --package watershed_editor

# Send one command to a running editor:
#   just drive observe water
#   just drive run scenarios/water_finds_the_lakes.txt
drive *ARGS:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	WATERSHED_CONTROL="{{CONTROL_SOCKET}}" \
	cargo run --release --quiet --bin watershed-ctl -- {{ARGS}}

# Prose the AI policy leaves to a human. Not part of `all`: a placeholder is a note to
# myself, not a broken build.
check-placeholders:
	@rg -n 'TODO\(jb-(doc|comment)\)' crates/ || echo "none outstanding"
