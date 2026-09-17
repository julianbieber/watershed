# Every way this repository is built, checked, run and driven.
#
# `all` is CI's order and CI runs nothing else, so a check that is not reachable from it
# is a check that does not happen.

# Use bash strict mode
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Shared env (same as CI)
RUSTFLAGS_BASE := "-Zshare-generics=y -Zthreads=0"
RUSTDOCFLAGS_BASE := "-Zshare-generics=y -Zthreads=0"
WASM_TARGET := "wasm32-unknown-unknown"
CONTROL_SOCKET := "/tmp/watershed-control.sock"
PROJECT := "/tmp/watershed-project"

# Default: list recipes
default:
    @just --list

# Install system libraries used by CI (Ubuntu/Debian)
deps:
	@sudo apt-get update
	@sudo apt-get install --no-install-recommends -y libasound2-dev libudev-dev libwayland-dev clang mesa-vulkan-drivers

# Put `watershed_editor` and `watershed-ctl` on PATH, built from this checkout.
install:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo install --locked --path crates/watershed_editor

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
all: fmt docs clippy bevy-lints test check-web check-goals

# Clean
clean:
	@cargo clean

# Run the editor on a project directory
run DIR:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo run --package watershed_editor -- {{DIR}}

# Start the editor with its control socket open, on a scratch project directory so the
# repository root is never made a project by accident. Release, because a scenario
# solves water over a whole document and a debug solve is minutes rather than seconds.
drive-start:
	@mkdir -p {{PROJECT}}
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	WATERSHED_CONTROL="{{CONTROL_SOCKET}}" \
	cargo run --release --package watershed_editor -- {{PROJECT}}

# Send one command to a running editor:
#   just drive observe water
#   just drive run scenarios/water_finds_the_lakes.txt
drive *ARGS:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	WATERSHED_CONTROL="{{CONTROL_SOCKET}}" \
	cargo run --release --quiet --bin watershed-ctl -- {{ARGS}}

# Build the example that spells out what the crate interface is meant to do.
check-goals:
	@env \
	RUSTFLAGS="{{RUSTFLAGS_BASE}}" \
	RUSTDOCFLAGS="{{RUSTDOCFLAGS_BASE}}" \
	cargo build --locked --package watershed --profile ci --example load_terrain

# Prose the AI policy leaves to a human. Not part of `all`: a placeholder is a note to
# myself, not a broken build.
check-placeholders:
	@rg -n 'TODO\(jb-(doc|comment)\)' crates/ || echo "none outstanding"
