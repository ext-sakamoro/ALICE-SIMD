#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml or security-audit.yml runs, with the same
# arguments. A step this script does not cover is a step that can only fail
# remotely, so a step added to a workflow is added here in the same commit.
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, clippy, no_std / wasm builds, docs, the
#            full test suites (every feature set), benches compile, MSRV, and
#            the security jobs (cargo audit / deny / machete)
#   --quick  static checks, clippy, no_std / wasm builds, docs and
#            `cargo test --lib`; skips the full suites, benches, MSRV and the
#            security jobs
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac
MSRV=1.87

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate root invalidates only this crate's fingerprints.
relint() { touch src/lib.rs; }
add_target() { rustup target list --installed | grep -qx "$1" || rustup target add "$1"; }

need actionlint "brew install actionlint"
need python3 "python 3.9+"

step "ci.yml / actionlint: workflow YAML"
actionlint .github/workflows/*.yml

step "ci.yml / fmt: cargo fmt --check"
cargo fmt -- --check

step "ci.yml / docs-lint: tests + public documents / CHANGELOG structure"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: default, all features, no_std tests (pedantic + nursery via Cargo.toml)"
relint
cargo clippy --all-targets -- -D warnings
relint
cargo clippy --all-targets --all-features -- -D warnings
relint
cargo clippy --tests --no-default-features --features libm -- -D warnings

step "ci.yml / no-std: thumbv7em-none-eabihf build + clippy"
add_target thumbv7em-none-eabihf
cargo build --lib --no-default-features --features libm --target thumbv7em-none-eabihf
relint
cargo clippy --lib --no-default-features --features libm --target thumbv7em-none-eabihf -- -D warnings

step "ci.yml / wasm: wasm32-unknown-unknown build"
add_target wasm32-unknown-unknown
cargo build --lib --target wasm32-unknown-unknown

step "ci.yml / doc: rustdoc -D warnings (default + all features)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --all-features

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick)"
  cargo test --lib
  echo; echo "preflight --quick OK (full test suites, benches, MSRV and security jobs skipped)"; exit 0
fi

step "ci.yml / test: every feature, default features, no_std code path via libm"
cargo test --all-features
cargo test
cargo test --lib --no-default-features --features libm

step "ci.yml / test: benches compile"
cargo bench --no-run

step "ci.yml / msrv: rust-version = $MSRV"
if rustup toolchain list | grep -q "^$MSRV"; then
  cargo +"$MSRV" check --lib
  cargo +"$MSRV" check --lib --all-features
else
  echo "toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
  exit 1
fi

step "security-audit.yml: cargo audit / cargo deny / cargo machete"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
cargo audit --db target/advisory-db --deny yanked
cargo deny --all-features check all
cargo machete

echo; echo "preflight OK"
