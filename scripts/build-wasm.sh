#!/usr/bin/env bash
#
# scripts/build-wasm.sh -- build numrs2 for WebAssembly
#
# Primary path: wasm-pack (if installed) drives the whole pipeline -- cargo
# build, wasm-bindgen, and npm packaging -- in one step.
#
# Fallback path: if wasm-pack is not installed, do the equivalent by hand
# with `cargo build --target wasm32-unknown-unknown` plus the `wasm-bindgen`
# CLI, degrading gracefully with a clear, actionable message if that CLI is
# missing too, or if its version does not match the `wasm-bindgen` crate
# version actually resolved in Cargo.lock (a mismatch otherwise fails late,
# inside wasm-bindgen's own schema check, with a message that never mentions
# version skew).
#
# Requires: bash (arrays only -- no associative arrays, no `${var,,}`, no
# `mapfile` -- so this also runs on the stock bash 3.2 shipped with macOS,
# matching scripts/ci-local.sh's own constraint).
#
# Usage:
#   ./scripts/build-wasm.sh [target] [profile]
#     target:  web (default) | bundler | nodejs
#     profile: dev (default) | release
#
# Output: pkg/ at the repo root, an npm package ready for `npm publish`
# (wasm-pack path) or the pieces of one (fallback path -- see the generated
# pkg/package.json).
#
# This script only builds/packages. Publishing to npm happens from CI
# (.github/workflows/npm-publish.yml, once it exists -- see the build report
# for this task) or by hand with `npm publish` inside the resulting pkg/
# directory. The repo-root package.json is a dev-convenience manifest only
# ("private": true) -- it is never what gets published.
#
# Testing: this script does not run the WASM test suite -- `tests/wasm/
# {test_wasm_array,test_wasm_linalg,test_wasm_stats}.rs` (wired into a real
# cargo test target by `tests/wasm_integration.rs`) are `wasm-bindgen-test`s,
# which need a real browser and so cannot run on a plain `cargo test`/`cargo
# nextest run` host target. Run them with:
#   wasm-pack test --headless --firefox --features wasm
#   wasm-pack test --headless --chrome  --features wasm
#   wasm-pack test --firefox --features wasm   # interactive, not headless
# This builds for wasm32-unknown-unknown and drives the suite inside a real
# (headless, for the first two) browser via wasm-bindgen-test-runner.
# Requires geckodriver/chromedriver on PATH -- wasm-pack reports the install
# command if neither is found. `cargo check --target wasm32-unknown-unknown
# --features wasm --all-targets` (see scripts/ci-local.sh's `wasm-check`
# step) is the closest host-only proxy: it proves the suite compiles for the
# target without a browser, but it does not run the tests.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and arguments
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CRATE_NAME="numrs2"
SCOPE="cooljapan"
OUT_DIR="pkg"

TARGET="${1:-web}"
PROFILE="${2:-dev}"

case "$TARGET" in
  web|bundler|nodejs) ;;
  *)
    echo "error: unknown target '$TARGET' (expected web|bundler|nodejs)" >&2
    exit 2
    ;;
esac
case "$PROFILE" in
  dev|release) ;;
  *)
    echo "error: unknown profile '$PROFILE' (expected dev|release)" >&2
    exit 2
    ;;
esac

echo "== numrs2 WASM build =="
echo "repo root: ${REPO_ROOT}"
echo "target:    ${TARGET}"
echo "profile:   ${PROFILE}"
echo "out dir:   ${OUT_DIR}"
echo

# ---------------------------------------------------------------------------
# Make sure the wasm32 target is installed (idempotent; no-op if present)
# ---------------------------------------------------------------------------
echo "-- ensuring wasm32-unknown-unknown target is installed --"
if command -v rustup >/dev/null 2>&1; then
  rustup target add wasm32-unknown-unknown
else
  echo "warning: rustup not found on PATH; assuming wasm32-unknown-unknown is already available" >&2
fi
echo

# ---------------------------------------------------------------------------
# Primary path: wasm-pack
# ---------------------------------------------------------------------------
if command -v wasm-pack >/dev/null 2>&1; then
  echo "-- wasm-pack found ($(wasm-pack --version)); building with it --"

  PROFILE_FLAGS=(--dev)
  if [ "$PROFILE" = "release" ]; then
    PROFILE_FLAGS=(--release)
  fi

  WASM_PACK_ARGS=(
    build
    --target "$TARGET"
    --out-dir "$OUT_DIR"
    --out-name "$CRATE_NAME"
    --scope "$SCOPE"
    "${PROFILE_FLAGS[@]}"
    --features wasm
  )

  echo "\$ CARGO_INCREMENTAL=0 wasm-pack ${WASM_PACK_ARGS[*]}"
  if CARGO_INCREMENTAL=0 wasm-pack "${WASM_PACK_ARGS[@]}"; then
    echo
    echo "wasm-pack build succeeded -- output in ${OUT_DIR}/"
    ls -la "$OUT_DIR"
    exit 0
  fi

  echo
  echo "error: wasm-pack build failed -- see the compiler output above for the" >&2
  echo "       real error. This is not a script/tooling problem; it means the" >&2
  echo "       crate itself does not currently compile for wasm32-unknown-unknown" >&2
  echo "       with --features wasm." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Fallback path: cargo build + the wasm-bindgen CLI, by hand
# ---------------------------------------------------------------------------
echo "-- wasm-pack not found; falling back to cargo build + wasm-bindgen CLI --"
echo "   (install wasm-pack for the smoother, self-contained path:"
echo "    cargo install wasm-pack)"
echo

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: neither wasm-pack nor the wasm-bindgen CLI is installed on PATH.

Install one of:
    cargo install wasm-pack
    cargo install wasm-bindgen-cli

then re-run this script.
EOF
  exit 1
fi

# The wasm-bindgen CLI's version must match the `wasm-bindgen` crate version
# actually resolved into Cargo.lock -- wasm-bindgen embeds a schema version
# in the generated .wasm and refuses to process artifacts built against a
# different one. That failure mode does not mention version skew at all, so
# check for it here with an actionable message instead of letting it happen
# lower down.
LOCKED_VERSION="$(awk '
  $0 == "name = \"wasm-bindgen\"" { want = 1; next }
  want && /^version = / {
    gsub(/version = "|"/, "");
    print;
    exit
  }
' Cargo.lock 2>/dev/null || true)"
CLI_VERSION="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')"

if [ -n "$LOCKED_VERSION" ] && [ -n "$CLI_VERSION" ] && [ "$LOCKED_VERSION" != "$CLI_VERSION" ]; then
  cat >&2 <<EOF
warning: version skew between the wasm-bindgen CLI and the 'wasm-bindgen'
crate resolved in Cargo.lock:
    Cargo.lock:        ${LOCKED_VERSION}
    wasm-bindgen CLI:  ${CLI_VERSION}

This commonly fails later with a schema-mismatch error from wasm-bindgen
itself, not with a version-related message. Fix with one of:
    cargo install wasm-bindgen-cli --version ${LOCKED_VERSION} --locked
    cargo install wasm-pack   # manages this pairing automatically

Continuing anyway...

EOF
fi

PROFILE_DIR="debug"
CARGO_PROFILE_FLAGS=()
if [ "$PROFILE" = "release" ]; then
  PROFILE_DIR="release"
  CARGO_PROFILE_FLAGS=(--release)
fi

echo "-- cargo build --target wasm32-unknown-unknown --features wasm (${PROFILE}) --"
CARGO_BUILD_ARGS=(
  build
  --target wasm32-unknown-unknown
  --features wasm
  --lib
  "${CARGO_PROFILE_FLAGS[@]}"
)
echo "\$ CARGO_INCREMENTAL=0 cargo ${CARGO_BUILD_ARGS[*]}"
if ! CARGO_INCREMENTAL=0 cargo "${CARGO_BUILD_ARGS[@]}"; then
  echo
  echo "error: cargo build failed -- see the compiler output above for the real" >&2
  echo "       error. This is not a script/tooling problem; it means the crate" >&2
  echo "       itself does not currently compile for wasm32-unknown-unknown" >&2
  echo "       with --features wasm." >&2
  exit 1
fi

WASM_ARTIFACT="target/wasm32-unknown-unknown/${PROFILE_DIR}/${CRATE_NAME}.wasm"
if [ ! -f "$WASM_ARTIFACT" ]; then
  echo "error: expected build artifact not found at ${WASM_ARTIFACT}" >&2
  echo "       (cargo build reported success but the .wasm cdylib is missing --" >&2
  echo "       check that [lib] crate-type in Cargo.toml includes \"cdylib\")" >&2
  exit 1
fi
echo

echo "-- wasm-bindgen --target ${TARGET} --out-dir ${OUT_DIR} --"
mkdir -p "$OUT_DIR"
echo "\$ wasm-bindgen --target ${TARGET} --out-dir ${OUT_DIR} --out-name ${CRATE_NAME} ${WASM_ARTIFACT}"
wasm-bindgen \
  --target "$TARGET" \
  --out-dir "$OUT_DIR" \
  --out-name "$CRATE_NAME" \
  "$WASM_ARTIFACT"
echo

# wasm-bindgen itself does not emit a package.json (only wasm-pack does), so
# the fallback path writes a minimal, target-aware one by hand -- otherwise
# pkg/ would not be an installable/publishable npm package at all.
CRATE_VERSION="$(awk -F'"' '/^version = / {print $2; exit}' Cargo.toml)"
PKG_NAME="@${SCOPE}/${CRATE_NAME}"

echo "-- writing a minimal ${OUT_DIR}/package.json (wasm-bindgen does not generate one) --"
if [ "$TARGET" = "nodejs" ]; then
  cat > "${OUT_DIR}/package.json" <<EOF
{
  "name": "${PKG_NAME}",
  "version": "${CRATE_VERSION}",
  "description": "WebAssembly bindings for NumRS2 (Node.js/CommonJS build)",
  "license": "Apache-2.0",
  "main": "${CRATE_NAME}.js",
  "types": "${CRATE_NAME}.d.ts",
  "files": [
    "${CRATE_NAME}_bg.wasm",
    "${CRATE_NAME}.js",
    "${CRATE_NAME}.d.ts"
  ]
}
EOF
else
  cat > "${OUT_DIR}/package.json" <<EOF
{
  "name": "${PKG_NAME}",
  "version": "${CRATE_VERSION}",
  "description": "WebAssembly bindings for NumRS2 (${TARGET} build)",
  "license": "Apache-2.0",
  "module": "${CRATE_NAME}.js",
  "types": "${CRATE_NAME}.d.ts",
  "sideEffects": false,
  "files": [
    "${CRATE_NAME}_bg.wasm",
    "${CRATE_NAME}.js",
    "${CRATE_NAME}.d.ts"
  ]
}
EOF
fi

echo
echo "wasm-bindgen fallback build succeeded -- output in ${OUT_DIR}/"
ls -la "$OUT_DIR"
