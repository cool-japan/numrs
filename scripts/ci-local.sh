#!/usr/bin/env bash
#
# scripts/ci-local.sh -- numrs2 local verification pipeline
#
# COOLJAPAN branch policy forbids any GitHub Actions workflow other than
# pypi-publish.yml / npm-publish.yml (see .github/workflows/rust.yml.disabled).
# This script IS the CI: run it locally before you consider a change done.
#
# Requires: bash (arrays; written to also work on the stock bash 3.2 shipped
# with macOS -- no associative arrays, no `${var,,}`, no `mapfile`).
#
# Usage:
#   ./scripts/ci-local.sh                      # run every step, in order
#   ./scripts/ci-local.sh all                  # same as above
#   ./scripts/ci-local.sh build test clippy deny
#   ./scripts/ci-local.sh -h | --help
#
# Steps (see usage() below for one-line descriptions):
#   fmt-check  build  clippy  test  doctest  deny  wasm-check  policy
#
# Exit status: 0 if every executed check passed (WARN/SKIP do not count as
# failure); 1 if any step FAILed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASELINE_DIR="${REPO_ROOT}/scripts/ci-baselines"
UNWRAP_BASELINE_FILE="${BASELINE_DIR}/unwrap_count.txt"
COW_GUARD_FILE="src/array/core.rs"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RED=$'\033[0;31m'
  C_GREEN=$'\033[0;32m'
  C_YELLOW=$'\033[0;33m'
  C_BLUE=$'\033[0;34m'
  C_BOLD=$'\033[1m'
  C_RESET=$'\033[0m'
else
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_RESET=""
fi

HR="================================================================"

banner() {
  printf '\n%s%s%s\n' "${C_BOLD}${C_BLUE}" "$HR" "$C_RESET"
  printf '%s%s%s\n' "${C_BOLD}${C_BLUE}" "$1" "$C_RESET"
  printf '%s%s%s\n' "${C_BOLD}${C_BLUE}" "$HR" "$C_RESET"
}

sub() {
  printf '\n%s-- %s --%s\n' "$C_BLUE" "$1" "$C_RESET"
}

info()  { printf '%s[info]%s %s\n' "$C_BLUE" "$C_RESET" "$1"; }
ok()    { printf '%s[ OK ]%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
fail_msg() { printf '%s[FAIL]%s %s\n' "$C_RED" "$C_RESET" "$1"; }
warn()  { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$1"; }

# ---------------------------------------------------------------------------
# Step bookkeeping (parallel indexed arrays -- bash 3.2 has no `declare -A`)
# ---------------------------------------------------------------------------
STEP_NAMES=()
STEP_STATUSES=()   # PASS | FAIL | WARN | SKIP
STEP_NOTES=()
OVERALL_FAIL=0

# record_result <label> <PASS|FAIL|WARN|SKIP> [note]
record_result() {
  local name="$1"
  local status="$2"
  local note="${3:-}"
  STEP_NAMES+=("$name")
  STEP_STATUSES+=("$status")
  STEP_NOTES+=("$note")
  if [ "$status" = "FAIL" ]; then
    OVERALL_FAIL=1
  fi
}

# run_cmd_step <label> <command...>
# Runs a command, prints a sub-banner, and records PASS/FAIL. Never lets a
# failing command abort the script (the command is the condition of an
# `if`, so `set -e` does not fire) -- that is the whole point: we want every
# requested step to run and be reported, not to stop at the first failure.
run_cmd_step() {
  local label="$1"
  shift
  sub "$label"
  info "\$ $*"
  if "$@"; then
    ok "$label"
    record_result "$label" PASS ""
  else
    local rc=$?
    fail_msg "$label (exit $rc)"
    record_result "$label" FAIL "exit $rc"
  fi
}

# ---------------------------------------------------------------------------
# Step 1: fmt-check
# ---------------------------------------------------------------------------
step_fmt_check() {
  banner "1) fmt-check  (cargo fmt --check)"

  local has_cfg=1
  if [ ! -f "rustfmt.toml" ] && [ ! -f ".rustfmt.toml" ]; then
    has_cfg=0
    warn "no rustfmt.toml/.rustfmt.toml checked in at the repo root -- formatting style is unpinned"
  fi

  sub "cargo fmt --check"
  info "\$ cargo fmt --check"
  if cargo fmt --check; then
    ok "cargo fmt --check: no diffs"
    record_result "fmt-check" PASS ""
    return
  fi

  if [ "$has_cfg" -eq 1 ]; then
    fail_msg "cargo fmt --check: diffs found against the checked-in rustfmt config"
    record_result "fmt-check" FAIL "diffs vs checked-in rustfmt config"
  else
    warn "cargo fmt --check: diffs found, but there is no rustfmt config checked in -- reporting only, not failing the build"
    record_result "fmt-check" WARN "diffs found; no rustfmt config checked in (report only)"
  fi
}

# ---------------------------------------------------------------------------
# Step 2: build matrix
# ---------------------------------------------------------------------------
step_build_matrix() {
  banner "2) build matrix  (cargo check across the feature matrix)"
  run_cmd_step "build: default features" \
    cargo check
  run_cmd_step "build: --no-default-features --features matrix_decomp,scirs" \
    cargo check --no-default-features --features matrix_decomp,scirs
  run_cmd_step "build: --features ci-safe" \
    cargo check --features ci-safe
  run_cmd_step "build: --features gpu" \
    cargo check --features gpu
  run_cmd_step "build: --features distributed" \
    cargo check --features distributed
  run_cmd_step "build: --features io-all" \
    cargo check --features io-all
  run_cmd_step "build: --features visualization" \
    cargo check --features visualization
  run_cmd_step "build: --features python" \
    cargo check --features python
  run_cmd_step "build: --all-features" \
    cargo check --all-features
}

# ---------------------------------------------------------------------------
# Step 3: clippy
# ---------------------------------------------------------------------------
step_clippy() {
  banner "3) clippy  (-D warnings, all targets, all features)"
  run_cmd_step "clippy: --all-targets --all-features -- -D warnings" \
    cargo clippy --all-targets --all-features -- -D warnings
}

# ---------------------------------------------------------------------------
# Step 4: test
# ---------------------------------------------------------------------------
step_test() {
  banner "4) test  (cargo nextest, falls back to cargo test)"

  # `--all-features` unconditionally turns on `python`, which builds pyo3
  # with its `extension-module` feature. `extension-module` deliberately
  # skips linking against libpython -- the hosting Python interpreter is
  # expected to supply those symbols itself at dlopen time -- so a normal
  # test binary that links the whole workspace can NEVER finish linking
  # while it is on; this is a link-time failure, not a flaky test, and no
  # amount of retrying fixes it. `cargo check` (below) stops before
  # codegen/linking, so it is unaffected and is how `python` gets verified
  # here. Actually exercising the extension module requires a real
  # `maturin develop`/`maturin build` (which sets up the link configuration
  # pyo3 needs) followed by running the Python side, e.g. `tests/
  # python_smoke.py` -- see pyproject.toml's `[tool.maturin]` and
  # README.md's Python section. That full maturin build is out of scope for
  # this cargo-only script.
  #
  # So: every OTHER feature is enabled explicitly (never `--all-features`)
  # for the actual test run, and `python` gets its own check-only step.
  local non_python_features="matrix_decomp,validation,unstable,fast,scirs,gpu,lapack,arrow,parquet,netcdf,matlab,messagepack,bson,io-all,wasm,distributed,visualization,ci-safe"

  if command -v cargo-nextest >/dev/null 2>&1; then
    run_cmd_step "test: nextest run --workspace (default features)" \
      cargo nextest run --workspace
    run_cmd_step "test: nextest run --workspace --features <all except python>" \
      cargo nextest run --workspace --features "$non_python_features"
  else
    warn "cargo-nextest not found on PATH -- falling back to 'cargo test' (install: cargo install cargo-nextest --locked)"
    record_result "test: cargo-nextest" SKIP "not installed; falling back to cargo test"
    run_cmd_step "test: cargo test --workspace (default features) [nextest fallback]" \
      cargo test --workspace
    run_cmd_step "test: cargo test --workspace --features <all except python> [nextest fallback]" \
      cargo test --workspace --features "$non_python_features"
  fi

  run_cmd_step "test: cargo check --features python (extension-module cannot link a test binary -- see comment above; real python testing goes through maturin)" \
    cargo check --features python
}

# ---------------------------------------------------------------------------
# Step 5: doctest
# ---------------------------------------------------------------------------
step_doctest() {
  banner "5) doctest  (cargo test --doc)"
  run_cmd_step "doctest: cargo test --doc" \
    cargo test --doc
}

# ---------------------------------------------------------------------------
# Step 6: deny
# ---------------------------------------------------------------------------
step_deny() {
  banner "6) deny  (cargo deny check bans)"
  if ! command -v cargo-deny >/dev/null 2>&1; then
    warn "cargo-deny not found on PATH -- skipping (install: cargo install cargo-deny --locked)"
    record_result "deny: cargo deny check bans" SKIP "cargo-deny not installed"
    return
  fi
  if [ ! -f "deny.toml" ]; then
    warn "deny.toml not found at the repo root -- skipping 'cargo deny check bans'. This is a known gap (see CLAUDE.md: prepare deny.toml at the workspace top level), not a pass."
    record_result "deny: cargo deny check bans" SKIP "deny.toml missing at repo root"
    return
  fi
  run_cmd_step "deny: cargo deny check bans" \
    cargo deny check bans
}

# ---------------------------------------------------------------------------
# Step 7: wasm-check
# ---------------------------------------------------------------------------
step_wasm_check() {
  banner "7) wasm-check  (wasm32-unknown-unknown)"
  if ! command -v rustup >/dev/null 2>&1; then
    warn "rustup not found on PATH -- skipping wasm-check"
    record_result "wasm-check: target" SKIP "rustup not installed"
    return
  fi
  # Capture rustup's output fully before grepping it (rather than piping
  # straight into `grep -q`): `grep -q` exits on its first match, and if
  # rustup were still writing when that happens, `pipefail` would turn its
  # SIGPIPE death into a nonzero pipeline status -- which `!` would then
  # misread as "target not installed" even when it is. Capturing first
  # removes rustup from the pipeline entirely, so there is nothing for
  # grep's early exit to race against.
  local installed
  installed="$( { rustup target list --installed 2>/dev/null || true; } )"
  if ! printf '%s\n' "$installed" | grep -q '^wasm32-unknown-unknown$'; then
    warn "wasm32-unknown-unknown target not installed -- skipping (install: rustup target add wasm32-unknown-unknown)"
    record_result "wasm-check: target" SKIP "wasm32-unknown-unknown target not installed"
    return
  fi
  # NOTE: intentionally NOT --no-default-features. Cargo.toml marks `scirs`
  # "MANDATORY: Always enabled by default per SCIRS2 POLICY - DO NOT
  # DISABLE", and two of the wasm-exported functions (determinant/inverse in
  # src/wasm/linalg.rs) are behind `#[cfg(feature = "lapack")]`, itself a
  # default feature -- disabling defaults here would silently stop checking
  # those two exports. `--features wasm` on top of the default set is also
  # the literal command this project's WASM build-story task verifies
  # against, so this step should match it exactly.
  #
  # 2026-08-25 update: the 7 `?`-on-non-Result errors formerly here (in the
  # `#[cfg(not(target_arch = "aarch64"))]` fallback impls of
  # src/simd_optimize/neon_enhanced/{arithmetic,comparison,exponential,
  # trigonometric}.rs -- unrelated to the wasm feature, only reachable
  # because that module has no arch cfg of its own and an aarch64 host masks
  # its "not aarch64" branch) are fixed: those 7 sites now use the same
  # `.unwrap_or_else(|e| panic!("{e}"))` pattern the sibling
  # `#[cfg(target_arch = "aarch64")]` impls in the same files already used.
  # `cargo check --target wasm32-unknown-unknown --features wasm` passes as
  # of this update (verified on this aarch64 host, which is why the fix
  # above matters at all -- this step would not have caught the regression
  # itself).
  #
  # Still an out-of-scope, out-of-ownership gap for a full `cargo build` /
  # `wasm-pack build` (this step only runs `cargo check`, which does not hit
  # this): src/memory_alloc/large_scale.rs:43 (`8 * 1024 * 1024 * 1024` as a
  # `usize` literal, "8GB default") and src/new_modules/fft_enhanced.rs:494
  # (`v |= v >> 32`, a shift equal to the type's bit width) both deny
  # `arithmetic_overflow` on any 32-bit target incl. wasm32-unknown-unknown
  # (usize is 32-bit there); both are unmodified at HEAD (not mid-edit by
  # another wave) and need a real per-`target_pointer_width` decision, not a
  # mechanical fix, so they are left for their owning module's wave.
  run_cmd_step "wasm-check: cargo check --target wasm32-unknown-unknown --features wasm" \
    cargo check --target wasm32-unknown-unknown --features wasm
}

# ---------------------------------------------------------------------------
# Step 8: policy greps
#
# NOTE: deliberately uses `grep -r`, never `rg`/ripgrep. ripgrep silently
# skips gitignored files -- that hid 102 unwrap() calls sitting in .bak
# files from a previous scan. `grep -r` walks everything under the given
# path, gitignored or not, which is what a policy gate needs. That fix is
# necessary but not sufficient by itself: a bare `--include='*.rs'` glob
# still excludes `foo.rs.bak` (it requires the name to *end* in ".rs"),
# which is exactly the pattern .gitignore's `**/*.rs.bak` rule names and
# the exact class of file that hid those 102 unwraps -- so the unwrap scan
# below also includes `*.rs.bak`, or `grep -r` alone would not have closed
# the gap it was chosen to close.
# ---------------------------------------------------------------------------
policy_unwrap_scan() {
  local label="policy: production unwrap() scan (src/, vs baseline)"
  sub "$label"

  local raw
  raw="$( { grep -rn '\.unwrap()' src/ --include='*.rs' --include='*.rs.bak' || true; } )"
  local count=0
  if [ -n "$raw" ]; then
    count="$(printf '%s\n' "$raw" | wc -l | tr -d ' ')"
  fi
  info "current unwrap() count in src/: $count"

  local baseline_exists=0
  local baseline=0
  if [ -f "$UNWRAP_BASELINE_FILE" ]; then
    baseline_exists=1
    baseline="$(tr -d '[:space:]' < "$UNWRAP_BASELINE_FILE")"
    case "$baseline" in
      ''|*[!0-9]*)
        warn "baseline file $UNWRAP_BASELINE_FILE does not contain a plain non-negative integer -- treating baseline as 0"
        baseline=0
        ;;
    esac
  fi

  if [ "$baseline_exists" -eq 0 ]; then
    warn "no baseline file at $UNWRAP_BASELINE_FILE -- treating current count ($count) as informational only, not gating on it"
    record_result "$label" WARN "no baseline on disk; count=$count"
    return
  fi

  if [ "$count" -gt "$baseline" ]; then
    fail_msg "unwrap() count regressed: $count > baseline $baseline (see scripts/ci-baselines/unwrap_count.txt)"
    printf '%s\n' "$raw" | head -20
    record_result "$label" FAIL "count=$count > baseline=$baseline"
  elif [ "$count" -lt "$baseline" ]; then
    ok "unwrap() count improved: $count < baseline $baseline"
    warn "consider tightening the baseline: printf '%s\n' $count > $UNWRAP_BASELINE_FILE"
    record_result "$label" PASS "count=$count < baseline=$baseline (baseline is stale; consider lowering it)"
  else
    ok "unwrap() count matches baseline ($count)"
    record_result "$label" PASS "count=$count == baseline=$baseline"
  fi
}

policy_file_size_check() {
  local label="policy: file size (no .rs file >= 2000 lines)"
  sub "$label"

  local violations
  violations="$( { find . \( -path './target' -o -path './.git' -o -path './patches' \) -prune \
        -o -type f -name '*.rs' -print0 2>/dev/null \
      | xargs -0 wc -l 2>/dev/null \
      | awk '$1 >= 2000 && $2 != "total" {print}'; } || true )"

  if [ -n "$violations" ]; then
    fail_msg "files >= 2000 lines found (COOLJAPAN policy: keep files < 2000 lines; split with splitrs, rslines 50 to find targets):"
    printf '%s\n' "$violations"
    record_result "$label" FAIL "$(printf '%s\n' "$violations" | wc -l | tr -d ' ') file(s) over the limit"
  else
    ok "no .rs file >= 2000 lines (repo-wide scan; target/, .git/, patches/ excluded)"
    record_result "$label" PASS ""
  fi
}

# Scan src/ for *calls* to one Arc primitive, ignoring prose.
#
# `Array<T>` is Arc-backed copy-on-write, and the whole invariant rests on
# there being exactly one unshare point (`Array::nd_mut`) and exactly one
# consuming extraction point (`Array::into_nd`), both in ${COW_GUARD_FILE}.
# A second `Arc::make_mut` anywhere would be a second, unreviewed place where
# a shared buffer silently forks; a second `Arc::try_unwrap` would be a second
# place that can deep-copy on extraction. So this counts occurrences rather
# than merely confining them.
#
# Comment lines are stripped before counting: the doc comments on `nd_mut` and
# `into_nd` (and the prose in src/array_ops/creation/contiguous.rs explaining
# why a clone is not reported as aliasing) name these primitives on purpose,
# and a guard that counted prose would punish documenting the invariant. The
# trailing `(` in the pattern keeps this anchored on call syntax as a second
# line of defence.
arc_primitive_call_sites() {
  local primitive="$1"
  { grep -rn -F "${primitive}(" src/ --include='*.rs' || true; } \
    | { grep -v -E '^[^:]*:[0-9]+:[[:space:]]*(//|/\*|\*)' || true; }
}

policy_arc_cow_guard() {
  local label="policy: Arc COW guard (exactly one make_mut + one try_unwrap, in ${COW_GUARD_FILE})"
  sub "$label"

  local status=PASS
  local detail=""
  local primitive raw count offenders

  for primitive in Arc::make_mut Arc::try_unwrap; do
    raw="$(arc_primitive_call_sites "$primitive")"
    count=0
    if [ -n "$raw" ]; then
      count="$(printf '%s\n' "$raw" | wc -l | tr -d ' ')"
    fi

    if [ "$count" -ne 1 ]; then
      fail_msg "expected exactly 1 ${primitive} call in src/, found ${count} (COW invariant guard):"
      if [ -n "$raw" ]; then printf '%s\n' "$raw"; fi
      status=FAIL
      detail="${detail}${primitive}=${count} (want 1); "
      continue
    fi

    offenders="$(printf '%s\n' "$raw" | grep -v -F "${COW_GUARD_FILE}:" || true)"
    if [ -n "$offenders" ]; then
      fail_msg "the single ${primitive} call is not in ${COW_GUARD_FILE} (COW invariant guard):"
      printf '%s\n' "$offenders"
      status=FAIL
      detail="${detail}${primitive} outside ${COW_GUARD_FILE}; "
    else
      ok "exactly one ${primitive} call, in ${COW_GUARD_FILE}"
      detail="${detail}${primitive}=1 ok; "
    fi
  done

  # `Arc::get_mut` is a third way to reach a uniquely-owned buffer and would
  # bypass the single unshare point entirely (it returns None instead of
  # copying when the buffer is shared), so it must not appear at all.
  raw="$(arc_primitive_call_sites Arc::get_mut)"
  if [ -n "$raw" ]; then
    fail_msg "Arc::get_mut must not be used in src/ -- all unsharing goes through ${COW_GUARD_FILE}'s nd_mut():"
    printf '%s\n' "$raw"
    status=FAIL
    detail="${detail}Arc::get_mut present"
  else
    ok "no Arc::get_mut calls in src/"
    detail="${detail}Arc::get_mut=0 ok"
  fi

  record_result "$label" "$status" "$detail"
}

step_policy() {
  banner "8) policy greps  (unwrap baseline / file size / Arc COW guard)"
  policy_unwrap_scan
  policy_file_size_check
  policy_arc_cow_guard
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ALL_STEPS=(fmt-check build clippy test doctest deny wasm-check policy)

usage() {
  cat <<EOF
Usage: $(basename "$0") [step ...]

Runs the numrs2 local CI pipeline. With no arguments, runs every step below
in order. Pass one or more step names to run only a subset (still executed
in the canonical order below, each at most once).

Steps:
  fmt-check   cargo fmt --check (report-only warning if no rustfmt config is checked in)
  build       cargo check across the feature matrix (default, matrix_decomp+scirs,
              ci-safe, gpu, distributed, io-all, visualization, python, all-features)
  clippy      cargo clippy --all-targets --all-features -- -D warnings
  test        cargo nextest run --workspace (default + --all-features);
              falls back to cargo test if cargo-nextest is not installed
  doctest     cargo test --doc
  deny        cargo deny check bans (skipped with a notice if cargo-deny or
              deny.toml is missing)
  wasm-check  cargo check --target wasm32-unknown-unknown --features wasm
              (skipped with a notice if the target is not installed)
  policy      grep-based COOLJAPAN policy gates: production unwrap() baseline,
              file-size limit (<2000 lines), Arc COW guard confinement

  all         same as passing no arguments

Examples:
  $(basename "$0")
  $(basename "$0") build test clippy deny
  $(basename "$0") policy

Exit status: 0 if every executed check passed; 1 if any step FAILed.
EOF
}

if [ "$#" -eq 0 ]; then
  SELECTED=("${ALL_STEPS[@]}")
elif [ "$#" -eq 1 ] && [ "$1" = "all" ]; then
  SELECTED=("${ALL_STEPS[@]}")
else
  SELECTED=()
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        usage
        exit 0
        ;;
      fmt-check|build|clippy|test|doctest|deny|wasm-check|policy)
        SELECTED+=("$arg")
        ;;
      *)
        echo "error: unknown step '$arg'" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

is_selected() {
  local needle="$1"
  local x
  for x in "${SELECTED[@]}"; do
    if [ "$x" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
banner "numrs2 local CI  --  scripts/ci-local.sh"
info "repo root: ${REPO_ROOT}"
info "rustc: $(rustc --version 2>/dev/null || echo 'not found')"
info "cargo: $(cargo --version 2>/dev/null || echo 'not found')"
if command -v cargo-nextest >/dev/null 2>&1; then
  info "cargo-nextest: available"
else
  info "cargo-nextest: not found (test step will fall back to cargo test)"
fi
if command -v cargo-deny >/dev/null 2>&1; then
  info "cargo-deny: available"
else
  info "cargo-deny: not found (deny step will be skipped)"
fi
info "selected steps: ${SELECTED[*]}"

for step in "${ALL_STEPS[@]}"; do
  if is_selected "$step"; then
    case "$step" in
      fmt-check)  step_fmt_check ;;
      build)      step_build_matrix ;;
      clippy)     step_clippy ;;
      test)       step_test ;;
      doctest)    step_doctest ;;
      deny)       step_deny ;;
      wasm-check) step_wasm_check ;;
      policy)     step_policy ;;
    esac
  fi
done

# ---------------------------------------------------------------------------
# Perf/bench (opt-in only -- NOT in ALL_STEPS, NOT run by `all`/no-args)
# ---------------------------------------------------------------------------
# matrix_decomp's lu/cholesky/qr `perf_verification` modules hold wall-clock
# COW-vs-precow ratio assertions that flake under parallel/shared-machine
# load, so they are `#[ignore]`d in the default suite (see each module for
# its exact reason string). This is the single place to invoke ignored perf
# tests with `--run-ignored`: if you add another wall-clock perf test
# elsewhere, add its own opt-in-gated `run_cmd_step` line here rather than a
# new section, so there is one shared, documented opt-in surface instead of
# several.
if [ "${RUN_PERF_VERIFICATION:-0}" = "1" ]; then
  banner "perf/bench  (opt-in via RUN_PERF_VERIFICATION=1; --run-ignored, wall-clock)"
  # `--run-ignored only` (nextest's actual flag value -- NOT `ignored-only`,
  # which nextest rejects) restricts to `#[ignore]`d tests; `test(cow_vs_precow)`
  # substring-matches all three matrix_decomp perf_verification benches
  # (`bench_lu_cow_vs_precow`, `bench_pivoted_cholesky_cow_vs_precow`,
  # `bench_householder_qr_cow_vs_precow`).
  run_cmd_step "perf: matrix_decomp COW-vs-precow wall-clock ratio (lu/cholesky/qr, --release --run-ignored)" \
    cargo nextest run --release --run-ignored only -E 'test(cow_vs_precow)'
fi

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
banner "summary"

printf '%-6s  %s\n' "STATUS" "STEP"
printf '%-6s  %s\n' "------" "----"

n=${#STEP_NAMES[@]}
pass_n=0
fail_n=0
warn_n=0
skip_n=0
i=0
while [ "$i" -lt "$n" ]; do
  st="${STEP_STATUSES[$i]}"
  nm="${STEP_NAMES[$i]}"
  note="${STEP_NOTES[$i]}"
  padded_status="$(printf '%-6s' "$st")"
  case "$st" in
    PASS) colored="${C_GREEN}${padded_status}${C_RESET}"; pass_n=$((pass_n + 1)) ;;
    FAIL) colored="${C_RED}${padded_status}${C_RESET}";   fail_n=$((fail_n + 1)) ;;
    WARN) colored="${C_YELLOW}${padded_status}${C_RESET}"; warn_n=$((warn_n + 1)) ;;
    SKIP) colored="${C_YELLOW}${padded_status}${C_RESET}"; skip_n=$((skip_n + 1)) ;;
    *)    colored="${padded_status}" ;;
  esac
  if [ -n "$note" ]; then
    printf '%s  %s  %s(%s)%s\n' "$colored" "$nm" "$C_BLUE" "$note" "$C_RESET"
  else
    printf '%s  %s\n' "$colored" "$nm"
  fi
  i=$((i + 1))
done

printf '\n%d step(s): %d passed, %d failed, %d warned, %d skipped\n' \
  "$n" "$pass_n" "$fail_n" "$warn_n" "$skip_n"

if [ "$OVERALL_FAIL" -eq 1 ]; then
  fail_msg "ci-local: FAILED"
  exit 1
fi

ok "ci-local: all executed steps passed (see WARN/SKIP rows above, if any)"
exit 0
