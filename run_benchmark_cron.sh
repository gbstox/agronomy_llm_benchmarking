#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-$REPO_ROOT/run_benchmark_until_complete.sh}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/benchmark_results/cron_logs}"
LOCK_DIR="${LOCK_DIR:-$REPO_ROOT/.benchmark-cron.lock}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"

log_runner() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_DIR/runner.log"
}

resolve_python_bin() {
    if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi

    local candidate
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/venv/bin/python" \
        "$(command -v python3 2>/dev/null || true)" \
        "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    log_runner "Missing Python interpreter. Checked .venv, venv, python3, and python."
    exit 1
fi

if [[ ! -x "$RUNNER_SCRIPT" ]]; then
    log_runner "Missing resilient runner script at $RUNNER_SCRIPT."
    exit 1
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log_runner "Skipping run because another benchmark job appears active."
    exit 0
fi

cleanup() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    log_runner "Skipping scheduled run because tracked git changes are present."
    exit 0
fi

log_runner "Pulling latest changes from $GIT_REMOTE/$GIT_BRANCH."
if ! git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH" >> "$LOG_DIR/git-$(date '+%Y-%m-%d').log" 2>&1; then
    log_runner "git pull failed. See git log for details."
    exit 1
fi

log_runner "Starting scheduled resilient benchmark run with $PYTHON_BIN."
if ! PYTHON_BIN="$PYTHON_BIN" LOG_DIR="$LOG_DIR" "$RUNNER_SCRIPT"; then
    log_runner "Benchmark run failed. See benchmark log for details."
    exit 1
fi

git add README.md benchmark_results >> "$LOG_DIR/git-$(date '+%Y-%m-%d').log" 2>&1 || true
if git diff --cached --quiet; then
    log_runner "Scheduled benchmark finished with no tracked output changes to commit."
    exit 0
fi

log_runner "Committing updated benchmark outputs."
if ! git commit -m "$(cat <<'EOF'
Update automated benchmark results.

EOF
)" >> "$LOG_DIR/git-$(date '+%Y-%m-%d').log" 2>&1; then
    log_runner "git commit failed. See git log for details."
    exit 1
fi

log_runner "Pushing updated benchmark outputs to $GIT_REMOTE/$GIT_BRANCH."
if ! git push "$GIT_REMOTE" "$GIT_BRANCH" >> "$LOG_DIR/git-$(date '+%Y-%m-%d').log" 2>&1; then
    log_runner "git push failed. See git log for details."
    exit 1
fi

log_runner "Finished scheduled benchmark run and pushed tracked output changes."
