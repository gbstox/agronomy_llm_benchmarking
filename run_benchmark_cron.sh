#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/benchmark_results/cron_logs}"
LOCK_DIR="${LOCK_DIR:-$REPO_ROOT/.benchmark-cron.lock}"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[$TIMESTAMP] Missing Python interpreter at $PYTHON_BIN" >> "$LOG_DIR/runner.log"
    exit 1
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$TIMESTAMP] Skipping run because another benchmark job appears active." >> "$LOG_DIR/runner.log"
    exit 0
fi

cleanup() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"

echo "[$TIMESTAMP] Starting scheduled benchmark run." >> "$LOG_DIR/runner.log"
"$PYTHON_BIN" -u main.py >> "$LOG_DIR/benchmark-$(date '+%Y-%m-%d').log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished scheduled benchmark run." >> "$LOG_DIR/runner.log"
