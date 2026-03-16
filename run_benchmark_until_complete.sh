#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/benchmark_results/cron_logs}"
LOCK_DIR="${LOCK_DIR:-$REPO_ROOT/.benchmark-resume.lock}"
HEARTBEAT_FILE="${HEARTBEAT_FILE:-$REPO_ROOT/benchmark_results/run_heartbeat.json}"
MAX_NO_PROGRESS_ATTEMPTS="${MAX_NO_PROGRESS_ATTEMPTS:-12}"
SLEEP_BETWEEN_ATTEMPTS_SECONDS="${SLEEP_BETWEEN_ATTEMPTS_SECONDS:-30}"
WATCHDOG_POLL_SECONDS="${WATCHDOG_POLL_SECONDS:-30}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1200}"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/resume-run-$(date '+%Y-%m-%d').log"

log_runner() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RUN_LOG"
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

refresh_reports() {
    "$PYTHON_BIN" - <<'PY'
import asyncio
import config
import report_generator

asyncio.run(
    report_generator.generate_reports(
        results_dir=config.BENCHMARK_RESULTS_DIR,
        graphs_base_dir=config.GRAPHS_BASE_DIR,
    )
)
PY
}

count_remaining_models() {
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import config
import main
import openrouter_catalog

repo_root = Path('.').resolve()
repo_benchmarked = main._load_repo_benchmarked_model_ids(repo_root / "README.md")
initial_models = list(config.MODELS_TO_RUN)
_, discovered_models = openrouter_catalog.build_new_openrouter_model_configs(
    project_root=repo_root,
    results_dir=config.BENCHMARK_RESULTS_DIR,
    api_key_env=config.OPENROUTER_DISCOVERY_API_KEY_ENV,
    existing_model_ids={model["id"] for model in initial_models},
)

remaining = 0
for model in initial_models + discovered_models:
    if main._get_model_identity_variants(model) & repo_benchmarked:
        continue
    results_path = Path(config.BENCHMARK_RESULTS_DIR) / main._get_safe_results_filename(model["id"])
    should_rerun, _ = main._should_rerun_result(results_path)
    if should_rerun:
        remaining += 1

print(remaining)
PY
}

PYTHON_BIN="$(resolve_python_bin || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    log_runner "Missing Python interpreter. Checked .venv, venv, python3, and python."
    exit 1
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log_runner "Another resume benchmark job appears active. Exiting."
    exit 0
fi

cleanup() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_ROOT"

remaining_before="$(count_remaining_models)"
log_runner "Starting resilient benchmark loop with $remaining_before models remaining."

no_progress_attempts=0
while [[ "$remaining_before" =~ ^[0-9]+$ ]] && (( remaining_before > 0 )); do
    log_runner "Launching benchmark pass with $remaining_before models remaining."
    rm -f "$HEARTBEAT_FILE"

    "$PYTHON_BIN" -u main.py >> "$RUN_LOG" 2>&1 &
    child_pid=$!
    pass_started_at_epoch="$(date +%s)"
    log_runner "Started benchmark worker pid $child_pid."

    pass_exit_code=0
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep "$WATCHDOG_POLL_SECONDS"

        now_epoch="$(date +%s)"
        if [[ -f "$HEARTBEAT_FILE" ]]; then
            heartbeat_epoch="$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || echo 0)"
            heartbeat_age="$(( now_epoch - heartbeat_epoch ))"
            if (( heartbeat_age > STALL_TIMEOUT_SECONDS )); then
                log_runner "Heartbeat stale for ${heartbeat_age}s; restarting worker pid $child_pid."
                kill "$child_pid" 2>/dev/null || true
                sleep 5
                kill -9 "$child_pid" 2>/dev/null || true
                wait "$child_pid" || true
                pass_exit_code=124
                break
            fi
        else
            startup_age="$(( now_epoch - pass_started_at_epoch ))"
            if (( startup_age > STALL_TIMEOUT_SECONDS )); then
                log_runner "No heartbeat created within ${startup_age}s; restarting worker pid $child_pid."
                kill "$child_pid" 2>/dev/null || true
                sleep 5
                kill -9 "$child_pid" 2>/dev/null || true
                wait "$child_pid" || true
                pass_exit_code=125
                break
            fi
        fi
    done

    if (( pass_exit_code == 0 )); then
        if wait "$child_pid"; then
            log_runner "Benchmark pass exited cleanly; will re-evaluate remaining queue."
        else
            pass_exit_code=$?
            log_runner "Benchmark pass exited non-zero with code $pass_exit_code; will re-evaluate remaining queue."
        fi
    fi

    log_runner "Refreshing README leaderboard and charts from latest saved results."
    if refresh_reports >> "$RUN_LOG" 2>&1; then
        log_runner "Report refresh completed."
    else
        log_runner "Report refresh failed; continuing to re-evaluate remaining queue."
    fi

    remaining_after="$(count_remaining_models)"
    log_runner "Remaining queue after pass: $remaining_after"

    if [[ ! "$remaining_after" =~ ^[0-9]+$ ]]; then
        log_runner "Could not determine remaining queue size. Exiting."
        exit 1
    fi

    if (( remaining_after == 0 )); then
        log_runner "Benchmark queue is fully complete."
        exit 0
    fi

    if (( remaining_after < remaining_before )); then
        no_progress_attempts=0
    else
        ((no_progress_attempts+=1))
        log_runner "No progress detected on this pass ($no_progress_attempts/$MAX_NO_PROGRESS_ATTEMPTS)."
    fi

    if (( no_progress_attempts >= MAX_NO_PROGRESS_ATTEMPTS )); then
        log_runner "Stopping after repeated no-progress passes. Manual intervention needed."
        exit 1
    fi

    remaining_before="$remaining_after"
    sleep "$SLEEP_BETWEEN_ATTEMPTS_SECONDS"
done

log_runner "Refreshing README leaderboard and charts from current results."
if refresh_reports >> "$RUN_LOG" 2>&1; then
    log_runner "Final report refresh completed."
else
    log_runner "Final report refresh failed."
    exit 1
fi

log_runner "Nothing remaining to benchmark."
