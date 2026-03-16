# benchmark_runner.py
import os
import json
import datetime
import re
import time
import random
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm
from pathlib import Path
import traceback

# Import shared config and NEW central API caller
import config
from llm_api_calls import (
    API_ERROR_SENTINEL,
    FATAL_API_ERROR_SENTINEL,
    RATE_LIMIT_SENTINEL,
    call_llm_api,
)


# --- Core Benchmark Logic ---

def load_benchmark_questions(filepath):
    """Loads and validates benchmark questions from a JSON file."""
    try:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark file not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        if "multiple_choice" not in data or not isinstance(data["multiple_choice"], dict):
             raise ValueError("Invalid format: 'multiple_choice' key missing or not a dictionary.")
        return data["multiple_choice"]
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading benchmark questions from {filepath}: {e}")
        return None

def parse_model_response(raw_response):
    """Cleans raw model output to extract a single-letter answer (a-z)."""
    if not raw_response or not isinstance(raw_response, str): return "fail"
    cleaned = raw_response.strip(' "\'*`\t\n')
    cleaned = re.sub(r"^(the correct answer is|answer is|correct answer:|answer:|key:)\s*", "", cleaned, flags=re.IGNORECASE).strip()
    match = re.match(r"^\s*([a-zA-Z])(?:[.)\s]|\b|$)", cleaned)
    if match: return match.group(1).lower()
    if len(cleaned) == 1 and cleaned.isalpha(): return cleaned.lower()
    return "fail"

def _get_safe_results_filename(model_id):
    """Generates a safe filename for results based on model ID."""
    return re.sub(r'[\\/*?:"<>|]', '_', model_id) + "_answers.json"


def _get_retry_delay(base_seconds, attempt_index):
    """Caps exponential backoff and adds light jitter to desynchronize retries."""
    delay = base_seconds * (2 ** max(attempt_index, 0))
    jittered = delay * random.uniform(0.85, 1.15)
    return min(jittered, config.RETRY_BACKOFF_MAX_SECONDS)


def _write_run_heartbeat(model_id, state, processed_questions=None, total_questions=None, details=None):
    """Writes a lightweight heartbeat file so the shell wrapper can detect stalls."""
    heartbeat_path = Path(config.RUN_HEARTBEAT_FILE)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_id": model_id,
        "state": state,
    }
    if processed_questions is not None:
        payload["processed_questions"] = processed_questions
    if total_questions is not None:
        payload["total_questions"] = total_questions
    if details is not None:
        payload["details"] = details

    temp_path = heartbeat_path.parent / (
        f"{heartbeat_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
        temp_path.replace(heartbeat_path)
    except OSError:
        try:
            temp_path.unlink()
        except OSError:
            pass


RETRYABLE_RESULT_ANSWERS = {
    "error_api",
    "error_provider",
    "error_unexpected",
    "error_task",
    "error_rate_limit",
}


def _make_question_key(category, question_identifier):
    if category is None or question_identifier is None:
        return None
    return f"{category}::{str(question_identifier)}"


def _is_terminal_result_entry(result_entry):
    model_answer = result_entry.get("model_answer")
    if not isinstance(model_answer, str):
        return False
    normalized = model_answer.strip().lower()
    if not normalized:
        return False
    if normalized == "fail":
        return True
    return len(normalized) == 1 and normalized.isalpha()


def _is_retryable_result_entry(result_entry):
    model_answer = result_entry.get("model_answer")
    if not isinstance(model_answer, str):
        return True
    normalized = model_answer.strip().lower()
    if normalized in RETRYABLE_RESULT_ANSWERS:
        return True
    if normalized == "error_fatal_api":
        return False
    return not _is_terminal_result_entry(result_entry)


def _build_multiple_choice_results(benchmark_questions, result_entries_by_key, extra_result_entries):
    multiple_choice = {category: [] for category in benchmark_questions}

    for category, questions in benchmark_questions.items():
        if not isinstance(questions, list):
            continue
        for question_data in questions:
            if not isinstance(question_data, dict):
                continue
            key = _make_question_key(category, question_data.get("id"))
            if key is None:
                continue
            result_entry = result_entries_by_key.get(key)
            if result_entry is not None:
                multiple_choice[category].append(result_entry)

    for category, result_entry in extra_result_entries:
        target_category = category if category in multiple_choice else "PROCESSING_ERRORS"
        multiple_choice.setdefault(target_category, []).append(result_entry)

    return multiple_choice


def _summarize_result_entries(result_entries_by_key, extra_result_entries):
    rate_limit_failures = 0
    api_failures = 0
    fatal_api_failures = 0

    all_entries = list(result_entries_by_key.values()) + [entry for _, entry in extra_result_entries]
    for entry in all_entries:
        if not isinstance(entry, dict):
            continue
        model_answer = entry.get("model_answer")
        if model_answer == "error_rate_limit":
            rate_limit_failures += 1
        elif model_answer == "error_api":
            api_failures += 1
        elif model_answer == "error_fatal_api":
            fatal_api_failures += 1

    return rate_limit_failures, api_failures, fatal_api_failures


def _write_model_results(results_filepath, model_results):
    temp_path = results_filepath.parent / (
        f"{results_filepath.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with open(temp_path, "w") as f:
        json.dump(model_results, f, indent=4)
    temp_path.replace(results_filepath)


def _count_total_questions(benchmark_questions):
    total_questions = 0
    for questions in benchmark_questions.values():
        if not isinstance(questions, list):
            continue
        total_questions += len(
            [question for question in questions if isinstance(question, dict) and "id" in question]
        )
    return total_questions


def _persist_model_results(
    results_filepath,
    model_results,
    benchmark_questions,
    result_entries_by_key,
    extra_result_entries,
    benchmark_status,
    processed_count,
    total_questions_count,
    checkpoint_reason=None,
):
    model_results["benchmark_status"] = benchmark_status
    model_results["multiple_choice"] = _build_multiple_choice_results(
        benchmark_questions,
        result_entries_by_key,
        extra_result_entries,
    )
    model_results["completed_questions"] = processed_count
    model_results["total_questions"] = total_questions_count
    if checkpoint_reason:
        model_results["checkpoint_reason"] = checkpoint_reason
    else:
        model_results.pop("checkpoint_reason", None)

    rate_limit_failures, api_failures, fatal_api_failures = _summarize_result_entries(
        result_entries_by_key,
        extra_result_entries,
    )
    model_results["rate_limit_failures"] = rate_limit_failures
    model_results["api_failures"] = api_failures
    model_results["fatal_api_failures"] = fatal_api_failures

    _write_model_results(results_filepath, model_results)


def _load_resume_result_entries(results_filepath, benchmark_questions):
    if not results_filepath.is_file():
        return {}

    try:
        existing_data = json.loads(results_filepath.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    if existing_data.get("retryable") is False:
        return {}

    valid_keys = set()
    for category, questions in benchmark_questions.items():
        if not isinstance(questions, list):
            continue
        for question_data in questions:
            if not isinstance(question_data, dict):
                continue
            key = _make_question_key(category, question_data.get("id"))
            if key is not None:
                valid_keys.add(key)

    resumed_entries = {}
    multiple_choice = existing_data.get("multiple_choice", {})
    if not isinstance(multiple_choice, dict):
        return resumed_entries

    for category, answers in multiple_choice.items():
        if not isinstance(answers, list):
            continue
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            key = _make_question_key(category, answer.get("id"))
            if key not in valid_keys:
                continue
            if _is_retryable_result_entry(answer):
                continue
            resumed_entries[key] = answer

    return resumed_entries


def _get_model_retry_state(model_config, benchmark_questions, results_dir):
    results_filepath = Path(results_dir) / _get_safe_results_filename(model_config["id"])
    total_questions_count = _count_total_questions(benchmark_questions)
    if not results_filepath.is_file():
        return {
            "should_retry": True,
            "status": "missing",
            "remaining_questions": total_questions_count,
        }

    try:
        existing_data = json.loads(results_filepath.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "should_retry": True,
            "status": "unreadable",
            "remaining_questions": total_questions_count,
        }

    if existing_data.get("retryable") is False or existing_data.get("benchmark_status") == "fatal_api_error":
        return {
            "should_retry": False,
            "status": existing_data.get("benchmark_status", "fatal_api_error"),
            "remaining_questions": 0,
        }

    remaining_questions = max(
        total_questions_count - len(_load_resume_result_entries(results_filepath, benchmark_questions)),
        0,
    )

    return {
        "should_retry": remaining_questions > 0,
        "status": existing_data.get("benchmark_status", "unknown"),
        "remaining_questions": remaining_questions,
    }


def _should_trip_circuit_breaker(
    attempted_count,
    terminal_count,
    transient_error_count,
    consecutive_transient_errors,
):
    if consecutive_transient_errors >= config.MODEL_CIRCUIT_BREAKER_MAX_CONSECUTIVE_TRANSIENT_ERRORS:
        return True, (
            "Circuit breaker triggered after "
            f"{consecutive_transient_errors} consecutive transient API failures."
        )

    if attempted_count < config.MODEL_CIRCUIT_BREAKER_MIN_ATTEMPTS:
        return False, None

    error_ratio = transient_error_count / max(attempted_count, 1)
    if (
        error_ratio >= config.MODEL_CIRCUIT_BREAKER_MAX_TRANSIENT_ERROR_RATIO
        and terminal_count < config.MODEL_CIRCUIT_BREAKER_MIN_TERMINAL_RESULTS
    ):
        return True, (
            "Circuit breaker triggered because transient API failures stayed above "
            f"{config.MODEL_CIRCUIT_BREAKER_MAX_TRANSIENT_ERROR_RATIO:.0%} "
            f"after {attempted_count} attempted questions."
        )

    return False, None


async def run_single_model_benchmark(model_config, benchmark_questions, base_prompts, results_dir):
    """Runs the full benchmark for one model configuration asynchronously and saves results."""
    model_id = model_config["id"]
    provider = model_config["provider"]
    model_name_reporting = model_config.get("model_name_api", model_id.split('/')[-1])
    access_type = model_config.get("access", "unknown") # Get access type or default
    question_concurrency = model_config.get("question_concurrency", config.QUESTIONS_CONCURRENCY_PER_MODEL)
    max_retries = model_config.get("max_retries", config.MAX_RETRIES)
    rate_limit_backoff_seconds = model_config.get("rate_limit_backoff_seconds", config.RATE_LIMIT_RETRY_BASE_DELAY_SECONDS)
    checkpoint_interval = max(1, int(config.CHECKPOINT_EVERY_N_QUESTIONS))

    sync_tqdm.write(f"---> Preparing Benchmark for: {model_id}")

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    safe_filename = _get_safe_results_filename(model_id)
    results_filepath = results_dir_path / safe_filename

    model_results = {
        "model_id": model_id,
        "model_name": model_name_reporting, # API model name for consistency
        "provider": provider,
        "access": access_type, # Store access type in results
        "date": datetime.datetime.now().isoformat(),
        "benchmark_file": config.BENCHMARK_QUESTIONS_FILE,
        "benchmark_status": "complete",
        "retryable": True,
        "question_concurrency": question_concurrency,
        "max_retries": max_retries,
        "multiple_choice": {},
        "resumed_from_checkpoint": False,
    }


    # Prepare Question Tasks
    question_semaphore = Semaphore(question_concurrency)
    all_question_data = []
    for category, questions in benchmark_questions.items():
        if not isinstance(questions, list):
             sync_tqdm.write(f"\nWarning ({model_id}): Invalid question format for category '{category}'. Skipping.")
             continue # Skip malformed category
        for q_data in questions:
             if isinstance(q_data, dict) and 'id' in q_data: # Basic validation
                 question_key = _make_question_key(category, q_data.get("id"))
                 if question_key is None:
                     continue
                 all_question_data.append({
                     "category": category,
                     "data": q_data,
                     "question_key": question_key,
                 })
             else:
                 sync_tqdm.write(f"\nWarning ({model_id}): Skipping invalid question data in category '{category}': {str(q_data)[:100]}...")

    result_entries_by_key = _load_resume_result_entries(results_filepath, benchmark_questions)
    resumed_count = len(result_entries_by_key)
    if resumed_count:
        model_results["resumed_from_checkpoint"] = True
        sync_tqdm.write(
            f"  -> Resuming {model_id} from checkpoint with {resumed_count} terminal answers already saved."
        )

    remaining_question_data = [
        item for item in all_question_data if item["question_key"] not in result_entries_by_key
    ]
    total_questions_count = len(all_question_data)
    if total_questions_count == 0:
        sync_tqdm.write(f"\n  Warning ({model_id}): No valid questions found to process. Saving empty results.")
        model_results["benchmark_status"] = "run_error"
        model_results["run_error"] = "No valid questions loaded or found."
        try:
            _write_model_results(results_filepath, model_results)
        except IOError as io_e: sync_tqdm.write(f"\n  Error saving empty results file {results_filepath}: {io_e}")

        return # Stop processing this model

    start_time = time.time()
    extra_result_entries = []
    processed_count = resumed_count
    _persist_model_results(
        results_filepath,
        model_results,
        benchmark_questions,
        result_entries_by_key,
        extra_result_entries,
        benchmark_status="in_progress",
        processed_count=processed_count,
        total_questions_count=total_questions_count,
        checkpoint_reason="Initial checkpoint before remaining questions run.",
    )
    _write_run_heartbeat(
        model_id,
        "model_started",
        processed_count,
        total_questions_count,
        {"resumed_from_checkpoint": resumed_count > 0},
    )

    # Inner Helper for Single Question Processing
    async def process_single_question(category, question_data, question_key):
        task_id = f"Q_{question_data.get('id', 'N/A')}"
        async with question_semaphore:
            current_user_prompt = base_prompts["USER_PROMPT_TEMPLATE"].format(
                question=question_data.get('question', 'MISSING QUESTION'),
                answer_options=question_data.get('answer_options', {})
            )
            raw_answer = None
            processed_answer = "fail" # Default to fail
            last_error = None
            api_error_attempts = 0
            parse_fail_attempts = 0
            rate_limit_attempts = 0
            for attempt in range(max_retries):
                try:
                    # --- USE THE NEW CENTRAL API CALLER ---
                    raw_answer = await call_llm_api(
                        model_config,
                        base_prompts["SYSTEM_PROMPT"],
                        current_user_prompt,
                        base_prompts.get("ASSISTANT_PROMPT") # Pass assistant prompt if available
                    )
                    # --- END NEW CALL ---

                    if raw_answer == RATE_LIMIT_SENTINEL:
                        rate_limit_attempts += 1
                        processed_answer = "error_rate_limit"
                        last_error = f"Rate limited on attempt {attempt + 1}."
                        if rate_limit_attempts >= max_retries:
                            break
                        delay = _get_retry_delay(
                            rate_limit_backoff_seconds,
                            rate_limit_attempts - 1,
                        )
                    elif raw_answer == FATAL_API_ERROR_SENTINEL:
                        processed_answer = "error_fatal_api"
                        last_error = "Encountered non-retryable API/provider error."
                        break
                    elif raw_answer not in {None, API_ERROR_SENTINEL}:
                        parsed = parse_model_response(raw_answer)
                        if parsed != "fail":
                            processed_answer = parsed # Got a valid letter
                            break # Success
                        else:
                            parse_fail_attempts += 1
                            processed_answer = "fail" # Still fail, record error for last attempt
                            last_error = f"Parsing failed on attempt {attempt + 1}. Raw: '{str(raw_answer)[:50]}...'"
                            if parse_fail_attempts >= config.PARSE_FAILURE_MAX_RETRIES:
                                break
                            delay = _get_retry_delay(
                                config.PARSE_FAILURE_RETRY_BASE_DELAY_SECONDS,
                                parse_fail_attempts - 1,
                            )
                    else:
                        # API call returned None (error already logged by call_llm_api)
                        api_error_attempts += 1
                        processed_answer = "error_api" # Mark as API error
                        last_error = f"API call failed or returned no content on attempt {attempt + 1} (see log above for details)."
                        if api_error_attempts >= config.API_ERROR_MAX_RETRIES:
                            break
                        delay = _get_retry_delay(
                            config.API_ERROR_RETRY_BASE_DELAY_SECONDS,
                            api_error_attempts - 1,
                        )

                    # If we are here, it means the attempt failed (parse fail or API error)
                    if attempt < max_retries - 1 and processed_answer != "error_fatal_api":
                        await asyncio.sleep(delay)
                    # On last attempt, processed_answer remains "fail" or "error_api"

                except Exception as e:
                    # Catch unexpected errors *around* the API call or processing
                    sync_tqdm.write(f"\n    {task_id}: Unexpected error during attempt {attempt + 1}: {e}\n{traceback.format_exc()}")
                    last_error = f"Unexpected error: {e}"; processed_answer = "error_unexpected"
                    # Optionally break here, or let retries continue if it might be transient
                    if attempt < max_retries - 1: await asyncio.sleep(1) # Short delay before next retry


            result_entry = question_data.copy()
            result_entry["model_answer"] = processed_answer
            result_entry["raw_model_response"] = str(raw_answer) if raw_answer is not None else ""
            # Only record last_error if the final answer is an error/fail state
            if processed_answer in ["fail", "error_api", "error_provider", "error_unexpected", "error_task", "error_rate_limit", "error_fatal_api"]:
                 result_entry["last_error"] = last_error if last_error else "Unknown error state"
            return question_key, category, result_entry

    # Create Tasks and Process with tqdm
    tasks = [
        asyncio.create_task(
            process_single_question(
                item["category"],
                item["data"],
                item["question_key"],
            )
        )
        for item in remaining_question_data
    ]
    position = model_config.get('_tqdm_position', 0)
    newly_attempted_count = 0
    current_pass_terminal_count = 0
    current_pass_transient_error_count = 0
    current_pass_consecutive_transient_errors = 0

    try: # Wrap the processing loop to catch potential tqdm issues
      async_as_completed_iterator = asyncio.as_completed(tasks)
      progress_bar_iterator = tqdm(
          async_as_completed_iterator, total=total_questions_count,
          desc=f"{model_id:<30}", initial=processed_count,
          unit="q", position=position, leave=False, ncols=100, ascii=" smooth" # Adjusted style
      )

      with progress_bar_iterator as bar:
          for future in bar:
              try:
                  question_key, category, result_entry = await future
              except Exception as e:
                  sync_tqdm.write(f"\n  Error captured processing future for {model_id}: {e}")
                  category = "PROCESSING_ERRORS"
                  question_key = None
                  result_entry = {
                        "id": f"TASK_ERROR_{time.time_ns()}",
                        "question": "A concurrent task failed",
                        "model_answer": "error_task", "raw_model_response": str(e),
                        "last_error": f"Task failed: {e}"
                  }

              if question_key is not None:
                  result_entries_by_key[question_key] = result_entry
              else:
                  extra_result_entries.append((category, result_entry))

              processed_count += 1
              newly_attempted_count += 1
              if (
                  processed_count % config.RUN_HEARTBEAT_EVERY_N_QUESTIONS == 0
                  or processed_count == total_questions_count
              ):
                  _write_run_heartbeat(
                      model_id,
                      "model_running",
                      processed_count,
                      total_questions_count,
                  )

              model_answer = result_entry.get("model_answer")
              if model_answer in RETRYABLE_RESULT_ANSWERS:
                  current_pass_transient_error_count += 1
                  current_pass_consecutive_transient_errors += 1
              else:
                  current_pass_consecutive_transient_errors = 0
                  if _is_terminal_result_entry(result_entry):
                      current_pass_terminal_count += 1

              if processed_count % checkpoint_interval == 0 or processed_count == total_questions_count:
                  _persist_model_results(
                      results_filepath,
                      model_results,
                      benchmark_questions,
                      result_entries_by_key,
                      extra_result_entries,
                      benchmark_status="in_progress",
                      processed_count=processed_count,
                      total_questions_count=total_questions_count,
                      checkpoint_reason="Periodic checkpoint during benchmark run.",
                  )

              if model_answer == "error_fatal_api":
                  model_results["run_error"] = result_entry.get(
                      "last_error",
                      "Encountered a non-retryable API/provider error.",
                  )
                  model_results["retryable"] = False
                  sync_tqdm.write(
                      f"\n  Fatal API error for {model_id}; cancelling remaining questions."
                  )
                  for task in tasks:
                      if not task.done():
                          task.cancel()
                  await asyncio.gather(*tasks, return_exceptions=True)
                  break

              should_break, break_reason = _should_trip_circuit_breaker(
                  attempted_count=newly_attempted_count,
                  terminal_count=current_pass_terminal_count,
                  transient_error_count=current_pass_transient_error_count,
                  consecutive_transient_errors=current_pass_consecutive_transient_errors,
              )
              if should_break:
                  model_results["run_error"] = break_reason
                  sync_tqdm.write(f"\n  {break_reason} Aborting remaining questions for {model_id}.")
                  for task in tasks:
                      if not task.done():
                          task.cancel()
                  await asyncio.gather(*tasks, return_exceptions=True)
                  break

    except Exception as e:
        sync_tqdm.write(f"\n!!!! Critical error during task processing loop for {model_id}: {e} !!!!")
        model_results["run_error"] = f"Task processing loop error: {e}"
        # Ensure remaining tasks are cancelled if the loop breaks
        for task in tasks:
            if not task.done(): task.cancel()
        # Gather cancelled tasks to avoid pending task warnings
        await asyncio.gather(*tasks, return_exceptions=True)


    if processed_count != total_questions_count and "run_error" not in model_results:
         sync_tqdm.write(f"\nWarning for {model_id}: Processed count ({processed_count}) != total tasks ({total_questions_count}).")
         # Don't mark as run_error unless catastrophic failure happened

    sync_tqdm.write(f"<--- Finished collecting results for: {model_id}") # Use sync_tqdm

    rate_limit_failures, api_failures, fatal_api_failures = _summarize_result_entries(
        result_entries_by_key,
        extra_result_entries,
    )

    if (
        "run_error" not in model_results
        and api_failures == total_questions_count
        and total_questions_count > 0
    ):
        model_results["run_error"] = "All benchmark questions failed at the API layer."

    if model_results.get("retryable") is False:
        model_results["benchmark_status"] = "fatal_api_error"
    elif "run_error" in model_results:
        model_results["benchmark_status"] = "run_error"
    elif rate_limit_failures > 0:
        model_results["benchmark_status"] = "rate_limited"
    elif api_failures > 0 or fatal_api_failures > 0:
        model_results["benchmark_status"] = "incomplete"
    elif processed_count != total_questions_count:
        model_results["benchmark_status"] = "incomplete"
    else:
        model_results["benchmark_status"] = "complete"

    model_results["rate_limit_failures"] = rate_limit_failures
    model_results["api_failures"] = api_failures
    model_results["fatal_api_failures"] = fatal_api_failures


    # Save Final Results
    end_time = time.time()
    duration = end_time - start_time
    model_results["multiple_choice"] = _build_multiple_choice_results(
        benchmark_questions,
        result_entries_by_key,
        extra_result_entries,
    )
    model_results["completed_questions"] = processed_count
    model_results["total_questions"] = total_questions_count
    model_results.pop("checkpoint_reason", None)
    sync_tqdm.write(f"  Finished {model_id}. Total time: {duration:.2f} seconds. Attempted {processed_count}/{total_questions_count} tasks.")

    try:
        _write_model_results(results_filepath, model_results)
        sync_tqdm.write(f"  Results saved to: {results_filepath}")
        _write_run_heartbeat(
            model_id,
            "model_finished",
            processed_count,
            total_questions_count,
            {"benchmark_status": model_results["benchmark_status"]},
        )
    except IOError as e:
        sync_tqdm.write(f"  Error saving results file {results_filepath}: {e}")
    except TypeError as e:
         sync_tqdm.write(f"  Error: Could not serialize results to JSON for {model_id}: {e}. Dumping raw object:")
         sync_tqdm.write(str(model_results)) # Try printing the structure


async def _run_model_task(model_config, benchmark_questions, base_prompts, results_dir):
    """Runs a single model benchmark and persists any outer task errors."""
    model_id = model_config['id']
    access_type = model_config.get("access", "unknown") # Get access type here too
    try:
        await run_single_model_benchmark(
            model_config, benchmark_questions, base_prompts, results_dir
        )
    except Exception as e:
        sync_tqdm.write(f"\n!!!!!!! CRITICAL ERROR running benchmark task for {model_id}: {e} !!!!!!!\n{traceback.format_exc()}")
        # Save error marker if a critical error occurs outside run_single_model_benchmark
        results_dir_path = Path(results_dir); results_dir_path.mkdir(parents=True, exist_ok=True)
        safe_filename = _get_safe_results_filename(model_id)
        results_filepath = results_dir_path / safe_filename
        error_data = {
            "model_id": model_id,
            "model_name": model_config.get("model_name_api", model_id),
            "provider": model_config.get("provider"),
            "access": access_type, # Include access type in error report
            "benchmark_status": "run_error",
            "retryable": True,
            "run_error": f"Critical task execution error: {type(e).__name__}: {e}",
            "date": datetime.datetime.now().isoformat(),
            "benchmark_file": config.BENCHMARK_QUESTIONS_FILE,
            "multiple_choice": {}
        }
        try:
            with open(results_filepath, 'w') as f: json.dump(error_data, f, indent=4)
            sync_tqdm.write(f"      Saved critical error marker to: {results_filepath}")
        except IOError as io_e:
            sync_tqdm.write(f"      Failed to save critical error marker for {model_id}: {io_e}")


async def run_model_with_semaphore(semaphore: Semaphore, model_config, benchmark_questions, base_prompts, results_dir):
    """Backward-compatible wrapper to run a single model inside a semaphore."""
    async with semaphore:
        await _run_model_task(model_config, benchmark_questions, base_prompts, results_dir)


async def _model_queue_worker(
    worker_idx,
    model_queue,
    benchmark_questions,
    base_prompts,
    results_dir,
    model_states,
):
    while True:
        model_config = await model_queue.get()
        try:
            if model_config is None:
                return

            model_id = model_config["id"]
            worker_model_config = dict(model_config)
            worker_model_config["_tqdm_position"] = worker_idx
            await _run_model_task(
                worker_model_config,
                benchmark_questions,
                base_prompts,
                results_dir,
            )

            retry_state = _get_model_retry_state(model_config, benchmark_questions, results_dir)
            model_state = model_states.setdefault(
                model_id,
                {
                    "attempts": 0,
                    "stagnant_requeues": 0,
                    "last_remaining_questions": retry_state["remaining_questions"],
                },
            )
            model_state["attempts"] += 1

            if not retry_state["should_retry"]:
                continue

            previous_remaining = model_state.get("last_remaining_questions", retry_state["remaining_questions"])
            remaining_questions = retry_state["remaining_questions"]
            made_progress = remaining_questions < previous_remaining
            if made_progress:
                model_state["stagnant_requeues"] = 0
            else:
                model_state["stagnant_requeues"] += 1
            model_state["last_remaining_questions"] = remaining_questions

            if (
                model_state["attempts"] < config.MODEL_MAX_IN_PROCESS_ATTEMPTS
                and model_state["stagnant_requeues"] <= config.MODEL_MAX_STAGNANT_REQUEUES
            ):
                sync_tqdm.write(
                    f"  Requeueing {model_id} immediately "
                    f"({remaining_questions} questions still retryable)."
                )
                await model_queue.put(model_config)
            else:
                sync_tqdm.write(
                    f"  Leaving {model_id} for a later wrapper pass "
                    f"(remaining={remaining_questions}, attempts={model_state['attempts']}, "
                    f"stagnant_requeues={model_state['stagnant_requeues']})."
                )
        finally:
            model_queue.task_done()

# --- Main Runner Function ---

async def run_benchmarks(models_to_run, benchmark_questions_file, results_dir):
    """Loads questions and runs benchmarks for the specified models."""
    if not models_to_run:
         print("No models selected to run.")
         return

    benchmark_questions = load_benchmark_questions(benchmark_questions_file)
    if not benchmark_questions:
        print("Critical Error: Could not load benchmark questions. Aborting benchmark run.")
        return

    base_prompts = {
        "SYSTEM_PROMPT": config.SYSTEM_PROMPT,
        "USER_PROMPT_TEMPLATE": config.USER_PROMPT_TEMPLATE,
        "ASSISTANT_PROMPT": config.ASSISTANT_PROMPT
    }

    print(f"\nStarting benchmarks for {len(models_to_run)} models with max concurrency {config.MAX_CONCURRENT_MODELS}...")
    worker_count = min(config.MAX_CONCURRENT_MODELS, len(models_to_run))
    model_queue = asyncio.Queue()
    model_states = {}
    for model_config in models_to_run:
        retry_state = _get_model_retry_state(model_config, benchmark_questions, results_dir)
        model_states[model_config["id"]] = {
            "attempts": 0,
            "stagnant_requeues": 0,
            "last_remaining_questions": retry_state["remaining_questions"],
        }
        await model_queue.put(model_config)

    workers = [
        asyncio.create_task(
            _model_queue_worker(
                worker_idx,
                model_queue,
                benchmark_questions,
                base_prompts,
                results_dir,
                model_states,
            )
        )
        for worker_idx in range(worker_count)
    ]

    await model_queue.join()
    for _ in workers:
        await model_queue.put(None)
    await asyncio.gather(*workers)
    print("\n--- All Model Benchmark Tasks Completed ---")

