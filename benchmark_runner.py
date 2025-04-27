# benchmark_runner.py
import os
import json
import datetime
import re
import time
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm
from tqdm import tqdm as sync_tqdm
from pathlib import Path
import traceback

# Import shared config and NEW central API caller
import config
from llm_api_calls import call_llm_api 


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


async def run_single_model_benchmark(model_config, benchmark_questions, base_prompts, results_dir):
    """Runs the full benchmark for one model configuration asynchronously and saves results."""
    model_id = model_config["id"]
    provider = model_config["provider"]
    model_name_reporting = model_config.get("model_name_api", model_id.split('/')[-1])
    access_type = model_config.get("access", "unknown") # Get access type or default

    sync_tqdm.write(f"---> Preparing Benchmark for: {model_id}")

    model_results = {
        "model_id": model_id,
        "model_name": model_name_reporting, # API model name for consistency
        "provider": provider,
        "access": access_type, # Store access type in results
        "date": datetime.datetime.now().isoformat(),
        "benchmark_file": config.BENCHMARK_QUESTIONS_FILE,
        "multiple_choice": {}
    }


    # Prepare Question Tasks
    tasks = []
    question_semaphore = Semaphore(config.QUESTIONS_CONCURRENCY_PER_MODEL)
    all_question_data = []
    for category, questions in benchmark_questions.items():
        model_results["multiple_choice"][category] = []
        if not isinstance(questions, list):
             sync_tqdm.write(f"\nWarning ({model_id}): Invalid question format for category '{category}'. Skipping.")
             continue # Skip malformed category
        for q_data in questions:
             if isinstance(q_data, dict) and 'id' in q_data: # Basic validation
                 all_question_data.append({"category": category, "data": q_data})
             else:
                 sync_tqdm.write(f"\nWarning ({model_id}): Skipping invalid question data in category '{category}': {str(q_data)[:100]}...")

    total_questions_count = len(all_question_data)
    if total_questions_count == 0:
        sync_tqdm.write(f"\n  Warning ({model_id}): No valid questions found to process. Saving empty results.")
        model_results["run_error"] = "No valid questions loaded or found."
        results_dir_path = Path(results_dir); results_dir_path.mkdir(parents=True, exist_ok=True)
        safe_filename = _get_safe_results_filename(model_id)
        results_filepath = results_dir_path / safe_filename
        try:
            with open(results_filepath, 'w') as f: json.dump(model_results, f, indent=4)
        except IOError as io_e: sync_tqdm.write(f"\n  Error saving empty results file {results_filepath}: {io_e}")

        return # Stop processing this model

    start_time = time.time()

    # Inner Helper for Single Question Processing
    async def process_single_question(category, question_data):
        task_id = f"Q_{question_data.get('id', 'N/A')}"
        async with question_semaphore:
            current_user_prompt = base_prompts["USER_PROMPT_TEMPLATE"].format(
                question=question_data.get('question', 'MISSING QUESTION'),
                answer_options=question_data.get('answer_options', {})
            )
            raw_answer = None
            processed_answer = "fail" # Default to fail
            last_error = None
            for attempt in range(config.MAX_RETRIES):
                try:
                    # --- USE THE NEW CENTRAL API CALLER ---
                    raw_answer = await call_llm_api(
                        model_config,
                        base_prompts["SYSTEM_PROMPT"],
                        current_user_prompt,
                        base_prompts.get("ASSISTANT_PROMPT") # Pass assistant prompt if available
                    )
                    # --- END NEW CALL ---

                    if raw_answer is not None:
                        parsed = parse_model_response(raw_answer)
                        if parsed != "fail":
                            processed_answer = parsed # Got a valid letter
                            break # Success
                        else:
                            processed_answer = "fail" # Still fail, record error for last attempt
                            last_error = f"Parsing failed on attempt {attempt + 1}. Raw: '{str(raw_answer)[:50]}...'"
                    else:
                        # API call returned None (error already logged by call_llm_api)
                        processed_answer = "error_api" # Mark as API error
                        last_error = f"API call failed or returned no content on attempt {attempt + 1} (see log above for details)."

                    # If we are here, it means the attempt failed (parse fail or API error)
                    if attempt < config.MAX_RETRIES - 1:
                         delay = 1.5 ** attempt # Exponential backoff
                         await asyncio.sleep(delay)
                    # On last attempt, processed_answer remains "fail" or "error_api"

                except Exception as e:
                    # Catch unexpected errors *around* the API call or processing
                    sync_tqdm.write(f"\n    {task_id}: Unexpected error during attempt {attempt + 1}: {e}\n{traceback.format_exc()}")
                    last_error = f"Unexpected error: {e}"; processed_answer = "error_unexpected"
                    # Optionally break here, or let retries continue if it might be transient
                    if attempt < config.MAX_RETRIES - 1: await asyncio.sleep(1) # Short delay before next retry


            result_entry = question_data.copy()
            result_entry["model_answer"] = processed_answer
            result_entry["raw_model_response"] = str(raw_answer) if raw_answer is not None else ""
            # Only record last_error if the final answer is an error/fail state
            if processed_answer in ["fail", "error_api", "error_provider", "error_unexpected", "error_task"]:
                 result_entry["last_error"] = last_error if last_error else "Unknown error state"
            return category, result_entry

    # Create Tasks and Process with tqdm
    tasks = [asyncio.create_task(process_single_question(item["category"], item["data"])) for item in all_question_data]
    position = model_config.get('_tqdm_position', 0)
    question_results_collected = []

    try: # Wrap the processing loop to catch potential tqdm issues
      async_as_completed_iterator = asyncio.as_completed(tasks)
      progress_bar_iterator = tqdm(
          async_as_completed_iterator, total=total_questions_count,
          desc=f"{model_id:<30}",
          unit="q", position=position, leave=False, ncols=100, ascii=" smooth" # Adjusted style
      )

      processed_count = 0
      with progress_bar_iterator as bar:
          for future in bar:
              try:
                  result = await future
                  question_results_collected.append(result)
              except Exception as e:
                  sync_tqdm.write(f"\n  Error captured processing future for {model_id}: {e}")
                  # Add a placeholder error result associated with a generic category or the first one
                  first_category = next(iter(benchmark_questions.keys()), "UNKNOWN_CATEGORY")
                  question_results_collected.append((first_category, {
                        "id": "TASK_ERROR", "question": "A concurrent task failed",
                        "model_answer": "error_task", "raw_model_response": str(e),
                        "last_error": f"Task failed: {e}"
                  }))
              processed_count += 1

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

    # Process collected results/exceptions
    category_counts = {cat: 0 for cat in model_results["multiple_choice"]}
    for item in question_results_collected:
        if isinstance(item, tuple) and len(item) == 2:
            category, result_entry = item
            if category in model_results["multiple_choice"]:
                model_results["multiple_choice"][category].append(result_entry)
                category_counts[category] += 1
            else:
                sync_tqdm.write(f"\n Warning: Processing result for unexpected category '{category}' in {model_id}. Storing under 'UNCATEGORIZED'.")
                if "UNCATEGORIZED" not in model_results["multiple_choice"]: model_results["multiple_choice"]["UNCATEGORIZED"] = []
                model_results["multiple_choice"]["UNCATEGORIZED"].append(result_entry)
        else: # Should mainly be the placeholder error added earlier if a future failed
            sync_tqdm.write(f"\n  Warning: Processing unexpected item type in collected results for {model_id}: {type(item)}")
            # Avoid crashing report generation for this model if possible
            if "PROCESSING_ERRORS" not in model_results["multiple_choice"]: model_results["multiple_choice"]["PROCESSING_ERRORS"] = []
            model_results["multiple_choice"]["PROCESSING_ERRORS"].append({ "id": "COLLECTION_ERROR", "model_answer": "error_task", "raw_model_response": str(item)})


    # Save Final Results
    end_time = time.time()
    duration = end_time - start_time
    sync_tqdm.write(f"  Finished {model_id}. Total time: {duration:.2f} seconds. Attempted {processed_count}/{total_questions_count} tasks.")

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    safe_filename = _get_safe_results_filename(model_id)
    results_filepath = results_dir_path / safe_filename
    try:
        with open(results_filepath, 'w') as f: json.dump(model_results, f, indent=4)
        sync_tqdm.write(f"  Results saved to: {results_filepath}")
    except IOError as e:
        sync_tqdm.write(f"  Error saving results file {results_filepath}: {e}")
    except TypeError as e:
         sync_tqdm.write(f"  Error: Could not serialize results to JSON for {model_id}: {e}. Dumping raw object:")
         sync_tqdm.write(str(model_results)) # Try printing the structure


# --- Task Execution Wrapper ---

async def run_model_with_semaphore(semaphore: Semaphore, model_config, benchmark_questions, base_prompts, results_dir):
    """Wrapper to run a single model's benchmark within semaphore limits."""
    model_id = model_config['id']
    access_type = model_config.get("access", "unknown") # Get access type here too
    async with semaphore:
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
        # No finally block needed here, semaphore is released automatically

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
    model_semaphore = Semaphore(config.MAX_CONCURRENT_MODELS)
    tasks = []
    for i, model_config in enumerate(models_to_run):
        model_config['_tqdm_position'] = i # Assign position for tqdm bar
        tasks.append(asyncio.create_task(
            run_model_with_semaphore(
                model_semaphore, model_config, benchmark_questions, base_prompts, results_dir
            )
        ))

    await asyncio.gather(*tasks)
    print("\n--- All Model Benchmark Tasks Completed ---")

