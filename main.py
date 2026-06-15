# main.py
import asyncio
import json
import time
import datetime
from pathlib import Path
import traceback
import re # Keep re for _get_safe_results_filename

# Import configurations and functions from other modules
import config
import benchmark_runner
import openrouter_catalog
import report_generator

def _get_safe_results_filename(model_id):
    """Generates a safe filename for results based on model ID.
       Needed here for the 'missing' mode check.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', model_id) + "_answers.json"


def _get_model_identity_variants(model_config):
    """Returns equivalent identifiers used across config, results, and README."""
    variants = set()
    for value in (
        model_config.get("id"),
        model_config.get("model_name_api"),
    ):
        if not value or not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        variants.add(normalized)
        variants.add(normalized.lower())
        variants.add(normalized.split("/")[-1])
        variants.add(normalized.split("/")[-1].lower())
    return variants


def _load_repo_benchmarked_model_ids(readme_path: Path):
    """Parses the tracked README leaderboard to find already-benchmarked models."""
    if not readme_path.exists():
        return set()

    benchmarked_ids = set()
    try:
        with open(readme_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line.startswith("|"):
                    continue
                columns = [column.strip() for column in line.split("|")[1:-1]]
                if not columns:
                    continue
                model_name = columns[0]
                if (
                    not model_name
                    or model_name == "Model Name"
                    or set(model_name) == {"-"}
                ):
                    continue

                benchmarked_ids.add(model_name)
                benchmarked_ids.add(model_name.lower())
                benchmarked_ids.add(model_name.split("/")[-1])
                benchmarked_ids.add(model_name.split("/")[-1].lower())
    except OSError as e:
        print(f"\n--- README Benchmark History Warning: Could not read {readme_path}: {e} ---")
        return set()

    return benchmarked_ids


def _should_rerun_result(results_filepath: Path):
    """Decides whether an existing results file should be rerun."""
    if not results_filepath.exists():
        return True, "No results file found"

    try:
        with open(results_filepath, "r") as f:
            existing_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return True, f"Existing results unreadable: {e}"

    status = existing_data.get("benchmark_status")
    completed_questions = existing_data.get("completed_questions")
    total_questions = existing_data.get("total_questions")
    if existing_data.get("retryable") is False or status == "fatal_api_error":
        return False, "Existing results marked non-retryable"
    if status in {"in_progress", "rate_limited", "incomplete", "run_error"}:
        api_failures = int(existing_data.get("api_failures", 0) or 0)
        fatal_api_failures = int(existing_data.get("fatal_api_failures", 0) or 0)
        if (
            status == "in_progress"
            and isinstance(completed_questions, int)
            and isinstance(total_questions, int)
            and total_questions > 0
            and completed_questions >= total_questions
            and api_failures == 0
            and fatal_api_failures == 0
            and "run_error" not in existing_data
        ):
            return False, "Existing results already finished all questions"
        return True, f"Existing results marked '{status}'"
    if "run_error" in existing_data:
        return True, "Existing results contain run_error"
    api_failures = int(existing_data.get("api_failures", 0) or 0)
    fatal_api_failures = int(existing_data.get("fatal_api_failures", 0) or 0)
    if api_failures > 0 or fatal_api_failures > 0:
        return True, "Existing results contain API failures"

    return False, "Results file exists"


def _read_result_json(results_filepath: Path):
    try:
        return json.loads(results_filepath.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _expected_question_keys(benchmark_questions, supports_vision):
    """Question keys a model is expected to answer (image questions need vision)."""
    keys = set()
    for category, questions in (benchmark_questions or {}).items():
        if not isinstance(questions, list):
            continue
        for q_data in questions:
            if not isinstance(q_data, dict) or "id" not in q_data:
                continue
            if q_data.get("image") and not supports_vision:
                continue
            key = benchmark_runner._make_question_key(category, q_data.get("id"))
            if key is not None:
                keys.add(key)
    return keys


def _image_question_keys(benchmark_questions):
    """Question keys that require image input."""
    keys = set()
    for category, questions in (benchmark_questions or {}).items():
        if not isinstance(questions, list):
            continue
        for q_data in questions:
            if isinstance(q_data, dict) and q_data.get("image") and "id" in q_data:
                key = benchmark_runner._make_question_key(category, q_data.get("id"))
                if key is not None:
                    keys.add(key)
    return keys


def _terminal_answered_keys(results_filepath: Path, benchmark_questions):
    """Question keys already answered with a terminal (non-retryable) result."""
    try:
        return set(
            benchmark_runner._load_resume_result_entries(
                results_filepath, benchmark_questions
            ).keys()
        )
    except Exception:
        return set()


def _decide_model_run(model_config, results_filepath: Path, benchmark_questions, in_readme):
    """Decides whether to run a model in 'missing' mode.

    Generalized over the question set: a model is run when it is missing terminal
    answers for any question it is expected to answer. This naturally picks up
    newly added questions (e.g. image questions for vision-capable models) for
    models that already have a results file, while leaving fully-answered models
    and documented-but-not-local models untouched.
    """
    supports_vision = bool(model_config.get("supports_vision", False))
    expected_keys = _expected_question_keys(benchmark_questions, supports_vision)

    if not results_filepath.exists():
        if in_readme:
            return False, "Already benchmarked in tracked repo leaderboard (no local answers)"
        if not expected_keys:
            return False, "No applicable questions to run"
        return True, "No results file found"

    data = _read_result_json(results_filepath)
    if data.get("retryable") is False or data.get("benchmark_status") == "fatal_api_error":
        return False, "Existing results marked non-retryable"

    answered_keys = _terminal_answered_keys(results_filepath, benchmark_questions)
    missing_keys = expected_keys - answered_keys
    if not missing_keys:
        return False, "All applicable questions already answered"

    if getattr(config, "IMAGE_BACKFILL_ONLY", False):
        image_keys = _image_question_keys(benchmark_questions)
        missing_image = missing_keys & image_keys
        missing_non_image = missing_keys - image_keys
        if missing_image and not missing_non_image:
            return True, f"Image backfill: {len(missing_image)} image question(s)"
        return False, "Image-backfill mode: skipping (text questions missing or no images)"

    return True, f"Missing {len(missing_keys)} question(s) (resume/append)"


def _build_rerun_configs_from_results(results_dir, existing_ids):
    """Reconstructs OpenRouter model configs from existing result files.

    Previously-benchmarked models (e.g. auto-discovered ones) are not present in
    MODELS_TO_RUN, so without this they could never be re-selected to answer
    newly added questions (such as image questions). Only OpenRouter-backed
    ('openai_compatible') results are reconstructed; non-vision models will be
    filtered out later by the missing-question / vision-gating logic.
    """
    rerun_configs = []
    seen = set(existing_ids)
    results_path = Path(results_dir)
    for result_file in sorted(results_path.glob("*_answers.json")):
        data = _read_result_json(result_file)
        model_id = data.get("model_id")
        if not model_id or model_id in seen:
            continue
        if data.get("provider") != "openai_compatible":
            continue
        model_name_api = data.get("model_name") or model_id
        rerun_configs.append({
            "id": model_id,
            "provider": "openai_compatible",
            "api_key_env": config.OPENROUTER_DISCOVERY_API_KEY_ENV,
            "base_url": "https://openrouter.ai/api/v1",
            "model_name_api": model_name_api,
            "access": data.get("access", "unknown"),
        })
        seen.add(model_id)
    return rerun_configs


def _apply_model_allowlist(model_configs, allowlist):
    """Filters model configs to an explicit allowlist when provided."""
    if not allowlist:
        return model_configs

    allowlist_set = set(allowlist)
    filtered_configs = [model for model in model_configs if model["id"] in allowlist_set]
    missing_allowlist_ids = [model_id for model_id in allowlist if model_id not in {model["id"] for model in model_configs}]

    print(f"\n--- Model Allowlist: Keeping {len(filtered_configs)} explicitly selected models. ---")
    for model_config in filtered_configs:
        print(f"  -> Allowed: {model_config['id']}")
    for missing_id in missing_allowlist_ids:
        print(f"  -> Warning: Allowlisted model not found in selected catalog: {missing_id}")

    return filtered_configs

async def main():
    """Main async function to orchestrate the benchmark run and report generation."""
    start_run_time = time.time()
    print(f"================================= Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =================================")
    print(f" Starting Agronomy Benchmark Run - Mode: {config.RUN_MODE.upper()}")
    print(f" Results Dir: {config.BENCHMARK_RESULTS_DIR}")
    print(f" Graphs Dir:  {config.GRAPHS_BASE_DIR}")
    print("==============================================================================================")


    # --- Create Directories ---
    try:
        results_path = Path(config.BENCHMARK_RESULTS_DIR)
        graphs_path = Path(config.GRAPHS_BASE_DIR)
        individual_graphs_path = graphs_path / 'individual_graphs'

        results_path.mkdir(parents=True, exist_ok=True)
        graphs_path.mkdir(parents=True, exist_ok=True)
        individual_graphs_path.mkdir(parents=True, exist_ok=True)
        print("Required directories ensured.")
    except OSError as e:
        print(f"CRITICAL ERROR: Could not create necessary directories: {e}. Check permissions. Exiting.")
        return

    # --- Benchmark Run (Conditional) ---
    models_to_run_this_session = []
    if config.RUN_MODE != 'reports_only':
        initial_models_to_run = list(config.MODELS_TO_RUN)
        repo_benchmarked_model_ids = _load_repo_benchmarked_model_ids(Path("README.md"))
        selection_mode = getattr(config, "MODEL_SELECTION_MODE", "configured_only")
        print(f"\n--- Model Selection Mode: {selection_mode} ---")

        if selection_mode == "configured_plus_new_openrouter":
            try:
                latest_result_date, discovered_openrouter_models = openrouter_catalog.build_new_openrouter_model_configs(
                    project_root=Path(__file__).resolve().parent,
                    results_dir=config.BENCHMARK_RESULTS_DIR,
                    api_key_env=config.OPENROUTER_DISCOVERY_API_KEY_ENV,
                    existing_model_ids={model["id"] for model in initial_models_to_run},
                )
                if discovered_openrouter_models:
                    print(
                        f"\n--- OpenRouter Discovery: Found {len(discovered_openrouter_models)} new text models "
                        f"since {latest_result_date.strftime('%Y-%m-%d %H:%M:%S')}. ---"
                    )
                    for model_config in discovered_openrouter_models:
                        print(f"  -> Added discovered model: {model_config['id']}")
                    initial_models_to_run.extend(discovered_openrouter_models)
                else:
                    print(
                        f"\n--- OpenRouter Discovery: No new text models found since "
                        f"{latest_result_date.strftime('%Y-%m-%d %H:%M:%S')}. ---"
                    )
            except Exception as e:
                print(f"\n--- OpenRouter Discovery Warning: {type(e).__name__} - {e}. Continuing with configured models only. ---")
        elif selection_mode != "configured_only":
            print(f"\n--- Model Selection Warning: Unknown MODEL_SELECTION_MODE '{selection_mode}'. Falling back to configured_only. ---")

        # In 'missing' mode, also reconstruct configs for already-benchmarked models
        # so they can be re-selected to answer newly added questions (e.g. images).
        if config.RUN_MODE == 'missing':
            rerun_configs = _build_rerun_configs_from_results(
                config.BENCHMARK_RESULTS_DIR,
                {model["id"] for model in initial_models_to_run},
            )
            if rerun_configs:
                print(
                    f"\n--- Reconstructed {len(rerun_configs)} model config(s) from existing "
                    "results for possible new-question reruns. ---"
                )
                initial_models_to_run.extend(rerun_configs)

        initial_models_to_run = _apply_model_allowlist(
            initial_models_to_run,
            getattr(config, "MODEL_ID_ALLOWLIST", []),
        )

        # Tag each model with vision capability so image questions can be gated.
        openrouter_catalog.annotate_vision_support(
            initial_models_to_run,
            project_root=Path(__file__).resolve().parent,
            api_key_env=config.OPENROUTER_DISCOVERY_API_KEY_ENV,
        )

        if config.RUN_MODE == 'all':
            print(f"\n--- Mode 'ALL': Preparing to run all {len(initial_models_to_run)} configured models. ---")
            models_to_run_this_session = initial_models_to_run
        elif config.RUN_MODE == 'missing':
            print("\n--- Mode 'MISSING': Checking for existing results... ---")
            results_dir_path = Path(config.BENCHMARK_RESULTS_DIR)
            benchmark_questions_for_selection = (
                benchmark_runner.load_benchmark_questions(config.BENCHMARK_QUESTIONS_FILE) or {}
            )
            for model_config in initial_models_to_run:
                in_readme = bool(
                    _get_model_identity_variants(model_config) & repo_benchmarked_model_ids
                )
                safe_filename = _get_safe_results_filename(model_config['id'])
                results_filepath = results_dir_path / safe_filename
                should_run, reason = _decide_model_run(
                    model_config,
                    results_filepath,
                    benchmark_questions_for_selection,
                    in_readme,
                )
                if should_run:
                    vision_tag = " [vision]" if model_config.get("supports_vision") else ""
                    print(f"  -> Will run: {model_config['id']}{vision_tag} ({reason})")
                    models_to_run_this_session.append(model_config)
                else:
                    print(f"  -> Skipping: {model_config['id']} ({reason})")
            if not models_to_run_this_session:
                 print("--- All configured models already have results. No new benchmarks to run. ---")

        # Execute benchmark runs if any models were selected``
        if models_to_run_this_session:
            await benchmark_runner.run_benchmarks(
                models_to_run=models_to_run_this_session,
                benchmark_questions_file=config.BENCHMARK_QUESTIONS_FILE,
                results_dir=config.BENCHMARK_RESULTS_DIR
            )
        #else: Proceed directly to reports if no models needed running in 'missing' mode
    else:
        print("\n--- Mode 'REPORTS_ONLY': Skipping model benchmarks. ---")

    # --- Report Generation ---
    # Always generate reports based on files present in results_dir
    await report_generator.generate_reports(
        results_dir=config.BENCHMARK_RESULTS_DIR,
        graphs_base_dir=config.GRAPHS_BASE_DIR
    )

    # --- Finalization ---
    end_run_time = time.time()
    total_duration = end_run_time - start_run_time
    print("\n==============================================================================================")
    print(f" Agronomy Benchmark Run Finished - Total Time: {total_duration:.2f} seconds")
    print(f"=================================== End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark run interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the main execution: {type(e).__name__} - {e}")
        traceback.print_exc()
