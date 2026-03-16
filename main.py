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
    if status in {"rate_limited", "incomplete", "run_error"}:
        return True, f"Existing results marked '{status}'"
    if "run_error" in existing_data:
        return True, "Existing results contain run_error"

    return False, "Results file exists"


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

        initial_models_to_run = _apply_model_allowlist(
            initial_models_to_run,
            getattr(config, "MODEL_ID_ALLOWLIST", []),
        )

        if config.RUN_MODE == 'all':
            print(f"\n--- Mode 'ALL': Preparing to run all {len(initial_models_to_run)} configured models. ---")
            models_to_run_this_session = initial_models_to_run
        elif config.RUN_MODE == 'missing':
            print("\n--- Mode 'MISSING': Checking for existing results... ---")
            results_dir_path = Path(config.BENCHMARK_RESULTS_DIR)
            for model_config in initial_models_to_run:
                safe_filename = _get_safe_results_filename(model_config['id'])
                results_filepath = results_dir_path / safe_filename
                should_rerun, reason = _should_rerun_result(results_filepath)
                if should_rerun:
                    print(f"  -> Will run: {model_config['id']} ({reason})")
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
