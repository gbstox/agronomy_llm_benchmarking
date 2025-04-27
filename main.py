# main.py
import asyncio
import time
import datetime
from pathlib import Path
import traceback
import re # Keep re for _get_safe_results_filename

# Import configurations and functions from other modules
import config
import benchmark_runner
import report_generator

def _get_safe_results_filename(model_id):
    """Generates a safe filename for results based on model ID.
       Needed here for the 'missing' mode check.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', model_id) + "_answers.json"

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
        initial_models_to_run = config.MODELS_TO_RUN

        if config.RUN_MODE == 'all':
            print(f"\n--- Mode 'ALL': Preparing to run all {len(initial_models_to_run)} configured models. ---")
            models_to_run_this_session = initial_models_to_run
        elif config.RUN_MODE == 'missing':
            print("\n--- Mode 'MISSING': Checking for existing results... ---")
            results_dir_path = Path(config.BENCHMARK_RESULTS_DIR)
            for model_config in initial_models_to_run:
                safe_filename = _get_safe_results_filename(model_config['id'])
                results_filepath = results_dir_path / safe_filename
                if not results_filepath.exists():
                    print(f"  -> Will run: {model_config['id']} (No results file found)")
                    models_to_run_this_session.append(model_config)
                else:
                    print(f"  -> Skipping: {model_config['id']} (Results file exists)")
            if not models_to_run_this_session:
                 print("--- All configured models already have results. No new benchmarks to run. ---")

        # Execute benchmark runs if any models were selected
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
