# report_generator.py
import os
import json
import datetime
import re
import asyncio
import aiohttp
from asyncio import Semaphore
from tqdm import tqdm as sync_tqdm # For writing messages
from pathlib import Path
import matplotlib.pyplot as plt


import config # For paths mostly


# --- Pricing Helper ---

_model_price_cache = {} # Cache for fetched prices { model_name_api: price_usd_per_mtok }
_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
PRICE_FETCH_CONCURRENCY = 5 # Concurrency limit for price API calls

async def get_model_price(session: aiohttp.ClientSession, model_config: dict) -> float | None:
    """
    Fetches the cheapest completion price for a model from the OpenRouter API,
    using the 'model_name_api' field. Handles 404s gracefully.
    """
    model_name_api = model_config.get('model_name_api')
    if not model_name_api:
        sync_tqdm.write(f"  Price lookup Warning: 'model_name_api' missing for ID {model_config.get('id', 'N/A')}. Cannot fetch price.")
        return None

    if model_name_api in _model_price_cache:
        return _model_price_cache[model_name_api]

    url = f"{_OPENROUTER_API_BASE}/models/{model_name_api}/endpoints"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status == 404:
                _model_price_cache[model_name_api] = None # Cache the miss (model not on OpenRouter)
                return None
            response.raise_for_status() # Raise for other errors (5xx, 429, etc.)

            text_content = await response.text()
            try: data = json.loads(text_content)
            except json.JSONDecodeError:
                sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): Failed to decode JSON from {url}. Content: {text_content[:200]}...")
                _model_price_cache[model_name_api] = None; return None

            endpoints = data.get('data', {}).get('endpoints', []) if isinstance(data.get('data'), dict) else data.get('endpoints', [])

            if not isinstance(endpoints, list):
                 sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): Expected 'endpoints' list, got {type(endpoints)}. URL: {url}")
                 _model_price_cache[model_name_api] = None; return None

            completion_prices = []
            for ep in endpoints:
                if not isinstance(ep, dict): continue
                price_str = ep.get('pricing', {}).get('completion') if isinstance(ep.get('pricing'), dict) else None
                if price_str is not None:
                    try: completion_prices.append(float(price_str))
                    except (ValueError, TypeError): pass # Ignore invalid price formats silently

            if not completion_prices:
                _model_price_cache[model_name_api] = None; return None # No valid prices found

            min_price_per_token = min(completion_prices)
            price_usd_per_mtok = min_price_per_token * 1_000_000
            _model_price_cache[model_name_api] = price_usd_per_mtok
            return price_usd_per_mtok

    # Log errors but still cache failure (None)
    except aiohttp.ClientResponseError as e:
        sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): HTTP Error {e.status} for {url}")
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): Connection/Client Error - {e}")
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): Request timed out for {url}")
    except Exception as e:
        sync_tqdm.write(f"\n  Price lookup Error ({model_name_api}): Unexpected error - {type(e).__name__}: {e}")

    _model_price_cache[model_name_api] = None # Cache failure
    return None


# --- Scoring Logic ---

def score_results(model_answers_data):
    """
    Calculates scores from the loaded results data for a single model.
    Every question attempt counts towards the total.
    Fails/errors count as incorrect (0 points).
    """
    category_scores = {}
    total_correct = 0
    total_questions_attempted = 0 # Renamed from total_questions_processed

    multiple_choice_data = model_answers_data.get("multiple_choice", {})
    if not isinstance(multiple_choice_data, dict): # Handle case where structure is incomplete
        sync_tqdm.write(f"Warning: Invalid or missing 'multiple_choice' data for model {model_answers_data.get('model_id','?')}")
        return category_scores, 0.0, 0, 0 # Return defaults indicating failure to score

    for category, answers in multiple_choice_data.items():
        if not isinstance(answers, list):
             sync_tqdm.write(f"Warning: Invalid answers format for category '{category}' in model {model_answers_data.get('model_id','?')}. Skipping category.")
             continue # Skip malformed category data

        correct_in_category = 0
        questions_in_category = 0

        for answer_data in answers:
            if not isinstance(answer_data, dict):
                 sync_tqdm.write(f"Warning: Skipping invalid answer data in category '{category}': {str(answer_data)[:100]}...")
                 continue # Skip malformed answer data

            questions_in_category += 1 # Count every entry as an attempted question
            model_ans = answer_data.get("model_answer")
            correct_ans = answer_data.get("correct_answer")

            # Check if the model produced the correct single-letter answer
            if (isinstance(model_ans, str) and
                model_ans.isalpha() and
                len(model_ans) == 1 and
                model_ans == correct_ans):
                 correct_in_category += 1
            # else: # Any other case (fail, error_*, None, incorrect letter) counts as 0 points
                 # No need for explicit else, just don't increment correct_in_category

        total_correct += correct_in_category
        total_questions_attempted += questions_in_category

        # Calculate score based on total questions attempted in the category
        category_score = round((correct_in_category / questions_in_category) * 100, 2) if questions_in_category > 0 else 0.0
        category_scores[category] = category_score

    # Calculate overall score based on total questions attempted across all categories
    overall_score = round((total_correct / total_questions_attempted) * 100, 2) if total_questions_attempted > 0 else 0.0

    # Return total_questions_attempted instead of scorable/errors
    return category_scores, overall_score, total_correct, total_questions_attempted



# --- Plotting Functions ---

def plot_category_scores(category_scores, model_name, output_dir):
    """Generates and saves a bar chart for a model's scores by category."""
    if not category_scores:
        sync_tqdm.write(f"Warning: No category scores to plot for {model_name}")
        return

    categories = sorted(list(category_scores.keys()))
    scores = [category_scores[cat] for cat in categories]
    safe_model_name = re.sub(r'[\\/*?:"<>|]', '_', model_name)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, scores, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Score (%) on Scorable Questions')
    plt.title(f'Scores by Category for: {model_name}')
    plt.ylim([0, 105])
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout(pad=2.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)

    output_dir_path = Path(output_dir); output_dir_path.mkdir(parents=True, exist_ok=True)
    filepath = output_dir_path / f'{safe_model_name}_by_category.png'

    try: plt.savefig(filepath)
    except Exception as e: sync_tqdm.write(f"  Error saving category plot {filepath}: {e}")
    plt.close()

def plot_overall_scores(all_scores_summary, output_dir):
    """Generates and saves a bar chart comparing overall scores of all models."""
    plottable_summary = [item for item in all_scores_summary if 'run_error' not in item]
    if not plottable_summary:
        sync_tqdm.write("Warning: No models without run errors found to plot overall scores.")
        return

    plottable_summary.sort(key=lambda item: item['overall_score'], reverse=True)
    model_names = [item['model_name'] for item in plottable_summary]
    scores = [item['overall_score'] for item in plottable_summary]

    plt.figure(figsize=(max(12, len(model_names) * 0.6), 7))
    bars = plt.bar(model_names, scores, color='lightblue')
    plt.xlabel('Model'); plt.ylabel('Overall Score (%) on Scorable Questions')
    plt.title('Overall Benchmark Scores (Excluding Run Errors)')
    plt.ylim([0, 105]); plt.xticks(rotation=75, ha='right', fontsize=9);
    plt.tight_layout(pad=2.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', va='bottom', ha='center', fontsize=8, rotation=0)

    output_dir_path = Path(output_dir); output_dir_path.mkdir(parents=True, exist_ok=True)
    filepath = output_dir_path / 'all_models_overall_score.png'
    if filepath.exists():
        try: os.remove(filepath)
        except OSError as e: sync_tqdm.write(f"  Warning: Could not remove old overall score plot: {e}")

    try:
        plt.savefig(filepath)
        sync_tqdm.write(f"  Saved overall score plot: {filepath}")
    except Exception as e:
        sync_tqdm.write(f"  Error saving overall plot {filepath}: {e}")
    plt.close()

def plot_performance_vs_price(all_scores_summary, output_dir):
    """Generates a scatter plot of Score vs. Price (Linear Scale), colored by score."""
    print("  Generating Performance vs. Price plot...") # Standard print ok here

    plot_data = []
    for item in all_scores_summary:
        # Include models even if price is zero, but skip negative prices
        if ('run_error' not in item and
            item.get('price_usd_per_mtok') is not None and isinstance(item.get('price_usd_per_mtok'), (int, float)) and item.get('price_usd_per_mtok') >= 0 and
            item.get('overall_score') is not None and isinstance(item.get('overall_score'), (int, float))):

             price = item['price_usd_per_mtok']
             score = item['overall_score']

             label = item['model_name']
             if '/' in label: label = label.split('/')[-1]
             if len(label) > 25: label = label[:22] + '...'

             plot_data.append({'name': label, 'score': score, 'price': price})

    if not plot_data:
        print("  Warning: No models found with valid scores and non-negative price information. Skipping Perf/Price plot.")
        return

    names = [item['name'] for item in plot_data]
    scores = [item['score'] for item in plot_data]
    prices = [item['price'] for item in plot_data]

    # --- Create the Scatter Plot ---
    plt.figure(figsize=(13, 8))
    # Color by score using viridis colormap
    cmap = plt.get_cmap('viridis')
    scatter = plt.scatter(prices, scores, alpha=0.7, s=60, c=scores, cmap=cmap)

    # --- Use Linear Scales ---
    plt.xscale('linear') # Explicitly linear (default)
    plt.yscale('linear') # Explicitly linear (default)
    plt.xlabel('Price (USD per Million Completion Tokens)')
    plt.ylabel('Overall Score (%) on Scorable Questions')
    plt.title('Model Performance vs. Price (Color Indicates Score)') # Updated title


    plt.grid(True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=0.6) # Simple grid

    # Adjust limits slightly for padding
    y_padding = max(5, (max(scores) - min(scores)) * 0.05) if len(scores) > 1 else 5
    x_padding = max(0.5, (max(prices) - min(prices)) * 0.05) if len(prices) > 1 else 0.5
    plt.ylim(bottom=max(0, min(scores)-y_padding if scores else 0), top=max(105, max(scores) + y_padding if scores else 105))
    plt.xlim(left=max(0, min(prices)-x_padding if prices else 0), right=(max(prices) + x_padding if prices else 1))


    # --- Add Labels to Points ---
    for i, name in enumerate(names):
         # Position label slightly above the point
        plt.text(prices[i], scores[i] + 0.5, name, fontsize=8, ha='center', va='bottom', alpha=0.85)


    plt.tight_layout(pad=1.5)

    # --- Save the Plot ---
    output_dir_path = Path(output_dir); output_dir_path.mkdir(parents=True, exist_ok=True)
    filepath = output_dir_path / 'performance_vs_price_score_color.png' # New filename
    if filepath.exists():
        try: os.remove(filepath)
        except OSError as e: print(f"  Warning: Could not remove old perf/price plot: {e}")

    try:
        plt.savefig(filepath)
        print(f"  Saved performance vs. price plot: {filepath}")
    except Exception as e:
        print(f"  Error saving performance vs. price plot {filepath}: {e}")
    plt.close()


# --- Main Report Generation Function ---

async def generate_reports(results_dir, graphs_base_dir):
    """Loads results, scores them, fetches prices, generates markdown, and triggers plots."""
    print("\n--- Generating Reports ---") # Standard print ok
    results_dir_path = Path(results_dir)
    results_files = sorted(list(results_dir_path.glob('*_answers.json')))

    if not results_files:
        print(f"No result files found in '{results_dir}'. Skipping report generation.")
        return

    all_scores_summary = []
    all_category_keys = set()

    print(f"Found {len(results_files)} results files. Processing scores and fetching prices...") # Standard print ok

    price_semaphore = Semaphore(PRICE_FETCH_CONCURRENCY)
    model_configs_for_pricing = [] # Will store info needed for pricing lookups
    price_tasks = []

    # --- Step 1: Read basic info and prepare price lookups ---
    for filepath in results_files:
        model_config_for_price = None
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            # Use model_name from results file for price lookup consistency
            model_name_api = data.get('model_name')
            model_id_reporting = data.get('model_id') # ID for reporting/warnings

            if not model_name_api:
                print(f"Warning: Skipping price check for {filepath.name} ('{model_id_reporting}') due to missing 'model_name' in results.")
                model_configs_for_pricing.append(None)
                continue

            model_config_for_price = {
                'id': model_id_reporting, # Store ID for reference
                'model_name_api': model_name_api # API name used for the run
            }
            model_configs_for_pricing.append(model_config_for_price)
        except Exception as e:
            print(f"Error reading initial data from {filepath.name}: {e}. Skipping file.")
            model_configs_for_pricing.append(None) # Mark as skipped

    # --- Step 2: Fetch prices asynchronously ---
    async with aiohttp.ClientSession() as session:
        async def fetch_price_with_limit(config):
            if config is None: return None # Skip if config failed earlier
            async with price_semaphore:
                await asyncio.sleep(0.1) # Small delay
                return await get_model_price(session, config)

        print(f"  Fetching prices (max concurrency: {PRICE_FETCH_CONCURRENCY})...")
        price_tasks = [asyncio.create_task(fetch_price_with_limit(config)) for config in model_configs_for_pricing]
        price_results_gathered = await asyncio.gather(*price_tasks, return_exceptions=True)
        print("  ...Price fetching complete.") # Standard print ok

    # --- Step 3: Score results and combine with price info ---
    for i, filepath in enumerate(results_files):
        if model_configs_for_pricing[i] is None: continue # Skip files that failed initial read

        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except Exception as e:
            print(f"Error re-reading {filepath.name} for scoring: {e}"); continue

        model_name_reporting = data.get("model_name", model_configs_for_pricing[i]['id']) # Fallback ID
        date_tested = data.get("date", "Unknown").split('T')[0]
        access_type = data.get("access", "unknown") # Read access type from results
        price_result = price_results_gathered[i]
        model_price = None

        if isinstance(price_result, Exception):
             # Log error, but don't crash the report
            sync_tqdm.write(f"\n  Warning: Exception during price fetch for {model_name_reporting}: {price_result}")
        elif price_result is not None:
            model_price = price_result

        if "run_error" in data:
            summary = {
                "model_name": model_name_reporting,
                "date_tested": date_tested,
                "access": access_type,
                "run_error": data['run_error'],
                "overall_score": 0.0, # Assign 0 score for run errors
                "category_scores": {},
                "price_usd_per_mtok": model_price,
                "correct": 0,
                "total_questions": 'N/A' # Indicate not applicable due to run error
            }
            all_scores_summary.append(summary)
            continue

        # Call the updated scoring function
        # Signature: category_scores, overall_score, total_correct, total_questions_attempted
        category_scores, overall_score, correct, total_qs_attempted = score_results(data)

        summary = {
            "model_name": model_name_reporting,
            "category_scores": category_scores,
            "overall_score": overall_score,
            "access": access_type, # Add access type to summary
            "date_tested": date_tested,
            "correct": correct,
            "total_questions": total_qs_attempted, # Store total questions attempted
            "price_usd_per_mtok": model_price
        }
        all_scores_summary.append(summary)
        all_category_keys.update(category_scores.keys())


    # --- Generate Table and Plots ---
    if not all_scores_summary:
        print("No valid score data processed. Cannot generate reports.") # Standard print ok
        return

    all_scores_summary.sort(key=lambda item: ('run_error' in item, -item.get('overall_score', -1)))
    ordered_categories = sorted(list(all_category_keys))

    # Markdown Table
    print("\n--- Benchmark Results Table (Markdown Format) ---") # Standard print ok
    if not all_scores_summary: print("| No models processed successfully. |"); return

    # Determine column widths
    model_col_width = max(12, max((len(str(item.get('model_name', ''))) for item in all_scores_summary), default=12))
    access_col_width = max(11, max((len(str(item.get('access', ''))) for item in all_scores_summary), default=11)) # Width for Access column
    price_col_width = 14 # Keep price width fixed for now
    date_col_width = 11 # Keep date width fixed

    # Build Header Row
    header = f"| {'Model Name':<{model_col_width}} | Overall Score | {'Access':<{access_col_width}} | Date Tested | Price ($/Mtok) |"
    separator = f"|{'-' * (model_col_width+1)}|---------------|{'-'*(access_col_width+1)}|-------------|----------------|" # Adjusted separator

    cat_col_widths = {}
    for cat in ordered_categories:
        cat_title = cat.replace('_', ' ').title()
        # Calculate width based on title and typical score format 'XX.X%'
        col_width = max(len(cat_title), 7) # Minimum width for score %
        cat_col_widths[cat] = col_width
        header += f" {cat_title:<{col_width}} |"
        separator += "-" * (col_width + 2) + "|"

    print(header)
    print(separator)

    # Build Data Rows
    for item in all_scores_summary:
        name = item.get('model_name', 'Unknown')
        access = item.get('access', 'unknown').replace('_', ' ').title() # Format access type

        if 'run_error' in item:
            overall_str = "N/A".center(13)
            access_str = access.ljust(access_col_width) # Show access even on error
            price_str = "N/A".center(price_col_width)
            date_str = item.get('date_tested','N/A').ljust(date_col_width)
            total_qs_str = item.get('total_questions', 'N/A') # Get total Qs if available
            run_failed_msg = f"RUN FAILED ({total_qs_str} Qs)"

            # Basic row structure for failed runs
            row = f"| {name:<{model_col_width}} | {overall_str} | {access_str} | {date_str} | {price_str} |"
            # Add placeholders for category scores
            for cat in ordered_categories:
                 col_width = cat_col_widths[cat]; row += f" {'N/A'.center(col_width)} |"
            print(row + f"  ({run_failed_msg})") # Add failure reason parenthesis

        else:
            overall_str = f"{item['overall_score']:>13.2f}%"
            access_str = access.ljust(access_col_width)
            price = item.get('price_usd_per_mtok')
            price_str = f"${price:.4f}".center(price_col_width) if price is not None else "N/A".center(price_col_width)
            date_str = item['date_tested'].ljust(date_col_width)
            correct = item['correct']
            total_qs = item['total_questions']
            # Correct/Total info can be added if desired, e.g. in parenthesis or a separate column
            correct_total_info = f"({correct}/{total_qs})"


            row = f"| {name:<{model_col_width}} | {overall_str} | {access_str} | {date_str} | {price_str} |"
            for cat in ordered_categories:
                col_width = cat_col_widths[cat]
                score = item['category_scores'].get(cat, 0.0)
                score_str = f"{score:>{col_width-1}.1f}%" if total_qs > 0 else "N/A".center(col_width) # Handle case with 0 questions
                row += f" {score_str} |"
            print(row + f" {correct_total_info}") # Append correct/total info


    print("--- End of Table ---")

    # Plots - No changes needed for plotting logic itself
    individual_graphs_dir = Path(graphs_base_dir) / 'individual_graphs'
    print(f"\nGenerating individual category plots in: {individual_graphs_dir}")
    for item in all_scores_summary:
         if 'run_error' not in item and item.get('category_scores'): # Check if scores exist
             plot_category_scores(item['category_scores'], item['model_name'], individual_graphs_dir)

    overall_graph_dir = Path(graphs_base_dir)
    print(f"Generating overall score plot in: {overall_graph_dir}")
    plot_overall_scores(all_scores_summary, overall_graph_dir)

    print(f"Generating performance vs. price plot in: {overall_graph_dir}")
    plot_performance_vs_price(all_scores_summary, overall_graph_dir)

    print("\n--- Reporting Complete ---")
