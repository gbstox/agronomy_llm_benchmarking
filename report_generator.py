# report_generator.py
import os
import json
import re
from tqdm import tqdm as sync_tqdm # For writing messages
from pathlib import Path
import matplotlib.pyplot as plt


import config # For paths mostly
import openrouter_catalog

FIXED_SUMMARY_COLUMNS = 5


# --- Scoring Logic ---

def _parse_percent(value):
    cleaned = str(value).strip().replace("%", "")
    if not cleaned or cleaned == "N/A":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_price(value):
    cleaned = str(value).strip()
    if not cleaned or cleaned == "N/A":
        return None
    cleaned = cleaned.replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_category_name(column_name):
    return column_name.strip().lower().replace(" ", "_")


def load_existing_readme_summaries(readme_path):
    """Parses the tracked README leaderboard so partial local runs can merge safely."""
    path = Path(readme_path)
    if not path.is_file():
        return [], set()

    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        print(f"Warning: Could not read existing README summaries from {readme_path}: {e}")
        return [], set()

    header_idx = next((idx for idx, line in enumerate(lines) if line.startswith("| Model Name")), None)
    if header_idx is None or header_idx + 1 >= len(lines):
        return [], set()

    headers = [column.strip() for column in lines[header_idx].split("|")[1:-1]]
    if len(headers) <= FIXED_SUMMARY_COLUMNS:
        return [], set()

    category_headers = headers[FIXED_SUMMARY_COLUMNS:]
    category_keys = {_normalize_category_name(header) for header in category_headers}
    summaries = []

    for line in lines[header_idx + 2:]:
        if not line.startswith("|"):
            break

        parts = line.split("|")
        if len(parts) < len(headers) + 1:
            continue

        columns = [parts[idx].strip() for idx in range(1, len(headers) + 1)]
        model_name = columns[0]
        if not model_name:
            continue

        category_scores = {}
        for header, value in zip(category_headers, columns[FIXED_SUMMARY_COLUMNS:]):
            parsed_score = _parse_percent(value)
            if parsed_score is not None:
                category_scores[_normalize_category_name(header)] = parsed_score

        score_match = re.search(r"\((\d+)/(\d+|N/A)\)\s*$", line)
        correct = int(score_match.group(1)) if score_match else 0
        total_questions = score_match.group(2) if score_match else "N/A"
        if isinstance(total_questions, str) and total_questions.isdigit():
            total_questions = int(total_questions)

        summary = {
            "model_name": model_name,
            "overall_score": _parse_percent(columns[1]) or 0.0,
            "access": columns[2].strip().lower(),
            "date_tested": columns[3].strip(),
            "price_usd_per_mtok": _parse_price(columns[4]),
            "category_scores": category_scores,
            "correct": correct,
            "total_questions": total_questions,
        }
        summaries.append(summary)

    return summaries, category_keys


def _build_markdown_table_lines(all_scores_summary, ordered_categories):
    model_col_width = max(
        12,
        max((len(str(item.get("model_name", ""))) for item in all_scores_summary), default=12),
    )
    access_col_width = max(
        11,
        max((len(str(item.get("access", ""))) for item in all_scores_summary), default=11),
    )
    price_col_width = 14
    date_col_width = 11

    header = (
        f"| {'Model Name':<{model_col_width}} | Overall Score | "
        f"{'Access':<{access_col_width}} | Date Tested | Price ($/Mtok) |"
    )
    separator = (
        f"|{'-' * (model_col_width + 1)}|---------------|"
        f"{'-' * (access_col_width + 1)}|-------------|----------------|"
    )

    cat_col_widths = {}
    for cat in ordered_categories:
        cat_title = cat.replace("_", " ").title()
        col_width = max(len(cat_title), 7)
        cat_col_widths[cat] = col_width
        header += f" {cat_title:<{col_width}} |"
        separator += "-" * (col_width + 2) + "|"

    table_lines = [header, separator]
    for item in all_scores_summary:
        name = item.get("model_name", "Unknown")
        access = item.get("access", "unknown").replace("_", " ").title()

        if "run_error" in item:
            overall_str = "N/A".center(13)
            access_str = access.ljust(access_col_width)
            price_str = "N/A".center(price_col_width)
            date_str = item.get("date_tested", "N/A").ljust(date_col_width)
            total_qs_str = item.get("total_questions", "N/A")
            run_failed_msg = f"RUN FAILED ({total_qs_str} Qs)"

            row = (
                f"| {name:<{model_col_width}} | {overall_str} | {access_str} | "
                f"{date_str} | {price_str} |"
            )
            for cat in ordered_categories:
                col_width = cat_col_widths[cat]
                row += f" {'N/A'.center(col_width)} |"
            table_lines.append(row + f"  ({run_failed_msg})")
            continue

        overall_str = f"{item['overall_score']:>13.2f}%"
        access_str = access.ljust(access_col_width)
        price = item.get("price_usd_per_mtok")
        price_str = (
            f"${price:.4f}".center(price_col_width)
            if price is not None
            else "N/A".center(price_col_width)
        )
        date_str = item["date_tested"].ljust(date_col_width)
        correct = item["correct"]
        total_qs = item["total_questions"]
        correct_total_info = f"({correct}/{total_qs})"

        row = (
            f"| {name:<{model_col_width}} | {overall_str} | {access_str} | "
            f"{date_str} | {price_str} |"
        )
        for cat in ordered_categories:
            col_width = cat_col_widths[cat]
            score = item["category_scores"].get(cat, 0.0)
            score_str = (
                f"{score:>{col_width - 1}.1f}%"
                if total_qs > 0
                else "N/A".center(col_width)
            )
            row += f" {score_str} |"
        table_lines.append(row + f" {correct_total_info}")

    return table_lines


def _write_readme_leaderboard(readme_path, table_lines):
    path = Path(readme_path)
    try:
        existing_lines = path.read_text().splitlines() if path.is_file() else []
    except OSError as e:
        print(f"Warning: Could not read README for leaderboard update: {e}")
        return

    start_idx = next(
        (idx for idx, line in enumerate(existing_lines) if line.startswith("| Model Name")),
        None,
    )
    if start_idx is not None:
        end_idx = start_idx + 1
        while end_idx < len(existing_lines) and existing_lines[end_idx].startswith("|"):
            end_idx += 1
        updated_lines = existing_lines[:start_idx] + table_lines + existing_lines[end_idx:]
    else:
        insert_idx = next(
            (idx for idx, line in enumerate(existing_lines) if line.startswith("# What is this?")),
            len(existing_lines),
        )
        updated_lines = existing_lines[:insert_idx]
        if updated_lines and updated_lines[-1] != "":
            updated_lines.append("")
        updated_lines.extend(table_lines)
        if insert_idx < len(existing_lines) and updated_lines and updated_lines[-1] != "":
            updated_lines.append("")
        updated_lines.extend(existing_lines[insert_idx:])

    try:
        path.write_text("\n".join(updated_lines).rstrip() + "\n")
        print(f"Updated README leaderboard: {path}")
    except OSError as e:
        print(f"Warning: Could not write updated README leaderboard: {e}")

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
    skipped_non_complete_results = 0

    print(f"Found {len(results_files)} results files. Processing scores and fetching metadata...") # Standard print ok

    openrouter_metadata = {}
    try:
        openrouter_metadata = openrouter_catalog.fetch_openrouter_model_metadata_map(
            project_root=Path(__file__).resolve().parent,
            api_key_env=config.OPENROUTER_DISCOVERY_API_KEY_ENV,
        )
        print(f"  Loaded OpenRouter metadata for {len(openrouter_metadata)} models.")
    except Exception as e:
        print(f"  Warning: Could not load OpenRouter metadata: {type(e).__name__} - {e}")

    # --- Score results and combine with price/access metadata ---
    for filepath in results_files:

        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except Exception as e:
            print(f"Error re-reading {filepath.name} for scoring: {e}"); continue

        model_name_api = data.get("model_name")
        model_name_reporting = model_name_api or data.get("model_id", filepath.stem) # Fallback ID
        date_tested = data.get("date", "Unknown").split('T')[0]
        live_metadata = openrouter_metadata.get(model_name_api, {}) if model_name_api else {}
        access_type = live_metadata.get("access") or data.get("access", "unknown")
        model_price = live_metadata.get("price_usd_per_mtok")

        if access_type == "unknown":
            access_type = data.get("access", "unknown")

        benchmark_status = data.get("benchmark_status", "complete")
        api_failures = int(data.get("api_failures", 0) or 0)
        fatal_api_failures = int(data.get("fatal_api_failures", 0) or 0)
        if (
            "run_error" in data
            or benchmark_status != "complete"
            or api_failures > 0
            or fatal_api_failures > 0
        ):
            skipped_non_complete_results += 1
            run_issue = data.get("run_error") or (
                f"Benchmark status '{benchmark_status}' "
                f"(api_failures={api_failures}, fatal_api_failures={fatal_api_failures})"
            )
            print(
                f"  Skipping incomplete leaderboard row for {model_name_reporting}: "
                f"{run_issue}"
            )
            continue

        # Call the updated scoring function
        # Signature: category_scores, overall_score, total_correct, total_questions_attempted
        category_scores, overall_score, correct, total_qs_attempted = score_results(data)
        normalized_category_scores = {
            _normalize_category_name(category): score
            for category, score in category_scores.items()
        }

        summary = {
            "model_name": model_name_reporting,
            "category_scores": normalized_category_scores,
            "overall_score": overall_score,
            "access": access_type, # Add access type to summary
            "date_tested": date_tested,
            "correct": correct,
            "total_questions": total_qs_attempted, # Store total questions attempted
            "price_usd_per_mtok": model_price
        }
        all_scores_summary.append(summary)
        all_category_keys.update(normalized_category_scores.keys())

    readme_path = Path(__file__).resolve().parent / "README.md"
    existing_readme_summaries, existing_category_keys = load_existing_readme_summaries(
        readme_path
    )
    for summary in existing_readme_summaries:
        live_metadata = openrouter_metadata.get(summary.get("model_name"), {})
        live_access = live_metadata.get("access")
        if live_access and live_access != "unknown":
            summary["access"] = live_access
        live_price = live_metadata.get("price_usd_per_mtok")
        if live_price is not None:
            summary["price_usd_per_mtok"] = live_price
    existing_names = {item.get("model_name") for item in all_scores_summary}
    for summary in existing_readme_summaries:
        if summary.get("model_name") in existing_names:
            continue
        all_scores_summary.append(summary)
        all_category_keys.update(summary.get("category_scores", {}).keys())
    all_category_keys.update(existing_category_keys)

    # --- Generate Table and Plots ---
    if not all_scores_summary:
        print("No valid score data processed. Cannot generate reports.") # Standard print ok
        return

    all_scores_summary.sort(key=lambda item: ('run_error' in item, -item.get('overall_score', -1)))
    ordered_categories = sorted(list(all_category_keys))

    if skipped_non_complete_results:
        print(
            f"Skipped {skipped_non_complete_results} non-complete result file(s) "
            "from the published README/charts refresh."
        )

    # Markdown Table
    print("\n--- Benchmark Results Table (Markdown Format) ---") # Standard print ok
    if not all_scores_summary: print("| No models processed successfully. |"); return
    table_lines = _build_markdown_table_lines(all_scores_summary, ordered_categories)
    for line in table_lines:
        print(line)


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

    _write_readme_leaderboard(readme_path, table_lines)

    print("\n--- Reporting Complete ---")
