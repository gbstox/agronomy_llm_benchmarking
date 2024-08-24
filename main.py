import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
import re
import glob
import matplotlib.pyplot as plt


import private_utils

load_dotenv() 


def format_multiple_choice_prompt(model_id, prompts):

    prompt_template = get_model_promt_template(model_id)

    formatted_system_prompt = prompts.get("system_prompt", "")
    formatted_user_prompt = prompts.get("user_prompt", "")
    formatted_assistant_prompt = prompts.get("assistant_prompt", "")

    if prompt_template["system_prompt_template"] and formatted_system_prompt:
        formatted_system_prompt = prompt_template["system_prompt_template"][0] + formatted_system_prompt + prompt_template["system_prompt_template"][1]
    
    if prompt_template["user_prompt_template"] and formatted_user_prompt:
        formatted_user_prompt = prompt_template["user_prompt_template"][0] + formatted_user_prompt + prompt_template["user_prompt_template"][1]
    
    if prompt_template["assistant_prompt_template"] and formatted_assistant_prompt:
        formatted_assistant_prompt = prompt_template["assistant_prompt_template"][0] + formatted_assistant_prompt + prompt_template["assistant_prompt_template"][1]
    
    formatted_prompts = {"system_prompt": formatted_system_prompt, "user_prompt": formatted_user_prompt, "assistant_prompt": formatted_assistant_prompt}
    
    return formatted_prompts
    
    
def get_model_promt_template(model_id):
    model_creator = model_id.split("/")[0]

    if model_creator == "Gryphe":
        prompt_template = {
            "system_prompt_template": [],
            "user_prompt_template": ["### Instruction:\n", "### Response: \n"], 
            "assistant_prompt_template": [],
            "stop_sequence": ["/s"]
            }

    elif model_creator in ["NousResearch", "teknium"]:
        prompt_template = {
            "system_prompt_template": ["<|im_start|>system\n", "<|im_end|>\n"],
            "user_prompt_template": ["<|im_start|>user\n", "<|im_end|>\n"], 
            "assistant_prompt_template": ["<|im_start|>assistant\n", "<|im_end|>\n"],
            "stop_sequence": ["<|im_start|>","<|im_end|>"]
            }

    elif model_creator in ["mistralai", "gbstox"]:
        prompt_template = {
            "system_prompt_template": ["[INST]\n", "[/INST]"],
            "user_prompt_template": [], 
            "assistant_prompt_template": [],
            "stop_sequence": ["</s>","[INST]"],
        }
    
    else:
        prompt_template = {
            "system_prompt_template": [],
            "user_prompt_template": [], 
            "assistant_prompt_template": [],
            "stop_sequence": [],
        }


    return prompt_template

def chat_prompt(client, model_id, formatted_prompts):
    #print (formatted_prompts)
    try:
        response = client.with_options(timeout=5).chat.completions.create(
            model= model_id,
            messages=[
                {
                    "role": "system",
                    "content": formatted_prompts["system_prompt"]
                },
                {
                    "role": "user",
                    "content": formatted_prompts["user_prompt"]
                },
                {
                    "role": "assistant",
                    "content": formatted_prompts["assistant_prompt"]
                },
            ],
            # functions   = funct,
            # function_call = "auto",
            # temperature = temperature,
            # stop        = "",
            # top_p       = top_p,
            # presence_penalty = 0.0,  # penalties -2.0 - 2.0
            # frequency_penalty = 0.0,  # frequency = cumulative score
            # stream      = True,
            # logit_bias  = {"100066": -1},  # example, '～\n\n' token
            #user        = "site_user-id",
            max_tokens=2000,
            n=1,
            temperature=0.5,
        )
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            return None
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)
        return None
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        return None
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        return None


    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)


def run_benchmark(model_id, benchmark_questions_file, prompts, benchmark_results_dir, retries=3):
    with open(benchmark_questions_file, 'r') as f:
        benchmark_questions = json.load(f)["multiple_choice"]

    model_answers = {"multiple_choice": {}}
    model_answers["date"] = datetime.datetime.now().isoformat()
    model_creator = model_id.split('/')[0]
    model_name = model_id.split('/')[-1]
    model_answers["model_name"] = model_name

    for category, questions in benchmark_questions.items():
        model_answers["multiple_choice"][category] = []
        for benchmark_mc_question in questions:
            prompts["user_prompt"] = f"Question: {benchmark_mc_question['question']}\n\nanswer_options: {benchmark_mc_question['answer_options']}"
            formatted_prompts = format_multiple_choice_prompt(model_id, prompts)

            temp_answer = "fail"
            model_answer = None  # Initialize model_answer
            for _ in range(retries):
                if model_creator == "fbn":
                    formatted_prompts["user_prompt"] = f"{formatted_prompts['system_prompt']} \n {formatted_prompts['user_prompt']}"
                    temp_answer = private_utils.prompt_fbn_norm(model_id, formatted_prompts)
                elif model_creator == "openai":
                    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
                    temp_answer = chat_prompt(client, model_name, formatted_prompts)
                elif model_creator == "pratik":
                    gradio_url = "https://huggingface.co/spaces/eswardivi/llama3-8b-dhenu-0.1"
                    temp_answer = private_utils.query_gradio(gradio_url, formatted_prompts)
                elif model_creator == "gbstox":
                    pod_url = "https://6lgv5h2aextq69-5000.proxy.runpod.net/v1/chat/completions"
                    temp_answer = private_utils.runpod_chat_prompt(pod_url, model_id, formatted_prompts)
                else:
                    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ['OPENROUTER_API_KEY'])
                    temp_answer = chat_prompt(client, model_id, formatted_prompts)

                if temp_answer:
                    temp_answer_stripped = temp_answer.strip(' "\'*')
                    temp_answer_stripped = re.sub(r'\s', '', temp_answer_stripped)
                    temp_answer_stripped = temp_answer_stripped.rstrip('*')  # This line removes trailing asterisks
                    correct = temp_answer_stripped == benchmark_mc_question.get('correct_answer')
                    print("Correct" if correct else "Incorrect", temp_answer_stripped, f" - try #{_}, correct: {benchmark_mc_question.get('correct_answer')}, q_id = {benchmark_mc_question.get('id')}")
                    if len(temp_answer_stripped) == 1 and temp_answer_stripped.isalpha():
                        model_answer = temp_answer_stripped.lower()
                        break
                else:
                    print(f"Invalid response received for question ID {benchmark_mc_question.get('id')}, retrying...")

            if model_answer is None:
                model_answer = "fail"  # Default value if no valid answer is found

            benchmark_mc_question["model_answer"] = model_answer
            model_answers["multiple_choice"][category].append(benchmark_mc_question)

    Path(benchmark_results_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{benchmark_results_dir}/{model_name}_answers.json', 'w') as f:
        f.write(json.dumps(model_answers, indent=4))


def score_multile_choice_answers(model_answers_file):
    with open(model_answers_file, 'r') as f:
        model_answers = json.load(f)["multiple_choice"]

    correct_answers = 0
    total_questions = len(model_answers)

    for model_answer in model_answers:
        if model_answer["model_answer"] == model_answer["correct_answer"]:
            correct_answers += 1

    score = round((correct_answers / total_questions) * 100, 2)
    return score

def score_multile_choice_answers_by_category(model_answers_file):
    with open(model_answers_file, 'r') as f:
        model_answers = json.load(f)["multiple_choice"]

    scores = {}
    for category, answers in model_answers.items():
        correct_answers = 0
        total_questions = len(answers)

        for model_answer in answers:
            if model_answer["model_answer"] == model_answer["correct_answer"]:
                correct_answers += 1

        score = round((correct_answers / total_questions) * 100, 2)
        scores[category] = score

    # Calculate the overall score weighted by the number of questions in each category
    total_questions = sum(len(answers) for answers in model_answers.values())
    weighted_score_sum = sum(score * len(model_answers[category]) for category, score in scores.items())
    overall_score = round(weighted_score_sum / total_questions, 2)

    return scores, overall_score


def graph_individual_model_by_category(model_scores, model_name, output_dir):
    categories = list(model_scores.keys())
    scores = list(model_scores.values())

    plt.figure(figsize=(10, 5))
    plt.bar(categories, scores, color='blue')
    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.title(f'Scores by Category for {model_name}')
    plt.ylim([0, 100])
    for i in range(len(scores)):
        plt.text(i, scores[i], scores[i], ha = 'center')

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(f'{output_dir}/{model_name}_by_category.png')
    plt.close()

def graph_all_models_by_overall_score(model_overall_scores, output_dir):
    model_names = [model_score[0] for model_score in model_overall_scores]
    scores = [model_score[1] for model_score in model_overall_scores]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, scores, color='blue')
    plt.xlabel('Model')
    plt.ylabel('Overall Score')
    plt.title('Overall Scores for All Models')
    plt.ylim([0, 100])
    for i in range(len(scores)):
        plt.text(i, scores[i], scores[i], ha='center')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/all_models_overall_score.png'
    
    # Remove the file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    plt.savefig(output_file)
    plt.close()

def compare_benchmark_scores(benchmark_results_dir, graphs_by_category_dir, graphs_overall_scores_dir):
    benchmark_results_files = glob.glob(f'{benchmark_results_dir}/*.json')
    scores = []
    for model_answers_file in benchmark_results_files:
        # Extract the filename from the path
        filename = os.path.basename(model_answers_file)
        model_scores, overall_score = score_multile_choice_answers_by_category(os.path.join(benchmark_results_dir, filename))
        # Extract the date and model_name from the model answers file
        with open(os.path.join(benchmark_results_dir, filename), 'r') as f:
            data = json.load(f)
            date_tested = data["date"]
            model_name = data["model_name"].split("/")[-1]
        # Only include the yyyy-mm-dd part of the date
        date_tested = date_tested.split('T')[0]
        scores.append((model_name, model_scores, overall_score, date_tested))

    # Sort the scores in descending order of the overall score
    scores.sort(key=lambda x: x[2], reverse=True)

    # Print the scores in a markdown table format
    print("| Model Name | Overall Score | Date Tested |", end="")
    # Print the category names
    for category in scores[0][1].keys():
        print(f" {category} |", end="")
    print()
    print("|------------|---------------|-------------|", end="")
    # Print the separator for each category
    for _ in scores[0][1].keys():
        print("-------|", end="")
    print()
    for model, model_scores, overall_score, date_tested in scores:
        print(f"| {model} | {overall_score}% | {date_tested} |", end="")
        # Print the score for each category
        for score in model_scores.values():
            print(f" {score}% |", end="")
        print()

    model_overall_scores = []
    for model, model_scores, overall_score, date_tested in scores:
        graph_individual_model_by_category(model_scores, model, graphs_by_category_dir)
        model_overall_scores.append((model, overall_score))

    graph_all_models_by_overall_score(model_overall_scores, graphs_overall_scores_dir)

system_prompt = """
        You are a helpful and brilliant agronomist. For the following multiple choice Question, answer  with the key of the correct answer_options value. 
        Your response must be ONLY the answer_options key of the correct answer_options value and no other text. respond with ONLY a single letter key from answer_options.

"""

user_prompt = ""

assistant_prompt = "Correct answer_options key:"

prompts = {"system_prompt": system_prompt, "user_prompt": user_prompt, "assistant_prompt": assistant_prompt}

model_ids = [
    #"01-ai/yi-34b-chat",
    #"anthropic/claude-2",
    #"anthropic/claude-3-haiku",
    #"anthropic/claude-3-opus",
    #"anthropic/claude-3.5-sonnet",
    #"fbn/norm",
    #"gbstox/agronomistral",
    #"gbstox/agronomYi-34b",
    #"google/gemini-flash-1.5",
    #"google/gemini-pro-1.5",
    #"google/gemini-pro-1.5-exp",
    #"meta-llama/llama-3-8b-instruct:nitro",
    #"meta-llama/llama-3-70b-instruct",
    #'meta-llama/llama-3.1-8b-instruct',
    #'meta-llama/llama-3.1-70b-instruct',
    #'meta-llama/llama-3.1-405b-instruct',
    #"microsoft/phi-3-medium-128k-instruct",
    #"microsoft/phi-3-mini-128k-instruct",
    #"mistralai/mixtral-8x7b-instruct",
    #"mistralai/mistral-7b-instruct",
    #"mistralai/mistral-large",
    #"mistralai/mistral-medium",
    #"nousresearch/hermes-2-pro-llama-3-8b",
    #"nousresearch/hermes-3-llama-3.1-405b",
    #"nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
    #"nousresearch/nous-hermes-yi-34b",
    #"openai/gpt-3.5-turbo",
    #"openai/gpt-4",
    #"openai/gpt-4o",
    #"openai/gpt-4o-mini",
    #"perplexity/llama-3-sonar-large-32k-chat",
    #"perplexity/llama-3.1-sonar-huge-128k-online",
    #"pratik/llama3-8b-dhenu-0.1",
    #"qwen/qwen-2-72b-instruct",
    #"teknium/openhermes-2.5-mistral-7b"
]


benchmark_questions_file = "./benchmark_questions/combined_benchmark.json"
#benchmark_results_dir = './benchmark_results_tests/model_results/benchmark_results_1'
benchmark_results_dir = 'benchmark_results/model_results'

graphs_by_category_dir = f'{benchmark_results_dir}/individual_graphs'
graphs_overall_scores_dir = f'benchmark_results/'

for model_id in model_ids:
    print ()
    print (model_id)
    print()


    run_benchmark(model_id, benchmark_questions_file, prompts, benchmark_results_dir)

print (benchmark_results_dir)
compare_benchmark_scores(benchmark_results_dir, graphs_by_category_dir, graphs_overall_scores_dir)