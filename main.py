import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
import re

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
        response_content = response.choices[0].message.content
        return response_content


    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)


def run_benchmark(model_id, benchmark_questions_file, prompts, retries = 3):
    with open(benchmark_questions_file, 'r') as f:
        benchmark_questions = json.load(f)

    model_answers = {"multiple_choice": []}  # Initialize model_answers dictionary
    model_answers["date"] = datetime.datetime.now().isoformat()
    model_creator = model_id.split('/')[0]
    model_name = model_id.split('/')[-1]
    model_answers["model_name"] = model_name

    # handle multiple choice questions
    for benchmark_mc_question in benchmark_questions["multiple_choice"]:

        prompts["user_prompt"] = f"Question: {benchmark_mc_question['question']}\n\nanswer_options: {benchmark_mc_question['answer_options']}"

        formatted_prompts = format_multiple_choice_prompt(model_id, prompts) 
        #print (json.dumps(formatted_prompts, indent =4))

        temp_answer = "fail"  # default answer if retries fail
        for _ in range(retries):  # retry n number of times
            if model_creator == "fbn":
                formatted_prompts["user_prompt"] = f"{formatted_prompts['system_prompt']} \n {formatted_prompts['user_prompt']}"
                temp_answer = private_utils.prompt_fbn_norm(model_id, formatted_prompts)
   
            elif model_creator == "openai":
                client = openai.OpenAI(api_key= os.environ['OPENAI_API_KEY'])
                temp_answer = chat_prompt(client, model_name, formatted_prompts)


            elif model_creator == "gbstox":
                #client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
                #temp_answer = chat_prompt(client, model_name, formatted_prompts)

                pod_url = "https://6lgv5h2aextq69-5000.proxy.runpod.net" + "/v1/chat/completions"
                temp_answer = private_utils.runpod_chat_prompt(pod_url, model_id, formatted_prompts)

                #hf_url = "https://fj9k7daknymi38df.us-east-1.aws.endpoints.huggingface.cloud"
                #private_utils.query_huggingface(hf_url, formatted_prompts)

            else:
                client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key= os.environ['OPENROUTER_API_KEY'])
                temp_answer = chat_prompt(client, model_id, formatted_prompts)

            temp_answer_stripped = temp_answer.strip(' "\'')
            temp_answer_stripped = re.sub(r'\s', '', temp_answer_stripped)
            print (temp_answer_stripped, f" try #{_}, correct: {benchmark_mc_question.get('correct_answer')}, q_id = {benchmark_mc_question.get('id')}")
            if len(temp_answer_stripped) == 1 and temp_answer_stripped.isalpha():
                model_answer = temp_answer_stripped.lower() # update the model answer
                break  

        benchmark_mc_question["model_answer"] = model_answer
        model_answers["multiple_choice"].append(benchmark_mc_question)  # Append the question to model_answers

    Path(benchmark_results_dir).mkdir(parents=True, exist_ok=True)
    model_name = model_name.split("/")[-1]
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


def compare_benchmark_scores():
    benchmark_results_files = os.listdir(benchmark_results_dir)
    scores = []
    for model_answers_file in benchmark_results_files:
        score = score_multile_choice_answers(os.path.join(benchmark_results_dir, model_answers_file))
        # Extract the date and model_name from the model answers file
        with open(os.path.join(benchmark_results_dir, model_answers_file), 'r') as f:
            data = json.load(f)
            date_tested = data["date"]
            model_name = data["model_name"].split("/")[-1]
        # Only include the yyyy-mm-dd part of the date
        date_tested = date_tested.split('T')[0]
        scores.append((model_name, score, date_tested))

    # Sort the scores in descending order
    scores.sort(key=lambda x: x[1], reverse=True)

    # Print the scores in a markdown table format
    print("| Model Name | Score | Date Tested |")
    print("|------------|-------|-------------|")
    for model, score, date_tested in scores:
        print(f"| {model} | {score}% | {date_tested} |")



benchmark_questions_file = "./agronomy_benchmark_questions.json"
benchmark_results_dir = './benchmark_results/'


system_prompt = """
        You are a helpful and brilliant agronomist. For the following multiple choice Question, answer  with the key of the correct answer_options value. 
        Your response must be ONLY the answer_options key of the correct answer_options value and no other text. respond with ONLY a single letter key from answer_options.

"""

user_prompt = ""

assistant_prompt = "Correct answer_options key:"

prompts = {"system_prompt": system_prompt, "user_prompt": user_prompt, "assistant_prompt": assistant_prompt}

model_ids = [
    #"openai/gpt-4", 
    #"openai/gpt-3.5-turbo", 
    #"openai/gpt-4-1106-preview", 
    #"anthropic/claude-2",
    #"anthropic/claude-3-opus"
    #"fbn/norm", 
    #"meta-llama/llama-3-8b-instruct:nitro",
    #"mistralai/mixtral-8x7b-instruct",
    #"mistralai/mistral-medium",
    #"mistralai/mistral-7b-instruct",
    #"01-ai/yi-34b-chat",
    #"teknium/openhermes-2.5-mistral-7b",
    #"nousresearch/nous-hermes-yi-34b",
    #"nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    #"gbstox/agronomistral",
    #"gbstox/agronomYi-34b"
]


for model_id in model_ids:
    print ()
    print (model_id)
    print()

    run_benchmark(model_id, benchmark_questions_file, prompts)

compare_benchmark_scores()




