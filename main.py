import os
import json
import time
import datetime
import together
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import private_utils


load_dotenv() 

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
together.api_key = os.environ['TOGETHER_API_KEY']

client = OpenAI()



def build_multiple_choice_prompt(benchmark_question):
    prompt = f"""
        For the following multiple choice question, answer with one of the answer_options.
        Your response must be ONLY the answer_options letter key of the correct answer and no other text. [No Prose]. 
        No explaination, formatting, or prose. ONLY a single character response.
        
        question: 
        {benchmark_question["question"]}

        answer_options: 
        {benchmark_question["answer_options"]}
        
        """
    return prompt
    
    
def get_model_promt_template(model_name, prompt):
    model_creator = model_name.split("/")[0]
    if model_creator == "Gryphe":
        model_prompt = f"### Instruction:\n{prompt}\n### Response: \n"
        stop_sequence = ["/s"]
    elif model_creator == "NousResearch" or "teknium":
        model_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        stop_sequence = ["<|im_start|>","<|im_end|>"]
    elif model_creator == "mistralai":
        model_prompt = f"[INST] {prompt} [/INST]"
        stop_sequence = ["</s>","[INST]"]

    model_promt_template = (model_prompt, stop_sequence)
    return model_promt_template

def prompt_together(model_name, prompt):
    time.sleep(1.1)
    prompt = f"You are a helpful and brilliant agronomist. {prompt}"
    model_promt_template = get_model_promt_template(model_name, prompt)
    formatted_prompt = model_promt_template[0]

    output = together.Complete.create(
        prompt = formatted_prompt,
        model = model_name, 
        max_tokens = 512,
        temperature = 0.2,
        top_k = 50,
        top_p = 0.7,
        repetition_penalty = 1,
        stop = model_promt_template[1]
        )
    
    response_text = output['output']['choices'][0]['text']
    return response_text


def prompt_openai(model_name, prompt):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {"role": "system", "content": "You are a helpful and brilliant agronomist."},
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content
    return response_text


def run_benchmark(model_name, benchmark_questions_file, retries = 3):
    with open(benchmark_questions_file, 'r') as f:
        benchmark_questions = json.load(f)

    model_answers = {"multiple_choice": []}  # Initialize model_answers dictionary
    model_answers["date"] = datetime.datetime.now().isoformat()
    model_answers["model_name"] = model_name

    # handle multiple choice questions
    for benchmark_mc_question in benchmark_questions["multiple_choice"]:
        prompt = build_multiple_choice_prompt(benchmark_mc_question)
        model_answer = "fail"  # default answer if retries fail
        for _ in range(retries):  # retry n number of times
            if model_name == "fbn-norm":
                temp_answer = private_utils.prompt_fbn_norm(prompt)
            # bad filter, figure out a better way
            elif "gpt-" not in model_name:
                temp_answer = prompt_together(model_name, prompt)
            else:
                temp_answer = prompt_openai(model_name, prompt)
            
            print (temp_answer.replace("'", "").replace('"', ""), f" try #{_}")
            if len(temp_answer.replace("'", "").replace('"', "")) == 1:  # if the answer is a single character
                model_answer = temp_answer.strip()  # update the model answer
                break  # exit the loop

        benchmark_mc_question["model_answer"] = model_answer
        model_answers["multiple_choice"].append(benchmark_mc_question)  # Append the question to model_answers

    Path('benchmark_results').mkdir(parents=True, exist_ok=True)
    model_name = model_name.split("/")[-1]
    with open(f'benchmark_results/{model_name}_answers.json', 'w') as f:
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
    benchmark_results_files = os.listdir('./benchmark_results')
    scores = []
    for model_answers_file in benchmark_results_files:
        score = score_multile_choice_answers(os.path.join('./benchmark_results', model_answers_file))
        # Extract the date and model_name from the model answers file
        with open(os.path.join('./benchmark_results', model_answers_file), 'r') as f:
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

model_name = "gpt-4-1106-preview"
model_names = [
    "gpt-4", 
    "gpt-3.5-turbo", 
    "gpt-4-1106-preview", 
    "fbn-norm", 
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Gryphe/MythoMax-L2-13b",
    "teknium/OpenHermes-2p5-Mistral-7B"
]


"""for model_name in model_names:
    print ()
    print (model_name)
    print()"""

run_benchmark(model_name, benchmark_questions_file)

compare_benchmark_scores()




