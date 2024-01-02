import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import private_utils


load_dotenv() 

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI()



def build_multiple_choice_prompt(benchmark_question):
    prompt = f"""
        For the following multiple choice question, answer with one of the anser_options.
        Reply with ONLY the answer_options key of the correct answer and nothing else. [No Prose]
        
        question: 
        {benchmark_question["question"]}

        answer_options: 
        {benchmark_question["answer_options"]}
        
        """
    return prompt
    

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
            else:
                temp_answer = prompt_openai(model_name, prompt)
            print (temp_answer.replace("'", "").replace('"', ""), f" try #{_}")
            if len(temp_answer.replace("'", "").replace('"', "")) == 1:  # if the answer is a single character
                model_answer = temp_answer.strip()  # update the model answer
                break  # exit the loop

        benchmark_mc_question["model_answer"] = model_answer
        model_answers["multiple_choice"].append(benchmark_mc_question)  # Append the question to model_answers

    Path('benchmark_results').mkdir(parents=True, exist_ok=True)
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
    for model_answers_file in benchmark_results_files:
        score = score_multile_choice_answers(os.path.join('./benchmark_results', model_answers_file))
        print(f'The model {model_answers_file} scored {score}% on the benchmark.')



#model_name = "gpt-4"
#model_name = 'gpt-3.5-turbo'
#model_name = 'gpt-4-1106-preview'
#model_name = "fbn-norm"
model_names = ["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "fbn-norm"]

benchmark_questions_file = "./agronomy_benchmark_questions.json"

for model_name in model_names:
    print ()
    print (model_name)
    print()
    run_benchmark(model_name, benchmark_questions_file)

compare_benchmark_scores()




