import json
import os

def analyze_results(benchmark_results_dir):
    model_stats = {}
    question_stats = {}

    for benchmark_file in os.listdir(benchmark_results_dir):
        with open(os.path.join(benchmark_results_dir, benchmark_file), 'r') as f:
            benchmark_results = json.load(f)

        for result in benchmark_results['multiple_choice']:
            model_answer = result['model_answer']
            question_id = result['id']
            correct = result['correct_answer'] == model_answer

            if model_answer not in model_stats:
                model_stats[model_answer] = {'correct': 0, 'incorrect': 0}

            if correct:
                model_stats[model_answer]['correct'] += 1
            else:
                model_stats[model_answer]['incorrect'] += 1
                if question_id not in question_stats:
                    question_stats[question_id] = 1
                else:
                    question_stats[question_id] += 1

    most_missed_questions = sorted(question_stats.items(), key=lambda x: x[1], reverse=True)

    return model_stats, most_missed_questions

import pandas as pd

benchmark_results_dir = './benchmark_results/model_results'
model_stats, most_missed_questions = analyze_results(benchmark_results_dir)
df = pd.DataFrame(most_missed_questions, columns=['Question ID', 'Missed Count'])
with open('./agronomy_benchmark_questions.json', 'r') as f:
    questions = json.load(f)['multiple_choice']
df['Question Text'] = df['Question ID'].apply(lambda x: next((item for item in questions if item["id"] == x), {'question': None})['question'])
df['Answer Options'] = df['Question ID'].apply(lambda x: next((item for item in questions if item["id"] == x), {'answer_options': None})['answer_options'])
df = df[['Question ID', 'Question Text', 'Answer Options', 'Missed Count']]
print(df.head(30).to_string(index=False))
