# Leaderboard


![Overall Scores for All Models](./benchmark_results/all_models_overall_score.png)


| Model Name | Overall Score | Date_Tested  | nutrient_management | soil_and_water | pest_management | crop_management | V1_benchmark_questions | License Type |
|------------|---------------|-------------|-------|-------|-------|-------|-------|-------|
| claude-3.5-sonnet | 89.81% | 2024-06-21 | 88.24% | 91.25% | 90.32% | 89.71% | 89.52% | Proprietary |
| gpt-4o | 89.56% | 2024-06-14 | 87.06% | 88.75% | 90.32% | 91.18% | 90.48% | Proprietary |
| claude-3-opus | 88.67% | 2024-06-14 | 89.41% | 87.5% | 87.1% | 92.65% | 86.67% | Proprietary |
| fbn-norm | 87.74% | 2024-06-14 | 84.71% | 87.5% | 88.71% | 88.24% | 89.52% | Proprietary |
| llama-3.1-405b-instruct | 86.87% | 2024-07-23 | 87.06% | 85.0% | 85.48% | 88.24% | 88.57% | Open Source |
| gemini-pro-1.5 | 85.01% | 2024-06-14 | 84.71% | 87.5% | 82.26% | 86.76% | 83.81% | Proprietary |
| gpt-4 | 84.92% | 2024-06-14 | 83.53% | 83.75% | 83.87% | 86.76% | 86.67% | Proprietary |
| llama-3.1-70b-instruct | 84.84% | 2024-07-23 | 77.65% | 83.75% | 90.32% | 86.76% | 85.71% | Open Source |
| qwen-2-72b-instruct | 84.15% | 2024-06-14 | 81.18% | 83.75% | 83.87% | 85.29% | 86.67% | Open Source |
| llama-3-70b-instruct | 81.49% | 2024-06-14 | 78.82% | 78.75% | 82.26% | 83.82% | 83.81% | Open Source |
| gpt-4o-mini | 80.92% | 2024-07-23 | 77.65% | 86.25% | 77.42% | 82.35% | 80.95% | Proprietary |
| gemini-flash-1.5 | 79.53% | 2024-06-14 | 74.12% | 76.25% | 83.87% | 85.29% | 78.1% | Proprietary |
| claude-3-haiku | 75.13% | 2024-06-14 | 71.76% | 73.75% | 79.03% | 72.06% | 79.05% | Proprietary |
| phi-3-medium-128k-instruct | 73.49% | 2024-06-14 | 67.06% | 77.5% | 72.58% | 77.94% | 72.38% | Open Source |
| yi-34b-chat | 71.23% | 2024-06-14 | 68.24% | 68.75% | 79.03% | 70.59% | 69.52% | Open Source |
| gpt-3.5-turbo | 68.44% | 2024-06-14 | 67.06% | 63.75% | 69.35% | 70.59% | 71.43% | Proprietary |
| phi-3-mini-128k-instruct | 67.53% | 2024-06-14 | 60.0% | 71.25% | 67.74% | 69.12% | 69.52% | Open Source |
| llama-3.1-8b-instruct | 64.53% | 2024-07-23 | 57.65% | 65.0% | 66.13% | 69.12% | 64.76% | Open Source |
| llama-3-8b-instruct:nitro | 63.82% | 2024-06-14 | 54.12% | 68.75% | 61.29% | 72.06% | 62.86% | Open Source |
| hermes-2-pro-llama-3-8b | 61.99% | 2024-06-14 | 57.65% | 57.5% | 62.9% | 66.18% | 65.71% | Open Source |



# What is this?
We are benchmarking the ability for different models to give correct answers to Agronomy questions. This is a simple, 98 multiple-choice question benchmark today, and I plan to make it more complete and challenging in the future.

# Why?
When building new models for agriculture, it's important to know if your model is getting better or worse. This is a simple benchmark to help us determine if we are improving the agronomic ability of new models and by how much.

# Roadmap
1. Make it harder! These are fairly basic questions. We should add short and long answer questions (to be evaluated against example correct answers)
2. Add questions for international regions
3. Add more models to the leaderboard


# Updates

## 2024-08-16
1. Thank you to , who contributed community questions!
2. Benchmarks have been run against the new community questions on select models.
3. Nous Hermes 3 405b added & benchmarked.

## 2024-07-24
1. Added Meta Llama 3.1 models
2. Added OpenAI GPT4o-mini

## 2024-06-15
1. Added 295 more questions to the benchmark.
2. Added quesiton cateogires
3. Re-ran with all models
4. Added graphs as output for visual comparison. 


## 2024-01-17
1. Updated benchmark questions to remove incorrectly formed questions (for eaxample, the most missed question across all models was "e. both symptoms occur across the field and stunted roots", which is clearly not a properly formed question). 

2. Included chat prompt templates for models that require chat templates. 

3. Re-ran benchmark against all models after fixes in place and updated leaderboard.



