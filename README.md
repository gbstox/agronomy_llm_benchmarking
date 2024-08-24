# Leaderboard


![Overall Scores for All Models](./benchmark_results/model_results/all_models_overall_score.png)


| Model Name | Overall Score | Date_Tested  | nutrient_management | soil_and_water | pest_management | crop_management | V1_benchmark_questions | community_questions_FBN | License Type |
|------------|---------------|-------------|-------|-------|-------|-------|-------|-------|-------|
| claude-3.5-sonnet | 89.41% | 2024-08-24 | 87.06% | 92.5% | 90.32% | 89.71% | 89.52% | 84.0% | Proprietary |
| gpt-4o | 88.71% | 2024-08-24 | 87.06% | 88.75% | 90.32% | 88.24% | 91.43% | 80.0% | Proprietary |
| FBN norm | 88.0% | 2024-08-24 | 82.35% | 90.0% | 88.71% | 89.71% | 89.52% | 88.0% | Proprietary |
| claude-3-opus | 87.53% | 2024-08-24 | 88.24% | 86.25% | 85.48% | 92.65% | 86.67% | 84.0% | Proprietary |
| llama-3.1-sonar-huge-128k-online | 85.41% | 2024-08-24 | 83.53% | 81.25% | 82.26% | 89.71% | 89.52% | 84.0% | Proprietary |
| gpt-4 | 85.0% | 2024-06-14 | 83.53% | 83.75% | 83.87% | 86.76% | 86.67% | | Proprietary |
| llama-3.1-405b-instruct | 84.0% | 2024-08-24 | 83.53% | 81.25% | 83.87% | 88.24% | 87.62% | 68.0% | Open Source |
| gemini-pro-1.5 | 83.53% | 2024-08-24 | 83.53% | 83.75% | 82.26% | 86.76% | 85.71% | 68.0% | Proprietary |
| hermes-3-llama-3.1-405b | 82.82% | 2024-08-24 | 81.18% | 82.5% | 87.1% | 85.29% | 83.81% | 68.0% | Open Source |
| qwen-2-72b-instruct | 82.59% | 2024-08-24 | 82.35% | 82.5% | 82.26% | 85.29% | 85.71% | 64.0% | Open Source |
| llama-3-70b-instruct | 81.5% | 2024-06-14 | 78.82% | 78.75% | 82.26% | 83.82% | 83.81% | | Open Source |
| gpt-4o-mini | 80.47% | 2024-08-24 | 77.65% | 85.0% | 75.81% | 82.35% | 81.9% | 76.0% | Proprietary |
| llama-3.1-70b-instruct | 80.23% | 2024-08-24 | 75.29% | 81.25% | 82.26% | 89.71% | 80.95% | 60.0% | Open Source |
| gemini-flash-1.5 | 79.0% | 2024-06-14 | 74.12% | 76.25% | 83.87% | 85.29% | 78.1% | | Proprietary |
| mistral-large | 78.12% | 2024-08-24 | 75.29% | 77.5% | 82.26% | 76.47% | 81.9% | 68.0% | Open Source |
| claude-3-haiku | 75.25% | 2024-06-14 | 71.76% | 73.75% | 79.03% | 72.06% | 79.05% | | Proprietary |
| phi-3-medium-128k-instruct | 74.35% | 2024-08-24 | 70.59% | 77.5% | 79.03% | 75.0% | 73.33% | 68.0% | Open Source |
| nous-hermes-yi-34b | 74.35% | 2024-08-24 | 70.59% | 76.25% | 83.87% | 72.06% | 74.29% | 64.0% | Open Source |
| yi-34b-chat | 70.75% | 2024-06-14 | 68.24% | 68.75% | 79.03% | 70.59% | 69.52% | | Open Source |
| phi-3-mini-128k-instruct | 67.5% | 2024-06-14 | 60.0% | 71.25% | 67.74% | 69.12% | 69.52% | | Open Source |
| gpt-3.5-turbo | 64.94% | 2024-08-24 | 62.35% | 61.25% | 70.97% | 72.06% | 70.48% | 28.0% | Proprietary |
| llama-3-8b-instruct:nitro | 63.5% | 2024-06-14 | 54.12% | 68.75% | 61.29% | 72.06% | 62.86% | | Open Source |
| hermes-2-pro-llama-3-8b | 62.0% | 2024-06-14 | 57.65% | 57.5% | 62.9% | 66.18% | 65.71% | | Open Source |
| llama-3.1-8b-instruct | 59.53% | 2024-08-24 | 51.76% | 58.75% | 61.29% | 66.18% | 59.05% | 68.0% | Open Source |
| mistral-7b-instruct | 51.53% | 2024-08-24 | 41.18% | 50.0% | 62.9% | 60.29% | 53.33% | 32.0% | Open Source |
| mistral-medium | 29.18% | 2024-08-24 | 30.59% | 23.75% | 20.97% | 41.18% | 34.29% | 8.0% | Open Source |
| mixtral-8x7b-instruct | 18.35% | 2024-08-24 | 16.47% | 13.75% | 17.74% | 14.71% | 26.67% | 16.0% | Open Source |


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



