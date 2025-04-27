# Leaderboard


![Overall Scores for All Models](./benchmark_results/all_models_overall_score.png)

## Price x Performance
![Overall Scores for All Models](./benchmark_results/performance_vs_price_score_color.png)



| Model Name                           | Overall Score | Access      | Date Tested | Price ($/Mtok) | V1 Benchmark Questions | Community Questions Fbn | Crop Management | Nutrient Management | Pest Management | Soil And Water |
|-------------------------------------|---------------|------------|-------------|----------------|------------------------|-------------------------|-----------------|---------------------|-----------------|----------------|
| google/gemini-2.5-pro-preview-03-25  |         92.94% | Proprietary | 2025-04-27  |    $10.0000    |                  90.5% |                   80.0% |           97.1% |               92.9% |           95.2% |          95.0% | (395/425)
| anthropic/claude-3.5-sonnet-20240620 |         89.18% | Proprietary | 2025-04-27  |    $15.0000    |                  89.5% |                   80.0% |           89.7% |               87.1% |           91.9% |          91.2% | (379/425)
| openai/gpt-4o                        |         88.00% | Proprietary | 2025-04-27  |    $10.0000    |                  86.7% |                   80.0% |           88.2% |               87.1% |           88.7% |          92.5% | (374/425)
| norm                                 |         87.53% | Proprietary | 2025-04-27  |      N/A       |                  89.5% |                   76.0% |           89.7% |               85.9% |           87.1% |          88.8% | (372/425)
| meta-llama/llama-4-maverick          |         87.53% | Open Source | 2025-04-27  |    $0.2000     |                  89.5% |                   84.0% |           88.2% |               84.7% |           88.7% |          87.5% | (372/425)
| deepseek/deepseek-chat               |         86.82% | Open Source | 2025-04-27  |    $0.8900     |                  89.5% |                   72.0% |           83.8% |               89.4% |           88.7% |          86.2% | (369/425)
| meta-llama/llama-3.1-405b-instruct   |         84.24% | Open Source | 2025-04-27  |    $0.8000     |                  88.6% |                   68.0% |           88.2% |               85.9% |           83.9% |          78.8% | (358/425)
| meta-llama/llama-3.1-70b-instruct    |         82.82% | Open Source | 2025-04-27  |    $0.2800     |                  83.8% |                   72.0% |           89.7% |               81.2% |           87.1% |          77.5% | (352/425)
| google/gemma-3-27b-it                |         82.59% | Open Source | 2025-04-27  |    $0.2000     |                  83.8% |                   64.0% |           82.3% |               82.3% |           83.9% |          86.2% | (351/425)
| meta-llama/llama-4-scout             |         79.06% | Open Source | 2025-04-27  |    $0.1000     |                  80.0% |                   60.0% |           79.4% |               78.8% |           82.3% |          81.2% | (336/425)
| openai/gpt-4o-mini                   |         78.59% | Proprietary | 2025-04-27  |    $0.6000     |                  78.1% |                   72.0% |           82.3% |               74.1% |           75.8% |          85.0% | (334/425)
| meta-llama/llama-3-70b-instruct      |         78.35% | Open Source | 2025-04-27  |    $0.4000     |                  83.8% |                   52.0% |           80.9% |               77.7% |           80.7% |          76.2% | (333/425)
| mistralai/mixtral-8x7b-instruct      |         73.65% | Open Source | 2025-04-27  |    $0.2400     |                  79.0% |                   48.0% |           76.5% |               68.2% |           77.4% |          75.0% | (313/425)
| dhenu2-in-8b-preview                 |         66.82% | Proprietary | 2025-04-27  |      N/A       |                  71.4% |                   52.0% |           64.7% |               61.2% |           71.0% |          70.0% | (284/425)
| openai/gpt-3.5-turbo                 |         65.65% | Proprietary | 2025-04-27  |    $1.5000     |                  73.3% |                   32.0% |           67.7% |               62.4% |           69.3% |          65.0% | (279/425)
| meta-llama/llama-3.1-8b-instruct     |         63.29% | Open Source | 2025-04-27  |    $0.0300     |                  66.7% |                   68.0% |           67.7% |               48.2% |           64.5% |          68.8% | (269/425)
| mistralai/mistral-7b-instruct        |         62.59% | Open Source | 2025-04-27  |    $0.0550     |                  61.9% |                   36.0% |           75.0% |               52.9% |           69.3% |          66.2% | (266/425)
| google/gemma-3-4b-it                 |         61.65% | Open Source | 2025-04-27  |    $0.0400     |                  62.9% |                   48.0% |           67.7% |               54.1% |           69.3% |          61.2% | (262/425)



# What is this?
We are benchmarking the ability for different models to give correct answers to Agronomy questions. This is a simple, 98 multiple-choice question benchmark today, and I plan to make it more complete and challenging in the future.

# Why?
When building new models for agriculture, it's important to know if your model is getting better or worse. This is a simple benchmark to help us determine if we are improving the agronomic ability of new models and by how much.

# Roadmap
1. Make it harder! These are fairly basic questions. We should add short and long answer questions (to be evaluated against example correct answers)
2. Add questions for international regions
3. Add more models to the leaderboard


# Updates

## 2025-04-27
1. Refactored code to simplify & make easier to maintain.
2. Added price vs performance graph
3. Fixed several formatting issues in the results display
4. Added models

## 2024-08-16
1. Thank you to Farmers Business Network, who contributed community questions!
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



