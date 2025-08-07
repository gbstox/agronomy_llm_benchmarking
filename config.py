# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- File/Directory Paths ---
BENCHMARK_QUESTIONS_FILE = "./benchmark_questions/combined_benchmark.json"
BENCHMARK_RESULTS_DIR = 'benchmark_results/model_results'
GRAPHS_BASE_DIR = 'benchmark_results'

# --- Execution Mode ---
# Controls which models are run during execution.
# Options:
#   'all': Run all models defined in MODELS_TO_RUN, overwriting existing results.
#   'missing': Only run models defined in MODELS_TO_RUN for which no results file exists.
#   'reports_only': Do not run any models, only generate reports from existing results.
RUN_MODE = 'missing' # Change this to 'missing' or 'reports_only' as needed

# --- Benchmark Settings ---
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30 # Slightly longer timeout for potentially slower models/retries
TOKENS_TO_REQUEST = 1000     # default max_tokens
DETERMINISTIC_TEMP = 0.0   # default temperature

# --- Concurrency Limits ---
# Max number of different MODELS to benchmark simultaneously
MAX_CONCURRENT_MODELS = 5
# Max number of QUESTIONS to process concurrently *within* a single model's run
QUESTIONS_CONCURRENCY_PER_MODEL = 10 # Adjust based on API rate limits

# --- Base Prompts ---
SYSTEM_PROMPT = """You are a helpful and brilliant agronomist. For the following multiple choice Question, answer with the key of the correct answer_options value. Your response must be ONLY the answer_options key of the correct answer_options value and no other text. respond with ONLY a single letter key from answer_options. If anything other than a single letter is submitted you will FAIL. DO NOT INCLUDE any Prose, explanation, chain of thought, thought process, or other comments. [NO PROSE] [NO REASONING]"""

USER_PROMPT_TEMPLATE = "Question: {question}\n\nanswer_options: {answer_options}"

ASSISTANT_PROMPT = "Correct answer_options key:" # For models supporting assistant priming

# --- Models to Run ---
# Define the models and how to interact with them

MODELS_TO_RUN = [

    # --- Centeotl ---
    # { # Uncomment for Centeotl provider
    #      "id": "centeotl/api",
    #      "provider": "centeotl",
    #      "api_key_env": "CENTEOTL_API_KEY",
    #      "model_name_api": "api" # Model name expected by Centeotl endpoint
    # },


    # --- FBN ---
    # Note: Auth details are read directly within the private_api_calls function
    { 
        "id": "fbn/norm",
        "provider": "fbn",
        "access": "proprietary"
    },

    # --- CropWizard (using custom aiohttp call) ---
    # { # Example CropWizard entry 
    #     "id": "UIUC_CropWizard/qwen2.5:14b-instruct-fp16", # Or other model they support
    #     "provider": "cropwizard",
    #     "api_key_env": "CROPWIZARD_API_KEY", 
    #     "model_name_api": "qwen2.5:14b-instruct-fp16", # Model name expected by their API
    #     "course_name": "benchmarking",
    #     "access": "proprietary" 
    # },

    # --- OpenAI Compatible (via OpenRouter unless specified) ---
    {
        "id": "openai/gpt-5", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "openai/gpt-5", # Specific model name expected by the API
        "access": "open source" 
    },
    {
        "id": "anthropic/claude-opus-4.1", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "anthropic/claude-opus-4.1", # Specific model name expected by the API
        "access": "open source" 
    },
    {
        "id": "openai/gpt-oss-20b", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "openai/gpt-oss-20b", # Specific model name expected by the API
        "access": "open source" 
    },
    {
        "id": "openai/gpt-oss-120b", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "openai/gpt-oss-120b", # Specific model name expected by the API
        "access": "open source" 
    },
    {
        "id": "gbstockdale/gemma3-4b-agrosirus200k-merged", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://v657ryaz7toltj-8000.proxy.runpod.net/v1", # API endpoint base
        "model_name_api": "gbstox/gemma3-4b-agrosirus200k-merged", # Specific model name expected by the API
        "access": "open source" 
    },
    {
        "id": "anthropic/claude-opus-4", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "anthropic/claude-opus-4", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "qwen/qwen3-235b-a22b", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "qwen/qwen3-235b-a22b", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "qwen/qwen3-32b", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "qwen/qwen3-32b", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "qwen/qwen3-8b", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "qwen/qwen3-8b", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "openai/o4-mini-high", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "openai/o4-mini-high", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "openai/gpt-4o", # Unique identifier for reporting/filenames
        "provider": "openai_compatible", # Type of interaction logic to use
        "api_key_env": "OPENROUTER_API_KEY", # Environment variable for the API key
        "base_url": "https://openrouter.ai/api/v1", # API endpoint base
        "model_name_api": "openai/gpt-4o", # Specific model name expected by the API
        "access": "proprietary" 
    },
    {
        "id": "meta-llama/llama-4-scout", 
        "provider": "openai_compatible", 
        "api_key_env": "OPENROUTER_API_KEY", 
        "base_url": "https://openrouter.ai/api/v1", 
        "model_name_api": "meta-llama/llama-4-scout",
        "access": "open source" 
    },
    {
        "id": "meta-llama/llama-4-maverick", 
        "provider": "openai_compatible", 
        "api_key_env": "OPENROUTER_API_KEY", 
        "base_url": "https://openrouter.ai/api/v1", 
        "model_name_api": "meta-llama/llama-4-maverick",
        "access": "open source" 
    },
    {
        "id": "google/gemma-3-4b-it",
        "provider": "openai_compatible", 
        "api_key_env": "OPENROUTER_API_KEY", 
        "base_url": "https://openrouter.ai/api/v1", 
        "model_name_api": "google/gemma-3-4b-it",
        "access": "open source" 
    },
    {
        "id": "google/gemma-3-27b-it", 
        "provider": "openai_compatible", 
        "api_key_env": "OPENROUTER_API_KEY", 
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "google/gemma-3-27b-it",
        "access": "open source"  
    },
    {
        "id": "google/gemini-2.5-pro-preview-03-25", 
        "provider": "openai_compatible", 
        "api_key_env": "OPENROUTER_API_KEY", 
        "base_url": "https://openrouter.ai/api/v1", 
        "model_name_api": "google/gemini-2.5-pro-preview-03-25",
        "access": "proprietary"  
    },
    {
        "id": "anthropic/claude-3.5-sonnet",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "anthropic/claude-3.5-sonnet-20240620",
        "access": "proprietary" 
    },
    {
        "id": "meta-llama/llama-3-70b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "meta-llama/llama-3-70b-instruct",
        "access": "open source" 
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "meta-llama/llama-3.1-8b-instruct",
        "access": "open source" 
    },
    {
        "id": "meta-llama/llama-3.1-70b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "meta-llama/llama-3.1-70b-instruct",
        "access": "open source" 
    },
    {
        "id": "meta-llama/llama-3.1-405b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "meta-llama/llama-3.1-405b-instruct",
        "access": "open source" 
    },
    {
        "id": "mistralai/mixtral-8x7b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "mistralai/mixtral-8x7b-instruct",
        "access": "open source" 
    },
    {
        "id": "mistralai/mistral-7b-instruct",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "mistralai/mistral-7b-instruct",
        "access": "open source" 
    },
    {
        "id": "openai/gpt-3.5-turbo",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "openai/gpt-3.5-turbo",
        "access": "proprietary" 
    },
    {
        "id": "openai/gpt-4o-mini",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "openai/gpt-4o-mini",
        "access": "proprietary" 
    }, 
    {
        "id": "pratik/llama3-8b-dhenu-0.1",
        "provider": "openai_compatible",
        "api_key_env": "DHENU_API_KEY",
        "base_url": "https://api.dhenu.ai/v1", 
        "model_name_api": "dhenu2-in-8b-preview", 
        "access": "proprietary" 
    },
    {
        "id": "deepseek/deepseek-chat",
        "provider": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name_api": "deepseek/deepseek-chat",
        "access": "open source" 
    }
]