# llm_api_calls.py
import os
import json
import uuid
import asyncio
import aiohttp
import openai
from openai import AsyncOpenAI
import traceback
from tqdm import tqdm as sync_tqdm # For consistent error logging

# Import shared config for settings like timeout
import config

# --- Client/Session Management ---

# Cache for initialized OpenAI-compatible clients
_openai_clients = {} # { (base_url, api_key_env): client }

def _get_openai_client(model_config):
    """Initializes and caches an Async OpenAI client based on config."""
    client_key = (model_config.get("base_url"), model_config["api_key_env"])
    if client_key in _openai_clients:
        return _openai_clients[client_key]

    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        raise ValueError(f"API Key environment variable '{model_config['api_key_env']}' not set for {model_config['id']}.")

    try:
        # Use explicit None for base_url if not provided or empty, OpenAI SDK handles default
        base_url = model_config.get("base_url") or None
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.REQUEST_TIMEOUT
        )
        _openai_clients[client_key] = client
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Async OpenAI client for {model_config['id']}: {e}") from e

# --- Provider-Specific Call Functions ---

async def _call_openai_sdk(client: AsyncOpenAI, model_config, system_prompt, user_prompt, assistant_prompt):
    """Calls an OpenAI-compatible API using an initialized async client."""
    model_name_api = model_config["model_name_api"]
    model_id = model_config.get('id', model_name_api)
    try:
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        if user_prompt: messages.append({"role": "user", "content": user_prompt})
        # Note: Assistant prompt might not be supported/used same way by all models
        if assistant_prompt: messages.append({"role": "assistant", "content": assistant_prompt})

        response = await client.chat.completions.create(
            model=model_name_api,
            messages=messages,
            max_tokens=config.TOKENS_TO_REQUEST,
            n=1,
            temperature=config.DETERMINISTIC_TEMP,
            # stream=False # Ensure non-streaming for this benchmark context
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            sync_tqdm.write(f"\nWarning ({model_id}): No response content received.")
            return None
    except openai.APIConnectionError as e:
        sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Connection Error - {e}")
    except openai.RateLimitError:
        sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Rate Limit Exceeded.")
    except openai.AuthenticationError:
         sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Authentication failed. Check API key ({model_config['api_key_env']}) and endpoint ({model_config.get('base_url')}).")
    except openai.NotFoundError:
         sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Model not found. Check API model name ('{model_name_api}').")
    except openai.APIStatusError as e:
        sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): API Error - Status {e.status_code}, Response: {e.response}")
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Request timed out.")
    except Exception as e:
        sync_tqdm.write(f"\nError ({model_id} / {model_name_api}): Unexpected API error - {type(e).__name__}: {e}")
        # sync_tqdm.write(traceback.format_exc()) # Optional: more detailed traceback
    return None

async def _call_fbn(model_config, system_prompt, user_prompt):
    """Asynchronously calls the FBN Norm API using aiohttp."""
    model_id = model_config.get('id', 'fbn/unknown')
    full_prompt = f"{system_prompt}\n{user_prompt}"
    session_id = str(uuid.uuid4())
    url = 'https://www.fbn.com/api/notifications/norm/chat/prompt'

    fbn_cookie = os.environ.get("FBN_COOKIE")
    fbn_token = os.environ.get("FBN_CSRF_TOKEN")

    if not fbn_cookie or not fbn_token:
        sync_tqdm.write(f"\nError ({model_id}): FBN_COOKIE or FBN_CSRF_TOKEN environment variables not set.")
        return None

    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'x-csrf-token': fbn_token,
        # Include fbnAuth in headers as per original example, if needed
        # 'fbnAuth': fbn_cookie
    }
    cookies = { 'fbnAuth': fbn_cookie }
    data = {
        "postal_code": "00000", # Assuming default needed
        'session_id': session_id,
        'prompt': full_prompt,
    }

    try:
        async with aiohttp.ClientSession(cookies=cookies) as session: # Pass cookies to session
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)) as response:
                response.raise_for_status()
                response_data = await response.json()

                # Adapt response parsing based on actual FBN API structure
                # Example from user's prompt_fbn_norm: response["messages"][-1]["content"]
                if response_data and "messages" in response_data and response_data["messages"]:
                    # Assuming the last message is the assistant's reply
                    last_message = response_data["messages"][-1]
                    if last_message.get("role") == "assistant": # Or check role if available
                        return last_message.get("content", "").strip()
                    else:
                         # Handle cases where last message isn't from assistant?
                         sync_tqdm.write(f"\nWarning ({model_id}): Last message structure unexpected: {last_message}")
                         # Fallback: try to find first assistant message? Or just return None?
                         # For now, return None if last isn't clearly the answer.
                         return None

                else:
                    resp_text = await response.text()
                    sync_tqdm.write(f"\nError ({model_id}): Unexpected response structure: {resp_text[:200]}")
                    return None
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
         sync_tqdm.write(f"\nError ({model_id}): HTTP Error - {e.status} {e.message}")
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
        try: resp_text = await response.text()
        except: resp_text = "[Could not read response text]"
        sync_tqdm.write(f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}")
    except Exception as e:
        sync_tqdm.write(f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}")
        # sync_tqdm.write(traceback.format_exc()) # Optional: more detailed traceback
    return None


async def _call_centeotl(model_config, system_prompt, user_prompt):
    """Calls the Centeotl API asynchronously using aiohttp."""
    model_id = model_config.get('id', 'centeotl/unknown')
    model_name_api = model_config["model_name_api"]
    api_key_env = model_config["api_key_env"]

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sync_tqdm.write(f"\nError ({model_id}): API Key env var '{api_key_env}' not set.")
        return None

    url = f'https://agquestion.com/{model_name_api}' # Use model_name_api from config
    headers = { 'Content-Type': 'application/json', 'X-API-Key': api_key }
    # Combine prompts as per user's example
    combined_prompt = f"system: {system_prompt} \n user: {user_prompt}"
    data = { 'question': combined_prompt }

    try:
        # Create session per call for simplicity
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)) as response:
                response.raise_for_status()
                response_data = await response.json()
                answer = response_data.get('answer')
                return answer.strip() if answer else None # Return None if no answer key/empty
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
         sync_tqdm.write(f"\nError ({model_id}): HTTP Error - {e.status} - {e.message}")
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
         try: resp_text = await response.text()
         except: resp_text = "[Could not read response text]"
         sync_tqdm.write(f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}")
    except Exception as e:
        sync_tqdm.write(f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}")
        # sync_tqdm.write(traceback.format_exc())
    return None


async def _call_cropwizard(model_config, system_prompt, user_prompt):
    model_id = model_config.get('id')
    # Use model_name_api from config, fall back to a default if necessary
    model_name_api = model_config.get("model_name_api", "qwen2.5:14b-instruct-fp16")
    api_key_env = model_config.get("api_key_env", "CROPWIZARD_API_KEY") # Default env var name
    course_name = model_config.get("course_name", "benchmarking") # Default course name

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sync_tqdm.write(f"\nError ({model_id}): API Key env var '{api_key_env}' not set.")
        return None

    url = "https://uiuc.chat/api/chat-api/chat"
    headers = { 'Content-Type': 'application/json' }
    data = {
        "model": model_name_api, # Use model name from config
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "api_key": api_key,
        "course_name": course_name,
        "stream": False,
        "temperature": config.DETERMINISTIC_TEMP,
        "retrieval_only": False
    }

    # Add error handling similar to other async functions
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)) as response:
                response.raise_for_status() # Raise HTTP Errors
                response_data = await response.json()

                # Extract the 'message' field, specific to CropWizard API
                answer = response_data.get('message', {}).get('content') # Updated based on likely structure
                # Or adjust if the structure is different, e.g., just response_data.get('message')
                if isinstance(answer, str):
                    return answer.strip()
                elif answer is not None:
                     sync_tqdm.write(f"\nWarning ({model_id}): Unexpected answer format: {type(answer)}")
                     return str(answer).strip() # Attempt to convert to string
                else:
                    sync_tqdm.write(f"\nWarning ({model_id}): No 'message' content found in response.")
                    # Log part of the response for debugging if needed
                    # sync_tqdm.write(f"Response structure: {str(response_data)[:200]}")
                    return None

    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
         sync_tqdm.write(f"\nError ({model_id}): HTTP Error - {e.status} {e.message}")
    except aiohttp.ClientError as e: # Catch other client errors (connection, etc.)
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
        try: resp_text = await response.text()
        except: resp_text = "[Could not read response text]"
        sync_tqdm.write(f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}")
    except Exception as e:
        sync_tqdm.write(f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}")
        # sync_tqdm.write(traceback.format_exc()) # Optional for debugging
    return None



# --- Main API Call Router ---

async def call_llm_api(model_config, system_prompt, user_prompt, assistant_prompt=None):
    """
    Routes the benchmark question to the appropriate LLM API based on model_config.

    Returns:
        str: The processed response content from the LLM.
        None: If the API call fails, times out, or returns no content after retries.
    """
    provider = model_config.get("provider")
    model_id = model_config.get("id", "Unknown Model")

    try:
        if provider == "openai_compatible":
            # Handles direct OpenAI, OpenRouter, Dhenu (if configured as compatible) etc.
            client = _get_openai_client(model_config)
            return await _call_openai_sdk(client, model_config, system_prompt, user_prompt, assistant_prompt)
        elif provider == "fbn":
            return await _call_fbn(model_config, system_prompt, user_prompt)
        elif provider == "centeotl":
            return await _call_centeotl(model_config, system_prompt, user_prompt)
        elif provider == "cropwizard":
             return await _call_cropwizard(model_config, system_prompt, user_prompt)
        # Add elif blocks for other providers (like a dedicated dhenu if not using openai_compatible)
        # elif provider == "dhenu": # Example if dhenu needs specific non-SDK handling
            # return await _call_specific_dhenu(...)
        else:
            sync_tqdm.write(f"\nError ({model_id}): Unknown provider type '{provider}' configured.")
            return None
    except (ValueError, RuntimeError) as e: # Catch client init errors
        sync_tqdm.write(f"\nError ({model_id}): Client initialization failed - {e}")
        return None
    except Exception as e:
        # Catch unexpected errors during routing itself
        sync_tqdm.write(f"\nError ({model_id}): Unexpected error during API call routing for provider '{provider}' - {type(e).__name__}: {e}")
        # sync_tqdm.write(traceback.format_exc())
        return None
