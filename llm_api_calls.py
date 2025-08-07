# llm_api_calls.py
import os
import json
import uuid
import asyncio
import aiohttp
import openai
from openai import AsyncOpenAI
import traceback
from tqdm import tqdm as sync_tqdm  # For consistent error logging

# Import shared config for settings like timeout / defaults
import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cache for initialized OpenAI-compatible clients
_openai_clients = {}  # { (base_url, api_key_env): client }


def _get_openai_client(model_config):
    """
    Initializes (and memo-izes) an AsyncOpenAI client based on the
    model’s base_url + api_key_env combination.
    """
    client_key = (model_config.get("base_url"), model_config["api_key_env"])
    if client_key in _openai_clients:
        return _openai_clients[client_key]

    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"API Key environment variable '{model_config['api_key_env']}' "
            f"not set for {model_config['id']}."
        )

    try:
        client = AsyncOpenAI(
            base_url=model_config.get("base_url") or None,
            api_key=api_key,
            timeout=config.REQUEST_TIMEOUT,
        )
        _openai_clients[client_key] = client
        return client
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize AsyncOpenAI client for {model_config['id']}: {e}"
        ) from e


def _param(model_config, key, default):
    """
    Convenience helper:   _param(model_cfg, "temperature", config.DETERMINISTIC_TEMP)
    Falls back to global default when the key is not present in the model entry.
    """
    return model_config.get(key, default)


# ---------------------------------------------------------------------------
# Provider-specific call functions
# ---------------------------------------------------------------------------


async def _call_openai_sdk(client: AsyncOpenAI,
                           model_config,
                           system_prompt,
                           user_prompt,
                           assistant_prompt):
    """
    Sends a chat completion request to an OpenAI-compatible endpoint.
    Respects per-model overrides for temperature / max_tokens.
    """
    model_name_api = model_config["model_name_api"]
    model_id = model_config.get("id", model_name_api)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        if assistant_prompt:
            messages.append({"role": "assistant", "content": assistant_prompt})

        response = await client.chat.completions.create(
            model=model_name_api,
            messages=messages,
            max_tokens=_param(model_config, "max_tokens", config.TOKENS_TO_REQUEST),
            n=1,
            temperature=_param(model_config, "temperature", config.DETERMINISTIC_TEMP),
            # stream=False
        )
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            return response.choices[0].message.content.strip()
        sync_tqdm.write(f"\nWarning ({model_id}): No response content received.")
        return None

    except openai.APIConnectionError as e:
        sync_tqdm.write(f"\nError ({model_id}): Connection Error - {e}")
    except openai.RateLimitError:
        sync_tqdm.write(f"\nError ({model_id}): Rate Limit Exceeded.")
    except openai.AuthenticationError:
        sync_tqdm.write(
            f"\nError ({model_id}): Authentication failed. "
            f"Check API key ({model_config['api_key_env']}) and endpoint "
            f"({model_config.get('base_url')})."
        )
    except openai.NotFoundError:
        sync_tqdm.write(
            f"\nError ({model_id}): Model not found "
            f"('{model_name_api}')."
        )
    except openai.APIStatusError as e:
        sync_tqdm.write(
            f"\nError ({model_id}): API Error - Status {e.status_code}, "
            f"Response: {e.response}"
        )
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except Exception as e:
        sync_tqdm.write(
            f"\nError ({model_id}): Unexpected API error - {type(e).__name__}: {e}"
        )
    return None


async def _call_fbn(model_config, system_prompt, user_prompt):
    """
    Asynchronously calls the FBN Norm API using aiohttp.
    (The FBN endpoint does not currently support temperature/max_tokens.)
    """
    model_id = model_config.get("id", "fbn/unknown")
    full_prompt = f"{system_prompt}\n{user_prompt}"
    session_id = str(uuid.uuid4())
    url = "https://www.fbn.com/api/notifications/norm/chat/prompt"

    fbn_cookie = os.environ.get("FBN_COOKIE")
    fbn_token = os.environ.get("FBN_CSRF_TOKEN")
    if not fbn_cookie or not fbn_token:
        sync_tqdm.write(
            f"\nError ({model_id}): FBN_COOKIE or FBN_CSRF_TOKEN env vars not set."
        )
        return None

    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "x-csrf-token": fbn_token,
    }
    cookies = {"fbnAuth": fbn_cookie}
    data = {
        "postal_code": "00000",
        "session_id": session_id,
        "prompt": full_prompt,
    }

    try:
        async with aiohttp.ClientSession(cookies=cookies) as session:
            async with session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
            ) as response:
                response.raise_for_status()
                response_data = await response.json()

                if (
                    response_data
                    and "messages" in response_data
                    and response_data["messages"]
                ):
                    last_message = response_data["messages"][-1]
                    if last_message.get("role") == "assistant":
                        return last_message.get("content", "").strip()
                    sync_tqdm.write(
                        f"\nWarning ({model_id}): "
                        f"Last message structure unexpected: {last_message}"
                    )
                else:
                    resp_text = await response.text()
                    sync_tqdm.write(
                        f"\nError ({model_id}): "
                        f"Unexpected response structure: {resp_text[:200]}"
                    )
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
        sync_tqdm.write(f"\nError ({model_id}): HTTP Error - {e.status} {e.message}")
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
        try:
            resp_text = await response.text()
        except Exception:
            resp_text = "[Could not read response text]"
        sync_tqdm.write(
            f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}"
        )
    except Exception as e:
        sync_tqdm.write(
            f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}"
        )
    return None


async def _call_centeotl(model_config, system_prompt, user_prompt):
    """
    Calls the Centeotl API asynchronously using aiohttp.
    (No temperature/max_tokens support in this endpoint.)
    """
    model_id = model_config.get("id", "centeotl/unknown")
    model_name_api = model_config["model_name_api"]
    api_key_env = model_config["api_key_env"]

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sync_tqdm.write(
            f"\nError ({model_id}): API Key env var '{api_key_env}' not set."
        )
        return None

    url = f"https://agquestion.com/{model_name_api}"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    combined_prompt = f"system: {system_prompt} \n user: {user_prompt}"
    data = {"question": combined_prompt}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                answer = response_data.get("answer")
                return answer.strip() if answer else None
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
        sync_tqdm.write(
            f"\nError ({model_id}): HTTP Error - {e.status} - {e.message}"
        )
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
        try:
            resp_text = await response.text()
        except Exception:
            resp_text = "[Could not read response text]"
        sync_tqdm.write(
            f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}"
        )
    except Exception as e:
        sync_tqdm.write(
            f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}"
        )
    return None


async def _call_cropwizard(model_config, system_prompt, user_prompt):
    """
    CropWizard custom endpoint (supports temperature/max_tokens).
    """
    model_id = model_config.get("id")
    api_key_env = model_config.get("api_key_env", "CROPWIZARD_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        sync_tqdm.write(
            f"\nError ({model_id}): API Key env var '{api_key_env}' not set."
        )
        return None

    url = "https://uiuc.chat/api/chat-api/chat"
    data = {
        "model": model_config.get("model_name_api", "qwen2.5:14b-instruct-fp16"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "api_key": api_key,
        "course_name": model_config.get("course_name", "benchmarking"),
        "stream": False,
        "temperature": _param(model_config, "temperature", config.DETERMINISTIC_TEMP),
        "max_tokens": _param(model_config, "max_tokens", config.TOKENS_TO_REQUEST),
        "retrieval_only": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                answer = response_data.get("message", {}).get("content")
                if isinstance(answer, str):
                    return answer.strip()
                if answer is not None:
                    sync_tqdm.write(
                        f"\nWarning ({model_id}): Unexpected answer format: {type(answer)}"
                    )
                    return str(answer).strip()
                sync_tqdm.write(
                    f"\nWarning ({model_id}): No 'message.content' found in response."
                )
    except asyncio.TimeoutError:
        sync_tqdm.write(f"\nError ({model_id}): Request timed out.")
    except aiohttp.ClientResponseError as e:
        sync_tqdm.write(f"\nError ({model_id}): HTTP Error - {e.status} {e.message}")
    except aiohttp.ClientError as e:
        sync_tqdm.write(f"\nError ({model_id}): Request failed - {e}")
    except json.JSONDecodeError:
        try:
            resp_text = await response.text()
        except Exception:
            resp_text = "[Could not read response text]"
        sync_tqdm.write(
            f"\nError ({model_id}): Failed to decode JSON response: {resp_text[:100]}"
        )
    except Exception as e:
        sync_tqdm.write(
            f"\nError ({model_id}): Unexpected error - {type(e).__name__}: {e}"
        )
    return None


# ---------------------------------------------------------------------------
# Main API router
# ---------------------------------------------------------------------------

async def call_llm_api(model_config,
                       system_prompt,
                       user_prompt,
                       assistant_prompt=None):
    """
    Routes a benchmark question to the correct provider handler.

    Returns:
        str | None
    """
    provider = model_config.get("provider")
    model_id = model_config.get("id", "Unknown Model")

    try:
        if provider == "openai_compatible":
            client = _get_openai_client(model_config)
            return await _call_openai_sdk(
                client,
                model_config,
                system_prompt,
                user_prompt,
                assistant_prompt,
            )
        elif provider == "fbn":
            return await _call_fbn(model_config, system_prompt, user_prompt)
        elif provider == "centeotl":
            return await _call_centeotl(model_config, system_prompt, user_prompt)
        elif provider == "cropwizard":
            return await _call_cropwizard(model_config, system_prompt, user_prompt)
        else:
            sync_tqdm.write(
                f"\nError ({model_id}): Unknown provider type '{provider}' configured."
            )
            return None
    except (ValueError, RuntimeError) as e:
        sync_tqdm.write(f"\nError ({model_id}): Client initialization failed - {e}")
    except Exception as e:
        sync_tqdm.write(
            f"\nError ({model_id}): Unexpected error in call_llm_api "
            f"({provider}) - {type(e).__name__}: {e}"
        )
    return None