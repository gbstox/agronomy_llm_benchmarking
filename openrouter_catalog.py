import json
import os
import re
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import config


_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_openrouter_catalog_cache = {}

_OPEN_SOURCE_PREFIXES = {
    "allenai/",
    "arcee-ai/",
    "deepseek/",
    "google/gemma",
    "meta-llama/",
    "mistralai/",
    "nvidia/",
    "openai/gpt-oss",
    "qwen/",
    "xiaomi/",
}

_PROPRIETARY_PREFIXES = {
    "aion-labs/",
    "anthropic/",
    "bytedance-seed/",
    "google/gemini",
    "inception/",
    "liquid/",
    "minimax/",
    "moonshotai/",
    "openai/",
    "stepfun/",
    "upstage/",
    "writer/",
    "x-ai/",
    "z-ai/",
}


def _read_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.is_file():
        return None

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, env_value = stripped.split("=", 1)
        if env_key.strip() == key:
            return env_value.strip().strip('"').strip("'")
    return None


def _load_openrouter_api_key(project_root: Path, api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env) or _read_env_value(project_root / ".env", api_key_env)
    if not api_key:
        raise ValueError(f"Missing OpenRouter API key in env var '{api_key_env}'.")
    return api_key


def _load_latest_result_date(results_dir: Path) -> datetime | None:
    latest_result_date = None

    for result_file in results_dir.glob("*_answers.json"):
        try:
            data = json.loads(result_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        raw_date = data.get("date")
        if not raw_date:
            continue

        try:
            result_date = datetime.fromisoformat(raw_date)
        except ValueError:
            continue

        if latest_result_date is None or result_date > latest_result_date:
            latest_result_date = result_date

    return latest_result_date


def _fetch_openrouter_models(api_key: str) -> list[dict]:
    request = urllib.request.Request(
        _OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Unexpected OpenRouter /models response format.")
    return data


def _safe_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_text_model(model_data: dict) -> bool:
    architecture = model_data.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []

    return "text" in input_modalities and "text" in output_modalities


def _is_free_variant(model_data: dict) -> bool:
    model_id = model_data.get("id", "")
    name = str(model_data.get("name", "")).lower()
    return model_id.endswith(":free") or "(free)" in name


def _is_cloaked_model(model_data: dict) -> bool:
    if not config.OPENROUTER_SKIP_CLOAKED_MODELS:
        return False

    model_id = model_data.get("id", "")
    canonical_slug = str(model_data.get("canonical_slug", ""))
    name = str(model_data.get("name", ""))
    description = str(model_data.get("description", ""))
    searchable_text = " ".join([model_id, canonical_slug, name, description]).lower()

    if model_id.startswith("openrouter/"):
        return True

    return "cloak" in searchable_text


def _infer_access_type(model_data: dict) -> str:
    model_id = model_data.get("id", "")

    if str(model_data.get("hugging_face_id", "")).strip():
        return "open source"

    if any(model_id.startswith(prefix) for prefix in _OPEN_SOURCE_PREFIXES):
        return "open source"

    if any(model_id.startswith(prefix) for prefix in _PROPRIETARY_PREFIXES):
        return "proprietary"

    return "unknown"


def _extract_price_usd_per_mtok(model_data: dict) -> float | None:
    pricing = model_data.get("pricing") or {}
    completion_price = _safe_float(pricing.get("completion"))
    if completion_price is None:
        return None
    return completion_price * 1_000_000


def _build_metadata_map_from_models(all_models: list[dict]) -> dict[str, dict]:
    metadata_map = {}
    for model_data in all_models:
        model_id = model_data.get("id")
        if not model_id:
            continue

        metadata_map[model_id] = {
            "id": model_id,
            "access": _infer_access_type(model_data),
            "price_usd_per_mtok": _extract_price_usd_per_mtok(model_data),
            "created": model_data.get("created"),
            "hugging_face_id": str(model_data.get("hugging_face_id", "")).strip() or None,
            "canonical_slug": str(model_data.get("canonical_slug", "")).strip() or model_id,
            "is_free_variant": _is_free_variant(model_data),
            "is_cloaked": _is_cloaked_model(model_data),
        }

    return metadata_map


def _get_safe_results_filename(model_id: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", model_id) + "_answers.json"


def _load_discovery_state(state_file: Path) -> dict | None:
    if not state_file.is_file():
        return None

    try:
        data = json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    return data


def _save_discovery_state(state_file: Path, state_data: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state_data, indent=4))


def _clear_discovery_state(state_file: Path) -> None:
    if state_file.exists():
        state_file.unlink()


def _result_is_complete(results_dir: Path, model_id: str) -> bool:
    result_file = results_dir / _get_safe_results_filename(model_id)
    if not result_file.is_file():
        return False

    try:
        data = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    status = data.get("benchmark_status")
    if status in {"rate_limited", "incomplete", "run_error"}:
        return False
    if "run_error" in data:
        return False
    return True


def _build_preferred_model_ids(metadata_map: dict[str, dict]) -> set[str]:
    family_members = {}
    for model_id, metadata in metadata_map.items():
        if metadata.get("is_cloaked"):
            continue
        if config.OPENROUTER_SKIP_ALL_FREE_VARIANTS and metadata.get("is_free_variant"):
            continue
        family_key = metadata.get("canonical_slug") or model_id
        family_members.setdefault(family_key, []).append(metadata)

    preferred_ids = set()
    for members in family_members.values():
        if config.OPENROUTER_SKIP_FREE_VARIANTS_WHEN_BASE_EXISTS:
            non_free_members = [member for member in members if not member.get("is_free_variant")]
            chosen_members = non_free_members or members
        else:
            chosen_members = members

        for member in chosen_members:
            preferred_ids.add(member["id"])

    return preferred_ids


def fetch_openrouter_model_metadata_map(
    project_root: str | Path,
    api_key_env: str,
) -> dict[str, dict]:
    root_path = Path(project_root)
    cache_key = (str(root_path.resolve()), api_key_env)
    if cache_key in _openrouter_catalog_cache:
        return _openrouter_catalog_cache[cache_key]

    api_key = _load_openrouter_api_key(root_path, api_key_env)
    all_models = _fetch_openrouter_models(api_key)
    metadata_map = _build_metadata_map_from_models(all_models)

    _openrouter_catalog_cache[cache_key] = metadata_map
    return metadata_map


def build_new_openrouter_model_configs(
    project_root: str | Path,
    results_dir: str | Path,
    api_key_env: str,
    existing_model_ids: set[str] | None = None,
) -> tuple[datetime | None, list[dict]]:
    """
    Returns OpenRouter-backed model configs for text-capable models created
    within the configured lookback window, while persisting unfinished discovery
    work across runs.
    """
    root_path = Path(project_root)
    results_path = Path(results_dir)
    known_ids = existing_model_ids or set()
    state_file = root_path / config.OPENROUTER_DISCOVERY_STATE_FILE

    api_key = _load_openrouter_api_key(root_path, api_key_env)
    all_models = _fetch_openrouter_models(api_key)
    metadata_map = _build_metadata_map_from_models(all_models)
    preferred_model_ids = _build_preferred_model_ids(metadata_map)
    state_data = _load_discovery_state(state_file)
    recent_model_cutoff = datetime.now() - timedelta(days=config.OPENROUTER_INITIAL_LOOKBACK_DAYS)

    latest_result_date = None
    candidate_ids = []
    if state_data and isinstance(state_data.get("pending_model_ids"), list):
        reference_date_raw = state_data.get("reference_result_date")
        if reference_date_raw:
            try:
                latest_result_date = datetime.fromisoformat(reference_date_raw)
            except ValueError:
                latest_result_date = None
        candidate_ids = [
            model_id
            for model_id in state_data["pending_model_ids"]
            if model_id not in known_ids and model_id in preferred_model_ids
        ]
    else:
        latest_result_date = recent_model_cutoff

        for model_data in sorted(all_models, key=lambda item: item.get("created", 0)):
            created_ts = model_data.get("created")
            model_id = model_data.get("id")

            if not created_ts or not model_id or model_id in known_ids:
                continue

            try:
                created_at = datetime.fromtimestamp(created_ts)
            except (TypeError, ValueError, OSError):
                continue

            if created_at <= latest_result_date or not _is_text_model(model_data):
                continue

            metadata = metadata_map.get(model_id, {})
            if model_id not in preferred_model_ids or metadata.get("is_cloaked"):
                continue

            candidate_ids.append(model_id)

        _save_discovery_state(
            state_file,
            {
                "reference_result_date": latest_result_date.isoformat(),
                "pending_model_ids": candidate_ids,
            },
        )

    new_model_configs = []
    remaining_candidate_ids = []
    for model_id in candidate_ids:
        if model_id in known_ids or _result_is_complete(results_path, model_id):
            continue
        if model_id not in metadata_map:
            continue
        if model_id not in preferred_model_ids or metadata_map.get(model_id, {}).get("is_cloaked"):
            continue
        remaining_candidate_ids.append(model_id)

        discovered_model_config = {
            "id": model_id,
            "provider": "openai_compatible",
            "api_key_env": api_key_env,
            "base_url": _OPENROUTER_BASE_URL,
            "model_name_api": model_id,
            "access": metadata_map.get(model_id, {}).get("access", "unknown"),
            "question_concurrency": config.OPENROUTER_DISCOVERED_MODEL_QUESTION_CONCURRENCY,
            "max_retries": config.OPENROUTER_DISCOVERED_MODEL_MAX_RETRIES,
            "rate_limit_backoff_seconds": config.OPENROUTER_DISCOVERED_MODEL_RATE_LIMIT_BACKOFF_SECONDS,
        }

        new_model_configs.append(discovered_model_config)

    if remaining_candidate_ids:
        _save_discovery_state(
            state_file,
            {
                "reference_result_date": latest_result_date.isoformat() if latest_result_date else None,
                "pending_model_ids": remaining_candidate_ids,
            },
        )
    else:
        _clear_discovery_state(state_file)

    return latest_result_date, new_model_configs
