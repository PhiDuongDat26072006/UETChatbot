"""Provider transport helpers (Custom API, OpenRouter, Groq, SambaNova, Gemini) and SSE decoding."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.utils.helpers import get_logger

logger = get_logger("llm")

DEFAULT_PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "custom_api": {
        "model": "gemma-4",
        "candidate_models": ["gemma-4"],
        "base_url": "https://multiple-effectively-lived-marketing.trycloudflare.com",
        "delay": 0.5,
    },
    "groq": {
        "model": "qwen/qwen3.8-27b",
        "candidate_models": ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
        "base_url": "https://api.groq.com/openai/v1",
        "delay": 2.0,  # ~30 requests/phút
    },
    "sambanova": {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "candidate_models": ["Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.3-70B-Instruct"],
        "base_url": "https://api.sambanova.ai/v1",
        "delay": 2.5,  # ~24 requests/phút
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "candidate_models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "delay": 4.0,  # ~15 requests/phút
    },
    "openrouter": {
        "model": "tencent/hy4-flash",
        "candidate_models": [
            "tencent/hy4-flash",
            "tencent/hy4",
            "tencent/hy4-preview",
        ],
        "base_url": "https://openrouter.ai/api/v1",
        "delay": 1.5,
    },
}


def extract_http_error(resp: Any) -> str:
    """Extract an error message from a JSON error payload, falling back to raw response text."""
    err_msg = getattr(resp, "text", "")
    try:
        err_msg = resp.json().get("error", {}).get("message", err_msg)
    except Exception:
        pass
    return err_msg


def iter_sse_deltas(
    response: Any,
    decode_bytes: bool = True,
) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
    """
    Parse OpenAI-compatible SSE streams and yield ``(delta_text, usage_or_none)``.

    When ``decode_bytes=True`` (for ``requests.Response``), raw bytes are decoded explicitly as UTF-8
    to prevent ISO-8859-1 mojibake on Vietnamese characters.
    """
    if decode_bytes:
        try:
            response.encoding = "utf-8"
        except Exception:
            pass
        raw_lines = response.iter_lines(decode_unicode=False)
    else:
        raw_lines = response.iter_lines()

    for raw_line in raw_lines:
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk_json = json.loads(data_str)
            usage = chunk_json.get("usage")
            choices = chunk_json.get("choices") or []
            delta = ""
            if choices:
                delta = choices[0].get("delta", {}).get("content") or ""
            if delta or usage:
                yield delta, usage
        except Exception:
            continue


def apply_openrouter_reasoning(payload: Dict[str, Any], reasoning_effort: Optional[str]) -> None:
    """Attach OpenRouter reasoning configuration to a chat completion payload."""
    if not reasoning_effort:
        return
    eff_low = str(reasoning_effort).strip().lower()
    if eff_low in ("none", "off", "false", "disabled"):
        payload["reasoning"] = {"enabled": False}
    elif eff_low in ("minimal", "low", "medium", "high"):
        payload["reasoning"] = {"effort": eff_low}


def apply_openrouter_provider_routing(
    payload: Dict[str, Any],
    provider_routing: Optional[Dict[str, Any]],
) -> None:
    """Attach OpenRouter provider routing rules when configured."""
    if isinstance(provider_routing, dict) and provider_routing:
        payload["provider"] = dict(provider_routing)


def call_custom_api(
    http_session: Any,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    connect_timeout_s: float,
    timeout_s: float,
    enable_thinking: bool,
    use_stream: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call an OpenAI-compatible custom endpoint (with optional SSE streaming + UTF-8 decode)."""
    if not base_url:
        raise RuntimeError("Chưa cấu hình llm.custom_api.base_url trong config.yaml hoặc LLM_API_BASE_URL")

    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if use_stream:
        payload["stream"] = True

    timeout = (connect_timeout_s, timeout_s)
    if use_stream:
        with http_session.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"CustomAPI [{resp.status_code}]: {extract_http_error(resp)}")

            content_type = str(getattr(resp, "headers", {}).get("Content-Type", "") or "").lower()
            if "application/json" in content_type:
                data = resp.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return answer.strip(), model, {"provider": "custom_api", "streamed": False, "usage": usage}

            parts: List[str] = []
            usage: Dict[str, Any] = {}
            for delta, chunk_usage in iter_sse_deltas(resp, decode_bytes=True):
                if delta:
                    parts.append(delta)
                if chunk_usage:
                    usage = chunk_usage
            return "".join(parts).strip(), model, {"provider": "custom_api", "streamed": True, "usage": usage}

    resp = http_session.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"CustomAPI [{resp.status_code}]: {extract_http_error(resp)}")

    data = resp.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return answer.strip(), model, {"provider": "custom_api", "streamed": False, "usage": usage}


def call_groq(
    http_session: Any,
    *,
    api_key: str,
    base_url: str,
    candidate_models: List[str],
    prompt_text: str,
    temperature: float,
    max_tokens: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call Groq chat completions with candidate model fallback on retryable HTTP statuses."""
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = ""
    for target_model in candidate_models:
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 25.0))
            if resp.status_code < 400:
                data = resp.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                usage = data.get("usage", {})
                return answer.strip(), target_model, {"provider": "groq", "usage": usage}

            last_error = f"Groq [{resp.status_code}]: {extract_http_error(resp)}"
            if resp.status_code in (404, 413, 429, 503):
                continue
            break
        except Exception as exc:
            last_error = f"Groq ({target_model}): {exc}"
            continue

    raise RuntimeError(last_error or "Groq API không phản hồi")


def call_sambanova(
    http_session: Any,
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    temperature: float,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call SambaNova OpenAI-compatible chat completions."""
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
    }

    resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 30.0))
    if resp.status_code >= 400:
        raise RuntimeError(f"SambaNova [{resp.status_code}]: {extract_http_error(resp)}")

    data = resp.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return answer.strip(), model, {"provider": "sambanova", "usage": usage}


def call_gemini(
    http_session: Any,
    *,
    api_key: str,
    candidate_models: List[str],
    prompt_text: str,
    temperature: float,
) -> Tuple[str, str, Dict[str, Any]]:
    """Call Google Gemini generateContent with candidate model fallback."""
    last_error = ""
    for target_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": temperature},
        }
        try:
            resp = http_session.post(url, headers=headers, json=payload, timeout=(10.0, 30.0))
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                answer = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    answer = "".join(p.get("text", "") for p in parts)
                usage = data.get("usageMetadata", {})
                return answer.strip(), target_model, {"provider": "gemini", "usage": usage}

            last_error = f"Gemini [{resp.status_code}]: {extract_http_error(resp)}"
            if resp.status_code in (404, 429, 503):
                continue
            break
        except Exception as exc:
            last_error = f"Gemini: {exc}"

    raise RuntimeError(last_error or "Gemini API không phản hồi")


def call_openrouter(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    provider_routing: Optional[Dict[str, Any]],
) -> Tuple[str, str, Dict[str, Any]]:
    """Call OpenRouter via ``httpx.post`` with reasoning and provider-routing options."""
    import httpx

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://uet.vnu.edu.vn",
        "X-Title": "UETChatbot",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    apply_openrouter_reasoning(payload, reasoning_effort)
    apply_openrouter_provider_routing(payload, provider_routing)

    resp = httpx.post(url, headers=headers, json=payload, timeout=150.0)
    if resp.status_code == 400 and "reasoning" in payload and "reasoning" in resp.text.lower():
        payload.pop("reasoning", None)
        resp = httpx.post(url, headers=headers, json=payload, timeout=150.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter [{resp.status_code}]: {extract_http_error(resp)}")

    data = resp.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    usage = data.get("usage", {})
    return answer.strip(), model, {"provider": "openrouter", "usage": usage}


def stream_custom_api(
    http_session: Any,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    connect_timeout_s: float,
    timeout_s: float,
    enable_thinking: bool,
) -> Generator[str, None, bool]:
    """Stream SSE text deltas from Custom API; returns whether streaming succeeded."""
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream": True,
    }
    with http_session.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(connect_timeout_s, timeout_s),
    ) as response:
        if response.status_code == 200:
            for delta, _ in iter_sse_deltas(response, decode_bytes=True):
                if delta:
                    yield delta
            return True
    return False


def stream_openrouter(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    provider_routing: Optional[Dict[str, Any]],
) -> Generator[str, None, bool]:
    """Stream SSE text deltas from OpenRouter via ``httpx.Client``."""
    import httpx

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://uet.vnu.edu.vn",
        "X-Title": "UETChatbot",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    apply_openrouter_reasoning(payload, reasoning_effort)
    apply_openrouter_provider_routing(payload, provider_routing)

    with httpx.Client(timeout=150.0) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code == 200:
                yielded_any = False
                for delta, _ in iter_sse_deltas(response, decode_bytes=False):
                    if delta:
                        yielded_any = True
                        yield delta
                return yielded_any
            logger.warning(f"Streaming qua OpenRouter trả về mã HTTP {response.status_code}")
    return False


def stream_groq(
    http_session: Any,
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt_text: str,
    temperature: float,
) -> Generator[str, None, bool]:
    """Stream SSE text deltas from Groq."""
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "stream": True,
    }
    with http_session.post(url, headers=headers, json=payload, stream=True, timeout=(10.0, 30.0)) as response:
        if response.status_code == 200:
            for delta, _ in iter_sse_deltas(response, decode_bytes=True):
                if delta:
                    yield delta
            return True
    return False
