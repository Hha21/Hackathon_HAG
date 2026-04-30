"""
Thin wrapper around vLLM's OpenAI-compatible API and the OpenRouter API.
Allows swapping backends (local vLLM, OpenRouter) by changing config.
"""

import json
import time
from openai import OpenAI
from config import TEMPERATURE, MAX_NEW_TOKENS
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, TARGET_SERVER_URL


def get_client(base_url: str = None) -> OpenAI:
    """Return an OpenAI-compatible client pointing at a local vLLM server."""
    return OpenAI(
        base_url=base_url or TARGET_SERVER_URL,
        api_key="not-needed",
    )


def get_openrouter_client() -> OpenAI:
    """Return a client pointing at the OpenRouter API."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/adversarial-alignment",
            "X-Title": "Adversarial Alignment Experiment",
        },
    )


def query_model(
    client: OpenAI,
    model_id: str,
    messages: list[dict],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_NEW_TOKENS,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Send a chat completion request and return the response text.
    Retries on transient errors.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"[query_model] Error on attempt {attempt+1}: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"[query_model] All {retries} attempts failed. Returning empty string.")
                return ""


def parse_json_response(response: str) -> dict:
    """
    Attempt to parse JSON from a model response.
    Handles common issues like markdown code fences.

    Returns parsed dict, or empty dict on failure.
    """
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        print(f"[parse_json_response] Failed to parse JSON:\n{response[:200]}")
        return {}