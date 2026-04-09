"""
VideoARM API client — unified OpenAI-compatible chat / tool-call wrapper.
"""

import base64
import copy
import json
import os
import random
import time
from mimetypes import guess_type

_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

import cv2
import requests


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 4,
    exponential_base: float = 2,
    jitter: bool = False,
    max_retries: int = 6,
):
    """
    Retry decorator with exponential back-off.

    Retries on rate-limit, timeout, server errors, and transient connection
    failures.  Other exceptions propagate immediately.

    Delays: 4 s, 8 s, 16 s, 32 s, 64 s, 128 s  (default)
    """

    def wrapper(*args, **kwargs):
        num_retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                retryable = (
                    "rate limit" in err_str
                    or "timed out" in err_str
                    or "too many requests" in err_str
                    or "forbidden for url" in err_str
                    or "internal" in err_str
                    or "503" in err_str
                    or "500" in err_str
                    or "upstream connect error" in err_str
                    or "connection termination" in err_str
                    or "overloaded" in err_str
                    or "something went wrong" in err_str
                    or "bad gateway" in err_str
                    or "sslerror" in err_str
                    or "ssleoferror" in err_str
                    or "unexpected_eof" in err_str
                    or "connection reset" in err_str
                    or "connectionerror" in err_str
                )
                if retryable:
                    num_retries += 1
                    if num_retries > max_retries:
                        print(f"{_RED}│ ✗  Max retries ({max_retries}) reached. Giving up.{_RESET}")
                        return None
                    delay = initial_delay * (exponential_base ** (num_retries - 1))
                    if jitter:
                        delay *= 1 + random.random()
                    print(f"{_RED}│ ✗  API call failed (retry {num_retries}/{max_retries}): {e}{_RESET}")
                    print(f"{_YELLOW}│ ⚠  Retrying in {delay:.1f} seconds...{_RESET}")
                    time.sleep(delay)
                else:
                    print(f"{_RED}│ ✗  {e}{_RESET}")
                    return None

    return wrapper


def local_image_to_data_url(image_path: str) -> str:
    """Encode a local image file as a base64 data URL."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


@retry_with_exponential_backoff
def call_openai_model_with_tools(
    messages,
    endpoints,
    model_name,
    api_key: str = None,
    tools: list = [],
    image_paths: list = [],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tool_choice: str = "auto",
    return_json: bool = False,
    request_timeout: float = 900.0,
) -> dict:
    """
    Unified chat / tool-call wrapper for OpenAI-compatible endpoints.

    Args:
        messages:        Conversation in OpenAI role/content format.
        endpoints:       Base URL string for the API endpoint.
        model_name:      Model name (e.g. "o3", "gpt-4.1").
        api_key:         API key; uses Authorization: Bearer header.
        tools:           List of function-tool definitions.
        image_paths:     Local image paths to attach as vision content.
        max_tokens:      Maximum tokens to generate.
        temperature:     Sampling temperature.
        tool_choice:     "auto", "none", or "required".
        return_json:     Whether to request JSON response format.
        request_timeout: HTTP request timeout in seconds.

    Returns:
        dict with keys ``content`` and ``tool_calls``.
    """
    if not api_key:
        raise ValueError(
            "api_key is required. Set OPENAI_API_KEY in your environment."
        )
    if not endpoints:
        raise ValueError(
            "endpoints (OPENAI_BASE_URL) is not set. "
            "Please configure the API base URL via environment."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    url = f"{endpoints}/chat/completions"

    payload = {
        "model": model_name,
        "messages": copy.deepcopy(messages),
    }
    if return_json:
        payload["response_format"] = {"type": "json_object"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    if image_paths:
        image_data_list = [local_image_to_data_url(p) for p in image_paths]
        payload["messages"].append({"role": "user", "content": []})
        for data_url in image_data_list:
            payload["messages"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

    response = requests.post(
        url, headers=headers, json=payload, timeout=request_timeout or 900.0
    )

    if response.status_code != 200:
        raise Exception(
            f"API returned status {response.status_code}: {response.text}"
        )

    message = response.json()["choices"][0]["message"]
    if "tool_calls" in message:
        return message
    return {"content": message["content"].strip(), "tool_calls": None}


def handle_json_parsing_error(content: str, context: str = "response") -> dict | None:
    """
    Robust JSON parser for AI model outputs.

    Handles:
    - Markdown code-block wrappers (```json ... ```)
    - Partial JSON with missing closing brackets
    - Arbitrary leading/trailing text

    This is the **project-wide standard** for parsing any JSON that comes from
    a model.  Never call ``json.loads()`` directly on model output.

    Args:
        content: Raw string from the model.
        context: Human-readable label used in warning messages.

    Returns:
        Parsed dict/list, or None on failure.
    """
    if not isinstance(content, str):
        print(f"{_RED}│ ✗  Expected string for {context}, got {type(content)}{_RESET}")
        return None
    if not content.strip():
        print(f"{_YELLOW}│ ⚠  Empty content for {context}{_RESET}")
        return None

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strip Markdown fences
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first JSON structure
    json_start = next(
        (i for i, c in enumerate(content) if c in "{["), -1
    )
    if json_start == -1:
        print(f"{_RED}│ ✗  No JSON structure found in {context}: {content[:200]}{_RESET}")
        return None

    # Walk to matching close bracket
    stack = []
    for i in range(json_start, len(content)):
        c = content[i]
        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                continue
            if (c == "}" and stack[-1] == "{") or (c == "]" and stack[-1] == "["):
                stack.pop()
                if not stack:
                    try:
                        return json.loads(content[json_start: i + 1])
                    except json.JSONDecodeError:
                        break

    # Attempt bracket completion
    for end in range(len(content), json_start, -1):
        candidate = content[json_start:end]
        open_b = candidate.count("{") - candidate.count("}")
        open_k = candidate.count("[") - candidate.count("]")
        if open_b > 0 or open_k > 0:
            completed = candidate + "}" * open_b + "]" * open_k
            try:
                result = json.loads(completed)
                print(
                    f"{_YELLOW}│ ⚠  Recovered partial JSON for {context} "
                    f"(added {open_b} braces, {open_k} brackets){_RESET}"
                )
                return result
            except json.JSONDecodeError:
                continue

    print(f"{_RED}│ ✗  Failed to parse JSON from {context}: {content[:200]}{_RESET}")
    return None


def extract_answer(message: dict) -> str | None:
    """
    Extract a plain-text answer from an assistant message.

    Checks ``content`` first; falls back to ``tool_calls[*].arguments.answer``.
    """
    if content := message.get("content"):
        return content.strip()

    for call in message.get("tool_calls") or []:
        args = handle_json_parsing_error(
            call["function"]["arguments"], "tool call arguments"
        )
        if args and (answer := args.get("answer")):
            return answer

    return None
