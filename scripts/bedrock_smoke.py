"""Smoke test for the OpenAI-compatible LLM endpoint.

Usage:
    uv run python scripts/bedrock_smoke.py

Reads OPENAI_API_KEY, OPENAI_BASE_URL from .env (via dotenv).
Reads OID_LLM_MODEL from .env / environment.

Tests:
  1. Basic text generation
  2. Prompt-based tool calling (the model outputs <tool_call> tags we parse)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from config.settings import get_settings

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call\s*>?|$)", re.DOTALL)


def test_basic(client: OpenAI, model: str) -> bool:
    """Verify basic text completion works."""
    print("── Test 1: basic text generation ──")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=64,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    print(f"Response: {text.strip()}")
    if usage:
        print(f"Tokens:   {usage.prompt_tokens} in / {usage.completion_tokens} out")
    ok = "hello" in text.lower()
    print(f"Result:   {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_tool_call(client: OpenAI, model: str) -> bool:
    """Verify the model follows prompt-based tool calling instructions."""
    print("── Test 2: prompt-based tool calling ──")

    system = (
        "You have access to tools. To call a tool, output exactly:\n"
        '<tool_call>{"name": "calculator", "arguments": {"expression": "..."}}</tool_call>\n'
        "Output the tool_call tag and nothing else."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=256,
    )
    text = resp.choices[0].message.content or ""
    print(f"Raw response: {text.strip()}")

    match = TOOL_CALL_RE.search(text)
    if not match:
        print("Result:   FAIL — no <tool_call> tag found")
        return False

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        print(f"Result:   FAIL — invalid JSON inside <tool_call>: {exc}")
        return False

    name = payload.get("name", "")
    args = payload.get("arguments", {})
    print(f"Parsed:   name={name!r}, arguments={args}")
    ok = name == "calculator" and "expression" in args
    print(f"Result:   {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> None:
    load_dotenv()
    settings = get_settings()

    model = settings.llm_model
    print(f"Model: {model}\n")

    client = OpenAI()

    pass1 = test_basic(client, model)
    pass2 = test_tool_call(client, model)

    if pass1 and pass2:
        print("All smoke tests passed.")
    else:
        print("Some tests FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
