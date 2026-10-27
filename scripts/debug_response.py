"""Debug script — test three approaches to get content from the model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from config.settings import get_settings

load_dotenv()
settings = get_settings()
client = OpenAI()


def dump_response(label: str, resp) -> None:
    msg = resp.choices[0].message
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  content:    {msg.content!r}")
    print(f"  tool_calls: {msg.tool_calls!r}")
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        print(f"  reasoning:  {reasoning!r}")
    print(f"  tokens:     {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out")


# Test 1: Simple instruction (baseline)
print("\n--- Test 1: Simple system prompt ---")
resp1 = client.chat.completions.create(
    model=settings.llm_model,
    messages=[
        {"role": "system", "content": (
            "IMPORTANT: Always write your response as visible output text. "
            "Do not use internal reasoning or thinking — output everything directly.\n\n"
            "You have a tool called get_schema. To use it, output:\n"
            '<tool_call>{"name": "get_schema", "arguments": {}}\n'
            "Output the tool_call tag directly in your response."
        )},
        {"role": "user", "content": "What tables are available?"},
    ],
    max_tokens=256,
)
dump_response("Prompt with no-reasoning instruction", resp1)

# Test 2: Direct question, no tools
print("\n--- Test 2: Direct question, no tool instructions ---")
resp2 = client.chat.completions.create(
    model=settings.llm_model,
    messages=[
        {"role": "user", "content": "Say 'hello world' and nothing else."},
    ],
    max_tokens=64,
)
dump_response("Direct question", resp2)

# Test 3: With extra_body reasoning_effort
print("\n--- Test 3: With reasoning_effort=none ---")
try:
    resp3 = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": (
                "You have a tool called get_schema. To use it, output:\n"
                '<tool_call>{"name": "get_schema", "arguments": {}}\n'
            )},
            {"role": "user", "content": "What tables are available?"},
        ],
        max_tokens=256,
        extra_body={"reasoning_effort": "none"},
    )
    dump_response("With reasoning_effort=none", resp3)
except Exception as e:
    print(f"  reasoning_effort=none failed: {e}")

# Test 4: With temperature=0
print("\n--- Test 4: temperature=0 ---")
resp4 = client.chat.completions.create(
    model=settings.llm_model,
    messages=[
        {"role": "system", "content": (
            "IMPORTANT: Do NOT use <think> or internal reasoning. "
            "Write your full response as output text.\n\n"
            "You have a tool called get_schema. To use it, output exactly this tag:\n"
            '<tool_call>{"name": "get_schema", "arguments": {}}\n'
        )},
        {"role": "user", "content": "What tables are available?"},
    ],
    max_tokens=256,
    temperature=0,
)
dump_response("temperature=0", resp4)
