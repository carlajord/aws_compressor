"""Text-to-SQL agent loop with prompt-based tool calling."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd

from oidpipe.agent.prompts import PROMPT_VERSION, build_system_prompt
from oidpipe.agent.tools import dispatch_tool
from oidpipe.agent.tracing import Tracer

logger = logging.getLogger(__name__)

_REASONING_PATTERNS = re.compile(
    r"(?:we need to|let's|let me|i should|i need to|i'll|i will|use get_schema|use run_sql|"
    r"call get_schema|call run_sql|query the|check the|need to run|need to query|"
    r"need to check|to answer|first,?\s+i|step \d)",
    re.IGNORECASE,
)

_NUDGE = (
    "You must write visible output now. To call a tool, write exactly:\n"
    '<tool_call>{"name": "run_sql", "arguments": {"sql": "YOUR SQL HERE"}}</tool_call>\n'
    "Or to check the schema:\n"
    '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>\n'
    "Output the tag and nothing else."
)


@dataclass
class AgentResult:
    """Result from a single agent invocation."""

    answer: str
    sql_list: list[str] = field(default_factory=list)
    last_result_df: pd.DataFrame | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    trace_id: str = ""
    error: str | None = None


def _parse_tool_call(text: str) -> tuple[dict | None, str]:
    """Extract a <tool_call> JSON from text using raw_decode for robust parsing."""
    marker = "<tool_call>"
    end_marker = "</tool_call>"
    idx = text.find(marker)
    if idx == -1:
        return None, text

    json_start = idx + len(marker)
    remaining = text[:idx].strip()

    try:
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text, json_start)
        if isinstance(payload, dict) and "name" in payload:
            return payload, remaining
    except (json.JSONDecodeError, ValueError):
        pass

    end_idx = text.find(end_marker, json_start)
    raw_json = text[json_start:end_idx].strip() if end_idx != -1 else text[json_start:].strip()
    raw_json = raw_json.rstrip("'\"` \n\t")

    for candidate in [raw_json, raw_json.replace("\n", "\\n")]:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict) and "name" in payload:
                return payload, remaining
        except json.JSONDecodeError:
            continue

    return None, text


def _tool_from_reasoning(reasoning: str) -> str:
    """Try to construct a simple tool call string from the model's reasoning field.

    Only handles get_schema (simple, reliable).  SQL extraction is deliberately
    omitted — regex-based SQL parsing from prose truncates CTEs and produces
    broken queries that waste agent steps.
    """
    if re.search(r"\bget_schema\b", reasoning, re.IGNORECASE):
        table_match = re.search(
            r"get_schema\s+(?:for|on|of|table)\s+['\"]?(\w+)", reasoning, re.IGNORECASE
        )
        if table_match:
            table = table_match.group(1)
            return f'{{"name": "get_schema", "arguments": {{"table": "{table}"}}}}'
        return '{"name": "get_schema", "arguments": {}}'

    if re.search(r"\brun.?sql\b|\bquery\b|\bcheck\b.*\btable", reasoning, re.IGNORECASE):
        return '{"name": "get_schema", "arguments": {}}'

    return ""


def _extract_response(resp) -> str:
    """Extract text from a chat completion, handling multiple response formats."""
    msg = resp.choices[0].message
    if msg.content:
        return msg.content

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tc = msg.tool_calls[0]
        name = tc.function.name
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                return ""
        else:
            args = json.dumps(args)
        return f'<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'

    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        constructed = _tool_from_reasoning(reasoning)
        if constructed:
            logger.info("Constructed tool call from reasoning: %s", constructed[:100])
            return f"<tool_call>{constructed}</tool_call>"

    return ""


def _looks_like_reasoning(text: str, has_tool_results: bool) -> bool:
    """Detect if text is model reasoning rather than a real answer."""
    if not text or len(text) > 500:
        return False
    if has_tool_results:
        return False
    return bool(_REASONING_PATTERNS.search(text))


def _call_bedrock(
    model: str,
    messages: list[dict[str, str]],
    region: str,
    max_tokens: int = 1024,
) -> tuple[str, int, int]:
    """Call Bedrock Converse API. Returns (text, input_tokens, output_tokens)."""
    import boto3

    br = boto3.client("bedrock-runtime", region_name=region)

    system_parts = [m for m in messages if m["role"] == "system"]
    chat_parts = [m for m in messages if m["role"] != "system"]

    converse_msgs = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in chat_parts
    ]

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": converse_msgs,
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if system_parts:
        kwargs["system"] = [{"text": system_parts[0]["content"]}]

    resp = br.converse(**kwargs)

    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


def ask(
    question: str,
    conn: duckdb.DuckDBPyConnection,
    client: Any = None,
    *,
    model: str = "openai.gpt-oss-120b",
    max_steps: int = 5,
    row_limit: int = 500,
    tracer: Tracer | None = None,
    source: str = "app",
    region: str = "us-east-1",
) -> AgentResult:
    """Run the agent loop: send question, parse tool calls, dispatch, repeat.

    If *client* is provided (OpenAI instance), uses the OpenAI-compatible API.
    If *client* is None, uses boto3 Bedrock Converse API with IAM credentials.
    """
    start_ns = time.perf_counter_ns()
    system_prompt = build_system_prompt(conn)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    result = AgentResult(answer="", trace_id=tracer.trace_id if tracer else "")
    sql_list: list[str] = []
    last_result_df: pd.DataFrame | None = None
    total_in = 0
    total_out = 0
    has_tool_results = False
    last_tool_key: str | None = None
    use_bedrock = client is None

    def _call_llm(msgs: list[dict]) -> tuple[str, int, int, str | None]:
        if use_bedrock:
            txt, s_in, s_out = _call_bedrock(model, msgs, region)
            return txt, s_in, s_out, None

        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=1024,
        )
        usage = resp.usage
        s_in = usage.prompt_tokens if usage else 0
        s_out = usage.completion_tokens if usage else 0
        txt = _extract_response(resp)
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        return txt, s_in, s_out, reasoning

    for step in range(1, max_steps + 1):
        result.steps = step

        try:
            if tracer:
                with tracer.span("llm", f"step-{step}", input={"messages_count": len(messages)}) as s:
                    text, step_in, step_out, reasoning = _call_llm(messages)
                    s.record(
                        output={"text": text},
                        usage={"input_tokens": step_in, "output_tokens": step_out},
                    )
            else:
                text, step_in, step_out, reasoning = _call_llm(messages)
        except Exception as exc:
            logger.error("LLM API error on step %d: %s", step, exc)
            result.error = str(exc)
            result.answer = "Sorry, I encountered an error calling the language model."
            break

        total_in += step_in
        total_out += step_out

        # Empty response — nudge and continue to next step
        if not text.strip():
            logger.warning("Empty response on step %d, adding nudge", step)
            if reasoning:
                nudge = (
                    "Your response was empty — your output went to internal reasoning "
                    "instead of visible text. Here is what you were thinking:\n"
                    f"---\n{reasoning[:400]}\n---\n"
                    "Now output the tool call as visible text.\n" + _NUDGE
                )
            else:
                nudge = _NUDGE
            messages.append({"role": "user", "content": nudge})
            continue

        tool_call, remaining_text = _parse_tool_call(text)

        if tool_call is not None:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            tool_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if tool_key == last_tool_key:
                logger.info("Duplicate tool call %s, nudging", tool_name)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": (
                    "You already called this tool with the same arguments and "
                    "got a result above. Use that result to write a SQL query "
                    "or answer the question directly. Do NOT call the same "
                    "tool again.\n" + _NUDGE
                )})
                continue
            last_tool_key = tool_key

            if tool_name == "run_sql":
                sql_list.append(tool_args.get("sql", ""))

            if tracer:
                with tracer.span("tool", tool_name, input=tool_args) as ts:
                    tool_result = dispatch_tool(conn, tool_name, tool_args, row_limit=row_limit)
                    ts.record(output=tool_result)
            else:
                tool_result = dispatch_tool(conn, tool_name, tool_args, row_limit=row_limit)

            if tool_name == "run_sql" and "columns" in tool_result:
                try:
                    last_result_df = pd.DataFrame(
                        tool_result["rows"], columns=tool_result["columns"]
                    )
                except Exception:
                    pass

            has_tool_results = True
            messages.append({"role": "assistant", "content": text})
            tool_result_str = json.dumps(tool_result, default=str)
            messages.append({"role": "user", "content": f"Tool result: {tool_result_str}"})
            continue

        # No tool call found — is this reasoning or a real answer?
        if _looks_like_reasoning(text, has_tool_results):
            logger.info("Detected reasoning text, nudging: %s", text[:100])
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": _NUDGE})
            continue

        # Real answer
        result.answer = text.strip()
        break
    else:
        result.answer = remaining_text if remaining_text else "I reached the maximum number of steps."

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    result.sql_list = sql_list
    result.last_result_df = last_result_df
    result.input_tokens = total_in
    result.output_tokens = total_out
    result.latency_ms = elapsed_ms

    if tracer:
        tracer.record_run(
            question=question,
            answer=result.answer,
            model_id=model,
            prompt_version=PROMPT_VERSION,
            steps=result.steps,
            total_tokens=total_in + total_out,
            latency_ms=elapsed_ms,
            source=source,
        )

    return result
