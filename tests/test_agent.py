"""Tests for agent/prompts.py and agent/sql_agent.py with mocked OpenAI/Bedrock client."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from oidpipe.agent.prompts import PROMPT_VERSION, build_system_prompt
from oidpipe.agent.sql_agent import AgentResult, _call_bedrock, ask
from oidpipe.agent.tracing import Tracer, init_traces_db


# ── Helpers ─────────────────────────────────────────────────────────────


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class FakeFunction:
    name: str = ""
    arguments: str = ""


@dataclass
class FakeToolCall:
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = ""
    tool_calls: list[FakeToolCall] | None = None
    reasoning: str | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage


def make_response(text: str, in_tok: int = 100, out_tok: int = 50) -> FakeResponse:
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=text))],
        usage=FakeUsage(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def make_reasoning_response(
    reasoning: str, in_tok: int = 100, out_tok: int = 50,
) -> FakeResponse:
    """Response where content is None but reasoning field has text."""
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(
            content=None, tool_calls=None, reasoning=reasoning,
        ))],
        usage=FakeUsage(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def make_structured_tc_response(
    name: str, arguments: str, in_tok: int = 100, out_tok: int = 50,
) -> FakeResponse:
    """Response where content is None but tool_calls has a structured call."""
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(
            content=None,
            tool_calls=[FakeToolCall(function=FakeFunction(name=name, arguments=arguments))],
        ))],
        usage=FakeUsage(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def make_client(responses: list[str]) -> MagicMock:
    """Create a mock OpenAI client that returns scripted responses in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        make_response(r) for r in responses
    ]
    return client


@pytest.fixture()
def db():
    """In-memory DuckDB with a minimal schema for agent testing."""
    c = duckdb.connect(":memory:")
    c.execute("""
        CREATE TABLE features_10m (
            ts TIMESTAMP NOT NULL,
            pressure_ratio DOUBLE,
            is_running BOOLEAN
        )
    """)
    c.execute("COMMENT ON TABLE features_10m IS '10-min features'")
    c.execute("COMMENT ON COLUMN features_10m.pressure_ratio IS 'Discharge/suction pressure ratio'")
    c.execute("""
        INSERT INTO features_10m VALUES
        ('2024-01-01 00:00', 3.5, true),
        ('2024-01-01 00:10', 3.6, true),
        ('2024-01-01 00:20', 3.7, true),
        ('2024-01-01 00:30', NULL, false),
        ('2024-01-01 00:40', 3.8, true)
    """)
    c.execute("""
        CREATE VIEW v_daily_kpis AS
        SELECT CAST(ts AS DATE) AS day, AVG(pressure_ratio) AS avg_pr
        FROM features_10m WHERE is_running GROUP BY day
    """)
    return c


@pytest.fixture()
def traces_conn():
    c = duckdb.connect(":memory:")
    init_traces_db(c)
    return c


# ── Prompt tests ────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_schema(self, db):
        prompt = build_system_prompt(db)
        assert "features_10m" in prompt
        assert "pressure_ratio" in prompt

    def test_contains_glossary(self, db):
        prompt = build_system_prompt(db)
        assert "23-KA-9101" in prompt
        assert "barg" in prompt

    def test_contains_tool_docs(self, db):
        prompt = build_system_prompt(db)
        assert "get_schema" in prompt
        assert "run_sql" in prompt
        assert "<tool_call>" in prompt

    def test_contains_few_shot(self, db):
        prompt = build_system_prompt(db)
        assert "average pressure ratio" in prompt.lower()

    def test_contains_rules(self, db):
        prompt = build_system_prompt(db)
        assert "Never invent numbers" in prompt

    def test_prompt_version_constant(self):
        assert PROMPT_VERSION == "v1"

    def test_view_in_schema(self, db):
        prompt = build_system_prompt(db)
        assert "v_daily_kpis" in prompt


# ── Agent loop tests ───────────────────────────────────────────────────


class TestAgentDirect:
    def test_direct_answer(self, db):
        """Model answers without calling any tool."""
        client = make_client(["The compressor is running fine."])
        result = ask("How is the compressor?", db, client, max_steps=5)

        assert result.answer == "The compressor is running fine."
        assert result.steps == 1
        assert result.sql_list == []
        assert result.error is None

    def test_tool_call_then_answer(self, db):
        """Model calls run_sql, gets result, then answers."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT AVG(pressure_ratio) AS avg_pr FROM features_10m WHERE is_running"}}</tool_call>',
            "The average pressure ratio is **3.65**.",
        ])
        result = ask("What is the avg pressure ratio?", db, client, max_steps=5)

        assert "3.65" in result.answer
        assert result.steps == 2
        assert len(result.sql_list) == 1
        assert "pressure_ratio" in result.sql_list[0]
        assert result.last_result_df is not None
        assert len(result.last_result_df) == 1

    def test_get_schema_then_query(self, db):
        """Model calls get_schema, then run_sql, then answers."""
        client = make_client([
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT count(*) AS n FROM features_10m"}}</tool_call>',
            "There are **5 rows** in features_10m.",
        ])
        result = ask("How many rows?", db, client, max_steps=5)

        assert "5 rows" in result.answer
        assert result.steps == 3
        assert len(result.sql_list) == 1

    def test_sql_error_retry(self, db):
        """Model sends bad SQL, gets error, fixes it."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT * FROM nonexistent"}}</tool_call>',
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT count(*) AS n FROM features_10m"}}</tool_call>',
            "There are 5 rows.",
        ])
        result = ask("Count the rows", db, client, max_steps=5)

        assert result.steps == 3
        assert len(result.sql_list) == 2
        assert result.error is None

    def test_step_cap(self, db):
        """Model keeps calling tools and hits the step limit."""
        client = make_client([
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
        ])
        result = ask("What tables?", db, client, max_steps=3)

        assert result.steps == 3
        assert "maximum number of steps" in result.answer.lower() or result.answer != ""

    def test_api_error(self, db):
        """LLM API throws an exception."""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("connection timeout")

        result = ask("Hello", db, client, max_steps=5)

        assert result.error is not None
        assert "error" in result.answer.lower()

    def test_token_counting(self, db):
        """Tokens from multiple steps are accumulated."""
        client = make_client([
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            "Done.",
        ])
        result = ask("What tables?", db, client, max_steps=5)

        assert result.input_tokens == 200
        assert result.output_tokens == 100

    def test_row_limit_passed(self, db):
        """Row limit is forwarded to dispatch_tool."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT * FROM features_10m"}}</tool_call>',
            "Got 2 rows.",
        ])
        result = ask("Get data", db, client, max_steps=5, row_limit=2)

        assert result.last_result_df is not None
        assert len(result.last_result_df) == 2

    def test_unknown_tool(self, db):
        """Model calls a tool that doesn't exist."""
        client = make_client([
            '<tool_call>{"name": "delete_all", "arguments": {}}</tool_call>',
            "Sorry, that tool doesn't exist. There are tables available.",
        ])
        result = ask("Delete everything", db, client, max_steps=5)

        assert result.steps == 2

    def test_tool_call_with_trailing_text(self, db):
        """Model outputs <tool_call>JSON followed by hallucinated text, no closing tag."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT count(*) AS n FROM features_10m"}}There are many rows in the table.',
            "There are **5** rows in features_10m.",
        ])
        result = ask("How many rows?", db, client, max_steps=5)

        assert "5" in result.answer
        assert result.steps == 2
        assert len(result.sql_list) == 1

    def test_tool_call_missing_closing_bracket(self, db):
        """Model outputs <tool_call>JSON</tool_call without > on closing tag."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT 1 AS n"}}</tool_call',
            "Result is 1.",
        ])
        result = ask("Select 1", db, client, max_steps=5)

        assert result.steps == 2
        assert len(result.sql_list) == 1


# ── Structured tool_calls fallback ──────────────────────────────────────


class TestStructuredToolCalls:
    def test_structured_run_sql(self, db):
        """Endpoint returns structured tool_calls instead of text <tool_call> tags."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_structured_tc_response(
                "run_sql",
                '{"sql": "SELECT count(*) AS n FROM features_10m"}',
            ),
            make_response("There are **5 rows**."),
        ]
        result = ask("How many rows?", db, client, max_steps=5)

        assert "5 rows" in result.answer
        assert result.steps == 2
        assert len(result.sql_list) == 1

    def test_structured_get_schema(self, db):
        """Structured tool_calls for get_schema."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_structured_tc_response("get_schema", "{}"),
            make_response("Tables: features_10m, v_daily_kpis."),
        ]
        result = ask("What tables?", db, client, max_steps=5)

        assert "features_10m" in result.answer
        assert result.steps == 2

    def test_mixed_structured_and_text(self, db):
        """First call returns structured tool_calls, second returns text answer."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_structured_tc_response(
                "run_sql",
                '{"sql": "SELECT AVG(pressure_ratio) AS avg_pr FROM features_10m WHERE is_running"}',
            ),
            make_response("Average pressure ratio is 3.65."),
        ]
        result = ask("Avg PR?", db, client, max_steps=5)

        assert result.steps == 2
        assert len(result.sql_list) == 1
        assert result.last_result_df is not None


# ── Reasoning field → tool call construction ───────────────────────────


class TestReasoningExtraction:
    def test_reasoning_get_schema(self, db):
        """Empty content with reasoning mentioning get_schema → tool call constructed."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_reasoning_response("We need to inspect schema. Use get_schema tool."),
            make_response("The database has features_10m and v_daily_kpis."),
        ]
        result = ask("What tables?", db, client, max_steps=5)

        assert "features_10m" in result.answer
        assert result.steps == 2

    def test_reasoning_with_sql_gets_relayed(self, db):
        """Empty content with reasoning containing SQL but no keywords → relay nudge."""
        captured_messages: list[list[dict]] = []

        def capture_create(**kwargs):
            captured_messages.append([m.copy() for m in kwargs["messages"]])
            responses = [
                make_reasoning_response(
                    "The user wants counts. SELECT count(*) AS n FROM v_daily_kpis"
                ),
                make_response(
                    '<tool_call>{"name": "run_sql", "arguments": '
                    '{"sql": "SELECT count(*) AS n FROM v_daily_kpis"}}</tool_call>'
                ),
                make_response("There are results."),
            ]
            return responses[len(captured_messages) - 1]

        client = MagicMock()
        client.chat.completions.create.side_effect = capture_create
        result = ask("Any events?", db, client, max_steps=5)

        assert result.steps == 3
        assert len(result.sql_list) == 1
        nudge_msg = captured_messages[1][-1]["content"]
        assert "internal reasoning" in nudge_msg
        assert "SELECT count" in nudge_msg

    def test_reasoning_query_keyword(self, db):
        """Empty content with reasoning mentioning 'query' → get_schema as fallback."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_reasoning_response("We should query the anomalies table."),
            make_response("Done."),
        ]
        result = ask("Check anomalies", db, client, max_steps=5)

        assert result.steps == 2


# ── Reasoning detection and empty response nudging ─────────────────────


class TestReasoningNudge:
    def test_reasoning_text_gets_nudged(self, db):
        """Model outputs reasoning text, gets nudged, then produces tool call."""
        client = make_client([
            "We need to query the database. Let's use get_schema.",
            '<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            "The database has features_10m and v_daily_kpis.",
        ])
        result = ask("What tables?", db, client, max_steps=5)

        assert "features_10m" in result.answer
        assert result.steps == 3

    def test_empty_response_gets_nudged(self, db):
        """Model returns empty content, gets nudged, then answers."""
        client = make_client([
            "",
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT count(*) AS n FROM features_10m"}}</tool_call>',
            "There are 5 rows.",
        ])
        result = ask("How many rows?", db, client, max_steps=5)

        assert "5 rows" in result.answer
        assert result.steps == 3

    def test_reasoning_after_tool_result_is_answer(self, db):
        """After tool results, model text is treated as an answer, not reasoning."""
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT count(*) AS n FROM features_10m"}}</tool_call>',
            "We need to report that there are 5 rows.",
        ])
        result = ask("Count rows", db, client, max_steps=5)

        assert "5 rows" in result.answer
        assert result.steps == 2

    def test_multiple_nudges_hit_step_cap(self, db):
        """Repeated reasoning hits the step cap."""
        client = make_client([
            "Let me check the schema.",
            "I should use get_schema.",
            "I need to query the database.",
        ])
        result = ask("What tables?", db, client, max_steps=3)

        assert result.steps == 3


# ── Agent with tracing ──────────────────────────────────────────────────


class TestAgentTracing:
    def test_spans_recorded(self, db, traces_conn):
        tracer = Tracer(traces_conn, trace_id="test-trace")
        client = make_client([
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT 1 AS n"}}</tool_call>',
            "Result is 1.",
        ])

        result = ask("What is 1?", db, client, max_steps=5, tracer=tracer)

        spans = traces_conn.execute(
            "SELECT kind, name FROM agent_spans WHERE trace_id='test-trace' ORDER BY started_at"
        ).fetchall()
        assert len(spans) == 3  # llm step-1, tool run_sql, llm step-2
        assert spans[0] == ("llm", "step-1")
        assert spans[1] == ("tool", "run_sql")
        assert spans[2] == ("llm", "step-2")

    def test_run_recorded(self, db, traces_conn):
        tracer = Tracer(traces_conn, trace_id="run-trace")
        client = make_client(["Hello!"])

        result = ask("Hi", db, client, max_steps=5, tracer=tracer, source="eval")

        run = traces_conn.execute(
            "SELECT question, answer, prompt_version, steps, source "
            "FROM agent_runs WHERE trace_id='run-trace'"
        ).fetchone()
        assert run[0] == "Hi"
        assert run[1] == "Hello!"
        assert run[2] == "v1"
        assert run[3] == 1
        assert run[4] == "eval"

    def test_trace_id_in_result(self, db, traces_conn):
        tracer = Tracer(traces_conn, trace_id="id-check")
        client = make_client(["Done."])

        result = ask("Test", db, client, max_steps=5, tracer=tracer)
        assert result.trace_id == "id-check"


# ── Bedrock Converse backend ─────────────────────────────────────────────


def _mock_converse(**kwargs):
    """Return a minimal Bedrock Converse response."""
    return {
        "output": {"message": {"content": [{"text": kwargs.get("_text", "Hello.")}]}},
        "usage": {"inputTokens": 100, "outputTokens": 50},
    }


class TestBedrockBackend:
    def test_call_bedrock_basic(self):
        mock_br = MagicMock()
        mock_br.converse.return_value = {
            "output": {"message": {"content": [{"text": "The answer is 42."}]}},
            "usage": {"inputTokens": 120, "outputTokens": 30},
        }

        with patch("boto3.client", return_value=mock_br):
            text, in_tok, out_tok = _call_bedrock(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 6 * 7?"},
                ],
                region="us-east-1",
            )

        assert text == "The answer is 42."
        assert in_tok == 120
        assert out_tok == 30
        call_kwargs = mock_br.converse.call_args.kwargs
        assert call_kwargs["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert call_kwargs["system"] == [{"text": "You are a helpful assistant."}]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_ask_with_bedrock(self, db):
        """Full agent loop using Bedrock backend (client=None)."""
        mock_br = MagicMock()
        mock_br.converse.side_effect = [
            {
                "output": {"message": {"content": [{"text":
                    '<tool_call>{"name": "run_sql", "arguments": '
                    '{"sql": "SELECT count(*) AS n FROM features_10m"}}</tool_call>'
                }]}},
                "usage": {"inputTokens": 200, "outputTokens": 40},
            },
            {
                "output": {"message": {"content": [{"text": "There are **5** rows."}]}},
                "usage": {"inputTokens": 300, "outputTokens": 20},
            },
        ]

        with patch("boto3.client", return_value=mock_br):
            result = ask("How many rows?", db, None, max_steps=5, region="us-east-1")

        assert "5" in result.answer
        assert result.steps == 2
        assert len(result.sql_list) == 1
        assert result.input_tokens == 500
        assert result.output_tokens == 60
