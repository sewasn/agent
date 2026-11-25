"""Tests for the data-analyst agent's tools, run without any model."""

from __future__ import annotations

import asyncio

import pytest
from mistralai import agents

from main import (
    _filter_match,
    _parse_rows,
    add_filter,
    aggregate,
    clear_filters,
    describe_columns,
    load_csv,
    preview_rows,
)


def run(coro):
    return asyncio.run(coro)


def ctx_for(session_id: str) -> agents.ToolCallbackContext:
    return agents.ToolCallbackContext(
        agents.CallbackSession(id=session_id, turn_id="t1"), tool_call_id="c1"
    )


CSV = """name,city,amount,signed_at
Alice,Paris,120.5,2024-01-15
Bob,Lyon,80,2024-02-01
Carol,Paris,220.75,2024-03-10
Dan,,40.25,2024-03-11
Eve,Paris,,2024-04-02
Frank,Paris,-10,2024-04-30
"""


def session() -> str:
    return "s-test"


def load() -> dict:
    return run(load_csv(ctx_for(session()), csv_text=CSV, file_name="sales.csv"))


class TestParse:
    def test_parse_rows_columns(self) -> None:
        columns, rows = _parse_rows(CSV)
        assert columns == ["name", "city", "amount", "signed_at"]
        assert len(rows) == 6

    def test_duplicate_headers_rejected(self) -> None:
        with pytest.raises(agents.ToolError):
            _parse_rows("a,a\n1,2\n")

    def test_missing_value_is_empty_string(self) -> None:
        _, rows = _parse_rows(CSV)
        assert rows[1]["name"] == "Bob"
        assert rows[3]["city"] == ""
        assert rows[4]["amount"] == ""


class TestFilterMatching:
    def test_numeric_comparison(self) -> None:
        _, rows = _parse_rows(CSV)
        kept = [r for r in rows if _filter_match(r, "amount", "gte", 80)]
        assert [r["name"] for r in kept] == ["Alice", "Bob", "Carol"]

    def test_eq_case_insensitive(self) -> None:
        _, rows = _parse_rows(CSV)
        kept = [r for r in rows if _filter_match(r, "city", "eq", "PARIS")]
        assert len(kept) == 4

    def test_is_null(self) -> None:
        _, rows = _parse_rows(CSV)
        kept = [r for r in rows if _filter_match(r, "city", "is_null", None)]
        assert [r["name"] for r in kept] == ["Dan"]

    def test_in(self) -> None:
        _, rows = _parse_rows(CSV)
        kept = [r for r in rows if _filter_match(r, "city", "in", ["Lyon", "Paris"])]
        assert len(kept) == 5


class TestTools:
    def test_load_reports_schema(self) -> None:
        result = load()
        assert result["rows"] == 6
        assert result["columns"] == ["name", "city", "amount", "signed_at"]
        assert len(result["first_rows"]) == 5

    def test_load_requires_exactly_one_source(self) -> None:
        with pytest.raises(agents.ToolError):
            run(load_csv(ctx_for("s2")))
        with pytest.raises(agents.ToolError):
            run(load_csv(ctx_for("s2"), csv_text="a\n1", url="https://example.com/x.csv"))
        with pytest.raises(agents.ToolError):
            run(load_csv(ctx_for("s2"), url="http://example.com/x.csv"))

    def test_describe_columns(self) -> None:
        load()
        result = run(describe_columns(ctx_for(session())))
        assert result["rows_in_scope"] == 6
        columns = result["columns"]
        assert columns["amount"]["type"] == "number"
        assert columns["amount"]["missing"] == 1
        assert columns["amount"]["min"] == -10
        assert columns["amount"]["max"] == 220.75
        assert columns["city"]["distinct"] == 2
        assert columns["city"]["top_values"][0] == ("Paris", 4)
        assert columns["signed_at"]["type"] == "date"

    def test_add_filter_accumulates(self) -> None:
        load()
        first = run(add_filter(ctx_for(session()), column="city", op="eq", value="Paris"))
        assert first["rows_after"] == 4
        second = run(add_filter(ctx_for(session()), column="amount", op="gt", value=0))
        assert second["rows_after"] == 2
        described = run(describe_columns(ctx_for(session())))
        assert described["rows_in_scope"] == 2
        assert len(described["active_filters"]) == 2

    def test_unknown_column_and_op(self) -> None:
        load()
        with pytest.raises(agents.ToolError, match="No column"):
            run(add_filter(ctx_for(session()), column="revenue", op="eq", value=1))
        with pytest.raises(agents.ToolError, match="Unknown op"):
            run(add_filter(ctx_for(session()), column="city", op="regex", value="x"))

    def test_clear_filters(self) -> None:
        load()
        run(add_filter(ctx_for(session()), column="city", op="eq", value="Paris"))
        result = run(clear_filters(ctx_for(session())))
        assert result["rows_in_scope"] == 6
        described = run(describe_columns(ctx_for(session())))
        assert described["active_filters"] == []

    def test_preview_rows(self) -> None:
        load()
        run(add_filter(ctx_for(session()), column="city", op="eq", value="Lyon"))
        result = run(preview_rows(ctx_for(session()), limit=20))
        assert result["rows_in_scope"] == 1
        assert "Bob" in result["table"]

    def test_aggregate_sum_per_group(self) -> None:
        load()
        result = run(aggregate(ctx_for(session()), group_by=["city"], aggregations=["count", "sum:amount"]))
        rows_text = result["table"]
        assert result["groups_total"] == 3
        assert "Paris" in rows_text
        paris_row = next(
            line for line in rows_text.splitlines() if "| Paris " in line + " " and line.endswith("|")
        )
        # 120.5 + 220.75 - 10 = 331.25
        assert "331.25" in paris_row

    def test_aggregate_overall_count(self) -> None:
        load()
        result = run(aggregate(ctx_for(session()), aggregations=["count"]))
        assert result["groups_total"] == 1

    def test_aggregate_respects_filters(self) -> None:
        load()
        run(add_filter(ctx_for(session()), column="amount", op="gt", value=0))
        run(add_filter(ctx_for(session()), column="city", op="eq", value="Paris"))
        result = run(aggregate(ctx_for(session()), aggregations=["mean:amount", "nunique:city"]))
        assert "170.625" in result["table"]

    def test_aggregate_median_and_min(self) -> None:
        load()
        result = run(aggregate(ctx_for(session()), group_by=["city"], aggregations=["median:amount"]))
        assert "120.5" in result["table"]

    def test_aggregate_validation(self) -> None:
        load()
        with pytest.raises(agents.ToolError, match="Unknown function"):
            run(aggregate(ctx_for(session()), aggregations=["avg:amount"]))
        with pytest.raises(agents.ToolError, match="needs a column"):
            run(aggregate(ctx_for(session()), aggregations=["sum"]))
        with pytest.raises(agents.ToolError, match="sort_by"):
            run(aggregate(ctx_for(session()), aggregations=["count"], sort_by="nope"))

    def test_tools_require_loaded_file(self) -> None:
        for tool in (describe_columns, preview_rows, clear_filters):
            with pytest.raises(agents.ToolError, match="No file is loaded"):
                run(tool(ctx_for("s-empty")))
        with pytest.raises(agents.ToolError, match="No file is loaded"):
            run(add_filter(ctx_for("s-empty"), column="a", op="eq", value=1))
        with pytest.raises(agents.ToolError, match="No file is loaded"):
            run(aggregate(ctx_for("s-empty")))


class TestAgent:
    def test_agent_configuration(self) -> None:
        from main import agent

        assert agent.name == "data-analyst"

    def test_scripted_turn(self) -> None:
        from main import (
            add_filter,
            aggregate,
            clear_filters,
            describe_columns,
            load_csv,
            preview_rows,
        )

        run(load_csv(ctx_for("local"), csv_text=CSV, file_name="sales.csv"))
        tested = agents.Agent(
            model="mistral-large-4",
            name="data-analyst-test",
            tools=[
                load_csv,
                describe_columns,
                add_filter,
                clear_filters,
                preview_rows,
                aggregate,
            ],
        ).override(
            model=agents.testing.ScriptedModel(
                [
                    agents.testing.call("add_filter", column="city", op="eq", value="Paris"),
                    agents.testing.call("aggregate", aggregations=["sum:amount"]),
                    agents.testing.reply("Paris total loaded."),
                ]
            )
        )
        result = run(tested.run("How much for Paris?"))
        assert isinstance(result, agents.Completed)
        assert "Paris" in result.text
