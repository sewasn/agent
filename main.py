"""Conversational data analyst agent.

The user shares a link to a CSV file in their Google Drive and asks questions
in plain language. The agent fetches the file through the user's Google Drive
connector and answers with numbers computed by a fixed set of analysis tools.
There is no sandbox and no shell: the model can only orchestrate the tools
defined here (and the Drive connector's tools). The Google Drive connector
requires programmatic tool calling, so custom fixed tools stay
`model_access="direct"` and remain in the model's tool list.
"""

from __future__ import annotations

import csv
import io
import statistics
import urllib.request
from typing import Annotated, Any

from pydantic import Field

from mistralai import agents

DRIVE_CONNECTOR = "google_drive_mcp"

MAX_CSV_TEXT_CHARS = 10_000_000
MAX_URL_BYTES = 100_000_000
PREVIEW_ROWS = 5
GROUP_ROW_LIMIT_DEFAULT = 25
GROUP_ROW_LIMIT_MAX = 100

AGG_FUNCS = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "median",
    "nunique",
)

FILTER_OPS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "not_contains",
    "in",
    "not_in",
    "is_null",
    "is_not_null",
)


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, dict[str, Any]] = {}


def _session_store(session_id: str) -> dict[str, Any]:
    store = _SESSIONS.get(session_id)
    if store is None:
        store = {"file_name": None, "columns": [], "rows": [], "filters": []}
        _SESSIONS[session_id] = store
    return store


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _sniff_dialect(text: str) -> csv.Dialect:
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _looks_number(values: list[str]) -> bool:
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return False
    try:
        [float(v) for v in non_empty]
    except ValueError:
        return False
    return True


def _looks_date(values: list[str]) -> bool:
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return False
    hits = sum(1 for v in non_empty if len(v) >= 8 and v[4] == "-" and v[7] == "-")
    return hits / len(non_empty) >= 0.9


def _column_type(name: str, columns: dict[str, list[str]]) -> str:
    values = columns[name]
    if _looks_number(values):
        return "number"
    if _looks_date(values):
        return "date"
    lowered = {v.strip().lower() for v in values if v != ""}
    if lowered and lowered <= {"true", "false"}:
        return "boolean"
    return "text"


def _parse_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.reader(io.StringIO(text), _sniff_dialect(text))
    raw = [row for row in reader if row]
    if not raw:
        raise agents.ToolError("The file contains no rows.")
    header = [h.strip() for h in raw[0]]
    if len(set(header)) != len(header):
        raise agents.ToolError("The CSV header has duplicate column names.")
    rows = []
    for row in raw[1:]:
        if len(row) == 1 and row[0].strip() == "":
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        rows.append(dict(zip(header, (c.strip() for c in row), strict=True)))
    return header, rows


def _column_values(columns: list[str], rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {name: [row[name] for row in rows] for name in columns}


def _filter_match(row: dict[str, str], column: str, op: str, value: Any) -> bool:
    cell = row[column]
    if op == "is_null":
        return cell == ""
    if op == "is_not_null":
        return cell != ""
    if op in ("in", "not_in"):
        items = value if isinstance(value, list) else [value]
        hits = any(str(item).strip().lower() == cell.strip().lower() for item in items)
        return hits if op == "in" else not hits
    if op == "eq":
        return str(value).strip().lower() == cell.strip().lower()
    if op == "ne":
        return str(value).strip().lower() != cell.strip().lower()
    if op in ("gt", "gte", "lt", "lte"):
        a, b = cell, str(value)
        try:
            a, b = float(cell), float(value)
        except ValueError:
            pass
        if op == "gt":
            return a > b
        if op == "gte":
            return a >= b
        if op == "lt":
            return a < b
        return a <= b
    if op == "contains":
        return str(value).upper() in cell.upper()
    return str(value).upper() not in cell.upper()


def _apply_filters(store: dict[str, Any]) -> list[dict[str, str]]:
    rows = store["rows"]
    for f in store["filters"]:
        rows = [row for row in rows if _filter_match(row, f["column"], f["op"], f["value"])]
    return rows


def _apply_one(rows: list[dict[str, str]], column: str, op: str, value: Any) -> list[dict[str, str]]:
    return [row for row in rows if _filter_match(row, column, op, value)]


def _num(value: float) -> int | float:
    return int(value) if value == int(value) else round(value, 6)


def _resolve_column(requested: str, columns: list[str]) -> str:
    for name in columns:
        if name == requested:
            return name
    lowered = requested.strip().lower()
    for name in columns:
        if name.lower() == lowered:
            return name
    raise agents.ToolError(
        f"No column {requested!r}. Available columns: {', '.join(columns)}."
    )


def _markdown_table(header: list[str], rows: list[list[Any]], limit: int) -> tuple[str, int]:
    shown = rows[:limit]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in shown:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines), len(rows)


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float] | None:
    non_empty = [row[column] for row in rows if row[column] != ""]
    try:
        return [float(v) for v in non_empty]
    except ValueError:
        return None


def _agg_entry(rows: list[dict[str, str]], func: str, column: str | None, label: str) -> Any:
    if func == "count" or column is None:
        return len(rows)
    text_values = [row[column] for row in rows]
    non_empty = [v for v in text_values if v != ""]
    numbers = _numeric_values(rows, column)
    if func == "nunique":
        return len(set(non_empty))
    if numbers is None:
        values: list[float | str] = non_empty
    else:
        values = numbers
    if not values:
        return None
    if func == "sum":
        return _num(sum(values))
    if func == "mean":
        return _num(statistics.fmean(values))
    if func == "median":
        return _num(statistics.median(values))
    if func == "min":
        return _num(min(values))
    return _num(max(values))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@agents.tool(model_access="direct")
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_text: Annotated[str | None, Field(description="The raw CSV file content. Use this when the Drive connector returned the file's text.")] = None,
    url: Annotated[str | None, Field(description="A direct download or export URL for the CSV. Use this when the Drive connector returned a URL instead of the content.")] = None,
    file_name: Annotated[str, Field(description="File name to remember for this dataset.")] = "data.csv",
) -> dict[str, Any]:
    """Load the CSV file to analyse. Exactly one of csv_text or url must be given."""
    if (csv_text is None) == (url is None):
        raise agents.ToolError("Give exactly one of csv_text or url.")
    if url is not None and not url.startswith("https://"):
        raise agents.ToolError("url must be an https:// URL.")
    if url is not None:
        request = urllib.request.Request(url, headers={"User-Agent": "data-analyst-agent"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_URL_BYTES + 1)
        if len(data) > MAX_URL_BYTES:
            raise agents.ToolError("The downloaded file exceeds 100 MB.")
        csv_text = data.decode("utf-8-sig", errors="replace")
    assert csv_text is not None
    if len(csv_text) > MAX_CSV_TEXT_CHARS:
        raise agents.ToolError(
            "The CSV content passed as text exceeds 10 MB. Pass a download url instead."
        )
    if not csv_text.strip():
        raise agents.ToolError("The CSV content is empty.")

    columns, rows = _parse_rows(csv_text)
    store = _session_store(ctx.session.id)
    store["file_name"] = file_name
    store["columns"] = columns
    store["rows"] = rows
    store["filters"] = []

    return {
        "file_name": file_name,
        "rows": len(rows),
        "columns": columns,
        "first_rows": [[row[c] for c in columns] for row in rows[:PREVIEW_ROWS]],
        "note": "File loaded. Follow-up questions now apply to this dataset.",
    }


@agents.tool(model_access="direct")
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Describe the columns of the loaded file: type, missing values, ranges for numbers, top values for text."""
    store = _session_store(ctx.session.id)
    if store["file_name"] is None:
        raise agents.ToolError("No file is loaded. Load it first with load_csv.")
    rows = _apply_filters(store)
    per_column = _column_values(store["columns"], rows)

    description: dict[str, Any] = {}
    for name in store["columns"]:
        values = per_column[name]
        non_empty = [v for v in values if v != ""]
        entry: dict[str, Any] = {
            "type": _column_type(name, per_column),
            "missing": len(values) - len(non_empty),
        }
        if entry["type"] == "number":
            numbers = [float(v) for v in non_empty]
            entry["min"] = _num(min(numbers)) if numbers else None
            entry["max"] = _num(max(numbers)) if numbers else None
            entry["mean"] = _num(statistics.fmean(numbers)) if numbers else None
        else:
            counts: dict[str, int] = {}
            for v in non_empty:
                counts[v] = counts.get(v, 0) + 1
            entry["distinct"] = len(counts)
            entry["top_values"] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:PREVIEW_ROWS]
        description[name] = entry

    return {
        "file_name": store["file_name"],
        "rows_in_scope": len(rows),
        "active_filters": [dict(f) for f in store["filters"]],
        "columns": description,
    }


@agents.tool(model_access="direct")
async def add_filter(
    ctx: agents.ToolCallbackContext,
    /,
    column: Annotated[str, Field(description="Column to filter on.")],
    op: Annotated[str, Field(description=f"One of {', '.join(FILTER_OPS)}.")],
    value: Annotated[Any, Field(description="Comparison value: a number, text, boolean, or a list of those for in/not_in. Ignored for is_null/is_not_null.")] = None,
) -> dict[str, Any]:
    """Keep only the rows matching a condition. Filters accumulate; call this once per condition."""
    if op not in FILTER_OPS:
        raise agents.ToolError(f"Unknown op {op!r}. Use one of: {', '.join(FILTER_OPS)}.")
    store = _session_store(ctx.session.id)
    if store["file_name"] is None:
        raise agents.ToolError("No file is loaded. Load it first with load_csv.")
    column = _resolve_column(column, store["columns"])
    if op not in ("is_null", "is_not_null") and value is None:
        raise agents.ToolError(f"op {op!r} needs a value.")
    if op in ("in", "not_in") and not isinstance(value, list):
        raise agents.ToolError(f"op {op!r} needs a list as value.")
    before = _apply_filters(store)
    filtered = _apply_one(before, column, op, value)
    store["filters"].append({"column": column, "op": op, "value": value})
    return {
        "file_name": store["file_name"],
        "active_filters": [dict(f) for f in store["filters"]],
        "rows_before": len(before),
        "rows_after": len(filtered),
    }


@agents.tool(model_access="direct")
async def clear_filters(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Remove all active filters so the whole file is in scope again."""
    store = _session_store(ctx.session.id)
    if store["file_name"] is None:
        raise agents.ToolError("No file is loaded. Load it first with load_csv.")
    removed = store["filters"]
    store["filters"] = []
    return {
        "file_name": store["file_name"],
        "removed_filters": removed,
        "rows_in_scope": len(store["rows"]),
    }


@agents.tool(model_access="direct")
async def preview_rows(
    ctx: agents.ToolCallbackContext,
    /,
    limit: Annotated[int, Field(description="How many rows to show (at most 20).")] = 10,
) -> dict[str, Any]:
    """Show a small sample of the rows currently in scope, as a markdown table."""
    limit = min(max(limit, 1), 20)
    store = _session_store(ctx.session.id)
    if store["file_name"] is None:
        raise agents.ToolError("No file is loaded. Load it first with load_csv.")
    rows = _apply_filters(store)
    header = store["columns"]
    table, total = _markdown_table(header, [[row[c] for c in header] for row in rows], limit)
    return {
        "file_name": store["file_name"],
        "rows_in_scope": total,
        "active_filters": [dict(f) for f in store["filters"]],
        "table": table,
    }


@agents.tool(model_access="direct")
async def aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="Column(s) to group by. Empty for a single overall group.")] = [],
    aggregations: Annotated[list[str], Field(description="Each entry is 'func' or 'func:column', for example 'count', 'sum:revenue', 'mean:price'. Available functions: count, sum, mean, min, max, median, nunique.")] = ["count"],
    sort_by: Annotated[str | None, Field(description="Aggregation entry to sort by, highest first. Defaults to the first aggregation.")] = None,
    limit: Annotated[int, Field(description="How many groups to return (at most 100).")] = GROUP_ROW_LIMIT_DEFAULT,
) -> dict[str, Any]:
    """Group the rows currently in scope and compute aggregates, e.g. total revenue per country."""
    limit = min(max(limit, 1), GROUP_ROW_LIMIT_MAX)
    store = _session_store(ctx.session.id)
    if store["file_name"] is None:
        raise agents.ToolError("No file is loaded. Load it first with load_csv.")
    if not aggregations:
        aggregations = ["count"]

    parsed: list[tuple[str, str | None, str]] = []
    for spec in aggregations:
        spec = spec.strip()
        func, _, column = spec.partition(":")
        func = func.strip().lower()
        if func == "count_distinct":
            func = "nunique"
        if func not in AGG_FUNCS:
            raise agents.ToolError(f"Unknown function {func!r}. Use one of: {', '.join(AGG_FUNCS)}.")
        if func != "count" and not column:
            raise agents.ToolError(f"{func!r} needs a column: use '{func}:<column>'.")
        if column:
            column = _resolve_column(column, store["columns"])
        parsed.append((func, column, func + (f":{column}" if column else "")))

    group_cols = [_resolve_column(col, store["columns"]) for col in group_by]
    rows = _apply_filters(store)
    if group_cols:
        buckets: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            key = "\x1f".join(row[c] for c in group_cols)
            buckets.setdefault(key, []).append(row)
    else:
        buckets = {"": rows}

    labels = [label for _, _, label in parsed]
    results: list[dict[str, Any]] = []
    for key, group_rows in buckets.items():
        entry: dict[str, Any] = dict(zip(group_cols, key.split("\x1f")))
        for func, column, label in parsed:
            entry[label] = _agg_entry(group_rows, func, column, label)
        results.append(entry)

    if sort_by:
        wanted = sort_by.strip()
        normalized = wanted.replace("count_distinct", "nunique")
        sort_label = next(
            (label for _, _, label in parsed if label == wanted or label == normalized),
            None,
        )
        if sort_label is None:
            raise agents.ToolError(
                f"sort_by {sort_by!r} is not one of the requested aggregations: {', '.join(labels)}."
            )
    else:
        sort_label = parsed[0][2]

    results.sort(
        key=lambda e: (
            e[sort_label] is None,
            -(e[sort_label] or 0) if isinstance(e[sort_label], (int, float)) else str(e[sort_label] or ""),
        )
    )
    total = len(results)
    header = group_cols + labels
    table, _ = _markdown_table(header, [[e[c] for c in header] for e in results], limit)

    return {
        "file_name": store["file_name"],
        "active_filters": [dict(f) for f in store["filters"]],
        "rows_in_scope": len(rows),
        "groups_total": total,
        "table": table,
        "note": "Table shows the top groups by " + sort_label + "." if total > limit else "All groups shown.",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You are a conversational data analyst. The user gives you a link to a CSV file in their Google Drive and asks questions about it in plain language.

Loading the file:
- When the user shares a Google Drive link or file name, use your Drive connector tools to locate that exact file and get its content: resolve the link to a file, then retrieve the CSV text or a download/export URL for it.
- If the connector returns the file's text, call load_csv with csv_text. If it returns a direct download or export URL, call load_csv with url.
- After loading, tell the user the file name, row count and columns. If the user asks for a different file, load it and say so.
- If a tool says the dataset is not loaded in this session (for example after a long pause), reload it from the link before answering.

Answering:
- Every number in an answer must come from a tool result computed on the loaded data. Never invent figures or estimate from memory.
- Use describe_columns to understand the structure before answering questions about columns.
- Use add_filter for conditions ("only rows where..."), clear_filters when the user wants the whole file again, preview_rows to show examples, and aggregate for group-by totals, averages, medians, min/max and distinct counts.
- Filters persist across turns. Say which filters are active when they shape an answer, and quote the row counts you observe.
- If a filter returns 0 rows, say so and check value spelling with describe_columns before concluding.
- When a question needs math beyond these tools (correlations, joins across files, regressions), say what you can and cannot compute.

Conversation:
- Remember the file and everything discussed this session; follow-ups refer to the loaded dataset and its active filters without re-explaining.
- Answer concisely, with concrete numbers, in the user's language.
"""

agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    description="Conversational analyst for CSV files in the user's Google Drive.",
    instructions=INSTRUCTIONS,
    tools=[
        load_csv,
        describe_columns,
        add_filter,
        clear_filters,
        preview_rows,
        aggregate,
    ],
    connectors=[agents.connector(DRIVE_CONNECTOR)],
    builtins=agents.Builtins(
        subagents=False,
        background_processes=False,
    ),
)


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        result = await agent.run(
            "I will share a CSV from my Google Drive and analyse it. "
            "For now, tell me what you can do and what you need from me."
        )
        print(result.text)

    asyncio.run(main())
