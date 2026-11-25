# data-analyst

A conversational data analyst agent built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart).
You give it a link to a CSV file in your Google Drive and ask questions in plain
language; it answers with real numbers computed from your data.

The model has a **fixed set of tools** — no sandbox, no shell, no arbitrary code
execution:

| Tool | Purpose |
| --- | --- |
| `load_csv` | Load the CSV (content from the Drive connector, or a download URL it returned) |
| `describe_columns` | Type, missing values, min/max/mean or distinct/top values per column |
| `add_filter` | Keep rows matching a condition; filters accumulate and persist across turns |
| `clear_filters` | Back to the whole file |
| `preview_rows` | Small sample of rows currently in scope |
| `aggregate` | Group-by with count/sum/mean/min/max/median/nunique, sorted, capped |

The file and the active filters are remembered **per session**, so follow-up
questions ("ok, now split that by month") need no re-explaining.

The agent reads your Google Drive through the `google_drive_mcp` Studio
connector, acting as you (the session user). When you have not connected Drive
yet, the session asks you to sign in.

## Fixed-tool guarantee

- No `environment=` (no shell, no file tools), no subagents, no background
  processes.
- Analysis tools are `model_access="direct"`: they live in the model's tool
  list with typed schemas; every number in an answer comes from one of them.
- Programmatic tool calling stays enabled only because the connector requires
  it; the TypeScript programs the model writes can only orchestrate the fixed
  tools — it cannot read or transform data on its own.

## Local run

```bash
uv sync
export MISTRAL_API_KEY="..."   # a key with Workflows access
uv run pytest -q               # tool tests, no model needed
uv run agents chat             # chat locally with the agent
```

## Deploy (Mistral-managed)

`agents deploy` registers the agent as a workflow, builds the pushed commit,
and hosts it. You need: a GitHub repo holding this project, the GitHub and
Google Drive connectors connected in
[Mistral Studio](https://console.mistral.ai/build/connectors), and a
`MISTRAL_API_KEY` for a workspace role with Workflows access.

```bash
git push                      # commit and push everything incl. Dockerfile
uv run agents deploy          # generates Dockerfile on first run if missing
```

After deploy, chat with the agent from Vibe at the URL the deploy prints, or
from a terminal: `uv run agents chat data-analyst`.

## Tests

`test_data_tools.py` exercises every tool (parsing, filters, aggregates,
validation) with a fixture CSV, without a model or API key, including one
scripted-model turn that verifies the model-driven tool loop.
