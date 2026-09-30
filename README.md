# mcp-laya

**Local Laya MCP server — System 1 decision model (open Jev reproduction).**

Runs entirely locally on [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (mmBERT-base, 322M params, 100+ languages). State + typed questions (choice / noul / score) → calibrated probabilities in a **single encoder pass**. No token generation, no API key, no network beyond the one-time model download.

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| `laya_guard_tool_call` | Gate a consequential tool call | `allow` / `confirm` / `deny` |
| `laya_route_model` | Pick a model from candidates | `best_model` |
| `laya_route_task` | Route an ambiguous/risky task | `proceed_fast` / `deep_review` / `split_task` / `block` |
| `laya_check_research` | Judge evidence for a claim | `accept` / `needs_more_verification` / `reject` |
| `laya_review_completion` | Check a task before reporting it done | `complete` / `complete_with_notes` / `incomplete` |
| `laya_decide` | Raw custom state + typed questions | calibrated probabilities |

The 5 presets use **calibrated decision rules** (short positive keyword questions + python-side thresholds) tuned for the 322M model: it is reliable on short yes/no keyword questions but weak on negation and multi-step reasoning, so verdicts are combined in code, not asked from the model.

## Install

```bash
pip install mcp laya
```

## Configure your MCP client

Point the client at **this file** (stdio):

```json
{
  "mcpServers": {
    "laya": {
      "command": "python",
      "args": ["/path/to/mcp-laya/laya_mcp_server.py"]
    }
  }
}
```

> ⚠️ If the `laya` package is installed in the same environment, make sure the client launches your file and not the package's own `laya-mcp-server` entry point — they are different servers.

The server preloads the model at startup (the first tool call then costs just one forward pass). First run downloads the checkpoint (~30 s).

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYA_CHECKPOINT` | `convaiinnovations/laya-multilingual` | HuggingFace checkpoint id or local path |

## Example call

```python
laya_guard_tool_call(
    tool="terminal",
    action="run rm -rf on production database backup directory",
    arguments_summary=["command: rm -rf /data/backups/prod"],
    side_effects=["deletes production backups"],
    reversibility="irreversible",
)
# -> {"answers": {"destr": 0.96, "prod": 0.94, "sends": 0.0}, "decision": "deny"}
```

## Notes

- Encoder-only: no generation, so answers are probabilities, not sentences.
- The model is multilingual — state can be written in 100+ languages.
- Presets were calibrated with a live battery of 6 tests per tool; rules and thresholds are commented in `laya_mcp_server.py`.
