# Spec: Serving / Configuration

**Module:** `run_pipeline.py`, environment variables

## Launch

```bash
python run_pipeline.py "topic string" [OPTIONS]

Options:
  --conference  Target venue (neurips|icml|iclr|acl|cvpr). Default: neurips
  --model       Base LLM model (sonnet|opus|haiku). Default: sonnet
  --topics      Path to .txt file for batch mode (one topic per line)
  --project-dir Output directory. Default: ~/research-projects/<slug>
  --max-budget  Max total USD spend. Default: 5.00
  --dry-run     Run health checks only; do not start pipeline
```

## Configuration Sources (priority order)

1. CLI arguments (highest priority)
2. Environment variables
3. Defaults in `run_pipeline.py`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key; passed to proxy as HTTP header |
| `PROXY_URL` | No | `http://localhost:3456` | LLM proxy endpoint |
| `SEMANTIC_SCHOLAR_API_KEY` | No | — | Semantic Scholar API key (higher rate limits) |
| `RESEARCH_PROJECTS_DIR` | No | `~/research-projects` | Root directory for all projects |
| `HTTP_PROXY` | — | Cleared by client | Must NOT be set for localhost proxy; client clears it |
| `HTTPS_PROXY` | — | Cleared by client | Same |

**Important:** The system may have `HTTP_PROXY`/`HTTPS_PROXY` set globally (e.g., corporate proxy). `AgentClient` explicitly clears these environment variables for `httpx` connections to `localhost:3456` to prevent interception.

## Secrets Policy

- API keys are read from environment variables only.
- Never passed as CLI arguments (would appear in process list).
- Never written to any file, log, or `progress.txt`.
- `progress.txt` writer applies a redaction regex before each write: strings matching `sk-ant-[a-zA-Z0-9-]{20,}` are replaced with `[REDACTED]`.

## Model Short Names

The proxy accepts short model names and normalises them:

| Short name | Full model ID |
|-----------|---------------|
| `sonnet` | `claude-sonnet-4-6` |
| `opus` | `claude-opus-4-6` |
| `haiku` | `claude-haiku-4-5-20251001` |

## Project Directory Layout

Each run creates:

```
~/research-projects/<project_slug>/
├── project.json          # Stage tracker
├── progress.txt          # Append-only log
├── context/papers.json   # Found papers
├── ideas/
│   ├── all_ideas.json
│   └── selected_idea.json
├── src/                  # Core method code
├── scripts/              # Experiment runners
├── configs/              # YAML configs
├── experiments/
│   ├── hypotheses.json   # Pre-registered
│   ├── all_results.json  # Signed results
│   └── analysis.json
├── verification/         # Per-hypothesis verdicts
├── figures/
├── paper/
│   ├── paper.json
│   ├── main.tex
│   └── main.pdf          # If compilation succeeds
└── review_1.json, review_2.json
```
