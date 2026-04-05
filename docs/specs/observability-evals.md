# Spec: Observability / Evals

**Module:** `src/agents/orchestrator.py`, `src/state/progress.py`, `src/agents/client.py`

## Metrics Collected

### Pipeline metrics (written to `project.json` at completion)

| Metric | Type | Description |
|--------|------|-------------|
| `total_wall_time_s` | int | Wall-clock seconds from start to COMPLETE/ABORT |
| `total_tokens_in` | int | Sum of input tokens across all LLM calls |
| `total_tokens_out` | int | Sum of output tokens across all LLM calls |
| `estimated_cost_usd` | float | Calculated from token counts and model pricing |
| `stage_times` | dict | Per-stage wall-clock times `{research: N, experiment: N, ...}` |
| `retry_count` | int | Total agent-level retries across all stages |
| `fallback_activations` | int | Number of times model was downgraded (opus→sonnet, etc.) |
| `papers_fetched` | int | Total papers returned by all three APIs |
| `hypotheses_registered` | int | Pre-registered hypotheses count |
| `hypotheses_passed` | int | Hypotheses that passed p < 0.05 |
| `hypotheses_blocked` | int | Hypotheses that failed the gate |
| `reviewer_attempts` | int | Number of review cycles (target ≤ 2) |
| `final_word_count` | int | Word count of `paper/main.tex` |

### Per-call metrics (written to `progress.txt`)

Each LLM call appends a metrics line:

```
[2026-04-06T14:32:01Z] [client] [METRICS] model=sonnet tokens_in=3421 tokens_out=1203 latency_ms=8432 cost_usd=0.0124
```

### Experiment metrics (written to `experiments/all_results.json`)

All numeric outputs from experiment scripts (loss, accuracy, latency, etc.) are stored verbatim with their SHA-256 hash.

## Logs

All observability data flows through `progress.txt`. There is no external logging service.

**Log levels used:**

| Level | Usage |
|-------|-------|
| `INFO` | Normal pipeline events |
| `WARN` | Non-fatal issues (API rate limit hit, section too short) |
| `ERROR` | Fatal issues before abort |
| `HASH` | SHA-256 integrity entries |
| `METRICS` | Per-call LLM metrics |
| `CONFIRM` | Irreversible actions (see governance.md) |

## Traces

No distributed tracing. The `progress.txt` file serves as the linear trace of all pipeline actions. Each entry includes a timestamp and agent name, providing a sequential audit trail.

## Evals

### Pipeline-level evals (PoC)

Run `python test_components.py all` to check:

| Check | What it validates |
|-------|-------------------|
| `health` | Proxy responds at localhost:3456 |
| `client` | SDK can make a minimal Anthropic API call |
| `effort` | Effort levels (low/medium/high) change response quality |
| `fallback` | Model fallback chain activates on simulated 429 |

### Quality checks (within pipeline)

| Check | When | Pass condition |
|-------|------|----------------|
| Novelty score | After idea generation | ≥ 0.5 (relaxed) or ≥ 0.7 (strict) |
| p-value gate | After each experiment | p < 0.05 |
| SHA-256 integrity | At review stage | All hashes match |
| Word count | After writing | ≥ 6 000 words |
| Reviewer verdict | Review stage | 2× PASS from opus |

### Future evals (post-PoC)

- **Human evaluation:** researcher rates generated paper quality (1–5) on novelty, correctness, clarity.
- **Venue acceptance rate:** track submitted-vs-accepted ratio over time.
- **Reproducibility:** third party re-runs `src/` + `scripts/` and checks result delta ≤ 5 %.
