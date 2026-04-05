# Spec: Agent / Orchestrator

**Module:** `src/agents/orchestrator.py`, `src/agents/base.py`

## OrchestratorAgent

### Responsibilities

1. Pre-flight proxy health check (abort if unhealthy).
2. Initialise project directory structure.
3. Sequence stages: Research → Experiment → Writing → Review.
4. Skip already-completed stages (resume support via `project.json`).
5. Handle stage-level errors: retry up to `max_retries`, then abort with structured error.
6. Collect and report metrics: total tokens, total cost, wall-clock time per stage.

### Stage transition rules

| From stage | To stage | Pre-condition |
|-----------|---------|---------------|
| (start) | Research | Proxy health OK |
| Research | Experiment | `ideas/selected_idea.json` exists and novelty ≥ 0.5 |
| Experiment | Writing | `experiments/all_results.json` exists; at least 1 hypothesis passes p < 0.05 |
| Writing | Review | `paper/main.tex` exists; word count ≥ 4 000 |
| Review | Complete | 2× PASS from ReviewerAgent |

If a pre-condition fails, the orchestrator does not advance the stage. It logs the condition failure and aborts.

### Stop conditions

| Condition | Result |
|-----------|--------|
| Stage completes successfully | Advance to next stage |
| Stage fails after `max_retries` (default 3) | `PIPELINE_ABORT` with reason |
| Budget exceeded | `PIPELINE_ABORT: BUDGET_EXCEEDED` |
| Proxy becomes unavailable mid-run | `PIPELINE_ABORT: PROXY_UNAVAILABLE` |
| Reviewer returns 3× FAIL | `PIPELINE_ABORT: REVIEW_FAILED` |
| SHA-256 integrity violation | `PIPELINE_ABORT: INTEGRITY_VIOLATION` |

### Retry / fallback

- **Agent-level retry:** `BaseAgent.call_with_retry()` retries up to 3 times on transient errors (429, 5xx, timeout).
- **Stage-level retry:** OrchestratorAgent retries a failed stage up to `stage_max_retries` (default 2) before aborting.
- **Model fallback:** `AgentClient` downgrades model on 429 (opus → sonnet → haiku) before stage-level retry counts.

## BaseAgent

All agents inherit `BaseAgent` which provides:

- `build_context()` — reads only the required files for this agent; returns structured context dict.
- `call_with_retry(prompt, system, **kwargs)` — wraps `AgentClient.call()` with retry logic.
- `log(level, message)` — appends to `progress.txt` via `ProgressState`.
- `mark_stage_complete(stage)` — updates `project.json`.

### Fresh context guarantee

`build_context()` always reads from the filesystem at call time. It never uses cached in-memory state from previous calls. This ensures that if a file was written by a previous agent, the current agent sees the updated content.

## Per-Agent Configuration

| Agent | Model | Effort | Budget cap | Fallback |
|-------|-------|--------|-----------|---------|
| ResearchAgent (analysis) | sonnet | medium | — | haiku |
| ResearchAgent (ideas) | sonnet | high | — | haiku |
| ExperimentAgent (impl) | sonnet | high | $0.50 | haiku |
| ExperimentAgent (design) | sonnet | high | $0.50 | haiku |
| ExperimentAgent (analysis) | sonnet | medium | — | haiku |
| WriterAgent (sections) | sonnet | high | — | haiku |
| WriterAgent (expansion) | sonnet | medium | — | haiku |
| ReviewerAgent | opus | high | — | sonnet |
