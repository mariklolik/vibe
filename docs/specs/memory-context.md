# Spec: Memory / Context Handling

**Module:** `src/state/progress.py`, `src/agents/base.py`

## Session State

Vibe has **no persistent in-process session state**. Each agent call is independent. Between calls, all state lives on the filesystem.

There is no vector memory, no conversation history accumulation, and no shared mutable state between agents.

## Memory Policy

| Data | Scope | TTL | Location |
|------|-------|-----|---------|
| Pipeline stage | Project lifetime | Until project deleted | `project.json` |
| Found papers | Project lifetime | Until project deleted | `context/papers.json` |
| Selected idea | Project lifetime | Until project deleted | `ideas/selected_idea.json` |
| Pre-registered hypotheses | Project lifetime | Until project deleted | `experiments/hypotheses.json` |
| Experiment results + hashes | Project lifetime | Until project deleted | `experiments/all_results.json` |
| Paper sections | Project lifetime | Until project deleted | `paper/paper.json` |
| Execution log | Project lifetime | Until project deleted | `progress.txt` (append-only) |
| In-process LLM response | Single call | Discarded after call | RAM only |
| Full prompt content | Single call | Discarded after call | RAM only |

Nothing is ever stored in a database, vector store, or cloud service.

## Context Budget

Each agent call is constructed with a minimal context: only the files it needs, not the full project state.

| Agent | Files read at call time | Approx. tokens |
|-------|------------------------|----------------|
| ResearchAgent (ideas) | `context/papers.json` (abstracts only) | ~8 000 |
| ExperimentAgent (impl) | `ideas/selected_idea.json` + top-3 paper methods | ~6 000 |
| ExperimentAgent (verify) | `experiments/hypotheses.json` + result JSONs | ~4 000 |
| WriterAgent (per section) | `experiments/analysis.json` + previous sections (summary) | ~10 000 |
| ReviewerAgent | `paper/main.tex` + `experiments/all_results.json` + last 100 lines of `progress.txt` | ~12 000 |

If a file exceeds the token budget, it is truncated to the most recent / most relevant N tokens with an explicit `[TRUNCATED]` marker so the agent is aware.

## Resume Behaviour

`project.json` tracks which stages have completed:

```json
{
  "stage": "writing",
  "completed_stages": ["research", "experiment"],
  "started_at": "2026-04-06T14:00:00Z"
}
```

If `run_pipeline.py` is called on an existing project, the orchestrator reads `project.json` and skips completed stages. This allows recovery from crashes without re-running expensive LLM and experiment stages.

## Progress.txt Format

Append-only log. Each line is:

```
[ISO8601_UTC] [agent_name] [LEVEL] message
```

Example:
```
[2026-04-06T14:32:01Z] [research] [INFO] Fetched 12 papers (query="efficient attention")
[2026-04-06T14:32:45Z] [experiment] [HASH] results/main.json sha256=a3f8c2...
[2026-04-06T14:33:10Z] [verification] [INFO] h1: p=0.003 effect_size=1.8 PASS
```

Writes are atomic: content written to `progress.tmp`, then `os.replace()` to `progress.txt`. A file lock (`fcntl.flock`) is acquired before each write.
