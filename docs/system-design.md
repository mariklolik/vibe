# System Design — Vibe

## 1. Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Context model** | Fresh context per call (no accumulated conversation) | Prevents context poisoning across pipeline stages; each agent gets only what it needs |
| **State persistence** | `progress.txt` + project JSON files | Human-readable; survives crashes; enables resume without re-running completed stages |
| **LLM access** | Local proxy at `localhost:3456` (Anthropic SDK passthrough) | Adds effort levels, budget caps, fallback chain, and rate limit detection transparently |
| **Code structure** | Two-phase: `src/` (importable classes) then `scripts/` (runners that import from src/) | Separation of method from experiment harness; `src/` is reusable, `scripts/` are ephemeral |
| **Review model** | Independent opus agent, requires 2× PASS | Reviewer has no shared state with writer; opus chosen for higher reasoning quality |
| **Statistical gate** | p < 0.05 mandatory, pre-registered hypotheses | No claim appears in the paper without statistical backing; pre-registration prevents p-hacking |
| **Fallback chain** | opus → sonnet → haiku per call | Resilience against rate limits; quality degrades gracefully |

---

## 2. Modules and Their Roles

| Module | File | Role |
|--------|------|------|
| **CLI** | `run_pipeline.py` | Entry point; parses args, initialises project, calls orchestrator |
| **Orchestrator** | `src/agents/orchestrator.py` | Coordinates all agents in sequence; health checks; collects metrics; handles retries |
| **Client** | `src/agents/client.py` | Anthropic SDK wrapper; adds proxy params (effort, budget, fallback); tracks token usage |
| **Base agent** | `src/agents/base.py` | Fresh-context loop; reads progress.txt; writes structured log entries |
| **Research agent** | `src/agents/research.py` | Literature search across 3 APIs; idea generation; auto-selection (novelty ≥ 0.7) |
| **Experiment agent** | `src/agents/experiment.py` | Method implementation; script generation; execution; SHA-256 signing; statistical verification |
| **Writer agent** | `src/agents/writer.py` | Section-by-section paper writing (7 calls); expansion loop if word count short |
| **Reviewer agent** | `src/agents/reviewer.py` | Independent paper review; opus model; 2× PASS required; writes `review_*.json` |
| **Progress state** | `src/state/progress.py` | Atomic writes to progress.txt; stage tracking; resume detection |
| **arXiv API** | `src/apis/arxiv.py` | Paper search with 3 s delay between requests |
| **HuggingFace API** | `src/apis/huggingface.py` | Trending paper search |
| **Semantic Scholar API** | `src/apis/semantic_scholar.py` | Semantic search with 1 s delay; graceful fallback on 429 |

---

## 3. Workflow — Main Execution Path

```
run_pipeline.py
    │
    ├─ [0] Proxy health check ──► abort if unhealthy
    │
    ├─ [1] ResearchAgent.run(topic)
    │       ├─ search_papers() × 3 APIs (parallel, rate-limited)
    │       ├─ extract style and citation norms from top-5 papers
    │       ├─ generate_ideas() — 3 candidates with novelty scores
    │       └─ auto_select(novelty ≥ 0.7) → writes ideas/selected_idea.json
    │
    ├─ [2] ExperimentAgent.run(idea)
    │       ├─ Phase 1: implement src/ (model.py, trainer.py, data_utils.py, metrics.py)
    │       ├─ Phase 2: generate scripts/ (run_main.py, run_baselines.py, run_ablations.py)
    │       ├─ execute scripts via subprocess (no shell=True)
    │       ├─ SHA-256 sign all result files → experiments/all_results.json
    │       └─ verify_hypotheses() — p < 0.05 gate per claim
    │
    ├─ [3] WriterAgent.run(results, context)
    │       ├─ write_section() × 7 (Introduction → Conclusion)
    │       ├─ check word count after each section; expansion pass if short
    │       └─ writes paper/paper.json + paper/main.tex
    │
    └─ [4] ReviewerAgent.run(paper)
            ├─ read paper/main.tex + experiments/all_results.json
            ├─ verify SHA-256 chain
            ├─ issue PASS or FAIL with reasons
            ├─ repeat until 2× PASS (max 3 attempts)
            └─ writes review_1.json, review_2.json
```

---

## 4. State / Memory / Context Handling

### Fresh context per call

Every agent call is independent. No conversation history is accumulated across calls. State is read from the filesystem at the start of each call and written back at the end.

### What persists between calls

| Artifact | Location | Written by | Read by |
|----------|----------|------------|---------|
| Pipeline log | `progress.txt` | Every agent (append-only) | Orchestrator, Reviewer |
| Stage tracking | `project.json` | Orchestrator | Orchestrator (resume) |
| Found papers | `context/papers.json` | ResearchAgent | ExperimentAgent, WriterAgent |
| Selected idea | `ideas/selected_idea.json` | ResearchAgent | ExperimentAgent |
| Pre-registered hypotheses | `experiments/hypotheses.json` | ExperimentAgent | ExperimentAgent (verification) |
| Experiment results + hashes | `experiments/all_results.json` | ExperimentAgent | WriterAgent, ReviewerAgent |
| Paper sections | `paper/paper.json` | WriterAgent | ReviewerAgent |
| Review verdicts | `review_*.json` | ReviewerAgent | Orchestrator |

### Context budget per agent call

Each agent call receives only the subset of files it needs — not the full project state. This keeps prompts small and prevents irrelevant context from polluting agent reasoning.

| Agent | Context passed in | Context size (approx.) |
|-------|------------------|------------------------|
| ResearchAgent | Topic string + paper abstracts (top 5) | ~8 000 tokens |
| ExperimentAgent (impl) | Selected idea + paper method sections | ~6 000 tokens |
| ExperimentAgent (verify) | hypotheses.json + result files | ~4 000 tokens |
| WriterAgent | results, context papers, section outline | ~10 000 tokens per section |
| ReviewerAgent | main.tex + all_results.json + progress.txt tail | ~12 000 tokens |

---

## 5. Retrieval Contour

The pipeline does not use a vector store or RAG system. Retrieval is structured API calls:

1. **arXiv API** — keyword search; returns metadata + abstract; 3 s inter-request delay.
2. **HuggingFace Papers API** — trending papers with topic filter.
3. **Semantic Scholar API** — semantic search with relevance scoring; 1 s delay; falls back gracefully on rate limit.

Papers are deduplicated by arXiv ID. The top 5 by citation count × recency score are passed to the research agent as structured JSON (title, abstract, authors, citation count, year). No raw PDF content is parsed in the PoC.

---

## 6. Tool / API Integrations

| Integration | Type | Auth | Rate limit handling |
|------------|------|------|---------------------|
| Anthropic API | REST via proxy (localhost:3456) | API key via env var | Proxy handles backoff; fallback chain |
| arXiv API | REST (no auth) | None | 3 s sleep between requests |
| HuggingFace Papers | REST (no auth) | None | No strict limit; 1 s sleep |
| Semantic Scholar | REST | Optional API key | 1 s sleep; 429 → skip with warning |
| Local subprocess | OS process | None | Timeout 300 s; non-zero exit → retry × 3 |
| Filesystem | Local I/O | None | Atomic writes via temp file + rename |

---

## 7. Failure Modes, Fallback, and Guardrails

| Failure | Detection | Fallback | Guardrail |
|---------|-----------|---------|-----------|
| Proxy down | Health check at pipeline start | Abort early (no API calls wasted) | `PROXY_UNAVAILABLE` error |
| Rate limit (429) | HTTP response code | Exponential backoff + model downgrade | Max 3 retries per call |
| Experiment script crash | Non-zero exit code | Retry with modified script × 3 | After 3 failures: `EXPERIMENT_FAILED`; optionally synthetic data with explicit flag |
| p-value fails gate | p ≥ 0.05 | Re-run with larger sample or alternative test | Claim blocked from paper unconditionally |
| SHA-256 mismatch | Hash check in ReviewerAgent | Pipeline aborted | `INTEGRITY_VIOLATION` — no paper produced |
| Writer produces short section | Word count < 4 096 tokens | Expansion prompt issued (up to 2 passes) | If still short after 2 passes: logged as `SECTION_SHORT` |
| Reviewer FAIL | Verdict field = FAIL | Re-run writer with reviewer feedback (max 3 attempts) | After 3 FAILs: `REVIEW_FAILED`; paper saved for human inspection |
| Budget exceeded | Proxy returns budget error | Pipeline aborted | `BUDGET_EXCEEDED` with cumulative spend logged |

---

## 8. Technical and Operational Constraints

### Latency

| Operation | P50 | P95 | Hard timeout |
|-----------|-----|-----|-------------|
| Single LLM call (sonnet) | 8 s | 25 s | 30 s |
| Single LLM call (opus) | 12 s | 45 s | 60 s |
| Experiment script execution | 20 s | 120 s | 300 s |
| Literature search (3 APIs) | 15 s | 40 s | 60 s |
| Full pipeline | 45 min | 90 min | 120 min |

### Cost

| Stage | Model | Estimated cost |
|-------|-------|---------------|
| Research (search + ideas) | sonnet | ~$0.10 |
| Experiment (impl + design) | sonnet | ~$0.50 per phase |
| Writing (7 sections) | sonnet | ~$0.35 |
| Review (2 passes) | opus | ~$0.40 |
| **Total** | mixed | **≤ $2.00** |

### Reliability

| Concern | Approach |
|---------|---------|
| Crash recovery | `project.json` stage tracker; pipeline skips completed stages on resume |
| Data loss | Atomic writes; progress.txt append-only |
| API unavailability | Fallback model chain; graceful degradation per API |
| Python version | Tested on 3.11; `sys.version_info` check at startup |
