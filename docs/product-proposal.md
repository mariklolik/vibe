# Product Proposal — Vibe

## 1. Problem & Motivation

ML researchers at top venues spend the majority of their non-research time on workflow overhead: finding related work, scaffolding experiments, verifying statistical soundness, and wrestling with LaTeX templates. This is not a tooling gap that Cursor or Copilot can close — they assist with individual code fragments but have no model of the research process as a whole. A researcher still has to orchestrate every step manually.

**Vibe** is a fully autonomous research pipeline that takes a topic string as input and produces a statistically verified, publication-ready paper as output. It is designed for the solo researcher or small lab team publishing 3–8 papers per year at NeurIPS / ICML / ICLR / ACL / CVPR who wants to offload the plumbing entirely.

---

## 2. Goals and Success Metrics

### Product metrics (does it deliver value?)

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end pipeline success rate | ≥ 80 % | Pipeline runs that reach `COMPLETE` without manual intervention |
| Researcher time saved per paper | ≥ 12 h | Survey of early users (target: PoC → user study) |
| Reviewer-pass rate (internal) | ≥ 90 % | Fraction of runs where opus reviewer returns 2× PASS |
| Paper accepted by venue (long-term) | ≥ 20 % | Tracked post-submission (outside PoC scope) |

### Agentic metrics (is the agent pipeline reliable?)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Agent retry rate | ≤ 15 % per stage | Retries logged in `progress.txt` |
| Fallback model activation rate | ≤ 10 % of calls | Tracked in proxy metrics |
| Statistical verification pass rate | 100 % | No unverified claim reaches the paper |
| SHA-256 log integrity violations | 0 | Checked at review stage |
| Average agent calls per pipeline | ≤ 60 | Tracked in orchestrator metrics |

### Technical metrics (does it run well?)

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end wall-clock time | ≤ 90 min | `progress.txt` timestamps |
| p95 latency per agent call | ≤ 30 s | Proxy request logs |
| Cost per full pipeline run | ≤ $2.00 | Proxy real token usage |
| Proxy health check success | 100 % before run start | `test_components.py health` |
| Pipeline crash rate (unhandled exception) | ≤ 5 % | Exception logs |

---

## 3. Use Cases and Edge Cases

### Primary use cases

| Use case | Actor | Trigger | Expected output |
|----------|-------|---------|----------------|
| Generate paper on standard ML topic | Researcher | `run_pipeline.py "topic"` | Verified, written paper draft |
| Batch paper generation | Lab manager | `--topics topics.txt` | Sequential pipeline runs, one paper per topic |
| PoC demo for stakeholders | PI / mentor | Same as above | Demonstrates end-to-end autonomy |

### Edge cases and how the system handles them

| Edge case | Detection | Handling |
|-----------|-----------|---------|
| No papers found for topic | Literature search returns 0 results | Pipeline aborts early with `NO_PAPERS_FOUND`; suggests broadening query |
| All generated ideas below novelty threshold (< 0.7) | Idea scorer returns max score < 0.7 | Relaxes threshold to 0.5; if still below, aborts with `LOW_NOVELTY` |
| Experiment script crashes | Non-zero exit code | Agent retries up to 3 times with modified script; falls back to synthetic results with explicit `SYNTHETIC_DATA` flag in paper |
| Statistical test fails (p ≥ 0.05) | Verification module | Claim is blocked from paper; experiment re-run with larger sample or alternative hypothesis |
| SHA-256 mismatch on result log | Reviewer reads progress.txt | Pipeline raises `INTEGRITY_VIOLATION`; reviewer agent marks paper as `FAIL` |
| Proxy rate limit (429) | Proxy returns 429 | Exponential backoff with jitter; fallback to lower model tier (opus → sonnet → haiku) |
| Writer produces < 4 096 tokens per section | Token count check after each section | Expansion prompt issued automatically; up to 2 additional passes per section |
| Reviewer returns FAIL twice | Two consecutive FAIL verdicts | Pipeline aborts; writes `review_fail.json` with reasons for human inspection |
| LaTeX compilation error | pdflatex non-zero exit | Logs error, skips PDF step; paper.json and main.tex still written |

---

## 4. Constraints

### Technical constraints (SLO)

| Constraint | Value | Reason |
|-----------|-------|--------|
| Max wall-clock time | 90 min | Practical limit for a demo session; longer runs lose user trust |
| p95 latency per LLM call | 30 s | Proxy timeout setting; calls over 30 s are retried |
| Max API budget per run | $2.00 USD | Set via `max_budget_usd` on expensive agents |
| Max retries per agent | 3 | Prevents infinite loops on broken experiments |
| Min p-value for a verified claim | 0.05 | Standard statistical significance threshold |
| Min paper word count | 6 000 words | Approximate NeurIPS/ICML acceptance bar |

### Operational constraints

| Constraint | Value |
|-----------|-------|
| Python version | 3.11+ |
| Local proxy required | localhost:3456 (Anthropic SDK passthrough) |
| GPU | Optional; pipeline designed for CPU-feasible toy experiments in PoC |
| Internet access | Required for literature search APIs |
| Disk space per project | ~500 MB (datasets + logs + generated code) |

---

## 5. Architectural Sketch

```
[CLI: run_pipeline.py]
        │
        ▼
[OrchestratorAgent]  ←── health check proxy ──► [localhost:3456]
        │                                              │
        ├──► [ResearchAgent]                    Anthropic API
        │      ├─ arXiv API                           │
        │      ├─ HuggingFace API                fallback chain:
        │      └─ Semantic Scholar API          opus→sonnet→haiku
        │
        ├──► [ExperimentAgent]
        │      ├─ src/  (core method code)
        │      ├─ scripts/  (experiment runners)
        │      └─ SHA-256 log + p<0.05 gate
        │
        ├──► [WriterAgent]
        │      └─ 7 sections × 4096 tok
        │
        └──► [ReviewerAgent]
               └─ opus model, 2× PASS required
```

**Modules and integrations:**

| Module | Role | External integration |
|--------|------|---------------------|
| `src/agents/client.py` | Anthropic SDK wrapper with proxy features | Anthropic API via localhost:3456 |
| `src/agents/research.py` | Literature search + idea selection | arXiv, HuggingFace, Semantic Scholar |
| `src/agents/experiment.py` | Method implementation + execution + verification | Local Python subprocess |
| `src/agents/writer.py` | Section-by-section paper writing | None (pure LLM) |
| `src/agents/reviewer.py` | Independent quality gate | None (pure LLM, opus) |
| `src/agents/orchestrator.py` | Pipeline coordination + health + metrics | proxy health endpoint |
| `src/state/progress.py` | State persistence (progress.txt) | Filesystem |
| `src/apis/` | Paper retrieval clients | arXiv, HuggingFace, Semantic Scholar |

---

## 6. Data Flow — What Is and Is Not Delegated to LLM/Agent

### Delegated to LLM / Agent

| Task | Agent | Notes |
|------|-------|-------|
| Generating research ideas from paper abstracts | ResearchAgent | Creative task; LLM provides novelty ranking |
| Writing all paper sections | WriterAgent | LLM generates text; no templated content |
| Implementing the core ML method (Python code) | ExperimentAgent | LLM writes `src/model.py`, `src/trainer.py`, etc. |
| Generating experiment scripts | ExperimentAgent | LLM writes `scripts/run_main.py`, etc. |
| Reviewing the paper and issuing PASS/FAIL | ReviewerAgent | Independent opus-model agent |
| Interpreting statistical results and drawing conclusions | ExperimentAgent | LLM reads p-values and effect sizes |

### NOT delegated to LLM / Agent (deterministic pipeline)

| Task | Handler | Notes |
|------|---------|-------|
| Fetching papers from external APIs | `src/apis/` | Deterministic HTTP calls |
| Running experiment scripts | `subprocess` | OS-level process execution |
| Computing statistical tests (t-test, etc.) | `scipy.stats` | Deterministic library |
| SHA-256 signing of result logs | `hashlib` | Deterministic integrity check |
| Checking p-value threshold (p < 0.05) | Verification module | Hard-coded rule, not LLM judgment |
| Tracking pipeline state (progress.txt) | `src/state/progress.py` | File I/O, no LLM involvement |
| Counting words / tokens in paper sections | Character/word counter | Deterministic check |
| Proxy health check | `src/agents/client.py` | HTTP ping to localhost:3456 |
