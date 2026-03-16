# CLAUDE.md - AI Agent Instructions

## Project Overview

Vibe is an autonomous AI research pipeline that generates publication-ready papers
from a research topic. It uses the Anthropic SDK via a local proxy (localhost:3456)
to coordinate specialized agents through a fresh-context-per-call pattern.

## Architecture

```
run_pipeline.py              # CLI entry point
│
├── src/agents/              # Agent infrastructure
│   ├── __init__.py          # Package exports
│   ├── client.py            # Anthropic SDK wrapper (proxy features: effort, budget, fallback)
│   ├── base.py              # Base agent with fresh-context loop + proxy params
│   ├── research.py          # Literature search + idea generation
│   ├── experiment.py        # Method implementation + experiment design/execution/verification
│   ├── writer.py            # Paper writing with expansion loop
│   ├── reviewer.py          # Independent verification (opus model, 2x PASS required)
│   └── orchestrator.py      # Full pipeline coordinator with health checks + metrics
│
├── src/state/               # State management
│   └── progress.py          # progress.txt handler (rom4ik pattern)
│
├── src/apis/                # External API clients (KEPT)
│   ├── arxiv.py             # arXiv paper search
│   ├── huggingface.py       # HuggingFace trending
│   └── semantic_scholar.py  # Semantic Scholar search
│
├── src/tools/               # Tool implementations (KEPT, used by agents)
├── src/db/                  # SQLite databases (KEPT)
├── src/context/             # Paper context extraction (KEPT)
├── src/paper/               # Paper IR and LaTeX rendering (KEPT)
├── src/project/             # Project management + git (KEPT)
│
├── test_components.py       # Component-level tests
└── run_server.py            # Legacy MCP server (still works)
```

## Generated Project Structure (rom4ik pattern)

Each pipeline run creates a self-contained project:

```
~/research-projects/<project_name>/
├── progress.txt              # Full log of all agent actions
├── project.json              # Stage tracking and config
│
├── context/                  # Research papers found
│   └── papers.json
├── ideas/                    # Generated research ideas
│   ├── all_ideas.json
│   └── selected_idea.json
│
├── src/                      # CORE METHOD CODE (importable classes)
│   ├── __init__.py
│   ├── model.py              # Main model/method (nn.Module)
│   ├── trainer.py            # Training loop
│   ├── data_utils.py         # Dataset loading
│   └── metrics.py            # Evaluation metrics
│
├── scripts/                  # EXPERIMENT RUNNERS (import from src/)
│   ├── run_main.py
│   ├── run_baselines.py
│   ├── run_ablations.py
│   ├── analyze_results.py
│   └── generate_figures.py
│
├── configs/                  # YAML/JSON configs
│   ├── default.yaml
│   ├── main_experiment.yaml
│   └── ablation_*.yaml
│
├── experiments/              # Execution outputs
│   ├── experiment_design.json
│   ├── hypotheses.json
│   ├── all_results.json
│   ├── analysis.json
│   ├── results/
│   └── logs/
│
├── verification/             # Statistical verification (p<0.05 gates)
│   └── h1.json, h2.json...
├── figures/                  # Generated visualizations
├── paper/                    # Final paper
│   ├── paper.json
│   ├── main.tex
│   └── main.pdf
├── data/                     # Datasets
├── baselines/                # Baseline implementations
├── requirements.txt          # Python dependencies
└── review_*.json             # Reviewer verdicts
```

## Proxy Features (localhost:3456)

The pipeline leverages all proxy capabilities:

| Feature | Usage |
|---------|-------|
| **Effort levels** | low/medium/high — controls quality/cost tradeoff per call |
| **Budget caps** | max_budget_usd per call — prevents runaway costs |
| **Fallback models** | opus→sonnet→haiku chain for rate limit resilience |
| **Session pool** | Warm sessions with prompt caching for repeat patterns |
| **Rate limit detection** | Proactive backoff when approaching limits |
| **Real token usage** | Actual counts tracked across all phases |
| **Health monitoring** | Check before pipeline, report after |
| **Model normalization** | Use short names: sonnet, opus, haiku |

### Per-Agent Proxy Settings

| Agent | Model | Effort | Budget | Fallback |
|-------|-------|--------|--------|----------|
| Research (analysis) | sonnet | medium | — | haiku |
| Research (ideas) | sonnet | high | — | haiku |
| Experiment (method impl) | sonnet | high | $0.50 | haiku |
| Experiment (design) | sonnet | high | $0.50 | haiku |
| Experiment (analysis) | sonnet | medium | — | haiku |
| Writer (sections) | sonnet | high | — | haiku |
| Writer (expansion) | sonnet | medium | — | haiku |
| Reviewer | opus | high | — | sonnet |

## Key Design Decisions

### 1. Fresh Context Per Call (from rom4ik)
Every agent call is independent — no accumulated conversation history.
State persists via progress.txt and project files only.

### 2. Skills as Agents (from research_claude_agents)
Each agent maps to a focused skill with its own system prompt.

### 3. Two-Phase Code Generation (from rom4ik)
- Phase 1: Implement core method in `src/` (importable classes)
- Phase 2: Generate experiment scripts in `scripts/` (import from src/)

### 4. Auto-Selection (from research_claude_agents)
Ideas auto-selected when novelty >= 0.7 — no human bottleneck.

### 5. Statistical Gates (p < 0.05 mandatory)
No claim can appear in the paper without statistical verification.
Anti-fabrication: experiment log signatures (SHA256).

### 6. Section-by-Section Writing
7 individual API calls × 4096 tokens each — more reliable than one massive call.

### 7. HTTP Proxy Bypass
System has HTTP_PROXY set which intercepts localhost. All httpx clients
must clear HTTP_PROXY/HTTPS_PROXY env vars before connecting to proxy.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Test components
python test_components.py health      # Proxy health check
python test_components.py client      # SDK connectivity
python test_components.py effort      # Effort levels + fallback
python test_components.py all         # All component tests

# Run full pipeline
python run_pipeline.py "efficient attention mechanisms for long sequences"

# With options
python run_pipeline.py "topic" --conference neurips --model sonnet

# Batch (sequential)
python run_pipeline.py --topics topics.txt
```

## Pipeline Stages

```
Topic → Research → Experiment → Writing → Review → Complete
         │           │            │          │
         │           │            │          └─ opus model, 2x PASS (rom4ik)
         │           │            └─ Section-by-section, expansion loop
         │           └─ Implement src/ → Design scripts/ → Execute → Verify (p<0.05)
         └─ Search 3 APIs → Generate ideas → Auto-select (novelty≥0.7)
```
