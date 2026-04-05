# Vibe — Autonomous AI Research Pipeline

> From a research topic to a publication-ready paper, fully automated.

---

## Problem

Publishing ML research requires a researcher to context-switch across a dozen tools — arXiv browsers, experiment trackers, statistical libraries, LaTeX editors — while maintaining creative momentum. A typical PhD student spends 40–60 % of their working time on tooling and workflow plumbing rather than on ideas. Existing AI coding assistants (Cursor, Copilot) write code in isolation but have no model of the research process: they don't know that every paper claim requires a p-value, that figures are mandatory, or that a paper will be desk-rejected if it falls short of the venue page target.

**Who feels this pain most:** solo researchers and small lab teams producing 3–8 papers per year at top ML venues. For them, even saving two days per submission is a meaningful competitive advantage.

---

## PoC: What It Demonstrates

A single `python run_pipeline.py "topic"` command runs the following end-to-end:

1. **Literature search** — fetches papers from arXiv, HuggingFace, and Semantic Scholar; extracts style and citation norms from the top results.
2. **Idea generation** — generates research ideas ranked by novelty score; auto-selects the top idea (novelty ≥ 0.7) without blocking on human approval.
3. **Method implementation** — writes production-quality PyTorch code in `src/` (importable model, trainer, data utilities, metrics).
4. **Experiment execution** — generates scripts in `scripts/`, runs them, captures outputs; all results signed with SHA-256 to prevent fabrication.
5. **Statistical verification** — every empirical claim is validated with a t-test or equivalent (p < 0.05 gate); unverified claims are blocked from the paper.
6. **Paper writing** — writes all seven sections independently (Introduction, Related Work, Method, Experiments, Ablations, Discussion, Conclusion) at ≥ 4 096 tokens each.
7. **Independent review** — a second Claude Opus agent reviews the full paper; the paper must pass two consecutive PASS verdicts before completion.

**PoC success criteria:**

| Metric | Target |
|--------|--------|
| End-to-end pipeline completion rate | ≥ 80 % on standard ML topics |
| All empirical claims statistically verified (p < 0.05) | 100 % |
| Paper word count | ≥ 6 000 words |
| Time from topic to draft | ≤ 90 min on a single machine |
| Cost per run (Anthropic API) | ≤ $2.00 |

---

## Out of Scope (PoC)

- **Multi-user collaboration** — single researcher per pipeline run; no concurrent edits or shared approval flows.
- **Proprietary / authenticated datasets** — only publicly available benchmarks; no internal data warehouses or licensed corpora.
- **Distributed GPU training** — local or single-node execution only; no SLURM, SageMaker, or Vertex AI.
- **Post-submission tasks** — no rebuttal generation, camera-ready revision, or reviewer response drafting.
- **Non-CS/ML venues** — templates calibrated to NeurIPS, ICML, ICLR, ACL, CVPR; no medical, humanities, or social science journals.
- **Real-time human-in-the-loop** — the PoC is fully autonomous; human review of intermediate outputs is not part of the current workflow.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Health check
python test_components.py health

# Run pipeline
python run_pipeline.py "efficient attention mechanisms for long sequences"

# With options
python run_pipeline.py "topic" --conference neurips --model sonnet
```

Requires a local proxy at `localhost:3456` (Anthropic SDK passthrough). See `docs/system-design.md` for setup.

---

## Architecture at a Glance

```
Topic → ResearchAgent → ExperimentAgent → WriterAgent → ReviewerAgent → Paper
           │                  │                │               │
        3 APIs           src/ + scripts/   7 sections     opus model
      (arXiv, HF,       SHA-256 log       per-section    2× PASS req.
       Sem. Scholar)    p<0.05 gate       4096 tok each
```

Full architecture: [`docs/system-design.md`](docs/system-design.md)

---

## Docs

| Document | Purpose |
|----------|---------|
| [`docs/product-proposal.md`](docs/product-proposal.md) | Product proposal: metrics, edge cases, data flow |
| [`docs/governance.md`](docs/governance.md) | Risk register, log policy, injection protection |
| [`docs/system-design.md`](docs/system-design.md) | Full architecture, modules, failure modes |
| [`docs/diagrams/`](docs/diagrams/) | C4 Context, Container, Component, Workflow, Data flow |
| [`docs/specs/`](docs/specs/) | Per-module technical specifications |
