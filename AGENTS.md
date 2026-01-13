# AGENTS.md - Multi-Agent Workflow Documentation

This document describes the agent-based workflow architecture of ResearchMCP.

## Agent Roles

### 1. Research Agent (Primary)

The main LLM agent in Cursor chat that:
- Interprets user research goals
- Calls MCP tools to gather papers
- Generates creative research ideas
- Writes paper content
- Coordinates the full workflow

### 2. Workflow Orchestrator (MCP)

The MCP server acts as a workflow orchestrator:
- Tracks current stage (context → ideas → experiments → writing)
- Enforces prerequisites between stages
- Blocks actions until requirements are met
- Provides `get_next_action()` guidance

### 3. User (Human-in-the-Loop)

Critical decision points require human approval:
- Idea selection (confirmation codes)
- Experiment validation
- Paper review before submission

## Agent Memory

### Project State Persistence

```
projects/<name>/
├── project.json      # Project metadata
├── context/          # Extracted paper contexts (agent memory)
├── ideas/            # Generated ideas + approval status
├── experiments/      # Experiment logs and results
├── figures/          # Generated visualizations
└── papers/           # Draft and final papers
```

### Workflow State

Stored in SQLite (`workflow.db`):
```python
@dataclass
class WorkflowState:
    project_id: str
    stage: str  # context_gathering, idea_generation, experimenting, etc.
    approved_idea_id: Optional[str]
    completed_experiments: list[str]
    figures_generated: list[str]
    paper_drafted: bool
```

### Paper Context Cache

Papers and their extracted contexts cached in `papers_cache.db`:
- Full abstracts
- Section structures
- Citation patterns
- Writing style metrics

## Workflow Stages

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: CONTEXT GATHERING                                       │
│  ─────────────────────────────                                    │
│  Tools: fetch_hf_trending, search_papers, extract_paper_context   │
│  Output: Paper abstracts cached, themes identified                │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2: IDEA GENERATION                                         │
│  ─────────────────────────                                        │
│  Tools: generate_ideas (returns context), submit_idea             │
│  Output: Ideas with novelty scores, awaiting approval             │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3: USER APPROVAL (BLOCKING)                                │
│  ────────────────────────────────                                 │
│  User types: APPROVE idea_xxx CODE 1234                           │
│  Workflow blocked until human approves                            │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4: EXPERIMENT SETUP                                        │
│  ─────────────────────────                                        │
│  Tools: create_experiment_env, install_dependencies, setup_datasets│
│  Output: Environment ready, data prepared                         │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5: EXPERIMENTATION                                         │
│  ────────────────────────                                         │
│  Tools: run_experiment, run_baseline, run_ablation                │
│  Output: Metrics collected, checkpoints saved                     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 6: ANALYSIS                                                │
│  ─────────────────                                                │
│  Tools: plot_comparison_bar, compare_to_baselines, check_significance│
│  Output: Figures generated, statistical validation complete       │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 7: WRITING                                                 │
│  ────────────────                                                 │
│  Tools: format_results_table, get_citations_for_topic, create_paper_skeleton│
│  Output: LaTeX content, BibTeX references                         │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 8: FORMATTING & COMPILATION                                │
│  ─────────────────────────────────                                │
│  Tools: cast_to_format, generate_poster, compile_paper            │
│  Output: Conference-ready PDF                                     │
└──────────────────────────────────────────────────────────────────┘
```

## Communication Protocol

### Agent → MCP

All tool calls return JSON with status:
```json
{
  "status": "SUCCESS" | "BLOCKED" | "AWAITING_USER_APPROVAL",
  "data": {...},
  "next_action": {"tool": "...", "description": "..."}
}
```

### MCP → Agent

Guidance provided through:
- `get_next_action()` - What tool to call next
- `get_workflow_checklist()` - Full progress overview
- Tool descriptions with prerequisites

### Agent ↔ User

Ideas presented with approval commands:
```
Idea 1: "Energy-Guided Routing" (★★★★★)
→ APPROVE idea_a13e83fb CODE 1548

Idea 2: "Control Variate Sparse Attention" (★★★★★)
→ APPROVE idea_13021310 CODE 8865
```

User types command, agent calls `approve_idea()`.

## Error Recovery

### API Failures
```python
# Graceful fallback in apis/
try:
    result = await arxiv_client.search(query)
except Exception:
    return []  # Empty results, don't crash
```

### Workflow Violations
```python
# Tools check prerequisites
if not workflow.approved_idea_id:
    return {"status": "BLOCKED", "reason": "Approve an idea first"}
```

### State Recovery
```python
# Workflow state persisted to disk
workflow = await workflow_db.get_project_workflow(project_id)
# Can resume from any stage
```

## Best Practices

1. **Always check `get_next_action()` before major operations**
2. **Never auto-approve ideas** - wait for user confirmation code
3. **Generate figures before writing** - needed for cast_to_format
4. **Use project isolation** - each research topic in its own project
5. **Commit experiment results** - git integration tracks changes
