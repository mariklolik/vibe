# ResearchMCP

End-to-end AI research pipeline MCP server for Cursor. Enables vibe-coding style research workflow from idea generation to paper publication.

## Quick Start

```bash
# 1. Install
cd research-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Add to ~/.cursor/mcp.json
{
  "mcpServers": {
    "research-mcp": {
      "command": "python3",
      "args": ["run_server.py"],
      "cwd": "/path/to/research-mcp"
    }
  }
}

# 3. Restart Cursor and start chatting!
```

## Project Structure

Projects are created in `~/research-projects/` (separate from MCP code):

```
~/research-projects/
└── my_research/
    ├── context/           # Extracted paper contexts
    ├── ideas/             # Generated and approved ideas
    ├── experiments/
    │   ├── logs/          # Experiment run logs
    │   └── metrics.db     # Tracked metrics (SQLite)
    ├── figures/           # Generated visualizations
    ├── papers/
    │   ├── drafts/
    │   └── final/
    ├── data/              # Datasets
    ├── .git/              # Auto-initialized
    └── project.json
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. CONTEXT         2. IDEAS           3. APPROVAL        4. SETUP          │
│  ─────────────      ─────────────      ─────────────      ─────────────     │
│  fetch_hf_trending  generate_ideas     USER TYPES:        create_env        │
│  search_papers      submit_idea        APPROVE <id>       install_deps      │
│  set_target_metrics (ranked by         CODE <code>        setup_datasets    │
│                      novelty)           ▲                                    │
│                                         │ BLOCKED                            │
│                                         │ until user                         │
│                                         │ approves                           │
│                                                                              │
│  5. EXPERIMENTS     6. ANALYSIS        7. WRITING         8. FORMAT         │
│  ─────────────      ─────────────      ─────────────      ─────────────     │
│  run_experiment     compare_baselines  format_table       cast_to_format    │
│  log_experiment     check_significance check_completeness create_github_repo│
│  collect_metrics    plot_comparison    expand_paper       compile_paper     │
│                     get_exp_history    create_skeleton    finalize_github   │
│                                                                              │
│                         ┌──────────────────────┐                            │
│                         │ EXPANSION LOOP       │                            │
│                         │ if paper too short   │                            │
│                         │ → more experiments   │                            │
│                         │ → more figures       │                            │
│                         └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Principles**:
- Human oversight: Ideas require explicit approval with confirmation codes
- Target-driven: Paper length/figures compared against reference papers
- Auto-tracking: Experiments logged to SQLite with git auto-commits
- GitHub integration: Create repo and add link to paper automatically

## Step-by-Step Usage

### Step 1: Create a Project

```
You: "Create a new research project on efficient attention"

AI calls: create_project(name="efficient_attention", description="...")
AI calls: create_workflow(project_id="efficient_attention")
```

Projects are created in `~/research-projects/efficient_attention/` with full directory structure.

### Step 2: Gather Papers & Set Targets

```
You: "Find papers on linear attention transformers"

AI calls: fetch_hf_trending(topic="linear attention transformers", max_results=10)
AI calls: search_papers(query="efficient attention mechanisms", max_results=10)
```

**New:** Set target metrics from reference papers:
```
You: "Use these papers as reference for paper length"

AI calls: set_target_metrics_from_papers(arxiv_ids=["2312.xxxxx", "2401.xxxxx"])

Response: {
  "target_metrics": {
    "word_count": 5200,
    "figure_count": 7,
    "table_count": 3
  }
}
```

### Step 3: Generate Ideas

```
You: "Generate research ideas from these papers"

AI calls: generate_ideas(paper_ids=[...], count=3, focus="efficiency")
```

The AI receives paper context and generates ideas creatively, then submits each:

```
AI calls: submit_idea(title="...", description="...", motivation="...")

Response: {
  "idea_id": "idea_abc123",
  "novelty_score": 0.85,
  "status": "AWAITING_USER_APPROVAL",
  "approval_command": "APPROVE idea_abc123 CODE 7382"
}
```

### Step 4: YOU Approve an Idea

**This is the critical step.** The AI cannot proceed without your approval.

```
You: "APPROVE idea_abc123 CODE 7382"

AI calls: approve_idea(idea_id="idea_abc123", confirmation_code="7382")

Response: "✅ Idea approved. Workflow unlocked."
```

### Step 5: Check What's Next

```
You: "What should we do next?"

AI calls: get_next_action()

Response: {
  "status": "IN_PROGRESS",
  "stage": "experiment_setup",
  "next_action": {"tool": "create_experiment_env", "description": "Create Python environment"}
}
```

### Step 6: Setup Environment & Run Experiments

```
You: "Setup the experiment environment"

AI calls: create_experiment_env(name="exp_env", python="3.11")
AI calls: install_dependencies(env_name="exp_env", requirements=["torch", "numpy"])
AI calls: define_hypotheses(idea_id="idea_abc123")
```

**New:** Experiments are auto-tracked and logged:
```
You: "Run the experiments"

AI calls: run_experiment(script="train.py", name="main_run")
# Auto-logs to experiments/metrics.db and auto-commits to git

AI calls: log_experiment(project_dir="...", name="main_run", config={...}, metrics={...})
AI calls: collect_metrics(experiments=["main_run"])
```

### Step 7: Analyze Results & Generate Figures

```
You: "Analyze results and create visualizations"

AI calls: get_experiment_history(project_dir="...")  # Get all tracked runs
AI calls: compare_to_baselines(method="ours", baselines=["baseline1", "baseline2"], results={...})
AI calls: plot_comparison_bar(results={...}, metric="accuracy")
AI calls: plot_training_curves(experiments=["main_run"], metrics=["loss"])
AI calls: check_significance(method1="ours", method2="baseline", results={...})
```

### Step 8: Write Paper & Check Completeness

**New:** Check paper against target metrics from reference papers:
```
You: "Write the paper for NeurIPS"

AI calls: get_citations_for_topic(topic="attention mechanisms")
AI calls: format_results_table(results={...})
AI calls: check_paper_completeness(paper_content={...}, target_word_count=5200)

Response: {
  "status": "NEEDS_EXPANSION",
  "current": {"words": 3200, "figures": 4},
  "target": {"words": 5200, "figures": 7},
  "suggestions": ["Add 3 more figures", "Need ~2000 more words"]
}
```

If paper is too short, run more experiments:
```
AI calls: expand_paper()  # Get suggestions for experiments to add
AI calls: run_ablation(script="train.py", ablation_params={"learning_rate": [0.001, 0.01]})
```

### Step 9: Format & Publish to GitHub

**New:** Create GitHub repo and add link to paper:
```
You: "Format for NeurIPS and publish to GitHub"

AI calls: create_github_repo(name="efficient-attention", private=True)

Response: {
  "repo_url": "https://github.com/user/efficient-attention"
}

AI calls: finalize_paper_with_github(latex_file="paper.tex")
# Adds \footnote{Code: \url{https://github.com/...}} to paper

AI calls: cast_to_format(conference="neurips", paper_content={...})
AI calls: compile_paper(tex_file="output/paper_neurips.tex")
```

## Workflow Enforcement

The system prevents skipping steps:

| If you try... | Without... | You get... |
|---------------|------------|------------|
| `run_experiment` | Approved idea | `BLOCKED: Must approve idea first` |
| `cast_to_format` | Generated figures | `BLOCKED: Generate figures first` |
| `cast_to_format` | GitHub linked | Workflow suggests `create_github_repo` first |
| `approve_idea` | Confirmation code | `ERROR: Code required` |
| `check_paper_completeness` | Target metrics | Uses default (5000 words, 6 figures) |

### Workflow Orchestrator

Always use `get_next_action()` to see what to do next:

```
AI calls: get_next_action()

Response: {
  "status": "IN_PROGRESS",
  "stage": "analysis",
  "progress": {
    "completed": ["create_experiment_env", "run_experiment"],
    "remaining": ["plot_comparison_bar", "compare_to_baselines"]
  },
  "next_action": {"tool": "plot_comparison_bar", "description": "Create comparison chart"}
}
```

## Idea Approval System

Ideas require a **confirmation code** that only you can see:

1. `generate_ideas()` returns ideas with codes: `APPROVE idea_x CODE 1234`
2. AI is instructed to STOP and wait
3. You type the approval command in chat
4. AI calls `approve_idea(idea_id, confirmation_code)`
5. Workflow unlocks for experiments

This prevents the AI from auto-approving low-quality ideas.

## Complete Tool Reference (80 tools)

### Paper Aggregation (6 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `fetch_arxiv_trending` | Fetch by arXiv category | `fetch_arxiv_trending(category="cs.LG", max_results=20)` |
| `fetch_hf_trending` | HuggingFace with topic search | `fetch_hf_trending(topic="gradient boosting", max_results=10)` |
| `search_papers` | Semantic search with relevance | `search_papers(query="attention", min_relevance=0.3)` |
| `get_paper_details` | Full paper metadata | `get_paper_details(paper_id="2409.07146")` |
| `clone_paper_code` | Clone GitHub repo | `clone_paper_code(paper_id="2409.07146")` |
| `extract_paper_context` | Extract structure/style | `extract_paper_context(arxiv_id="2409.07146")` |

### Project & Workflow (6 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `create_project` | Create project structure | `create_project(name="my_research")` |
| `list_projects` | List all projects | `list_projects()` |
| `set_current_project` | Set active project | `set_current_project(project_id="my_research")` |
| `create_workflow` | Create workflow tracker | `create_workflow(project_id="my_research")` |
| `get_next_action` | **Get next required step** | `get_next_action()` |
| `get_workflow_checklist` | Full workflow status | `get_workflow_checklist()` |

### Paper Metrics (2 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `extract_paper_metrics` | Get word/figure counts from paper | `extract_paper_metrics(arxiv_id="2312.xxxxx")` |
| `set_target_metrics_from_papers` | Average metrics as targets | `set_target_metrics_from_papers(arxiv_ids=[...])` |

### Idea Generation (8 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `generate_ideas` | Get paper context for ideas | `generate_ideas(paper_ids=[...], count=3)` |
| `submit_idea` | Submit a generated idea | `submit_idea(title="...", description="...", motivation="...")` |
| `approve_idea` | **Requires confirmation code** | `approve_idea(idea_id="...", confirmation_code="1234")` |
| `reject_idea` | Reject with feedback | `reject_idea(idea_id="...", reason="...")` |
| `list_ideas` | List ideas by status | `list_ideas(status="approved")` |
| `check_novelty` | Check against literature | `check_novelty(idea="...")` |
| `define_hypotheses` | Create testable hypotheses | `define_hypotheses(idea_id="...")` |
| `create_research_plan` | Create phased plan | `create_research_plan(idea_id="...")` |

### Environment Setup (6 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `create_experiment_env` | Create venv/conda | `create_experiment_env(name="exp", python="3.11")` |
| `install_dependencies` | Install packages | `install_dependencies(env_name="exp", requirements=["torch"])` |
| `setup_docker` | Generate Dockerfile | `setup_docker(requirements_file="requirements.txt")` |
| `check_gpu_availability` | Check GPU resources | `check_gpu_availability()` |
| `clone_baseline_repos` | Clone baselines | `clone_baseline_repos(paper_ids=[...])` |
| `setup_datasets` | Prepare datasets | `setup_datasets(datasets=["mnist", "cifar10"])` |

### Experiment Execution (6 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `run_experiment` | Run with logging | `run_experiment(script="train.py", name="run1")` |
| `run_baseline` | Run baseline method | `run_baseline(baseline_dir="baselines/method1")` |
| `run_ablation` | Run ablation study | `run_ablation(script="train.py", ablation_params={...})` |
| `monitor_training` | Monitor progress | `monitor_training(experiment_name="run1")` |
| `save_checkpoint` | Save checkpoint | `save_checkpoint(experiment_name="run1")` |
| `resume_experiment` | Resume from checkpoint | `resume_experiment(experiment_name="run1", checkpoint="...")` |

### Experiment Tracking (2 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `log_experiment` | Log run to SQLite + JSON | `log_experiment(project_dir="...", name="run1", config={...}, metrics={...})` |
| `get_experiment_history` | Get all tracked runs | `get_experiment_history(project_dir="...")` |

### Data Collection (6 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `collect_metrics` | Collect from logs | `collect_metrics(experiments=["run1", "run2"])` |
| `parse_tensorboard` | Parse TB logs | `parse_tensorboard(log_dir="logs/")` |
| `parse_wandb` | Parse W&B runs | `parse_wandb(project="my_project")` |
| `aggregate_results` | Aggregate across runs | `aggregate_results(experiments=[...])` |
| `export_to_csv` | Export to CSV | `export_to_csv(results={...}, output_path="results.csv")` |
| `compute_statistics` | Compute mean, std, CI | `compute_statistics(results={...})` |

### Visualization (8 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `plot_training_curves` | **Required for papers** | `plot_training_curves(experiments=[...], metrics=["loss"])` |
| `plot_comparison_bar` | **Required for papers** | `plot_comparison_bar(results={...}, metric="accuracy")` |
| `plot_ablation_table` | LaTeX ablation table | `plot_ablation_table(results={...})` |
| `plot_scatter` | Scatter with regression | `plot_scatter(x_data=[...], y_data=[...])` |
| `plot_heatmap` | Heatmap visualization | `plot_heatmap(data=[[...]])` |
| `plot_qualitative` | Side-by-side images | `plot_qualitative(images=[...])` |
| `generate_architecture_diagram` | Model diagram | `generate_architecture_diagram(components=[...])` |
| `style_for_conference` | Apply conference style | `style_for_conference(figure_path="...", conference="neurips")` |

### Results Verification (5 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `verify_hypothesis` | Statistical test | `verify_hypothesis(hypothesis="...", results={...})` |
| `check_significance` | Significance test | `check_significance(method1="ours", method2="baseline", results={...})` |
| `detect_anomalies` | Find outliers | `detect_anomalies(results={...})` |
| `compare_to_baselines` | **Required for papers** | `compare_to_baselines(method="ours", baselines=[...], results={...})` |
| `generate_results_summary` | Summary report | `generate_results_summary(experiments=[...])` |

### Paper Writing (13 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `estimate_paper_structure` | Word/figure estimates | `estimate_paper_structure(conference="neurips")` |
| `format_results_table` | LaTeX results table | `format_results_table(results={...})` |
| `format_ablation_table` | LaTeX ablation table | `format_ablation_table(ablations={...})` |
| `get_citations_for_topic` | Get BibTeX citations | `get_citations_for_topic(topic="attention")` |
| `format_figure` | Figure LaTeX | `format_figure(figure_path="...", caption="...")` |
| `format_algorithm` | Algorithm LaTeX | `format_algorithm(steps=[...], caption="...")` |
| `format_equation` | Equation LaTeX | `format_equation(equation="...", label="...")` |
| `create_paper_skeleton` | Paper structure | `create_paper_skeleton(title="...", conference="neurips")` |
| `get_paper_context` | Get writing context | `get_paper_context(paper_ids=[...])` |
| `validate_latex` | Validate syntax | `validate_latex(latex_content="...")` |
| `save_to_file` | Save content | `save_to_file(content="...", filename="...")` |
| `check_paper_completeness` | **Compare to target metrics** | `check_paper_completeness(paper_content={...})` |
| `expand_paper` | Get expansion suggestions | `expand_paper()` |

### Conference Formatting (8 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `list_conferences` | List all conferences | `list_conferences()` |
| `get_conference_requirements` | Format requirements | `get_conference_requirements(conference="neurips")` |
| `cast_to_format` | Convert to format | `cast_to_format(conference="neurips", paper_content={...})` |
| `generate_poster` | Generate poster | `generate_poster(conference="neurips", paper_content={...})` |
| `generate_supplementary` | Generate supplementary | `generate_supplementary(include_code=true)` |
| `compile_paper` | Compile to PDF | `compile_paper(tex_file="paper.tex")` |
| `create_github_repo` | **Create GitHub repo** | `create_github_repo(name="my-research", private=True)` |
| `finalize_paper_with_github` | Add GitHub link to paper | `finalize_paper_with_github(latex_file="paper.tex")` |

## Supported Conferences

| Conference | Pages | Columns | Abstract | Style |
|------------|-------|---------|----------|-------|
| NeurIPS | 9 | 1 | 250 words | Numeric |
| ICML | 9 | 2 | 200 words | Author-year |
| ICLR | 9 | 1 | 250 words | Author-year |
| CVPR | 8 | 2 | 300 words | Numeric |
| ICCV | 8 | 2 | 300 words | Numeric |
| ECCV | 14 | 1 | 300 words | Numeric |
| ACL | 8 | 2 | 200 words | Author-year |

## LaTeX Setup

```bash
# macOS
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install environ units multirow algorithms algorithm2e lastpage

# Ubuntu/Debian
sudo apt install texlive-latex-extra texlive-fonts-recommended
```

## Troubleshooting

### "BLOCKED: Must approve idea first"
You need to type the approval command shown after `generate_ideas()`:
```
APPROVE idea_xyz CODE 1234
```

### arXiv/Semantic Scholar returns 403
Rate limiting is active. Wait a few seconds between requests.

### PDF compilation fails with missing package
```bash
sudo tlmgr install <package_name>
```

### AI keeps trying to auto-approve ideas
The confirmation code system should prevent this. If it happens, the `approve_idea` call will fail without the correct code.

## Key Features

### Target-Driven Paper Length

Reference papers are analyzed to set word/figure targets:

```
AI calls: set_target_metrics_from_papers(arxiv_ids=["2312.xxxxx", "2401.xxxxx"])
# Averages: 5200 words, 7 figures, 3 tables

AI calls: check_paper_completeness(paper_content={...})
# Returns: NEEDS_EXPANSION if paper is too short
```

### Experiment Tracking

All experiments logged to SQLite with auto-commit to git:

```python
# Stored in ~/research-projects/<project>/experiments/metrics.db
# JSON logs in experiments/logs/<run_id>/
# Auto-commits after each run
```

### Expansion Loop

If paper is too short compared to targets, the workflow suggests:
- More experiments to run
- Additional figures to generate
- Analysis sections to expand

### GitHub Integration

```
AI calls: create_github_repo(name="my-research", private=True)
AI calls: finalize_paper_with_github(latex_file="paper.tex")
# Adds: \footnote{Code: \url{https://github.com/user/my-research}}
```

## Design Philosophy

1. **Human approval required**: Ideas need explicit approval with confirmation codes
2. **Workflow enforcement**: Can't skip steps (no experiments without approved idea)
3. **Target-driven writing**: Paper length compared against reference papers
4. **Auto-tracking**: Experiments logged to SQLite + git auto-commits
5. **GitHub integration**: Code repo linked in paper automatically
6. **Tools, not content**: MCP provides utilities; LLM generates actual content
7. **Project isolation**: Each project in `~/research-projects/<name>/`

## License

MIT
