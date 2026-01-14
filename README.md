# ResearchMCP

End-to-end AI research pipeline MCP server for Cursor. Enables vibe-coding style research workflow from idea generation to paper publication.

## Quick Start (Local)

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

---

## Remote Machine / Server Setup

For remote machines (servers, VMs, cloud instances), use the automated setup script to install all system dependencies including LaTeX and PDF tools.

### One-Line Setup

```bash
# Clone and run full setup
git clone <your-repo>/research-mcp.git
cd research-mcp
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

### Manual Setup

#### Ubuntu/Debian

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    git python3 python3-pip python3-venv \
    texlive-full \
    poppler-utils ghostscript

# Or minimal LaTeX (smaller download)
sudo apt-get install -y \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    texlive-publishers \
    latexmk biber

# Python setup
cd research-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### RHEL/CentOS/Fedora

```bash
# System dependencies
sudo dnf install -y \
    git python3 python3-pip \
    texlive-scheme-full \
    poppler-utils ghostscript

# Python setup
cd research-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# System dependencies
brew install git python@3.11
brew install --cask basictex
brew install poppler

# LaTeX packages
sudo tlmgr update --self
sudo tlmgr install \
    collection-fontsrecommended \
    collection-latexrecommended \
    collection-latexextra \
    booktabs algorithms algorithmicx multirow microtype \
    xcolor hyperref natbib subcaption nicefrac latexmk

# Python setup
cd research-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verify Installation

```bash
# Run verification script
python3 verify_setup.py
```

This checks:
- Python 3.11+ installed
- pdflatex and bibtex available
- All required LaTeX packages present
- All Python dependencies installed
- Write permissions to project directories

### Required System Dependencies

| Component | Purpose | Install Command |
|-----------|---------|-----------------|
| Python 3.11+ | Runtime | `apt install python3` |
| Git | Version control | `apt install git` |
| pdflatex | LaTeX compilation | `apt install texlive-latex-base` |
| bibtex | Bibliography | `apt install texlive-bibtex-extra` |
| poppler-utils | PDF processing | `apt install poppler-utils` |

### Required LaTeX Packages

These are needed for conference paper templates:

| Package | Purpose |
|---------|---------|
| `booktabs` | Professional tables |
| `amsmath`, `amssymb`, `amsfonts` | Math typesetting |
| `algorithm`, `algorithmic` | Algorithm blocks |
| `graphicx` | Images |
| `xcolor` | Colors |
| `hyperref` | Links |
| `natbib` | Citations |
| `subcaption` | Subfigures |
| `microtype` | Microtypography |
| `multirow` | Table spanning |

Install missing packages:
```bash
# Find missing package
pdflatex paper.tex 2>&1 | grep "not found"

# Install it
sudo tlmgr install <package_name>
```

### Docker Setup (Alternative)

```dockerfile
FROM python:3.11-slim

# Install LaTeX and dependencies
RUN apt-get update && apt-get install -y \
    git \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    texlive-bibtex-extra \
    texlive-publishers \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python3", "run_server.py"]
```

```bash
docker build -t research-mcp .
docker run -v ~/.cursor:/root/.cursor research-mcp
```

---

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
    │   │   └── paper_ir.json  # Universal paper format
    │   └── final/
    ├── data/              # Datasets
    ├── worktrees/         # Git worktrees for parallel experiments
    ├── .git/              # Auto-initialized
    └── project.json
```

---

## Agent Instructions

This section provides explicit instructions for AI agents on how to correctly use the ResearchMCP workflow.

### Persona System

The server uses **dynamic tool filtering** based on workflow stage. You only see tools relevant to your current task:

| Persona | Active Stages | Tool Count | Purpose |
|---------|---------------|------------|---------|
| **Researcher** | context_gathering, idea_generation, idea_approval | ~21 | Paper discovery, idea generation |
| **Experimenter** | experiment_setup, experimenting, analysis | ~37 | Running and verifying experiments |
| **Writer** | writing, formatting, complete | ~31 | Paper writing and publication |

**Check your active persona:**
```
AI calls: get_active_persona()

Response: {
  "active_persona": "researcher",
  "workflow_stage": "context_gathering",
  "available_tools": ["fetch_hf_trending", "search_papers", ...]
}
```

**Switch persona manually if needed:**
```
AI calls: switch_persona(persona="experimenter")
```

### CRITICAL: HARD_BLOCK Before Planning

**DO NOT create a plan or proceed with experiments until an idea is approved.**

When you call `get_next_action()` during idea generation, you will receive a `HARD_BLOCK`:

```json
{
  "status": "HARD_BLOCK",
  "message": "STOP. Do NOT create a plan or proceed with any experiments yet.",
  "do_not_proceed": true,
  "do_not_create_plan": true,
  "ai_instruction": "Present ideas to user. Wait for: APPROVE <id> CODE <code>."
}
```

**Correct behavior:**
1. Generate ideas with `generate_ideas()` and `submit_idea()`
2. Present ideas to user with approval commands
3. **STOP and wait** for user to type `APPROVE <idea_id> CODE <code>`
4. Only AFTER approval, call `get_next_action()` again to proceed

### Mandatory Hypothesis Verification

**Every claim in your paper MUST be statistically verified before inclusion.**

```
AI calls: verify_and_record_hypothesis(
    hypothesis_id="h1_attention_faster",
    hypothesis_statement="Our attention mechanism is 2x faster than baseline",
    experiment_id="exp_001",
    results={"ours": [0.12, 0.11, 0.13], "baseline": [0.25, 0.24, 0.26]},
    test_type="t-test"
)

Response: {
  "can_include_in_paper": true,
  "p_value": 0.0012,
  "effect_size": 2.34
}
```

**Before formatting the paper:**
```
AI calls: check_claims_verified(claims=["h1_attention_faster", "h2_accuracy_same"])

Response: {
  "all_verified": true,
  "can_proceed": true
}
```

If unverified claims exist, the paper formatting will be blocked.

### Paper Expansion Loop

**Papers must meet target metrics. If too short, you MUST expand.**

```
AI calls: check_and_expand_paper()

Response: {
  "status": "NEEDS_EXPANSION",
  "iteration": 1,
  "completeness": {
    "current_words": 3200,
    "target_words": 5200,
    "current_figures": 4,
    "target_figures": 7
  },
  "expansion_suggestions": [
    {"type": "ablation_study", "estimated_words": 500, "estimated_figures": 1},
    {"type": "sensitivity", "estimated_words": 300, "estimated_figures": 2}
  ]
}
```

**Expansion loop procedure:**
1. Call `check_and_expand_paper()`
2. If `status: NEEDS_EXPANSION`:
   - Generate new hypotheses for suggested experiments
   - Run experiments (use git worktrees for parallel runs)
   - Verify results with `verify_and_record_hypothesis()`
   - Add verified results to paper
   - Call `check_and_expand_paper()` again
3. Repeat until `status: COMPLETE`

### Style-Consistent Writing

**Before writing each section, get style context from reference papers:**

```
AI calls: get_paper_context_for_writing(paper_ids=["arxiv:2312.xxxxx", "arxiv:2401.xxxxx"])

Response: {
  "target_metrics": {"word_count": 5200, "figure_count": 7},
  "style_guidance": {
    "avg_sentence_length": 22,
    "passive_voice_ratio": 0.35,
    "first_person_usage": true,
    "writing_instruction": "Write in academic style matching the reference papers..."
  },
  "example_paragraphs": ["Recent advances in attention mechanisms...", ...]
}
```

**Use this context to match the writing style of top papers in your field.**

### Git Worktrees for Parallel Experiments

Run multiple experiments simultaneously using git worktrees:

```
AI calls: create_worktree(experiment_id="ablation_lr")
# Creates: worktrees/ablation_lr/ with branch exp/ablation_lr

AI calls: commit_in_worktree(experiment_id="ablation_lr", message="Add LR ablation results")

# After verification:
AI calls: merge_worktree(experiment_id="ablation_lr", delete_after=true)
```

### Complete Agent Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AGENT WORKFLOW CHECKLIST                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  [ ] 1. Create project and workflow                                           │
│      create_project() → create_workflow()                                     │
│                                                                               │
│  [ ] 2. Gather papers and set targets                                         │
│      fetch_hf_trending_with_metrics() → set_target_metrics_from_papers()      │
│      get_paper_context_for_writing() → save style context                     │
│                                                                               │
│  [ ] 3. Generate and submit ideas                                             │
│      generate_ideas() → submit_idea() for each                                │
│                                                                               │
│  [ ] 4. STOP - Present ideas to user                                          │
│      ⚠️  DO NOT PROCEED until user types: APPROVE <id> CODE <code>            │
│      ⚠️  DO NOT CREATE A PLAN before approval                                 │
│                                                                               │
│  [ ] 5. After approval - setup experiments                                    │
│      define_hypotheses() → create_experiment_env() → setup_datasets()         │
│                                                                               │
│  [ ] 6. Run experiments with tracking                                         │
│      run_experiment() → log_experiment() → collect_metrics()                  │
│                                                                               │
│  [ ] 7. VERIFY every result (MANDATORY)                                       │
│      verify_and_record_hypothesis() for EACH claim                            │
│      get_verified_claims() → only these go in paper                           │
│                                                                               │
│  [ ] 8. Generate visualizations                                               │
│      plot_comparison_bar() → plot_training_curves() → save figures            │
│                                                                               │
│  [ ] 9. Write paper with style context                                        │
│      get_paper_context_for_writing() → write in same style                    │
│      INCLUDE ALL FIGURES - papers must have figures                           │
│                                                                               │
│  [ ] 10. Check completeness and expand                                        │
│      check_and_expand_paper() → if NEEDS_EXPANSION → run more experiments     │
│      LOOP until paper meets target metrics                                    │
│                                                                               │
│  [ ] 11. Verify all claims before formatting                                  │
│      check_claims_verified() → must pass before cast_to_format()              │
│                                                                               │
│  [ ] 12. Format and publish                                                   │
│      create_github_repo() → cast_to_format() → compile_paper()                │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

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
│  get_style_context   novelty)           ▲                                   │
│                                         │ HARD_BLOCK                        │
│                                         │ until user                        │
│                                         │ approves                          │
│                                                                              │
│  5. EXPERIMENTS     6. ANALYSIS        7. WRITING         8. FORMAT         │
│  ─────────────      ─────────────      ─────────────      ─────────────     │
│  run_experiment     compare_baselines  format_table       cast_to_format    │
│  log_experiment     VERIFY_HYPOTHESIS  check_completeness create_github_repo│
│  collect_metrics    plot_comparison    expand_paper       compile_paper     │
│  (git worktrees)    get_exp_history    (MUST ADD FIGS)    finalize_github   │
│                                                                              │
│                    ┌──────────────────────────────────────┐                 │
│                    │ EXPANSION LOOP (MANDATORY)           │                 │
│                    │ if paper too short:                  │                 │
│                    │ 1. Generate new hypotheses           │                 │
│                    │ 2. Run experiments (worktrees)       │                 │
│                    │ 3. VERIFY results                    │                 │
│                    │ 4. Add to paper with figures         │                 │
│                    │ 5. Check again → repeat if needed    │                 │
│                    └──────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Principles**:
- **HARD_BLOCK**: Cannot plan or experiment without approved idea
- **Mandatory verification**: Every claim requires `verify_and_record_hypothesis()`
- **Expansion loop**: Papers must meet target metrics - keep adding content
- **Style matching**: Write in the same style as reference papers
- **Figures required**: Papers MUST include figures - generate and add them

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

AI calls: fetch_hf_trending_with_metrics(topic="linear attention", max_results=10)
AI calls: search_papers(query="efficient attention mechanisms", max_results=10)
```

**Set target metrics and style context:**
```
AI calls: set_target_metrics_from_papers(arxiv_ids=["2312.xxxxx", "2401.xxxxx"])

Response: {
  "target_metrics": {
    "word_count": 5200,
    "figure_count": 7,
    "table_count": 3
  }
}

AI calls: get_paper_context_for_writing(paper_ids=["2312.xxxxx", "2401.xxxxx"])
# Save style context for later writing
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

**Run experiments with tracking:**
```
You: "Run the experiments"

AI calls: run_experiment(script="train.py", name="main_run")
AI calls: log_experiment(project_dir="...", name="main_run", config={...}, metrics={...})
AI calls: collect_metrics(experiments=["main_run"])
```

### Step 7: VERIFY Results (Mandatory)

```
You: "Verify the experiment results"

AI calls: verify_and_record_hypothesis(
    hypothesis_id="h1_faster",
    hypothesis_statement="Our method is faster than baseline",
    experiment_id="main_run",
    results={"ours": [...], "baseline": [...]},
    test_type="t-test"
)

Response: {
  "can_include_in_paper": true,
  "p_value": 0.001,
  "effect_size": 1.8
}
```

### Step 8: Generate Figures & Write Paper

```
You: "Create visualizations and write the paper"

AI calls: plot_comparison_bar(results={...}, metric="speed")
AI calls: plot_training_curves(experiments=["main_run"], metrics=["loss"])

# Get style context before writing
AI calls: get_paper_context_for_writing(paper_ids=[...])

# Write in matching style, INCLUDE FIGURES
```

### Step 9: Check Completeness & Expand

```
AI calls: check_and_expand_paper()

Response: {
  "status": "NEEDS_EXPANSION",
  "completeness": {"current_words": 3200, "target_words": 5200}
}
```

**If NEEDS_EXPANSION - run more experiments:**
```
AI calls: define_hypotheses(...)  # New hypotheses for ablations
AI calls: run_ablation(script="train.py", ablation_params={"lr": [0.001, 0.01]})
AI calls: verify_and_record_hypothesis(...)  # Verify new results
# Add to paper, check again
```

### Step 10: Format & Publish

```
You: "Format for NeurIPS and publish to GitHub"

AI calls: check_claims_verified(claims=["h1_faster", "h2_accurate"])
# Must pass before formatting

AI calls: create_github_repo(name="efficient-attention", private=True)
AI calls: cast_to_format(conference="neurips", paper_content={...})
AI calls: compile_paper(tex_file="output/paper_neurips.tex")
```

## Workflow Enforcement

The system prevents skipping steps:

| If you try... | Without... | You get... |
|---------------|------------|------------|
| Any experiment tool | Approved idea | `HARD_BLOCK: Must approve idea first` |
| `cast_to_format` | Verified claims | `BLOCKED: Unverified claims exist` |
| `cast_to_format` | Figures generated | `BLOCKED: Generate figures first` |
| Create a plan | Approved idea | `HARD_BLOCK: Do NOT create a plan yet` |
| `approve_idea` | Confirmation code | `ERROR: Code required` |

### Workflow Orchestrator

Always use `get_next_action()` to see what to do next:

```
AI calls: get_next_action()

Response: {
  "status": "IN_PROGRESS",
  "stage": "analysis",
  "progress": {
    "completed": ["create_experiment_env", "run_experiment"],
    "remaining": ["verify_and_record_hypothesis", "plot_comparison_bar"]
  },
  "next_action": {"tool": "verify_and_record_hypothesis", "description": "Verify experiment results"}
}
```

## Complete Tool Reference (90+ tools)

### Persona Management (2 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `get_active_persona` | Get current persona and tools | `get_active_persona()` |
| `switch_persona` | Switch to different persona | `switch_persona(persona="experimenter")` |

### Paper Aggregation (7 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `fetch_arxiv_trending` | Fetch by arXiv category | `fetch_arxiv_trending(category="cs.LG", max_results=20)` |
| `fetch_hf_trending` | HuggingFace with topic search | `fetch_hf_trending(topic="gradient boosting", max_results=10)` |
| `fetch_hf_trending_with_metrics` | **With paper metrics** | `fetch_hf_trending_with_metrics(topic="attention", max_results=10)` |
| `search_papers` | Semantic search with relevance | `search_papers(query="attention", min_relevance=0.3)` |
| `get_paper_details` | Full paper metadata | `get_paper_details(paper_id="2409.07146")` |
| `clone_paper_code` | Clone GitHub repo | `clone_paper_code(paper_id="2409.07146")` |
| `extract_paper_context` | Extract structure/style | `extract_paper_context(arxiv_id="2409.07146")` |

### Project & Workflow (8 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `create_project` | Create project structure | `create_project(name="my_research")` |
| `list_projects` | List all projects | `list_projects()` |
| `set_current_project` | Set active project | `set_current_project(project_id="my_research")` |
| `create_workflow` | Create workflow tracker | `create_workflow(project_id="my_research")` |
| `get_next_action` | **Get next required step** | `get_next_action()` |
| `get_workflow_checklist` | Full workflow status | `get_workflow_checklist()` |
| `check_and_expand_paper` | **Check and trigger expansion** | `check_and_expand_paper()` |
| `get_verified_claims` | Get verified claims only | `get_verified_claims()` |

### Paper Metrics & Style (4 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `extract_paper_metrics` | Get word/figure counts from paper | `extract_paper_metrics(arxiv_id="2312.xxxxx")` |
| `set_target_metrics_from_papers` | Average metrics as targets | `set_target_metrics_from_papers(arxiv_ids=[...])` |
| `get_writing_style_context` | **Extract writing style** | `get_writing_style_context(arxiv_ids=[...])` |
| `get_paper_context_for_writing` | **Full context for writing** | `get_paper_context_for_writing(paper_ids=[...])` |

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

### Results Verification (7 tools)

| Tool | Description | Example |
|------|-------------|---------|
| `verify_hypothesis` | Statistical test | `verify_hypothesis(hypothesis="...", results={...})` |
| `verify_and_record_hypothesis` | **MANDATORY: Verify and record** | `verify_and_record_hypothesis(hypothesis_id="...", ...)` |
| `check_claims_verified` | **Check all claims verified** | `check_claims_verified(claims=[...])` |
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
| EMNLP | 8 | 2 | 200 words | Author-year |
| AAAI | 7 | 2 | 150 words | Numeric |

## LaTeX Setup

See [Remote Machine / Server Setup](#remote-machine--server-setup) for complete instructions.

**Quick reference:**

```bash
# macOS (BasicTeX)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install booktabs algorithms algorithmicx multirow microtype \
    xcolor hyperref natbib subcaption nicefrac latexmk

# Ubuntu/Debian (full)
sudo apt install texlive-full

# Ubuntu/Debian (minimal)
sudo apt install texlive-latex-base texlive-latex-extra texlive-science \
    texlive-fonts-recommended texlive-bibtex-extra latexmk

# Verify installation
python3 verify_setup.py
```

## Troubleshooting

### "HARD_BLOCK: Do NOT create a plan yet"
You tried to proceed before the user approved an idea. Present the ideas and wait for:
```
APPROVE idea_xyz CODE 1234
```

### "BLOCKED: Unverified claims exist"
You tried to format a paper with claims that weren't verified. Call:
```
verify_and_record_hypothesis(hypothesis_id="...", ...)
```
for each claim before `cast_to_format()`.

### "NEEDS_EXPANSION" after writing
Your paper is too short. Run more experiments, verify results, add figures, and call `check_and_expand_paper()` again.

### arXiv/Semantic Scholar returns 403
Rate limiting is active. Wait a few seconds between requests.

### PDF compilation fails with missing package
```bash
sudo tlmgr install <package_name>
```

## Design Philosophy

1. **HARD_BLOCK before planning**: Cannot create a plan without approved idea
2. **Mandatory verification**: Every claim requires statistical verification
3. **Expansion loop**: Papers must meet target metrics - keep adding content
4. **Style matching**: Write in the same style as reference papers
5. **Figures required**: Papers MUST include figures
6. **Persona-based tools**: Only see relevant tools for current stage
7. **Human approval required**: Ideas need explicit approval with confirmation codes
8. **Auto-tracking**: Experiments logged to SQLite + git auto-commits
9. **GitHub integration**: Code repo linked in paper automatically
10. **Project isolation**: Each project in `~/research-projects/<name>/`

## License

MIT
