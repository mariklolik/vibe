# ResearchMCP

End-to-end AI research pipeline MCP server for Cursor. Enables vibe-coding style research workflow from idea generation to paper publication.

## Overview

ResearchMCP provides 67+ tools for the complete research workflow:

```
Papers → Ideas → Approval → Experiments → Results → Paper → PDF
```

**Key Features:**
- 🔍 **Smart Paper Discovery**: Search HuggingFace trending with topic queries, arXiv by category
- 💡 **Idea Generation with Approval**: Generate ideas ranked by novelty, requiring user approval before experiments
- 🔬 **Experiment Management**: Full lifecycle with Git tracking, checkpoints, and environment isolation
- 📊 **Publication-Ready Output**: Generate figures, tables, and full LaTeX papers
- 📝 **A* Conference Support**: Auto-format for NeurIPS, ICML, ICLR, CVPR, ACL, and 10+ more
- 📄 **PDF Compilation**: Direct LaTeX to PDF compilation with automatic package management

## Installation

```bash
cd research-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### LaTeX Setup (for PDF compilation)

```bash
# macOS
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install environ units multirow algorithms algorithm2e

# Ubuntu/Debian
sudo apt install texlive-latex-extra texlive-fonts-recommended
```

## Configuration for Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "research-mcp": {
      "command": "python3",
      "args": ["run_server.py"],
      "cwd": "/path/to/research-mcp"
    }
  }
}
```

## Complete Workflow Example

### 1. Discover Trending Papers

```
"Find gradient boosting papers on HuggingFace"
→ fetch_hf_trending(topic="gradient boosting", max_results=10)

Results: GBRL, GrowNet, FairGBM, NGBoost, GRANDE...
```

The `topic` parameter enables smart search on HuggingFace's trending page, returning only relevant papers.

```
"Fetch trending ML papers from arXiv"
→ fetch_arxiv_trending(category="cs.LG", days=7, max_results=20)

"Search for papers on attention mechanisms"
→ search_papers(query="attention mechanisms transformers", min_relevance=0.3)
```

### 2. Extract Paper Context

```
"Extract structure from this paper"
→ extract_paper_context(arxiv_id="2502.14678")

Returns:
- Section structure and word counts
- Figure/table analysis
- Citation patterns
- Writing style metrics
```

### 3. Create a Research Project

```
"Create a new project for MLX boosting research"
→ create_project(name="mlx_boosting", description="Gradient boosting on Apple Silicon")

Creates:
projects/mlx_boosting/
├── context/      # Extracted paper contexts
├── ideas/        # Generated and approved ideas
├── experiments/  # Experiment runs and logs
├── papers/       # Generated LaTeX files
├── data/         # Datasets
└── figures/      # Generated plots
```

### 4. Generate Research Ideas (with Approval)

```
"Generate ideas based on these papers"
→ generate_ideas(paper_ids=["arxiv:2407.08250", "arxiv:2209.07850"], count=3, focus="efficiency")

Returns ranked ideas with novelty scores:
1. ⭐⭐⭐⭐☆ (0.8) "Unified Framework for..." - RECOMMENDED
2. ⭐⭐⭐☆☆ (0.6) "Efficient Variant..."
3. ⭐⭐☆☆☆ (0.4) "Extension to..."

ACTION REQUIRED: Call approve_idea(idea_id) to proceed
```

**Ideas require explicit approval before experiments can run:**

```
"Approve this idea"
→ approve_idea(idea_id="idea_abc123", user_feedback="Proceed with MLX focus")

"Reject this idea"
→ reject_idea(idea_id="idea_xyz789", reason="Too similar to existing work")
```

### 5. Define Hypotheses and Plan

```
"Define testable hypotheses"
→ define_hypotheses(idea_id="idea_abc123")

Returns:
- H1: Primary performance hypothesis
- H2: Ablation hypothesis
- H3: Efficiency hypothesis
- H4: Generalization hypothesis

"Create research plan"
→ create_research_plan(idea_id="idea_abc123")

Returns:
- Phase 1: Literature (1-2 weeks)
- Phase 2: Methodology (2-3 weeks)
- Phase 3: Experiments (3-4 weeks)
- Phase 4: Analysis (1-2 weeks)
- Phase 5: Writing (2 weeks)
```

### 6. Setup Experiment Environment

```
"Create a venv for experiments"
→ create_experiment_env(name="mlx_exp", python="3.11", use_conda=false)

"Install dependencies"
→ install_dependencies(env_name="mlx_exp", requirements=["mlx", "numpy", "pandas"])

"Check GPU availability"
→ check_gpu_availability()

"Setup datasets"
→ setup_datasets(datasets=["california_housing", "iris"])
```

### 7. Run Experiments

```
"Run the main experiment"
→ run_experiment(script="train.py", config="config.yaml", name="main_run")

"Monitor training"
→ monitor_training(experiment_name="main_run")

"Save checkpoint"
→ save_checkpoint(experiment_name="main_run", checkpoint_name="best_model")

"Run ablation study"
→ run_ablation(script="train.py", ablation_params={"learning_rate": [0.01, 0.001], "depth": [3, 5, 7]})
```

### 8. Collect and Analyze Results

```
"Collect metrics from experiments"
→ collect_metrics(experiments=["baseline", "ours", "ablation"])

"Compute statistics"
→ compute_statistics(results={"ours": [0.92, 0.91, 0.93], "baseline": [0.85, 0.84, 0.86]})

Returns:
- Mean, std, standard error
- 95% confidence intervals

"Check statistical significance"
→ check_significance(method1="ours", method2="baseline", results=..., test="t-test")

Returns:
- t-statistic, p-value
- Effect size (Cohen's d)
- Significance decision
```

### 9. Generate Visualizations

```
"Plot training curves"
→ plot_training_curves(experiments=["exp1", "exp2"], metrics=["loss", "accuracy"])

"Create comparison bar chart"
→ plot_comparison_bar(results={"sklearn": 0.85, "xgboost": 0.92, "ours": 0.94}, metric="R² Score")

"Generate LaTeX results table"
→ format_results_table(results={"Method A": {"Acc": 0.92, "F1": 0.91}, "Method B": {"Acc": 0.89, "F1": 0.88}})
```

### 10. Write the Paper

```
"Estimate paper structure for 9 pages"
→ estimate_paper_structure(conference="neurips", target_pages=9)

Returns:
- Total words: 4950
- Introduction: 742 words (15%)
- Method: 1386 words (28%)
- Experiments: 1584 words (32%)
- Recommended: 8 figures, 4 tables, 57 citations

"Create paper skeleton"
→ create_paper_skeleton(title="MLX-Boost: Efficient Gradient Boosting on Apple Silicon", conference="neurips")

"Format equation"
→ format_equation(equation="\\nabla L(y, f(x)) = -2(y - f(x))", label="eq:gradient")

"Format algorithm"
→ format_algorithm(steps=["Initialize f_0", "For m=1 to M:", "  Compute residuals", "  Fit tree", "  Update"], caption="Gradient Boosting")

"Get citations for topic"
→ get_citations_for_topic(topic="gradient boosting", max_citations=10)
```

### 11. Format for Conference Submission

```
"List supported conferences"
→ list_conferences()

"Get NeurIPS requirements"
→ get_conference_requirements(conference="neurips")

"Convert to NeurIPS format"
→ cast_to_format(conference="neurips", paper_content={...})

"Generate poster"
→ generate_poster(conference="neurips", paper_content={...})

"Generate supplementary materials"
→ generate_supplementary(include_code=true, include_data=false)
```

### 12. Compile to PDF

```
"Compile the paper"
→ compile_paper(tex_file="projects/my_project/papers/main.tex")

Returns:
- PDF path: /path/to/paper.pdf
- Compilation logs
- Missing package suggestions if any
```

## Supported Conferences

| Conference | Page Limit | Columns | Abstract Limit | Citation Style |
|------------|------------|---------|----------------|----------------|
| NeurIPS    | 9          | 1       | 250 words      | Numeric        |
| ICML       | 9          | 2       | 200 words      | Author-year    |
| ICLR       | 9          | 1       | 250 words      | Author-year    |
| CVPR       | 8          | 2       | 300 words      | Numeric        |
| ICCV       | 8          | 2       | 300 words      | Numeric        |
| ECCV       | 14         | 1       | 300 words      | Numeric        |
| ACL        | 8          | 2       | 200 words      | Author-year    |
| EMNLP      | 9          | 2       | 200 words      | Author-year    |
| AAAI       | 7          | 2       | 150 words      | Author-year    |

## Complete Tool Reference

### Paper Aggregation (6 tools)
| Tool | Description |
|------|-------------|
| `fetch_arxiv_trending` | Fetch trending papers by arXiv category |
| `fetch_hf_trending` | Fetch HuggingFace papers with optional topic search |
| `search_papers` | Semantic search with relevance scoring |
| `get_paper_details` | Get full paper metadata |
| `clone_paper_code` | Clone paper's GitHub repository |
| `extract_paper_context` | Extract structure and style from arXiv paper |

### Project Management (4 tools)
| Tool | Description |
|------|-------------|
| `create_project` | Create project with directory structure |
| `list_projects` | List all research projects |
| `get_current_project` | Get active project |
| `set_current_project` | Set active project |

### Workflow Tracking (2 tools)
| Tool | Description |
|------|-------------|
| `create_workflow` | Create workflow for a project |
| `get_workflow_status` | Get workflow progress |

### Idea Generation (7 tools)
| Tool | Description |
|------|-------------|
| `generate_ideas` | Generate ideas with novelty ranking |
| `approve_idea` | Approve idea for experiments |
| `reject_idea` | Reject idea with feedback |
| `list_ideas` | List all ideas by status |
| `check_novelty` | Check idea against literature |
| `create_research_plan` | Create structured plan |
| `define_hypotheses` | Define testable hypotheses |

### Environment Setup (6 tools)
| Tool | Description |
|------|-------------|
| `create_experiment_env` | Create conda/venv environment |
| `install_dependencies` | Install packages |
| `setup_docker` | Generate Dockerfile |
| `check_gpu_availability` | Check GPU resources |
| `clone_baseline_repos` | Clone baseline code |
| `setup_datasets` | Download and prepare datasets |

### Experiment Execution (6 tools)
| Tool | Description |
|------|-------------|
| `run_experiment` | Run experiment with logging |
| `run_baseline` | Run baseline method |
| `run_ablation` | Run ablation study |
| `monitor_training` | Monitor experiment progress |
| `save_checkpoint` | Save experiment checkpoint |
| `resume_experiment` | Resume from checkpoint |

### Data Collection (6 tools)
| Tool | Description |
|------|-------------|
| `collect_metrics` | Collect from log files |
| `parse_tensorboard` | Parse TensorBoard logs |
| `parse_wandb` | Parse W&B runs |
| `aggregate_results` | Aggregate across runs |
| `export_to_csv` | Export results to CSV |
| `compute_statistics` | Compute mean, std, CI |

### Visualization (8 tools)
| Tool | Description |
|------|-------------|
| `plot_training_curves` | Loss/accuracy curves |
| `plot_comparison_bar` | Method comparison bars |
| `plot_ablation_table` | LaTeX ablation table |
| `plot_scatter` | Scatter plot with regression |
| `plot_heatmap` | Heatmap visualization |
| `plot_qualitative` | Side-by-side images |
| `generate_architecture_diagram` | Model architecture |
| `style_for_conference` | Apply conference styling |

### Results Verification (5 tools)
| Tool | Description |
|------|-------------|
| `verify_hypothesis` | Statistical hypothesis test |
| `check_significance` | Significance between methods |
| `detect_anomalies` | Detect result anomalies |
| `compare_to_baselines` | Compare to all baselines |
| `generate_results_summary` | Summary of all experiments |

### Paper Writing (11 tools)
| Tool | Description |
|------|-------------|
| `estimate_paper_structure` | Estimate words, figures, citations |
| `format_results_table` | Generate LaTeX results table |
| `format_ablation_table` | Generate LaTeX ablation table |
| `get_citations_for_topic` | Get BibTeX citations |
| `format_figure` | Generate figure LaTeX |
| `format_algorithm` | Generate algorithm LaTeX |
| `format_equation` | Generate equation LaTeX |
| `create_paper_skeleton` | Create paper structure |
| `get_paper_context` | Get context for writing |
| `validate_latex` | Validate LaTeX syntax |
| `save_to_file` | Save content to file |

### Conference Formatting (6 tools)
| Tool | Description |
|------|-------------|
| `list_conferences` | List all supported conferences |
| `get_conference_requirements` | Get format requirements |
| `cast_to_format` | Convert to conference format |
| `generate_poster` | Generate poster LaTeX |
| `generate_supplementary` | Generate supplementary |
| `compile_paper` | Compile LaTeX to PDF |

## Project Structure

```
research-mcp/
├── src/
│   ├── server.py              # MCP server (67 tools)
│   ├── tools/
│   │   ├── aggregation.py     # Paper fetching + search
│   │   ├── ideas.py           # Idea generation + approval
│   │   ├── environment.py     # Environment setup
│   │   ├── experiments.py     # Experiment execution
│   │   ├── data_collection.py # Metrics collection
│   │   ├── visualization.py   # Plotting
│   │   ├── verification.py    # Statistical tests
│   │   ├── writing.py         # LaTeX formatting utilities
│   │   └── formatting.py      # Conference formatting + PDF
│   ├── apis/
│   │   ├── arxiv.py           # arXiv API (with rate limiting)
│   │   ├── huggingface.py     # HF API (with topic search)
│   │   └── semantic_scholar.py # S2 API (with fallback)
│   ├── context/
│   │   ├── extractor.py       # Paper structure extraction
│   │   ├── profiles.py        # Context profile storage
│   │   └── styles.py          # Conference style guidelines
│   ├── project/
│   │   ├── manager.py         # Project directory management
│   │   └── git_ops.py         # Git integration
│   └── db/
│       ├── papers_cache.py    # Paper caching (SQLite + FTS5)
│       ├── experiments_db.py  # Experiment + ideas storage
│       ├── conferences.py     # Conference requirements
│       └── workflow.py        # Workflow state persistence
├── templates/
│   └── neurips/               # Official NeurIPS templates
├── data/
│   ├── conferences.json       # Conference metadata
│   └── figure_styles.json     # Figure styling rules
├── projects/                  # Created research projects
├── output/                    # Generated papers
├── run_server.py              # Server launcher
├── requirements.txt
└── pyproject.toml
```

## Design Philosophy

This MCP follows **vibe-coding** principles:

1. **Tools, not content**: The MCP provides formatting utilities. The LLM in Cursor chat generates the actual paper content.

2. **User approval required**: Ideas must be explicitly approved before experiments can run, ensuring human oversight.

3. **Workflow enforcement**: Cannot skip steps (e.g., can't run experiments without an approved idea).

4. **Context gathering**: Extract and save paper contexts to inform better writing.

5. **Project isolation**: Each research project has its own directory structure to prevent overwriting.

## Troubleshooting

### arXiv returns 403
Rate limiting is enabled (3s delay between requests). If issues persist, wait a few minutes.

### Semantic Scholar returns 403
Graceful fallback is implemented. Results will come from arXiv and HuggingFace instead.

### PDF compilation fails
Run the missing package install command shown in the error:
```bash
sudo tlmgr install <package_name>
```

### HuggingFace topic search returns irrelevant papers
The topic search parses the embedded JSON from `huggingface.co/papers/trending?q=<query>`. If no results, try different keywords.

## License

MIT
