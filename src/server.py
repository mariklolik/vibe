"""ResearchMCP Server - End-to-end AI research pipeline."""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Data collection and aggregation
from src.tools.aggregation import (
    fetch_arxiv_trending,
    fetch_hf_trending,
    search_papers,
    get_paper_details,
    clone_paper_code,
)

# Ideas generation and verification
from src.tools.ideas import (
    generate_ideas,
    submit_idea,
    approve_idea,
    reject_idea,
    check_novelty,
    create_research_plan,
    define_hypotheses,
    list_ideas,
)

# Environment setup
from src.tools.environment import (
    create_experiment_env,
    install_dependencies,
    setup_docker,
    check_gpu_availability,
    clone_baseline_repos,
    setup_datasets,
)

# Experiments
from src.tools.experiments import (
    run_experiment,
    run_baseline,
    run_ablation,
    monitor_training,
    save_checkpoint,
    resume_experiment,
)

# Data collection
from src.tools.data_collection import (
    collect_metrics,
    parse_tensorboard,
    parse_wandb,
    aggregate_results,
    export_to_csv,
    compute_statistics,
)

# Visualization
from src.tools.visualization import (
    plot_training_curves,
    plot_comparison_bar,
    plot_ablation_table,
    plot_scatter,
    plot_heatmap,
    plot_qualitative,
    generate_architecture_diagram,
    style_for_conference,
)

# Verification
from src.tools.verification import (
    verify_hypothesis,
    check_significance,
    detect_anomalies,
    compare_to_baselines,
    generate_results_summary,
)

# Writing utilities (simple formatting tools, not content generation)
from src.tools.writing import (
    estimate_paper_structure,
    format_results_table,
    format_ablation_table,
    get_citations_for_topic,
    format_figure,
    format_algorithm,
    format_equation,
    create_paper_skeleton,
    get_paper_context,
    validate_latex,
    save_to_file,
    check_paper_completeness,
    expand_paper,
)

# Formatting for conferences
from src.tools.formatting import (
    list_conferences,
    get_conference_requirements,
    cast_to_format,
    generate_poster,
    generate_supplementary,
    compile_paper,
    create_github_repo,
    finalize_paper_with_github,
)

# Context extraction
from src.context.extractor import extract_paper_context

# Workflow orchestration
from src.tools.workflow_tools import (
    get_next_action,
    get_workflow_checklist,
    mark_step_complete,
    set_target_metrics_from_papers,
)

# Experiment tracking
from src.tools.tracking import (
    log_experiment,
    get_experiment_history,
)

# Paper metrics extraction
from src.context.extractor import (
    extract_paper_metrics,
    extract_metrics_from_papers,
)

# Project management
from src.project.manager import project_manager

# Workflow state
from src.db.workflow import workflow_db


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-mcp")


# Project management tools
async def create_project(name: str, description: str = "") -> str:
    """Create a new research project with directory structure."""
    import json
    project = await project_manager.create_project(name, description)
    return json.dumps({
        "success": True,
        "project_id": project.project_id,
        "path": str(project.root_path),
        "directories": {
            "context": str(project.context_dir),
            "ideas": str(project.ideas_dir),
            "experiments": str(project.experiments_dir),
            "papers": str(project.papers_dir),
            "data": str(project.data_dir),
            "figures": str(project.figures_dir),
        },
    }, indent=2)


async def list_projects() -> str:
    """List all research projects."""
    import json
    projects = await project_manager.list_projects()
    return json.dumps({"projects": projects}, indent=2)


async def get_current_project() -> str:
    """Get the current active project."""
    import json
    project = await project_manager.get_current_project()
    if project:
        return json.dumps(project.to_dict(), indent=2)
    return json.dumps({"error": "No active project"})


async def set_current_project(project_id: str) -> str:
    """Set the current active project."""
    import json
    await project_manager.set_current_project(project_id)
    return json.dumps({"success": True, "project_id": project_id})


# Workflow tools
async def create_workflow(project_id: str) -> str:
    """Create a new workflow for a project."""
    import json
    workflow = await workflow_db.create_workflow(project_id)
    return json.dumps({
        "workflow_id": workflow.workflow_id,
        "stage": workflow.stage,
        "next_steps": workflow.next_steps,
    }, indent=2)


async def get_workflow_status(workflow_id: str) -> str:
    """Get current workflow status and progress."""
    import json
    workflow = await workflow_db.get_workflow(workflow_id)
    if not workflow:
        return json.dumps({"error": f"Workflow not found: {workflow_id}"})
    return json.dumps(workflow.get_progress_summary(), indent=2)


TOOL_HANDLERS = {
    # Aggregation
    "fetch_arxiv_trending": fetch_arxiv_trending,
    "fetch_hf_trending": fetch_hf_trending,
    "search_papers": search_papers,
    "get_paper_details": get_paper_details,
    "clone_paper_code": clone_paper_code,
    "extract_paper_context": extract_paper_context,
    
    # Ideas
    "generate_ideas": generate_ideas,
    "submit_idea": submit_idea,
    "approve_idea": approve_idea,
    "reject_idea": reject_idea,
    "check_novelty": check_novelty,
    "create_research_plan": create_research_plan,
    "define_hypotheses": define_hypotheses,
    "list_ideas": list_ideas,
    
    # Project management
    "create_project": create_project,
    "list_projects": list_projects,
    "get_current_project": get_current_project,
    "set_current_project": set_current_project,
    
    # Workflow
    "create_workflow": create_workflow,
    "get_workflow_status": get_workflow_status,
    "get_next_action": get_next_action,
    "get_workflow_checklist": get_workflow_checklist,
    
    # Environment
    "create_experiment_env": create_experiment_env,
    "install_dependencies": install_dependencies,
    "setup_docker": setup_docker,
    "check_gpu_availability": check_gpu_availability,
    "clone_baseline_repos": clone_baseline_repos,
    "setup_datasets": setup_datasets,
    
    # Experiments
    "run_experiment": run_experiment,
    "run_baseline": run_baseline,
    "run_ablation": run_ablation,
    "monitor_training": monitor_training,
    "save_checkpoint": save_checkpoint,
    "resume_experiment": resume_experiment,
    
    # Data collection
    "collect_metrics": collect_metrics,
    "parse_tensorboard": parse_tensorboard,
    "parse_wandb": parse_wandb,
    "aggregate_results": aggregate_results,
    "export_to_csv": export_to_csv,
    "compute_statistics": compute_statistics,
    
    # Visualization
    "plot_training_curves": plot_training_curves,
    "plot_comparison_bar": plot_comparison_bar,
    "plot_ablation_table": plot_ablation_table,
    "plot_scatter": plot_scatter,
    "plot_heatmap": plot_heatmap,
    "plot_qualitative": plot_qualitative,
    "generate_architecture_diagram": generate_architecture_diagram,
    "style_for_conference": style_for_conference,
    
    # Verification
    "verify_hypothesis": verify_hypothesis,
    "check_significance": check_significance,
    "detect_anomalies": detect_anomalies,
    "compare_to_baselines": compare_to_baselines,
    "generate_results_summary": generate_results_summary,
    
    # Writing utilities
    "estimate_paper_structure": estimate_paper_structure,
    "format_results_table": format_results_table,
    "format_ablation_table": format_ablation_table,
    "get_citations_for_topic": get_citations_for_topic,
    "format_figure": format_figure,
    "format_algorithm": format_algorithm,
    "format_equation": format_equation,
    "create_paper_skeleton": create_paper_skeleton,
    "get_paper_context": get_paper_context,
    "validate_latex": validate_latex,
    "save_to_file": save_to_file,
    
    # Conference formatting
    "list_conferences": list_conferences,
    "get_conference_requirements": get_conference_requirements,
    "cast_to_format": cast_to_format,
    "generate_poster": generate_poster,
    "generate_supplementary": generate_supplementary,
    "compile_paper": compile_paper,
    
    # Paper completeness and expansion
    "check_paper_completeness": check_paper_completeness,
    "expand_paper": expand_paper,
    
    # GitHub integration
    "create_github_repo": create_github_repo,
    "finalize_paper_with_github": finalize_paper_with_github,
    
    # Experiment tracking
    "log_experiment": log_experiment,
    "get_experiment_history": get_experiment_history,
    
    # Paper metrics
    "extract_paper_metrics": extract_paper_metrics,
    "set_target_metrics_from_papers": set_target_metrics_from_papers,
}

TOOLS = [
    # === AGGREGATION ===
    Tool(
        name="fetch_arxiv_trending",
        description="Fetch trending papers from arXiv by category. Returns recent high-impact papers.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "arXiv category (e.g., 'cs.LG', 'cs.CL', 'cs.CV')"},
                "days": {"type": "integer", "default": 7},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["category"],
        },
    ),
    Tool(
        name="fetch_hf_trending",
        description="Fetch trending papers from HuggingFace daily papers.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "days": {"type": "integer", "default": 365},
                "max_results": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="search_papers",
        description="Semantic search across cached papers with relevance scoring.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query - papers are ranked by keyword match"},
                "max_results": {"type": "integer", "default": 10},
                "min_relevance": {"type": "number", "default": 0.0, "description": "Minimum relevance score (0.0-1.0) to filter papers"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_paper_details",
        description="Get full details of a paper including abstract, citations, and code links.",
        inputSchema={
            "type": "object",
            "properties": {"paper_id": {"type": "string"}},
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="clone_paper_code",
        description="Clone the GitHub repository associated with a paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "target_dir": {"type": "string"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="extract_paper_context",
        description="Extract structure and style from a paper (sections, figures, citations, writing style).",
        inputSchema={
            "type": "object",
            "properties": {"arxiv_id": {"type": "string", "description": "arXiv ID (e.g., '2502.14678')"}},
            "required": ["arxiv_id"],
        },
    ),
    
    # === IDEAS ===
    Tool(
        name="generate_ideas",
        description=(
            "Get paper context for idea generation. "
            "Returns full abstracts from papers. After calling this, "
            "read the papers and call submit_idea() for each creative idea you generate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}},
                "count": {"type": "integer", "default": 3},
                "focus": {"type": "string"},
            },
            "required": ["paper_ids"],
        },
    ),
    Tool(
        name="submit_idea",
        description=(
            "Submit a research idea for user approval. "
            "Call this after generate_ideas() returns paper context. "
            "Generate creative, specific ideas based on the paper content."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Clear, specific title"},
                "description": {"type": "string", "description": "Detailed description (2-3 paragraphs)"},
                "motivation": {"type": "string", "description": "Why this is novel"},
                "source_papers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "description", "motivation"],
        },
    ),
    Tool(
        name="approve_idea",
        description=(
            "HUMAN USER ONLY - Do NOT call this automatically. "
            "Requires a confirmation code that is ONLY shown to the human user. "
            "AI assistants MUST wait for the user to type the approval command."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "idea_id": {"type": "string", "description": "The idea ID to approve"},
                "confirmation_code": {
                    "type": "string",
                    "description": "4-digit code shown only to human user. AI must NOT guess this.",
                },
                "user_feedback": {"type": "string", "description": "Optional user feedback"},
            },
            "required": ["idea_id", "confirmation_code"],
        },
    ),
    Tool(
        name="reject_idea",
        description="Reject an idea with optional feedback.",
        inputSchema={
            "type": "object",
            "properties": {
                "idea_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["idea_id"],
        },
    ),
    Tool(
        name="list_ideas",
        description="List all generated ideas, optionally filtered by status.",
        inputSchema={
            "type": "object",
            "properties": {"status": {"type": "string", "description": "Filter by status (pending_approval, approved, rejected)"}},
        },
    ),
    Tool(
        name="check_novelty",
        description="Check if a research idea is novel against existing literature.",
        inputSchema={
            "type": "object",
            "properties": {"idea": {"type": "string"}},
            "required": ["idea"],
        },
    ),
    Tool(
        name="create_research_plan",
        description="Create a structured research plan from an approved idea.",
        inputSchema={
            "type": "object",
            "properties": {"idea_id": {"type": "string"}},
            "required": ["idea_id"],
        },
    ),
    Tool(
        name="define_hypotheses",
        description="Define testable hypotheses for an approved idea.",
        inputSchema={
            "type": "object",
            "properties": {"idea_id": {"type": "string"}},
            "required": ["idea_id"],
        },
    ),
    
    # === PROJECT MANAGEMENT ===
    Tool(
        name="create_project",
        description="Create a new research project with directory structure and git repo.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="list_projects",
        description="List all research projects.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_current_project",
        description="Get the current active project.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="set_current_project",
        description="Set the current active project.",
        inputSchema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    ),
    
    # === WORKFLOW ===
    Tool(
        name="create_workflow",
        description="Create a new workflow for a project to track progress.",
        inputSchema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    ),
    Tool(
        name="get_workflow_status",
        description="Get current workflow status and progress.",
        inputSchema={
            "type": "object",
            "properties": {"workflow_id": {"type": "string"}},
            "required": ["workflow_id"],
        },
    ),
    Tool(
        name="get_next_action",
        description=(
            "ALWAYS call this before taking any action. "
            "Returns what step you must do next in the workflow. "
            "Prevents skipping required steps like environment setup, visualization, etc."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="get_workflow_checklist",
        description="Get complete checklist of all workflow stages and their status.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    
    # === ENVIRONMENT ===
    Tool(
        name="create_experiment_env",
        description="Create a conda or venv environment for experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "python": {"type": "string", "default": "3.10"},
                "use_conda": {"type": "boolean", "default": True},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="install_dependencies",
        description="Install dependencies in the experiment environment.",
        inputSchema={
            "type": "object",
            "properties": {
                "env_name": {"type": "string"},
                "requirements": {"type": "array", "items": {"type": "string"}},
                "requirements_file": {"type": "string"},
            },
            "required": ["env_name"],
        },
    ),
    Tool(
        name="setup_docker",
        description="Generate a Dockerfile for reproducible experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "base_image": {"type": "string", "default": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"},
                "requirements_file": {"type": "string"},
                "output_path": {"type": "string", "default": "Dockerfile"},
            },
        },
    ),
    Tool(
        name="check_gpu_availability",
        description="Check available GPU resources.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="clone_baseline_repos",
        description="Clone repositories for baseline methods.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}},
                "target_dir": {"type": "string", "default": "./baselines"},
            },
            "required": ["paper_ids"],
        },
    ),
    Tool(
        name="setup_datasets",
        description="Download and prepare datasets for experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "datasets": {"type": "array", "items": {"type": "string"}},
                "data_dir": {"type": "string", "default": "./data"},
            },
            "required": ["datasets"],
        },
    ),
    
    # === EXPERIMENTS ===
    Tool(
        name="run_experiment",
        description=(
            "Execute an experiment script with logging. "
            "PREREQUISITE: Must have approved idea and created environment first. "
            "Call get_next_action() if blocked."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string"},
                "config": {"type": "string"},
                "env_name": {"type": "string"},
                "gpu_ids": {"type": "string", "default": "0"},
                "name": {"type": "string"},
            },
            "required": ["script"],
        },
    ),
    Tool(
        name="run_baseline",
        description="Run a baseline method for comparison.",
        inputSchema={
            "type": "object",
            "properties": {
                "baseline_dir": {"type": "string"},
                "config": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["baseline_dir"],
        },
    ),
    Tool(
        name="run_ablation",
        description="Run an ablation study with multiple configurations.",
        inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string"},
                "base_config": {"type": "string"},
                "ablation_params": {"type": "object"},
            },
            "required": ["script", "ablation_params"],
        },
    ),
    Tool(
        name="monitor_training",
        description="Monitor an ongoing training run.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["experiment_name"],
        },
    ),
    Tool(
        name="save_checkpoint",
        description="Save experiment checkpoint.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "checkpoint_name": {"type": "string"},
            },
            "required": ["experiment_name"],
        },
    ),
    Tool(
        name="resume_experiment",
        description="Resume experiment from a checkpoint.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "checkpoint": {"type": "string"},
            },
            "required": ["experiment_name", "checkpoint"],
        },
    ),
    
    # === DATA COLLECTION ===
    Tool(
        name="collect_metrics",
        description="Collect metrics from experiment logs.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["experiments"],
        },
    ),
    Tool(
        name="parse_tensorboard",
        description="Extract data from TensorBoard logs.",
        inputSchema={
            "type": "object",
            "properties": {
                "log_dir": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["log_dir"],
        },
    ),
    Tool(
        name="parse_wandb",
        description="Extract data from Weights & Biases runs.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["project"],
        },
    ),
    Tool(
        name="aggregate_results",
        description="Aggregate results across multiple runs.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {"type": "array", "items": {"type": "string"}},
                "group_by": {"type": "string"},
            },
            "required": ["experiments"],
        },
    ),
    Tool(
        name="export_to_csv",
        description="Export results to CSV file.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "output_path": {"type": "string"},
            },
            "required": ["output_path"],
        },
    ),
    Tool(
        name="compute_statistics",
        description="Compute mean, std, confidence intervals for results.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "confidence_level": {"type": "number", "default": 0.95},
            },
            "required": ["results"],
        },
    ),
    
    # === VISUALIZATION ===
    Tool(
        name="plot_training_curves",
        description=(
            "REQUIRED for papers: Plot training loss/accuracy curves. "
            "Must be called before cast_to_format. Generates publication-quality figures."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "output_path": {"type": "string"},
                "conference": {"type": "string"},
            },
            "required": ["experiments"],
        },
    ),
    Tool(
        name="plot_comparison_bar",
        description=(
            "REQUIRED for papers: Create bar chart comparing methods. "
            "Must be called before cast_to_format. Generates publication-quality figures."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "metric": {"type": "string"},
                "output_path": {"type": "string"},
                "conference": {"type": "string"},
            },
            "required": ["results", "metric"],
        },
    ),
    Tool(
        name="plot_ablation_table",
        description="Generate ablation study table (LaTeX format).",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "output_path": {"type": "string"},
                "conference": {"type": "string"},
            },
            "required": ["results"],
        },
    ),
    Tool(
        name="plot_scatter",
        description="Create scatter plot with optional regression line.",
        inputSchema={
            "type": "object",
            "properties": {
                "x_data": {"type": "array"},
                "y_data": {"type": "array"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "output_path": {"type": "string"},
                "regression": {"type": "boolean", "default": False},
            },
            "required": ["x_data", "y_data"],
        },
    ),
    Tool(
        name="plot_heatmap",
        description="Create heatmap (confusion matrix, attention, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "x_labels": {"type": "array", "items": {"type": "string"}},
                "y_labels": {"type": "array", "items": {"type": "string"}},
                "output_path": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["data"],
        },
    ),
    Tool(
        name="plot_qualitative",
        description="Create side-by-side qualitative comparison.",
        inputSchema={
            "type": "object",
            "properties": {
                "images": {"type": "array", "items": {"type": "string"}},
                "labels": {"type": "array", "items": {"type": "string"}},
                "output_path": {"type": "string"},
            },
            "required": ["images"],
        },
    ),
    Tool(
        name="generate_architecture_diagram",
        description="Generate model architecture diagram.",
        inputSchema={
            "type": "object",
            "properties": {
                "components": {"type": "array", "items": {"type": "object"}},
                "connections": {"type": "array", "items": {"type": "object"}},
                "output_path": {"type": "string"},
            },
            "required": ["components"],
        },
    ),
    Tool(
        name="style_for_conference",
        description="Apply conference-specific styling to a figure.",
        inputSchema={
            "type": "object",
            "properties": {
                "figure_path": {"type": "string"},
                "conference": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["figure_path", "conference"],
        },
    ),
    
    # === VERIFICATION ===
    Tool(
        name="verify_hypothesis",
        description="Run statistical test to verify a hypothesis.",
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "results": {"type": "object"},
                "test_type": {"type": "string", "default": "t-test"},
            },
            "required": ["hypothesis", "results"],
        },
    ),
    Tool(
        name="check_significance",
        description="Check statistical significance between methods.",
        inputSchema={
            "type": "object",
            "properties": {
                "method1": {"type": "string"},
                "method2": {"type": "string"},
                "results": {"type": "object"},
                "test": {"type": "string", "default": "t-test"},
                "alpha": {"type": "number", "default": 0.05},
            },
            "required": ["method1", "method2", "results"],
        },
    ),
    Tool(
        name="detect_anomalies",
        description="Detect anomalous results in experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "threshold": {"type": "number", "default": 2.0},
            },
            "required": ["results"],
        },
    ),
    Tool(
        name="compare_to_baselines",
        description=(
            "REQUIRED for papers: Statistical comparison of your method to baselines. "
            "Must be called during analysis phase. Provides evidence for paper claims."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "baselines": {"type": "array", "items": {"type": "string"}},
                "results": {"type": "object"},
            },
            "required": ["method", "baselines", "results"],
        },
    ),
    Tool(
        name="generate_results_summary",
        description="Generate a summary of all experiment results.",
        inputSchema={
            "type": "object",
            "properties": {"experiments": {"type": "array", "items": {"type": "string"}}},
            "required": ["experiments"],
        },
    ),
    
    # === WRITING UTILITIES (formatting only, content generated by LLM) ===
    Tool(
        name="estimate_paper_structure",
        description="Estimate word counts and figures for a paper of given length.",
        inputSchema={
            "type": "object",
            "properties": {
                "target_pages": {"type": "integer", "default": 9},
                "conference": {"type": "string", "default": "neurips"},
            },
        },
    ),
    Tool(
        name="format_results_table",
        description="Format experiment results as a LaTeX table.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object", "description": "Dict of {method: {metric: value}}"},
                "caption": {"type": "string", "default": "Experimental results"},
                "label": {"type": "string", "default": "tab:results"},
                "bold_best": {"type": "boolean", "default": True},
            },
            "required": ["results"],
        },
    ),
    Tool(
        name="format_ablation_table",
        description="Format ablation results as a LaTeX table.",
        inputSchema={
            "type": "object",
            "properties": {
                "ablations": {"type": "object"},
                "full_model_name": {"type": "string", "default": "Full model"},
                "caption": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["ablations"],
        },
    ),
    Tool(
        name="get_citations_for_topic",
        description="Get relevant BibTeX citations from cached papers for a topic.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "max_citations": {"type": "integer", "default": 10},
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="format_figure",
        description="Generate LaTeX code for including a figure.",
        inputSchema={
            "type": "object",
            "properties": {
                "figure_path": {"type": "string"},
                "caption": {"type": "string"},
                "label": {"type": "string"},
                "width": {"type": "string", "default": "0.8\\textwidth"},
                "position": {"type": "string", "default": "t"},
            },
            "required": ["figure_path", "caption", "label"],
        },
    ),
    Tool(
        name="format_algorithm",
        description="Format algorithm steps as LaTeX algorithm environment.",
        inputSchema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}},
                "caption": {"type": "string"},
                "label": {"type": "string", "default": "alg:main"},
            },
            "required": ["steps", "caption"],
        },
    ),
    Tool(
        name="format_equation",
        description="Format a mathematical equation in LaTeX.",
        inputSchema={
            "type": "object",
            "properties": {
                "equation": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["equation"],
        },
    ),
    Tool(
        name="create_paper_skeleton",
        description="Create a paper skeleton with section headers.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "conference": {"type": "string", "default": "neurips"},
                "sections": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        },
    ),
    Tool(
        name="get_paper_context",
        description="Get context from cached papers to inform writing.",
        inputSchema={
            "type": "object",
            "properties": {"paper_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["paper_ids"],
        },
    ),
    Tool(
        name="validate_latex",
        description="Validate LaTeX content for common issues.",
        inputSchema={
            "type": "object",
            "properties": {"latex_content": {"type": "string"}},
            "required": ["latex_content"],
        },
    ),
    Tool(
        name="save_to_file",
        description="Save content to a file.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "filename": {"type": "string"},
                "output_dir": {"type": "string", "default": "./output"},
            },
            "required": ["content", "filename"],
        },
    ),
    
    # === CONFERENCE FORMATTING ===
    Tool(
        name="list_conferences",
        description="List all supported A* conferences with their requirements.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_conference_requirements",
        description="Get format requirements for a specific conference.",
        inputSchema={
            "type": "object",
            "properties": {"conference": {"type": "string"}},
            "required": ["conference"],
        },
    ),
    Tool(
        name="cast_to_format",
        description=(
            "Convert paper content to conference-specific LaTeX format. "
            "PREREQUISITE: Must generate figures first (plot_comparison_bar, plot_training_curves). "
            "Call get_next_action() if blocked."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_content": {"type": "object"},
                "conference": {"type": "string"},
                "output_dir": {"type": "string", "default": "./output"},
            },
            "required": ["conference"],
        },
    ),
    Tool(
        name="generate_poster",
        description=(
            "Generate a conference poster from paper content. "
            "PREREQUISITE: Must have figures and paper content. "
            "Call get_next_action() if blocked."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_content": {"type": "object"},
                "conference": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["conference"],
        },
    ),
    Tool(
        name="generate_supplementary",
        description="Generate supplementary materials document.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_content": {"type": "object"},
                "include_code": {"type": "boolean", "default": True},
                "include_data": {"type": "boolean", "default": False},
            },
        },
    ),
    Tool(
        name="compile_paper",
        description="Compile LaTeX paper to PDF using pdflatex. Runs pdflatex + bibtex + pdflatex x2.",
        inputSchema={
            "type": "object",
            "properties": {
                "tex_file": {"type": "string", "description": "Path to the .tex file to compile"},
                "output_dir": {"type": "string", "description": "Optional output directory for PDF"},
            },
            "required": ["tex_file"],
        },
    ),
    
    # === PAPER COMPLETENESS ===
    Tool(
        name="check_paper_completeness",
        description=(
            "Check if paper meets target length/figure/table requirements. "
            "Compare current paper against target metrics from reference papers. "
            "Returns NEEDS_EXPANSION if paper is too short."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_content": {"type": "object", "description": "Dict of {section: content}"},
                "latex_file": {"type": "string", "description": "Path to LaTeX file to analyze"},
                "target_word_count": {"type": "integer", "default": 5000},
                "target_figure_count": {"type": "integer", "default": 6},
                "target_table_count": {"type": "integer", "default": 3},
            },
        },
    ),
    Tool(
        name="expand_paper",
        description=(
            "Get suggestions for expanding a paper that is too short. "
            "Suggests additional experiments, figures, and analyses to add."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    
    # === GITHUB INTEGRATION ===
    Tool(
        name="create_github_repo",
        description=(
            "Create a GitHub repository for the current project using gh CLI. "
            "Commits all files and pushes to GitHub."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Repository name (defaults to project name)"},
                "private": {"type": "boolean", "default": True},
                "description": {"type": "string", "default": ""},
            },
        },
    ),
    Tool(
        name="finalize_paper_with_github",
        description=(
            "Add GitHub link to paper, compile to PDF, and commit. "
            "Final step before submission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "latex_file": {"type": "string", "description": "Path to LaTeX file"},
                "repo_url": {"type": "string", "description": "GitHub URL (auto-fetched if not provided)"},
            },
            "required": ["latex_file"],
        },
    ),
    
    # === EXPERIMENT TRACKING ===
    Tool(
        name="log_experiment",
        description=(
            "Log a completed experiment with config and metrics. "
            "Stores in SQLite and JSON for paper generation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Path to project directory"},
                "name": {"type": "string", "description": "Experiment name"},
                "config": {"type": "object", "description": "Experiment configuration"},
                "metrics": {"type": "object", "description": "Results metrics"},
            },
            "required": ["project_dir", "name", "config", "metrics"],
        },
    ),
    Tool(
        name="get_experiment_history",
        description=(
            "Get all logged experiment runs for a project. "
            "Returns summary and full run details."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Path to project directory"},
            },
            "required": ["project_dir"],
        },
    ),
    
    # === PAPER METRICS ===
    Tool(
        name="extract_paper_metrics",
        description=(
            "Extract word count, figure count, table count from a paper. "
            "Use to set targets for your paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID (e.g., '2502.14678')"},
            },
            "required": ["arxiv_id"],
        },
    ),
    Tool(
        name="set_target_metrics_from_papers",
        description=(
            "Extract metrics from multiple papers and set as targets. "
            "Averages word count, figures, tables across papers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_ids": {"type": "array", "items": {"type": "string"}, "description": "List of arXiv IDs"},
            },
            "required": ["arxiv_ids"],
        },
    ),
]


server = Server("research-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in TOOL_HANDLERS:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        handler = TOOL_HANDLERS[name]
        result = await handler(**arguments)
        
        # Handle case where result might be a dict (from extract_paper_context)
        import json
        if isinstance(result, dict):
            result = json.dumps(result, indent=2, ensure_ascii=False)
        
        return [TextContent(type="text", text=str(result))]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
