"""ResearchMCP Server - End-to-end AI research pipeline."""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Persona management for dynamic tool filtering
from src.personas import (
    persona_manager,
    PersonaType,
    get_tools_for_persona,
    PERSONAS,
)

# Data collection and aggregation
from src.tools.aggregation import (
    fetch_arxiv_trending,
    fetch_hf_trending,
    fetch_hf_trending_with_metrics,
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
    plot_verified_comparison,
    get_verified_experiment_data,
    plot_ablation_table,
    plot_scatter,
    plot_heatmap,
    plot_qualitative,
    generate_architecture_diagram,
    style_for_conference,
)

# Verification (includes mandatory verification before paper claims)
from src.tools.verification import (
    verify_hypothesis,
    check_significance,
    detect_anomalies,
    compare_to_baselines,
    generate_results_summary,
    verify_and_record_hypothesis,
    check_claims_verified,
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
    get_project_writing_context,
    extract_style_from_context,
    get_verified_claims_for_writing,
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

# Workflow orchestration (includes expansion loop)
from src.tools.workflow_tools import (
    get_next_action,
    get_workflow_checklist,
    mark_step_complete,
    set_target_metrics_from_papers,
    check_and_expand_paper,
    get_verified_claims,
)

# Experiment tracking
from src.tools.tracking import (
    log_experiment,
    get_experiment_history,
    get_real_metrics,
)

# Paper metrics and style extraction
from src.context.extractor import (
    extract_paper_metrics,
    extract_metrics_from_papers,
    extract_writing_style_context,
    get_paper_context_for_writing,
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
    workflow = await workflow_db.get_workflow(workflow_id)
    if not workflow:
        return json.dumps({"error": f"Workflow not found: {workflow_id}"})
    return json.dumps(workflow.get_progress_summary(), indent=2)


def _generate_persona_code() -> str:
    """Generate a 4-digit confirmation code for persona switching."""
    import random
    return str(random.randint(1000, 9999))


async def start_new_research() -> str:
    """
    Start a fresh research session. Call this at the beginning of a NEW chat/conversation.
    
    This tool:
    - Clears the current project selection (does NOT delete projects)
    - Resets persona to 'researcher' 
    - Prepares for a new research topic
    
    IMPORTANT: Call this when starting research on a NEW topic to avoid
    contamination from previous research sessions.
    
    Returns:
        Instructions for starting fresh research
    """
    # Clear current project selection
    await project_manager.set_current_project(None)
    
    # Reset persona to default
    persona_manager.clear_override()
    
    # List existing projects for reference
    projects = await project_manager.list_projects()
    project_names = [p.name for p in projects] if projects else []
    
    return json.dumps({
        "status": "FRESH_SESSION",
        "message": "Session reset for new research. Previous project deselected.",
        "current_project": None,
        "persona": "researcher",
        "existing_projects": project_names,
        "next_steps": [
            "1. Define your research topic",
            "2. Call create_project(name='your-topic-name') to create a new project",
            "3. Call create_workflow(project_id) to initialize workflow",
            "4. Use fetch_hf_trending or search_papers to gather context",
        ],
        "warning": (
            "If you want to continue previous research instead, "
            "call set_current_project(project_id) with one of the existing projects."
        ),
    }, indent=2)


async def switch_persona(persona: str) -> str:
    """
    Request to switch to a different persona. Requires user confirmation.
    
    This tool initiates a persona switch that MUST be confirmed by the user.
    Similar to idea approval, the model cannot switch personas without user consent.
    
    Personas:
    - researcher: Paper discovery, idea generation, approval workflow
    - experimenter: Experiment setup, execution, verification
    - writer: Paper writing, formatting, publication
    
    Args:
        persona: One of 'researcher', 'experimenter', 'writer'
    
    Returns:
        Instructions for user to confirm the switch
    """
    try:
        persona_type = PersonaType(persona.lower())
    except ValueError:
        return json.dumps({
            "error": f"Invalid persona: {persona}",
            "valid_personas": ["researcher", "experimenter", "writer"],
        })
    
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({
            "error": "No active project",
            "action_required": "Create a project first",
        })
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({
            "error": "No workflow found",
            "action_required": "Create a workflow first",
        })
    
    if workflow.current_persona == persona.lower():
        persona_info = PERSONAS[persona_type]
        return json.dumps({
            "message": f"Already in {persona} persona",
            "current_persona": persona,
            "available_tools": persona_info.tools,
        }, indent=2)
    
    confirmation_code = _generate_persona_code()
    workflow.pending_persona_switch = persona.lower()
    workflow.persona_confirmation_code = confirmation_code
    await workflow_db.save_workflow(workflow)
    
    persona_info = PERSONAS[persona_type]
    
    return json.dumps({
        "status": "AWAITING_USER_CONFIRMATION",
        "message": (
            f"Persona switch to '{persona}' requested. "
            "USER must confirm by typing the command below."
        ),
        "current_persona": workflow.current_persona,
        "requested_persona": persona,
        "confirmation_command": f"SWITCH {persona} CODE {confirmation_code}",
        "new_tools_available": persona_info.tools[:10],
        "ai_instruction": (
            "STOP. Do NOT call confirm_persona_switch yourself. "
            "Wait for the human user to type the confirmation command."
        ),
    }, indent=2)


async def confirm_persona_switch(persona: str, confirmation_code: str) -> str:
    """
    HUMAN USER ONLY - Confirm a persona switch with the confirmation code.
    
    This tool requires a confirmation code that is ONLY shown to the human user.
    AI assistants MUST NOT call this tool - wait for user to type the command.
    
    Args:
        persona: The persona to switch to
        confirmation_code: The 4-digit code shown to the user
    
    Returns:
        Confirmation of successful switch or error
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    if not workflow.pending_persona_switch:
        return json.dumps({
            "error": "NO_PENDING_SWITCH",
            "message": "No persona switch is pending. Call switch_persona() first.",
        })
    
    if not confirmation_code:
        return json.dumps({
            "error": "CONFIRMATION_CODE_REQUIRED",
            "message": "Confirmation code is required. Only the human user has the code.",
            "ai_instruction": "STOP. Wait for user to type the confirmation command.",
        })
    
    if confirmation_code != workflow.persona_confirmation_code:
        return json.dumps({
            "error": "INVALID_CODE",
            "message": "The confirmation code is incorrect.",
            "ai_instruction": "STOP. Do not retry. Wait for user.",
        })
    
    if persona.lower() != workflow.pending_persona_switch:
        return json.dumps({
            "error": "PERSONA_MISMATCH",
            "message": f"Requested switch was to '{workflow.pending_persona_switch}', not '{persona}'",
        })
    
    try:
        persona_type = PersonaType(persona.lower())
    except ValueError:
        return json.dumps({"error": f"Invalid persona: {persona}"})
    
    workflow.current_persona = persona.lower()
    workflow.pending_persona_switch = None
    workflow.persona_confirmation_code = None
    await workflow_db.save_workflow(workflow)
    
    persona_manager.set_override(persona_type)
    
    persona_info = PERSONAS[persona_type]
    
    return json.dumps({
        "success": True,
        "message": f"✅ Persona switched to '{persona}'",
        "active_persona": persona,
        "description": persona_info.description,
        "available_tools": persona_info.tools,
        "tool_count": len(persona_info.tools),
        "next_steps": _get_persona_next_steps(persona_type),
    }, indent=2)


def _get_persona_next_steps(persona_type: PersonaType) -> list[str]:
    """Get next steps for a persona."""
    if persona_type == PersonaType.RESEARCHER:
        return [
            "Use fetch_arxiv_trending or search_papers to gather papers",
            "Use generate_ideas to create research ideas",
            "Wait for user to approve an idea",
        ]
    elif persona_type == PersonaType.EXPERIMENTER:
        return [
            "Use create_experiment_env to set up environment",
            "Use run_experiment to execute experiments",
            "Use verify_and_record_hypothesis to verify results",
        ]
    else:  # WRITER
        return [
            "Use get_project_writing_context to read gathered papers",
            "Use extract_style_from_context to analyze writing style",
            "Use get_verified_claims_for_writing to get verifiable claims",
            "Use cast_to_format to generate conference paper",
        ]


async def get_active_persona() -> str:
    """Get the currently active persona and its available tools."""
    current_project_obj = await project_manager.get_current_project()
    
    workflow_stage = "context_gathering"
    workflow_persona = None
    if current_project_obj:
        workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
        if workflow:
            workflow_stage = workflow.stage
            workflow_persona = workflow.current_persona
    
    active_persona = persona_manager.get_active_persona(workflow_stage, workflow_persona)
    
    return json.dumps({
        "active_persona": active_persona.name.value,
        "workflow_stage": workflow_stage,
        "persisted_persona": workflow_persona,
        "description": active_persona.description,
        "available_tools": active_persona.tools,
        "tool_count": len(active_persona.tools),
        "message": (
            "Tools are automatically filtered based on workflow stage. "
            "Use switch_persona() to manually override."
        ),
    }, indent=2)


TOOL_HANDLERS = {
    # Aggregation
    "fetch_arxiv_trending": fetch_arxiv_trending,
    "fetch_hf_trending": fetch_hf_trending,
    "fetch_hf_trending_with_metrics": fetch_hf_trending_with_metrics,
    "search_papers": search_papers,
    "get_paper_details": get_paper_details,
    "clone_paper_code": clone_paper_code,
    "extract_paper_context": extract_paper_context,
    
    # Style and metrics extraction
    "extract_writing_style_context": extract_writing_style_context,
    "get_paper_context_for_writing": get_paper_context_for_writing,
    "get_writing_style_context": extract_writing_style_context,
    
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
    
    # Workflow orchestration
    "create_workflow": create_workflow,
    "get_workflow_status": get_workflow_status,
    "get_next_action": get_next_action,
    "get_workflow_checklist": get_workflow_checklist,
    "check_and_expand_paper": check_and_expand_paper,
    
    # Persona management
    "switch_persona": switch_persona,
    "confirm_persona_switch": confirm_persona_switch,
    "get_active_persona": get_active_persona,
    
    # Session management
    "start_new_research": start_new_research,
    
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
    "plot_verified_comparison": plot_verified_comparison,
    "get_verified_experiment_data": get_verified_experiment_data,
    "plot_ablation_table": plot_ablation_table,
    "plot_scatter": plot_scatter,
    "plot_heatmap": plot_heatmap,
    "plot_qualitative": plot_qualitative,
    "generate_architecture_diagram": generate_architecture_diagram,
    "style_for_conference": style_for_conference,
    
    # Verification (mandatory before paper claims)
    "verify_hypothesis": verify_hypothesis,
    "verify_and_record_hypothesis": verify_and_record_hypothesis,
    "check_claims_verified": check_claims_verified,
    "check_significance": check_significance,
    "detect_anomalies": detect_anomalies,
    "compare_to_baselines": compare_to_baselines,
    "generate_results_summary": generate_results_summary,
    "get_verified_claims": get_verified_claims,
    
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
    "get_project_writing_context": get_project_writing_context,
    "extract_style_from_context": extract_style_from_context,
    "get_verified_claims_for_writing": get_verified_claims_for_writing,
    
    # GitHub integration
    "create_github_repo": create_github_repo,
    "finalize_paper_with_github": finalize_paper_with_github,
    
    # Experiment tracking
    "log_experiment": log_experiment,
    "get_experiment_history": get_experiment_history,
    "get_real_metrics": get_real_metrics,
    
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
            "Create bar chart comparing methods. Accepts arbitrary data. "
            "WARNING: Use plot_verified_comparison() instead to ensure data comes from real experiments."
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
        name="plot_verified_comparison",
        description=(
            "🔒 RECOMMENDED: Create bar chart using ONLY verified experiment data. "
            "This function pulls metrics directly from experiment logs - no fabrication possible. "
            "Use this instead of plot_comparison_bar for integrity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of run_ids from run_experiment()",
                },
                "metric": {
                    "type": "string",
                    "description": "Metric name (e.g., 'accuracy', 'loss', 'f1')",
                },
                "output_path": {"type": "string"},
                "conference": {"type": "string"},
            },
            "required": ["run_ids", "metric"],
        },
    ),
    Tool(
        name="get_verified_experiment_data",
        description=(
            "Get verified experiment metrics for use in visualizations. "
            "Returns ONLY data parsed from actual experiment logs. "
            "Use this to build custom visualizations with verified data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metric": {"type": "string"},
            },
            "required": ["run_ids", "metric"],
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
    Tool(
        name="get_project_writing_context",
        description=(
            "Get all papers gathered for the current project from the context folder. "
            "Use this to inform your writing with the actual papers you gathered during research."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="extract_style_from_context",
        description=(
            "Analyze gathered papers to extract writing style patterns. "
            "Returns sentence length, first person usage, formality level, common phrases."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_verified_claims_for_writing",
        description=(
            "Get only the verified claims that can be included in the paper. "
            "Returns claims verified through verify_and_record_hypothesis with real experiment data."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_real_metrics",
        description=(
            "Get only the metrics parsed from actual experiment log files. "
            "Returns ONLY metrics that were extracted from real stdout logs, not user-provided."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Path to the project directory"},
                "run_id": {"type": "string", "description": "The run_id from run_experiment()"},
            },
            "required": ["project_dir", "run_id"],
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
    
    # === PERSONA MANAGEMENT ===
    Tool(
        name="switch_persona",
        description=(
            "Request to switch persona. Requires user confirmation. "
            "Personas: researcher (paper discovery, ideas), experimenter (experiments, verification), writer (paper writing)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "enum": ["researcher", "experimenter", "writer"],
                    "description": "The persona to switch to",
                },
            },
            "required": ["persona"],
        },
    ),
    Tool(
        name="confirm_persona_switch",
        description=(
            "HUMAN USER ONLY - Confirm a persona switch with the confirmation code. "
            "AI must NOT call this - wait for user to type the command."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "enum": ["researcher", "experimenter", "writer"],
                    "description": "The persona to switch to",
                },
                "confirmation_code": {
                    "type": "string",
                    "description": "4-digit code shown only to human user. AI must NOT guess this.",
                },
            },
            "required": ["persona", "confirmation_code"],
        },
    ),
    Tool(
        name="get_active_persona",
        description="Get the currently active persona and its available tools.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="start_new_research",
        description=(
            "Start a fresh research session. CALL THIS at the beginning of a NEW chat/conversation "
            "to avoid contamination from previous research. Clears current project selection and "
            "resets persona to 'researcher'. Does NOT delete existing projects."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    
    # === EXPANSION LOOP ===
    Tool(
        name="check_and_expand_paper",
        description=(
            "Check paper completeness and trigger expansion if too short. "
            "Generates new hypotheses and queues experiments if paper doesn't meet target metrics."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_verified_claims",
        description=(
            "Get all verified claims that can be included in the paper. "
            "Only claims that passed hypothesis verification can be cited."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    
    # === MANDATORY VERIFICATION ===
    Tool(
        name="verify_and_record_hypothesis",
        description=(
            "MANDATORY: Verify a hypothesis and record the result. "
            "Must be called before any claim can be added to the paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "description": "Unique identifier for the hypothesis"},
                "hypothesis_statement": {"type": "string", "description": "The claim being tested"},
                "experiment_id": {"type": "string", "description": "Which experiment produced the results"},
                "results": {"type": "object", "description": "Dict with method names as keys and result arrays"},
                "test_type": {
                    "type": "string",
                    "enum": ["t-test", "paired-t", "wilcoxon", "mann-whitney"],
                    "default": "t-test",
                },
            },
            "required": ["hypothesis_id", "hypothesis_statement", "experiment_id", "results"],
        },
    ),
    Tool(
        name="check_claims_verified",
        description=(
            "Check if all claims have been verified before paper submission. "
            "Blocks paper if unverified claims exist."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "string"}, "description": "List of hypothesis IDs to check"},
            },
            "required": ["claims"],
        },
    ),
    
    # === STYLE CONTEXT ===
    Tool(
        name="get_writing_style_context",
        description=(
            "Extract writing style context from reference papers for style-consistent writing. "
            "Returns style metrics and example paragraphs for prompting."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_ids": {"type": "array", "items": {"type": "string"}, "description": "List of arXiv IDs to analyze"},
            },
            "required": ["arxiv_ids"],
        },
    ),
    Tool(
        name="get_paper_context_for_writing",
        description=(
            "Get comprehensive context for paper writing including style and structure. "
            "Call before writing each section."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "List of paper IDs"},
            },
            "required": ["paper_ids"],
        },
    ),
    
    # === HF TRENDING WITH METRICS ===
    Tool(
        name="fetch_hf_trending_with_metrics",
        description=(
            "Fetch trending papers from HuggingFace with paper metrics (word count, figures). "
            "Specifically designed for setting target metrics for paper writing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to filter papers by"},
                "max_results": {"type": "integer", "default": 10},
            },
        },
    ),
]


server = Server("research-mcp")


# Create a mapping from tool names to Tool objects for filtering
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

# Core tools always available regardless of persona
CORE_TOOLS = [
    "get_next_action",
    "get_workflow_status",
    "get_workflow_checklist",
    "switch_persona",
    "confirm_persona_switch",
    "get_active_persona",
    "create_project",
    "list_projects",
    "get_current_project",
    "set_current_project",
    "create_workflow",
    "approve_idea",
    "start_new_research",
]


async def get_current_workflow_info() -> tuple[str, str]:
    """Get the current workflow stage and persona for filtering.
    
    Returns (stage, persona)
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return "context_gathering", "researcher"
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return "context_gathering", "researcher"
    
    return workflow.stage, workflow.current_persona or "researcher"


async def get_current_workflow_stage() -> str:
    """Get the current workflow stage for persona filtering."""
    stage, _ = await get_current_workflow_info()
    return stage


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Return tools filtered by the active persona.
    
    This reduces tool overload by only showing relevant tools for the current
    workflow stage. Core tools (workflow management, persona switching) are
    always available.
    """
    workflow_stage, workflow_persona = await get_current_workflow_info()
    active_persona = persona_manager.get_active_persona(workflow_stage, workflow_persona)
    
    allowed_tool_names = set(active_persona.tools) | set(CORE_TOOLS)
    
    filtered_tools = [
        tool for tool in TOOLS 
        if tool.name in allowed_tool_names
    ]
    
    return filtered_tools


ALWAYS_ALLOWED_TOOLS = {
    "get_next_action",
    "switch_persona",
    "confirm_persona_switch",
    "get_active_persona",
    "create_project",
    "list_projects",
    "get_current_project",
    "set_current_project",
    "create_workflow",
    "get_workflow_status",
    "get_workflow_checklist",
    "approve_idea",  # User action
    "start_new_research",  # Fresh session
}


async def _check_persona_access(tool_name: str) -> tuple[bool, str]:
    """Check if the current persona can access this tool.
    
    Returns (is_allowed, error_message)
    """
    if tool_name in ALWAYS_ALLOWED_TOOLS:
        return True, ""
    
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return True, ""
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return True, ""
    
    if workflow.pending_persona_switch:
        return False, json.dumps({
            "error": "PERSONA_SWITCH_PENDING",
            "message": (
                f"A persona switch to '{workflow.pending_persona_switch}' is pending confirmation. "
                "Wait for user to confirm before calling any tools."
            ),
            "current_persona": workflow.current_persona,
            "pending_switch": workflow.pending_persona_switch,
            "action_required": (
                f"User must type: SWITCH {workflow.pending_persona_switch} CODE <confirmation_code>"
            ),
        }, indent=2)
    
    current_persona = workflow.current_persona
    
    try:
        persona_type = PersonaType(current_persona)
        persona_info = PERSONAS.get(persona_type)
        
        if persona_info and tool_name not in persona_info.tools:
            return False, json.dumps({
                "error": "TOOL_NOT_AVAILABLE",
                "message": (
                    f"Tool '{tool_name}' is not available in '{current_persona}' persona. "
                    "You need to switch persona to access this tool."
                ),
                "current_persona": current_persona,
                "current_tools": persona_info.tools[:10],
                "action_required": f"Call switch_persona() to change persona",
            }, indent=2)
    except ValueError:
        pass
    
    return True, ""


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in TOOL_HANDLERS:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    is_allowed, error_msg = await _check_persona_access(name)
    if not is_allowed:
        return [TextContent(type="text", text=error_msg)]

    try:
        handler = TOOL_HANDLERS[name]
        result = await handler(**arguments)
        
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
