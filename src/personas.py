"""Agent personas - dynamic tool filtering based on workflow stage."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PersonaType(str, Enum):
    RESEARCHER = "researcher"
    EXPERIMENTER = "experimenter"
    WRITER = "writer"


@dataclass
class Persona:
    name: PersonaType
    description: str
    active_stages: list[str]
    tools: list[str]


RESEARCHER_TOOLS = [
    # Core workflow
    "get_next_action",
    "get_workflow_status",
    "get_workflow_checklist",
    "create_project",
    "list_projects",
    "get_current_project",
    "set_current_project",
    "create_workflow",
    
    # Paper discovery
    "fetch_arxiv_trending",
    "fetch_hf_trending",
    "search_papers",
    "get_paper_details",
    "extract_paper_context",
    "extract_paper_metrics",
    "set_target_metrics_from_papers",
    
    # Ideas
    "generate_ideas",
    "submit_idea",
    "list_ideas",
    "check_novelty",
    "approve_idea",
    "reject_idea",
]

EXPERIMENTER_TOOLS = [
    # Core workflow
    "get_next_action",
    "get_workflow_status",
    "get_workflow_checklist",
    
    # Environment
    "create_experiment_env",
    "install_dependencies",
    "setup_docker",
    "check_gpu_availability",
    "clone_baseline_repos",
    "setup_datasets",
    
    # Experiments
    "run_experiment",
    "run_baseline",
    "run_ablation",
    "monitor_training",
    "save_checkpoint",
    "resume_experiment",
    "log_experiment",
    "get_experiment_history",
    "get_real_metrics",
    
    # Data collection
    "collect_metrics",
    "parse_tensorboard",
    "parse_wandb",
    "aggregate_results",
    "export_to_csv",
    "compute_statistics",
    
    # Verification (MANDATORY - uses metrics from logs)
    "verify_hypothesis",
    "verify_and_record_hypothesis",
    "check_significance",
    "detect_anomalies",
    "compare_to_baselines",
    "generate_results_summary",
    "check_claims_verified",
    
    # Visualization
    "plot_training_curves",
    "plot_comparison_bar",
    "plot_ablation_table",
    "plot_scatter",
    "plot_heatmap",
    "plot_qualitative",
    "generate_architecture_diagram",
    
    # Ideas (for hypotheses)
    "create_research_plan",
    "define_hypotheses",
]

WRITER_TOOLS = [
    # Core workflow
    "get_next_action",
    "get_workflow_status",
    "get_workflow_checklist",
    
    # Context (from research phase)
    "get_project_writing_context",
    "extract_style_from_context",
    "get_verified_claims_for_writing",
    
    # Paper structure
    "estimate_paper_structure",
    "create_paper_skeleton",
    "get_paper_context",
    "get_writing_style_context",
    
    # Content formatting
    "format_results_table",
    "format_ablation_table",
    "format_figure",
    "format_algorithm",
    "format_equation",
    "get_citations_for_topic",
    "validate_latex",
    "save_to_file",
    
    # Paper IR
    "write_paper_section",
    "add_figure_to_paper",
    "add_table_to_paper",
    "get_paper_ir",
    "save_paper_ir",
    
    # Completeness
    "check_paper_completeness",
    "expand_paper",
    
    # Conference formatting
    "list_conferences",
    "get_conference_requirements",
    "cast_to_format",
    "style_for_conference",
    "generate_poster",
    "generate_supplementary",
    "compile_paper",
    
    # GitHub
    "create_github_repo",
    "finalize_paper_with_github",
]


PERSONAS = {
    PersonaType.RESEARCHER: Persona(
        name=PersonaType.RESEARCHER,
        description="Literature review, idea generation, and approval workflow",
        active_stages=["context_gathering", "idea_generation", "idea_approval"],
        tools=RESEARCHER_TOOLS,
    ),
    PersonaType.EXPERIMENTER: Persona(
        name=PersonaType.EXPERIMENTER,
        description="Experiment setup, execution, and result verification",
        active_stages=["experiment_setup", "experimenting", "analysis"],
        tools=EXPERIMENTER_TOOLS,
    ),
    PersonaType.WRITER: Persona(
        name=PersonaType.WRITER,
        description="Paper writing, formatting, and publication",
        active_stages=["writing", "formatting", "complete"],
        tools=WRITER_TOOLS,
    ),
}


STAGE_TO_PERSONA = {
    "context_gathering": PersonaType.RESEARCHER,
    "idea_generation": PersonaType.RESEARCHER,
    "idea_approval": PersonaType.RESEARCHER,
    "experiment_setup": PersonaType.EXPERIMENTER,
    "experimenting": PersonaType.EXPERIMENTER,
    "analysis": PersonaType.EXPERIMENTER,
    "writing": PersonaType.WRITER,
    "formatting": PersonaType.WRITER,
    "complete": PersonaType.WRITER,
}


def get_persona_for_stage(stage: str) -> Persona:
    """Get the appropriate persona for a workflow stage."""
    persona_type = STAGE_TO_PERSONA.get(stage, PersonaType.RESEARCHER)
    return PERSONAS[persona_type]


def get_tools_for_persona(persona_type: PersonaType) -> list[str]:
    """Get list of tool names available for a persona."""
    return PERSONAS[persona_type].tools


def filter_tools_by_persona(all_tools: list, persona_type: PersonaType) -> list:
    """Filter MCP Tool objects to only those available for the persona."""
    allowed_tool_names = set(get_tools_for_persona(persona_type))
    return [tool for tool in all_tools if tool.name in allowed_tool_names]


def get_all_tool_names() -> set[str]:
    """Get set of all tool names across all personas."""
    all_tools = set()
    for persona in PERSONAS.values():
        all_tools.update(persona.tools)
    return all_tools


class PersonaManager:
    """Manages active persona and tool filtering."""
    
    def __init__(self):
        self._override_persona: Optional[PersonaType] = None
    
    def set_override(self, persona_type: Optional[PersonaType]) -> None:
        """Manually override the active persona."""
        self._override_persona = persona_type
    
    def clear_override(self) -> None:
        """Clear manual persona override."""
        self._override_persona = None
    
    def get_active_persona(self, workflow_stage: str) -> Persona:
        """Get active persona based on workflow stage or override."""
        if self._override_persona:
            return PERSONAS[self._override_persona]
        return get_persona_for_stage(workflow_stage)
    
    def get_active_tools(self, workflow_stage: str, all_tools: list) -> list:
        """Get filtered tools for the active persona."""
        persona = self.get_active_persona(workflow_stage)
        return filter_tools_by_persona(all_tools, persona.name)


persona_manager = PersonaManager()
