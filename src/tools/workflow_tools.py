"""Workflow orchestration tools - guide the agent through required steps."""

import json
from pathlib import Path
from typing import Optional

from src.db.workflow import workflow_db, WorkflowState
from src.db.experiments_db import experiments_db
from src.project.manager import project_manager


STAGE_PERSONA_MAP = {
    "context_gathering": "researcher",
    "idea_generation": "researcher",
    "idea_approval": "researcher",
    "experiment_setup": "experimenter",
    "experimenting": "experimenter",
    "analysis": "experimenter",
    "writing": "writer",
    "formatting": "writer",
    "complete": "writer",
}


def _get_required_persona_for_stage(stage: str) -> Optional[str]:
    """Get the required persona for a workflow stage."""
    return STAGE_PERSONA_MAP.get(stage)


# Minimum thresholds for paper completeness
MIN_WORD_COUNT_RATIO = 0.85  # Paper must be 85% of target length
MIN_FIGURE_COUNT = 3
MIN_TABLE_COUNT = 1


# Stage definitions with required completions and next actions
WORKFLOW_STAGES = {
    "context_gathering": {
        "description": "Gather relevant papers, extract context, and set target metrics",
        "required_tools": [
            "fetch_arxiv_trending",
            "fetch_hf_trending",
            "search_papers",
            "extract_paper_metrics",
            "set_target_metrics_from_papers",
        ],
        "completion_check": lambda w: len(w.gathered_papers) >= 3 and w.target_metrics is not None,
        "next_stage": "idea_generation",
    },
    "idea_generation": {
        "description": "Generate research ideas from gathered context",
        "required_tools": ["generate_ideas", "submit_idea"],
        "completion_check": lambda w: len(w.generated_ideas) > 0,
        "next_stage": "idea_approval",
    },
    "idea_approval": {
        "description": "BLOCKED - Waiting for user to approve an idea",
        "required_tools": [],  # User must manually approve
        "completion_check": lambda w: w.approved_idea_id is not None,
        "next_stage": "experiment_setup",
        "is_blocking": True,
    },
    "experiment_setup": {
        "description": "Set up experiment environment and datasets",
        "required_tools": ["create_experiment_env", "setup_datasets", "define_hypotheses"],
        "completion_check": lambda w: (
            "env_created" in w.completed_steps and 
            "datasets_setup" in w.completed_steps
        ),
        "next_stage": "experimenting",
    },
    "experimenting": {
        "description": "Run experiments with proper tracking and logging",
        "required_tools": [
            "run_experiment",
            "run_baseline",
            "log_experiment",
            "collect_metrics",
        ],
        "completion_check": lambda w: len(w.completed_experiments) > 0 and len(w.experiment_runs) > 0,
        "next_stage": "analysis",
    },
    "analysis": {
        "description": "Analyze results and generate visualizations",
        "required_tools": [
            "compute_statistics",
            "check_significance",
            "plot_comparison_bar",
            "plot_training_curves",
            "compare_to_baselines",
            "get_experiment_history",
        ],
        "completion_check": lambda w: len(w.figures_generated) >= 2,
        "next_stage": "writing",
    },
    "writing": {
        "description": "Write the paper, check completeness against targets",
        "required_tools": [
            "estimate_paper_structure",
            "format_results_table",
            "get_citations_for_topic",
            "create_paper_skeleton",
            "check_paper_completeness",
        ],
        "completion_check": lambda w: len(w.paper_sections) >= 3,
        "next_stage": "formatting",
    },
    "formatting": {
        "description": "Format paper, create GitHub repo, and compile PDF",
        "required_tools": [
            "cast_to_format",
            "compile_paper",
            "create_github_repo",
            "finalize_paper_with_github",
        ],
        "completion_check": lambda w: w.target_conference is not None and w.github_url is not None,
        "next_stage": "complete",
    },
}


async def get_next_action() -> str:
    """
    Get the next required action based on current workflow state.
    
    ALWAYS call this before taking any action to ensure proper workflow progression.
    This tool guides you through all required steps and prevents skipping.
    
    Returns the current stage, what's been completed, and what must be done next.
    """
    # Get current project and workflow
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({
            "status": "NO_PROJECT",
            "error": "No project is currently active",
            "action_required": "Call create_project(name) first",
            "example": "create_project(name='my_research')",
        }, indent=2)
    
    current_project_id = current_project_obj.project_id
    workflow = await workflow_db.get_project_workflow(current_project_id)
    if not workflow:
        return json.dumps({
            "status": "NO_WORKFLOW",
            "error": "No workflow exists for this project",
            "action_required": "Call create_workflow(project_id) first",
            "example": f"create_workflow(project_id='{current_project_id}')",
        }, indent=2)
    
    current_stage = workflow.stage
    stage_info = WORKFLOW_STAGES.get(current_stage, {})
    
    # HARD BLOCK: If there's a pending persona switch awaiting confirmation
    if workflow.pending_persona_switch:
        return json.dumps({
            "status": "PERSONA_SWITCH_PENDING",
            "message": (
                "STOP. A persona switch is pending user confirmation. "
                "Wait for the user to confirm before proceeding."
            ),
            "current_persona": workflow.current_persona,
            "pending_switch_to": workflow.pending_persona_switch,
            "user_action_required": (
                f"User must type: SWITCH {workflow.pending_persona_switch} CODE <confirmation_code>\n"
                "The confirmation code was shown to the user in the switch_persona response."
            ),
            "ai_instruction": (
                "STOP. Do NOT proceed with any other actions. "
                "Wait for the user to confirm the persona switch."
            ),
            "do_not_proceed": True,
        }, indent=2)
    
    # HARD BLOCK: Check if stage requires a different persona and force switch
    required_persona = _get_required_persona_for_stage(current_stage)
    if required_persona and workflow.current_persona != required_persona:
        return json.dumps({
            "status": "PERSONA_SWITCH_REQUIRED",
            "message": (
                f"STOP. Stage '{current_stage}' requires '{required_persona}' persona. "
                f"Current persona is '{workflow.current_persona}'."
            ),
            "current_persona": workflow.current_persona,
            "required_persona": required_persona,
            "action_required": f"Call switch_persona('{required_persona}') and wait for user confirmation",
            "ai_instruction": (
                f"1. Call switch_persona('{required_persona}')\n"
                "2. Present the confirmation command to the user\n"
                "3. WAIT for the user to confirm the switch\n"
                "4. Do NOT proceed until persona is switched"
            ),
            "do_not_proceed": True,
        }, indent=2)
    
    # HARD BLOCK: Before ANY planning or experiments, require idea approval
    # This prevents the model from creating a plan that includes idea approval as step 1
    if current_stage in ["idea_generation", "idea_approval"]:
        if not workflow.approved_idea_id:
            ideas_info = []
            ideas = await experiments_db.list_ideas(status="pending_approval")
            for idea in ideas:
                ideas_info.append({
                    "idea_id": idea.idea_id,
                    "title": idea.title,
                    "novelty_score": getattr(idea, "novelty_score", 0),
                })
            
            return json.dumps({
                "status": "HARD_BLOCK",
                "stage": current_stage,
                "message": (
                    "STOP. Do NOT create a plan or proceed with any experiments yet. "
                    "An idea MUST be approved by the user BEFORE any planning."
                ),
                "reason": "User approval required before planning phase",
                "pending_ideas": ideas_info,
                "user_action_required": (
                    "User must type: APPROVE <idea_id> CODE <confirmation_code>\n"
                    "The confirmation code is shown only to the human user."
                ),
                "ai_instruction": (
                    "1. If no ideas exist, call generate_ideas() and submit_idea() first.\n"
                    "2. Present the ideas to the user with their approval commands.\n"
                    "3. WAIT for the user to approve one idea.\n"
                    "4. Do NOT write a plan until an idea is approved.\n"
                    "5. Do NOT call any experiment or analysis tools."
                ),
                "do_not_proceed": True,
                "do_not_create_plan": True,
            }, indent=2)
    
    # Check if current stage is blocking (requires user action)
    if stage_info.get("is_blocking"):
        return json.dumps({
            "status": "BLOCKED",
            "stage": current_stage,
            "description": stage_info.get("description", ""),
            "reason": "This stage requires human user action",
            "user_action_required": _get_user_action_instruction(workflow, current_stage),
            "ai_instruction": (
                "STOP HERE. Do NOT proceed with any other tools. "
                "Wait for the user to complete the required action."
            ),
            "do_not_proceed": True,
        }, indent=2)
    
    # Check stage completion
    completion_check = stage_info.get("completion_check", lambda w: False)
    is_complete = completion_check(workflow)
    
    if is_complete:
        # Advance to next stage
        next_stage = stage_info.get("next_stage", current_stage)
        workflow.stage = next_stage
        await workflow_db.save_workflow(workflow)
        
        return await get_next_action()  # Recurse to get next stage info
    
    # Check if in writing stage and paper completeness
    if current_stage == "writing" and len(workflow.paper_sections) > 0:
        completeness = await _check_paper_completeness_internal(workflow)
        
        if not completeness.get("sufficient", False):
            return json.dumps({
                "status": "NEEDS_EXPANSION",
                "stage": current_stage,
                "description": "Paper is too short compared to target metrics",
                "completeness": completeness,
                "next_action": {
                    "tool": "expand_paper",
                    "description": "Run more experiments or add more content to reach target length",
                },
                "suggestions": completeness.get("suggestions", []),
                "paper_iteration": workflow.paper_iterations + 1,
            }, indent=2)
    
    # Current stage not complete - return required actions
    required_tools = stage_info.get("required_tools", [])
    completed_tools = _get_completed_tools(workflow, required_tools)
    missing_tools = [t for t in required_tools if t not in completed_tools]
    
    # Add target metrics info if available
    target_info = {}
    if workflow.target_metrics:
        target_info = {
            "target_words": workflow.target_metrics.get("word_count", 5000),
            "target_figures": workflow.target_metrics.get("figure_count", 6),
            "target_tables": workflow.target_metrics.get("table_count", 3),
        }
    
    return json.dumps({
        "status": "IN_PROGRESS",
        "stage": current_stage,
        "stage_description": stage_info.get("description", ""),
        "progress": {
            "completed": completed_tools,
            "remaining": missing_tools,
        },
        "next_action": {
            "tool": missing_tools[0] if missing_tools else None,
            "description": _get_tool_description(missing_tools[0]) if missing_tools else "Stage complete",
        },
        "workflow_summary": {
            "papers_gathered": len(workflow.gathered_papers),
            "contexts_extracted": len(workflow.extracted_contexts),
            "target_metrics_set": workflow.target_metrics is not None,
            "ideas_generated": len(workflow.generated_ideas),
            "idea_approved": workflow.approved_idea_id is not None,
            "experiments_completed": len(workflow.completed_experiments),
            "tracked_runs": len(workflow.experiment_runs),
            "figures_generated": len(workflow.figures_generated),
            "github_linked": workflow.github_url is not None,
        },
        "target_metrics": target_info if target_info else None,
    }, indent=2)


def _get_user_action_instruction(workflow: WorkflowState, stage: str) -> str:
    """Get specific instruction for user action required."""
    if stage == "idea_approval":
        if workflow.generated_ideas:
            return (
                f"User must approve one of the generated ideas. "
                f"Type: APPROVE <idea_id> CODE <confirmation_code>"
            )
        return "Generate ideas first with generate_ideas()"
    return "User action required"


def _get_completed_tools(workflow: WorkflowState, required_tools: list[str]) -> list[str]:
    """Determine which required tools have been completed based on workflow state."""
    completed = []
    
    for tool in required_tools:
        if tool in ["fetch_arxiv_trending", "fetch_hf_trending", "search_papers"]:
            if len(workflow.gathered_papers) > 0:
                completed.append(tool)
        elif tool in ["extract_paper_metrics", "set_target_metrics_from_papers"]:
            if workflow.target_metrics is not None:
                completed.append(tool)
        elif tool in ["generate_ideas", "submit_idea"]:
            if len(workflow.generated_ideas) > 0:
                completed.append(tool)
        elif tool == "create_experiment_env":
            if "env_created" in workflow.completed_steps:
                completed.append(tool)
        elif tool == "setup_datasets":
            if "datasets_setup" in workflow.completed_steps:
                completed.append(tool)
        elif tool == "define_hypotheses":
            if workflow.hypotheses:
                completed.append(tool)
        elif tool == "run_experiment":
            if len(workflow.completed_experiments) > 0:
                completed.append(tool)
        elif tool == "run_baseline":
            if len(workflow.completed_experiments) > 1:
                completed.append(tool)
        elif tool == "log_experiment":
            if len(workflow.experiment_runs) > 0:
                completed.append(tool)
        elif tool == "get_experiment_history":
            if len(workflow.experiment_runs) > 0:
                completed.append(tool)
        elif tool in ["plot_comparison_bar", "plot_training_curves"]:
            if len(workflow.figures_generated) > 0:
                completed.append(tool)
        elif tool in ["compute_statistics", "check_significance", "compare_to_baselines"]:
            if workflow.experiment_results:
                completed.append(tool)
        elif tool == "check_paper_completeness":
            if len(workflow.paper_sections) > 0:
                completed.append(tool)
        elif tool in ["estimate_paper_structure", "format_results_table", "get_citations_for_topic", "create_paper_skeleton"]:
            if len(workflow.paper_sections) > 0:
                completed.append(tool)
        elif tool == "cast_to_format":
            if workflow.target_conference is not None:
                completed.append(tool)
        elif tool == "compile_paper":
            if workflow.target_conference is not None:
                completed.append(tool)
        elif tool == "create_github_repo":
            if workflow.github_url is not None:
                completed.append(tool)
        elif tool == "finalize_paper_with_github":
            if workflow.github_url is not None and workflow.target_conference is not None:
                completed.append(tool)
    
    return completed


def _get_tool_description(tool_name: str) -> str:
    """Get a brief description of what a tool does."""
    descriptions = {
        "fetch_arxiv_trending": "Fetch trending papers from arXiv",
        "fetch_hf_trending": "Fetch trending papers from HuggingFace",
        "search_papers": "Search for papers by topic",
        "extract_paper_metrics": "Extract word/figure counts from a paper",
        "set_target_metrics_from_papers": "Set target metrics from reference papers",
        "generate_ideas": "Generate research ideas from papers",
        "submit_idea": "Submit a generated idea for approval",
        "create_experiment_env": "Create Python environment for experiments",
        "setup_datasets": "Download and prepare datasets",
        "define_hypotheses": "Define testable hypotheses",
        "run_experiment": "Run an experiment script",
        "run_baseline": "Run baseline comparison",
        "log_experiment": "Log experiment results to database",
        "get_experiment_history": "Get all tracked experiment runs",
        "collect_metrics": "Collect metrics from experiment logs",
        "compute_statistics": "Compute statistical measures on results",
        "check_significance": "Run statistical significance tests",
        "plot_comparison_bar": "Generate comparison bar chart",
        "plot_training_curves": "Generate training curves plot",
        "compare_to_baselines": "Compare method to baselines",
        "estimate_paper_structure": "Estimate paper word/figure counts",
        "format_results_table": "Format results as LaTeX table",
        "get_citations_for_topic": "Get relevant citations",
        "create_paper_skeleton": "Create paper LaTeX structure",
        "check_paper_completeness": "Check if paper meets target length/figures",
        "expand_paper": "Get suggestions to expand a short paper",
        "cast_to_format": "Convert paper to conference format",
        "compile_paper": "Compile LaTeX to PDF",
        "create_github_repo": "Create GitHub repo and push code",
        "finalize_paper_with_github": "Add GitHub link to paper and compile",
    }
    return descriptions.get(tool_name, f"Execute {tool_name}")


async def mark_step_complete(step_name: str) -> str:
    """
    Mark a workflow step as complete.
    
    This is called internally by tools when they successfully complete.
    Users and AI should not need to call this directly.
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    if step_name not in workflow.completed_steps:
        workflow.completed_steps.append(step_name)
    
    await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "success": True,
        "step_completed": step_name,
        "total_completed": len(workflow.completed_steps),
    })


async def get_workflow_checklist() -> str:
    """
    Get a complete checklist of all workflow stages and their status.
    
    Useful for understanding overall progress and what remains to be done.
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    checklist = []
    current_found = False
    
    for stage_name, stage_info in WORKFLOW_STAGES.items():
        completion_check = stage_info.get("completion_check", lambda w: False)
        is_complete = completion_check(workflow)
        
        if stage_name == workflow.stage:
            current_found = True
            status = "IN_PROGRESS" if not is_complete else "COMPLETE"
        elif not current_found:
            status = "COMPLETE" if is_complete else "SKIPPED"
        else:
            status = "PENDING"
        
        checklist.append({
            "stage": stage_name,
            "description": stage_info.get("description", ""),
            "status": status,
            "required_tools": stage_info.get("required_tools", []),
            "is_blocking": stage_info.get("is_blocking", False),
        })
    
    return json.dumps({
        "project": current_project_obj.project_id,
        "current_stage": workflow.stage,
        "paper_iterations": workflow.paper_iterations,
        "github_url": workflow.github_url,
        "target_metrics": workflow.target_metrics,
        "checklist": checklist,
    }, indent=2)


async def _check_paper_completeness_internal(workflow: WorkflowState) -> dict:
    """Internal check for paper completeness against target metrics."""
    target_metrics = workflow.target_metrics or {}
    target_words = target_metrics.get("word_count", 5000)
    target_figures = target_metrics.get("figure_count", 6)
    target_tables = target_metrics.get("table_count", 3)
    
    current_words = 0
    for section_content in workflow.paper_sections.values():
        if isinstance(section_content, str):
            current_words += len(section_content.split())
    
    current_figures = len(workflow.figures_generated)
    
    word_ratio = current_words / max(1, target_words)
    sufficient = word_ratio >= MIN_WORD_COUNT_RATIO and current_figures >= MIN_FIGURE_COUNT
    
    suggestions = []
    if word_ratio < MIN_WORD_COUNT_RATIO:
        words_needed = int(target_words * MIN_WORD_COUNT_RATIO) - current_words
        suggestions.append(f"Add ~{words_needed} more words ({int((1 - word_ratio) * 100)}% short of target)")
    
    if current_figures < MIN_FIGURE_COUNT:
        suggestions.append(f"Add {MIN_FIGURE_COUNT - current_figures} more figures")
    
    if len(workflow.completed_experiments) < 3:
        suggestions.append("Run more experiments for additional results")
    
    return {
        "sufficient": sufficient,
        "current_words": current_words,
        "target_words": target_words,
        "word_ratio": round(word_ratio, 2),
        "current_figures": current_figures,
        "target_figures": target_figures,
        "suggestions": suggestions,
    }


async def set_target_metrics_from_papers(arxiv_ids: list[str]) -> str:
    """
    Extract and set target metrics from reference papers.
    
    This calculates average word count, figures, tables from given papers
    and sets them as targets for the current paper.
    
    Args:
        arxiv_ids: List of arXiv IDs to extract metrics from
    
    Returns:
        JSON with extracted and averaged metrics
    """
    from src.context.extractor import extract_metrics_from_papers
    
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    metrics = await extract_metrics_from_papers(arxiv_ids)
    
    workflow.target_metrics = metrics.to_dict()
    await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "success": True,
        "target_metrics": metrics.to_dict(),
        "papers_analyzed": len(arxiv_ids),
        "message": (
            f"Target metrics set: {metrics.word_count} words, "
            f"{metrics.figure_count} figures, {metrics.table_count} tables"
        ),
    }, indent=2)


async def check_and_expand_paper() -> str:
    """
    Main expansion loop - checks paper completeness and triggers expansion if needed.
    
    This should be called after each writing pass. If the paper is too short,
    it generates new hypotheses and queues background experiments.
    
    Returns:
        JSON with expansion status and queued experiments
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    completeness = await _check_paper_completeness_internal(workflow)
    
    if completeness.get("sufficient", False):
        return json.dumps({
            "status": "COMPLETE",
            "message": "Paper meets target metrics. Ready for formatting.",
            "completeness": completeness,
            "next_action": {
                "tool": "cast_to_format",
                "description": "Convert paper to conference LaTeX format",
            },
        }, indent=2)
    
    workflow.paper_iterations += 1
    await workflow_db.save_workflow(workflow)
    
    expansion_suggestions = _generate_expansion_suggestions(workflow, completeness)
    
    return json.dumps({
        "status": "NEEDS_EXPANSION",
        "iteration": workflow.paper_iterations,
        "completeness": completeness,
        "expansion_suggestions": expansion_suggestions,
        "ai_instruction": (
            "Paper is too short. You must:\n"
            "1. Generate additional hypotheses for unexplored aspects\n"
            "2. Run experiments to test these hypotheses (use git worktrees for parallel runs)\n"
            "3. Verify results with verify_hypothesis before adding claims\n"
            "4. Add verified results to paper with figures and analysis\n"
            "5. Call check_and_expand_paper() again to verify completeness"
        ),
        "required_actions": [
            "define_hypotheses - Generate new testable hypotheses",
            "run_experiment - Execute experiments in background",
            "verify_hypothesis - Statistically verify results",
            "plot_comparison_bar - Generate figures for new results",
            "write_paper_section - Add content to paper",
        ],
    }, indent=2)


def _generate_expansion_suggestions(workflow: WorkflowState, completeness: dict) -> list[dict]:
    """Generate suggestions for expanding a short paper."""
    suggestions = []
    
    word_gap = completeness.get("target_words", 5000) - completeness.get("current_words", 0)
    figure_gap = completeness.get("target_figures", 6) - completeness.get("current_figures", 0)
    
    if word_gap > 1000:
        suggestions.append({
            "type": "ablation_study",
            "description": "Add ablation study section analyzing component contributions",
            "estimated_words": 500,
            "estimated_figures": 1,
        })
    
    if word_gap > 500:
        suggestions.append({
            "type": "analysis",
            "description": "Add qualitative analysis with examples and visualizations",
            "estimated_words": 400,
            "estimated_figures": 2,
        })
    
    if figure_gap > 2:
        suggestions.append({
            "type": "sensitivity",
            "description": "Add hyperparameter sensitivity analysis with plots",
            "estimated_words": 300,
            "estimated_figures": 2,
        })
    
    if len(workflow.completed_experiments) < 5:
        suggestions.append({
            "type": "additional_baselines",
            "description": "Compare against additional baseline methods",
            "estimated_words": 300,
            "estimated_figures": 1,
        })
    
    if word_gap > 800:
        suggestions.append({
            "type": "failure_analysis",
            "description": "Add failure case analysis section",
            "estimated_words": 400,
            "estimated_figures": 1,
        })
    
    return suggestions


async def get_verified_claims() -> str:
    """
    Get all verified claims that can be included in the paper.
    
    Only claims that have passed hypothesis verification can be cited.
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    verified = getattr(workflow, "verified_hypotheses", {})
    
    claims = []
    for hypo_id, result in verified.items():
        if result.get("can_claim", False):
            claims.append({
                "hypothesis_id": hypo_id,
                "claim": result.get("statement", ""),
                "p_value": result.get("p_value"),
                "effect_size": result.get("effect_size"),
                "experiment_id": result.get("experiment_id"),
                "verified_at": result.get("timestamp"),
            })
    
    return json.dumps({
        "verified_claims": claims,
        "count": len(claims),
        "message": (
            "Only these verified claims can be made in the paper. "
            "Unverified results must go through verify_hypothesis first."
        ),
    }, indent=2)
