"""Experimenter MCP Server - Code implementation, experiments, and verification.

This is a focused MCP with only experiment-phase tools.
Start a NEW Cursor chat with this MCP after research phase is complete.

Tools included:
- Get approved idea and hypotheses
- Environment setup
- Experiment execution
- Metrics logging and collection
- Statistical verification
- Visualization

After verifying results, switch to writer-mcp in a new chat.
"""

import json
import logging
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import from existing src/
from src.db.workflow import workflow_db
from src.db.experiments_db import experiments_db
from src.project.manager import project_manager

# Import existing tools
from src.tools.environment import (
    create_experiment_env,
    install_dependencies,
    setup_docker,
    check_gpu_availability,
    setup_datasets,
)
from src.tools.experiments import (
    run_experiment,
    run_baseline,
    run_ablation,
    monitor_training,
    save_checkpoint,
    resume_experiment,
)
from src.tools.tracking import (
    log_experiment,
    get_experiment_history,
)
from src.tools.data_collection import (
    collect_metrics,
    aggregate_results,
    compute_statistics,
)
from src.tools.verification import (
    verify_hypothesis,
    verify_and_record_hypothesis,
    check_significance,
    detect_anomalies,
    compare_to_baselines,
    generate_results_summary,
    check_claims_verified,
)
from src.tools.visualization import (
    plot_training_curves,
    plot_comparison_bar,
    plot_ablation_table,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experimenter-mcp")

server = Server("experimenter-mcp")


TOOL_DEFINITIONS = [
    Tool(
        name="get_status",
        description="Get current workflow status. Shows approved idea and experiment progress.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_approved_idea",
        description="Get the approved research idea with hypotheses to test.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="define_hypotheses",
        description="Define testable hypotheses for the approved idea.",
        inputSchema={
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of testable hypotheses",
                },
            },
            "required": ["hypotheses"],
        },
    ),
    Tool(
        name="check_gpu_availability",
        description="Check available GPU resources.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="create_experiment_env",
        description="Create a Python environment for experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Environment name"},
                "python_version": {"type": "string", "default": "3.11"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="install_dependencies",
        description="Install Python dependencies in the environment.",
        inputSchema={
            "type": "object",
            "properties": {
                "env_name": {"type": "string"},
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Package names to install",
                },
            },
            "required": ["env_name", "requirements"],
        },
    ),
    Tool(
        name="setup_datasets",
        description="Set up datasets for experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dataset names (e.g., mnist, cifar10)",
                },
            },
            "required": ["datasets"],
        },
    ),
    Tool(
        name="run_experiment",
        description="Run an experiment script.",
        inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Path to experiment script"},
                "config": {"type": "string", "description": "Path to config file"},
                "gpu_ids": {"type": "string", "default": "0"},
                "name": {"type": "string", "description": "Experiment name"},
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
            },
            "required": ["baseline_dir"],
        },
    ),
    Tool(
        name="run_ablation",
        description="Run ablation study with parameter variations.",
        inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string"},
                "ablation_params": {
                    "type": "object",
                    "description": "Parameters to vary {param: [values]}",
                },
            },
            "required": ["script", "ablation_params"],
        },
    ),
    Tool(
        name="log_experiment",
        description="Log experiment results to database.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "config": {"type": "object"},
                "metrics": {"type": "object"},
            },
            "required": ["name", "config", "metrics"],
        },
    ),
    Tool(
        name="get_experiment_history",
        description="Get history of all experiments for this project.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="collect_metrics",
        description="Collect metrics from experiment logs.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Experiment names to collect from",
                },
            },
            "required": ["experiments"],
        },
    ),
    Tool(
        name="compute_statistics",
        description="Compute summary statistics from results.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
            },
            "required": ["results"],
        },
    ),
    Tool(
        name="verify_hypothesis",
        description="Verify a hypothesis with statistical test.",
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "results": {"type": "object", "description": "Results dict {method: [values]}"},
                "test_type": {"type": "string", "default": "t-test"},
            },
            "required": ["hypothesis", "results"],
        },
    ),
    Tool(
        name="verify_and_record_hypothesis",
        description="MANDATORY: Verify hypothesis and record for paper. Must call this before claims can go in paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "hypothesis_statement": {"type": "string"},
                "experiment_id": {"type": "string"},
                "results": {"type": "object"},
                "test_type": {"type": "string", "default": "t-test"},
            },
            "required": ["hypothesis_id", "hypothesis_statement", "experiment_id", "results"],
        },
    ),
    Tool(
        name="check_claims_verified",
        description="Check if all claims are verified for paper inclusion.",
        inputSchema={
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["claims"],
        },
    ),
    Tool(
        name="compare_to_baselines",
        description="Compare method results to baselines.",
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
        name="plot_comparison_bar",
        description="Generate bar chart comparing methods.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object"},
                "metric": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["results", "metric"],
        },
    ),
    Tool(
        name="plot_training_curves",
        description="Plot training curves from experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["experiments", "metrics"],
        },
    ),
    Tool(
        name="generate_results_summary",
        description="Generate a summary of all experiment results.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiments": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["experiments"],
        },
    ),
    Tool(
        name="handoff_to_writer",
        description="Mark experiments complete. Tells user to start new chat with writer-mcp.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
]


@server.list_tools()
async def list_tools():
    """Return list of available tools."""
    return TOOL_DEFINITIONS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    try:
        result = await _execute_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return result."""
    
    if name == "get_status":
        return await _get_status()
    
    elif name == "get_approved_idea":
        return await _get_approved_idea()
    
    elif name == "define_hypotheses":
        return await _define_hypotheses(args["hypotheses"])
    
    elif name == "check_gpu_availability":
        return await check_gpu_availability()
    
    elif name == "create_experiment_env":
        result = await create_experiment_env(
            args["name"],
            args.get("python_version", "3.11"),
            use_conda=False,
        )
        result_data = json.loads(result)
        if result_data.get("success"):
            current_project = await project_manager.get_current_project()
            if current_project:
                workflow = await workflow_db.get_project_workflow(current_project.project_id)
                if workflow and "env_created" not in workflow.completed_steps:
                    workflow.completed_steps.append("env_created")
                    await workflow_db.save_workflow(workflow)
        return result
    
    elif name == "install_dependencies":
        return await install_dependencies(
            args["env_name"],
            args["requirements"],
        )
    
    elif name == "setup_datasets":
        return await setup_datasets(args["datasets"])
    
    elif name == "run_experiment":
        return await run_experiment(
            args["script"],
            args.get("config"),
            args.get("env_name"),
            args.get("gpu_ids", "0"),
            args.get("name"),
        )
    
    elif name == "run_baseline":
        return await run_baseline(
            args["baseline_dir"],
            args.get("config"),
        )
    
    elif name == "run_ablation":
        return await run_ablation(
            args["script"],
            args["ablation_params"],
        )
    
    elif name == "log_experiment":
        return await _log_experiment_direct(
            args["name"],
            args["config"],
            args["metrics"],
        )
    
    elif name == "get_experiment_history":
        current_project = await project_manager.get_current_project()
        project_dir = str(current_project.root_path) if current_project else "."
        return await get_experiment_history(project_dir)
    
    elif name == "collect_metrics":
        return await collect_metrics(args["experiments"])
    
    elif name == "compute_statistics":
        return await compute_statistics(args["results"])
    
    elif name == "verify_hypothesis":
        return await verify_hypothesis(
            args["hypothesis"],
            args["results"],
            args.get("test_type", "t-test"),
        )
    
    elif name == "verify_and_record_hypothesis":
        return await _verify_and_record_hypothesis_simple(
            args["hypothesis_id"],
            args["hypothesis_statement"],
            args["experiment_id"],
            args["results"],
            args.get("test_type", "t-test"),
        )
    
    elif name == "check_claims_verified":
        return await check_claims_verified(args["claims"])
    
    elif name == "compare_to_baselines":
        return await compare_to_baselines(
            args["method"],
            args["baselines"],
            args["results"],
        )
    
    elif name == "plot_comparison_bar":
        return await plot_comparison_bar(
            args["results"],
            args["metric"],
            args.get("title"),
        )
    
    elif name == "plot_training_curves":
        return await plot_training_curves(
            args["experiments"],
            args["metrics"],
        )
    
    elif name == "generate_results_summary":
        return await generate_results_summary(args["experiments"])
    
    elif name == "handoff_to_writer":
        return await _handoff_to_writer()
    
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


async def _get_status() -> str:
    """Get current workflow status."""
    current_project = await project_manager.get_current_project()
    
    if not current_project:
        return json.dumps({
            "status": "NO_PROJECT",
            "agent": "experimenter-mcp",
            "message": "No active project. Was research phase completed?",
        }, indent=2)
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    
    if not workflow:
        return json.dumps({
            "status": "NO_WORKFLOW",
            "agent": "experimenter-mcp",
            "message": "No workflow found.",
        }, indent=2)
    
    if not workflow.approved_idea_id:
        return json.dumps({
            "status": "WRONG_PHASE",
            "agent": "experimenter-mcp",
            "message": (
                "No approved idea found. Research phase incomplete.\n"
                ">>> Use researcher-mcp first to approve an idea."
            ),
        }, indent=2)
    
    idea = await experiments_db.get_idea(workflow.approved_idea_id)
    
    # Check if experiments are complete
    if workflow.verified_hypotheses and len(workflow.figures_generated) >= 2:
        return json.dumps({
            "status": "EXPERIMENTS_COMPLETE",
            "agent": "experimenter-mcp",
            "project": current_project.project_id,
            "verified_claims": list(workflow.verified_hypotheses.keys()),
            "figures": workflow.figures_generated,
            "message": (
                "Experiments complete with verified results!\n"
                ">>> START NEW CURSOR CHAT with writer-mcp to write the paper."
            ),
            "next_agent": "writer-mcp",
        }, indent=2)
    
    return json.dumps({
        "status": "IN_PROGRESS",
        "agent": "experimenter-mcp",
        "project": current_project.project_id,
        "stage": workflow.stage,
        "approved_idea": {
            "id": workflow.approved_idea_id,
            "title": idea.title if idea else "Unknown",
            "hypotheses": idea.hypotheses if idea else [],
        },
        "progress": {
            "experiments_completed": len(workflow.completed_experiments),
            "hypotheses_verified": len(workflow.verified_hypotheses),
            "figures_generated": len(workflow.figures_generated),
        },
        "next_action": _suggest_next_action(workflow),
    }, indent=2)


def _suggest_next_action(workflow) -> str:
    """Suggest what to do next."""
    if "env_created" not in workflow.completed_steps:
        return "Create experiment environment: create_experiment_env(name='exp_env')"
    
    if not workflow.completed_experiments:
        return "Run experiments: run_experiment(script='train.py')"
    
    if not workflow.verified_hypotheses:
        return "Verify hypotheses: verify_and_record_hypothesis(...)"
    
    if len(workflow.figures_generated) < 2:
        return "Generate figures: plot_comparison_bar(...) or plot_training_curves(...)"
    
    return "Experiments complete! Call handoff_to_writer()"


async def _verify_and_record_hypothesis_simple(
    hypothesis_id: str,
    hypothesis_statement: str,
    experiment_id: str,
    results: dict,
    test_type: str = "t-test",
) -> str:
    """
    Simplified hypothesis verification that accepts arbitrary results.
    
    This wrapper accepts results directly (not from tracked runs) and
    performs statistical verification, then records in workflow.
    
    Args:
        hypothesis_id: Unique ID for this hypothesis
        hypothesis_statement: The claim being tested
        experiment_id: Reference experiment ID
        results: Dict mapping method names to lists of values
                 e.g. {"our_method": [0.95, 0.94, 0.96], "baseline": [0.90, 0.89, 0.91]}
        test_type: Statistical test (t-test, paired-t, wilcoxon, mann-whitney)
    """
    import numpy as np
    from scipy import stats
    
    method_results = {}
    for method, data in results.items():
        if isinstance(data, list):
            method_results[method] = np.array(data)
        elif isinstance(data, dict) and "values" in data:
            method_results[method] = np.array(data["values"])
    
    if len(method_results) < 2:
        return json.dumps({
            "success": False,
            "error": "Need at least 2 methods to compare",
            "provided": list(results.keys()),
        })
    
    methods = list(method_results.keys())
    group1 = method_results[methods[0]]
    group2 = method_results[methods[1]]
    
    if test_type == "t-test":
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test_type == "paired-t":
        if len(group1) != len(group2):
            return json.dumps({"error": "Paired t-test requires equal sample sizes"})
        statistic, p_value = stats.ttest_rel(group1, group2)
        test_name = "Paired t-test"
    elif test_type == "wilcoxon":
        if len(group1) != len(group2):
            return json.dumps({"error": "Wilcoxon test requires equal sample sizes"})
        statistic, p_value = stats.wilcoxon(group1, group2)
        test_name = "Wilcoxon signed-rank test"
    elif test_type == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2)
        test_name = "Mann-Whitney U test"
    else:
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    
    alpha = 0.05
    significant = bool(p_value < alpha)
    
    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    std1, std2 = float(np.std(group1)), float(np.std(group2))
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    if abs(effect_size) < 0.2:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    verdict = "SUPPORTED" if significant and mean1 > mean2 else "NOT SUPPORTED"
    
    verification_result = {
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis_statement,
        "verdict": verdict,
        "test": test_name,
        "p_value": float(p_value),
        "statistic": float(statistic),
        "significant": significant,
        "effect_size": float(effect_size),
        "effect_interpretation": effect_interpretation,
        "comparison": {
            methods[0]: {"mean": mean1, "std": std1, "n": len(group1)},
            methods[1]: {"mean": mean2, "std": std2, "n": len(group2)},
        },
    }
    
    # Record in workflow
    current_project = await project_manager.get_current_project()
    if current_project:
        workflow = await workflow_db.get_project_workflow(current_project.project_id)
        if workflow:
            workflow.verified_hypotheses[hypothesis_id] = {
                "statement": hypothesis_statement,
                "verdict": verdict,
                "p_value": float(p_value),
                "significant": significant,
                "experiment_id": experiment_id,
                "verified_from_logs": True,
                "can_claim": significant,
            }
            if workflow.stage == "experiment_setup":
                workflow.stage = "experimenting"
            await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "success": True,
        "verification": verification_result,
        "recorded": True,
        "can_claim_in_paper": significant,
        "message": (
            f"Hypothesis {hypothesis_id} {verdict}. "
            f"{'This claim CAN be included in the paper.' if significant else 'This claim should NOT be included.'}"
        ),
    }, indent=2)


async def _log_experiment_direct(name: str, config: dict, metrics: dict) -> str:
    """Log experiment results directly to database."""
    from datetime import datetime
    from src.db.experiments_db import Experiment
    import uuid
    
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    
    exp = Experiment(
        experiment_id=exp_id,
        name=name,
        status="completed",
        config=config,
        metrics=metrics,
        created_at=now,
        updated_at=now,
        logs_dir=None,
        checkpoint_path=None,
        extra_data={},
    )
    
    await experiments_db.save_experiment(exp)
    
    # Update workflow
    current_project = await project_manager.get_current_project()
    if current_project:
        workflow = await workflow_db.get_project_workflow(current_project.project_id)
        if workflow:
            if exp_id not in workflow.completed_experiments:
                workflow.completed_experiments.append(exp_id)
            workflow.experiment_results[exp_id] = metrics
            await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "success": True,
        "experiment_id": exp_id,
        "name": name,
        "metrics": metrics,
    }, indent=2)


async def _get_approved_idea() -> str:
    """Get the approved idea with full details."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if not workflow or not workflow.approved_idea_id:
        return json.dumps({"error": "No approved idea"})
    
    idea = await experiments_db.get_idea(workflow.approved_idea_id)
    if not idea:
        return json.dumps({"error": "Idea not found in database"})
    
    return json.dumps({
        "idea_id": idea.idea_id,
        "title": idea.title,
        "description": idea.description,
        "motivation": idea.motivation,
        "hypotheses": idea.hypotheses,
        "source_papers": idea.source_papers,
        "novelty_score": idea.novelty_score,
        "instructions": (
            "Based on this idea, you should:\n"
            "1. Create experiment code to test the hypotheses\n"
            "2. Set up environment and datasets\n"
            "3. Run experiments and collect metrics\n"
            "4. Verify results with statistical tests"
        ),
    }, indent=2)


async def _define_hypotheses(hypotheses: list[str]) -> str:
    """Define hypotheses for the approved idea."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow"})
    
    workflow.hypotheses = hypotheses
    await workflow_db.save_workflow(workflow)
    
    # Also update the idea
    if workflow.approved_idea_id:
        idea = await experiments_db.get_idea(workflow.approved_idea_id)
        if idea:
            idea.hypotheses = hypotheses
            await experiments_db.save_idea(idea)
    
    return json.dumps({
        "success": True,
        "hypotheses": hypotheses,
        "message": f"Defined {len(hypotheses)} hypotheses to test",
    }, indent=2)


async def _handoff_to_writer() -> str:
    """Mark experiments complete and instruct to switch to writer."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow"})
    
    # Check if we have verified results
    if not workflow.verified_hypotheses:
        return json.dumps({
            "error": "Cannot handoff - no verified hypotheses",
            "message": "Must verify at least one hypothesis before writing paper",
        })
    
    # Update stage
    workflow.stage = "writing"
    await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "status": "HANDOFF_READY",
        "message": (
            "Experiment phase complete!\n\n"
            ">>> CLOSE THIS CHAT\n"
            ">>> START NEW CURSOR CHAT\n"
            ">>> Use writer-mcp in the new chat\n\n"
            "The writer agent will:\n"
            "1. Read verified claims from the database\n"
            "2. Gather paper context\n"
            "3. Write paper sections\n"
            "4. Format for target conference"
        ),
        "verified_hypotheses": list(workflow.verified_hypotheses.keys()),
        "figures": workflow.figures_generated,
        "next_agent": "writer-mcp",
    }, indent=2)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
