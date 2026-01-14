"""Writer MCP Server - Paper writing and conference formatting.

This is a focused MCP with only writing-phase tools.
Start a NEW Cursor chat with this MCP after experiments are complete.

Tools included:
- Get verified results and figures
- Paper structure planning
- LaTeX formatting helpers
- Citation management
- Conference formatting (ICML, NeurIPS, etc.)
- PDF compilation
"""

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import from existing src/
from src.db.workflow import workflow_db
from src.db.experiments_db import experiments_db
from src.db.papers_cache import papers_cache
from src.project.manager import project_manager

# Import existing tools
from src.tools.writing import (
    get_project_writing_context,
    extract_style_from_context,
    estimate_paper_structure,
    format_results_table,
    format_algorithm,
    get_citations_for_topic,
    get_verified_claims_for_writing,
    create_paper_skeleton,
    check_paper_completeness,
)
from src.tools.formatting import (
    cast_to_format,
    compile_paper,
    list_conferences,
    get_conference_requirements,
)
from src.context.extractor import (
    extract_paper_context,
    extract_paper_metrics,
    extract_metrics_from_papers,
    extract_writing_style_context,
    get_paper_context_for_writing,
    PaperMetrics,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("writer-mcp")

server = Server("writer-mcp")


TOOL_DEFINITIONS = [
    Tool(
        name="get_status",
        description="Get current workflow status. Shows verified results and writing progress.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_verified_claims",
        description="Get all verified claims from experiments. ONLY these can go in the paper.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_project_writing_context",
        description="Get all gathered papers and context for writing. Use to match style.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="extract_style_from_context",
        description="Analyze gathered papers to extract writing style patterns.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="extract_paper_metrics",
        description="Extract figures/tables/formulas/word counts per section from a specific paper. Use to understand paper structure.",
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID (e.g., 2401.12345)"},
            },
            "required": ["arxiv_id"],
        },
    ),
    Tool(
        name="extract_target_metrics",
        description="IMPORTANT: Extract and average metrics from ALL gathered papers. Returns target word counts, figure counts, table counts per section. Call this FIRST before writing.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="extract_writing_style",
        description="IMPORTANT: Extract writing style from gathered papers. Returns sentence length, voice (passive/active), formality, AND example paragraphs to match. Call this BEFORE writing each section.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific paper IDs to analyze. If empty, uses all gathered papers.",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="get_full_writing_context",
        description="Get COMPLETE writing context: metrics + style + example paragraphs from gathered papers. Call this once before starting to write the paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific paper IDs. If empty, uses all gathered papers.",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="estimate_paper_structure",
        description="Estimate recommended paper structure based on target pages.",
        inputSchema={
            "type": "object",
            "properties": {
                "target_pages": {"type": "integer", "default": 9},
                "conference": {"type": "string", "default": "icml"},
            },
            "required": [],
        },
    ),
    Tool(
        name="create_paper_skeleton",
        description="Create a paper skeleton with sections structure.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Paper title"},
                "conference": {"type": "string", "default": "icml", "description": "Conference format"},
                "sections": {"type": "array", "items": {"type": "string"}, "description": "Optional list of section names"},
            },
            "required": ["title"],
        },
    ),
    Tool(
        name="format_results_table",
        description="Format experiment results as a LaTeX table.",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {"type": "object", "description": "Results dict {method: {metric: value}}"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "caption": {"type": "string"},
            },
            "required": ["results", "metrics"],
        },
    ),
    Tool(
        name="format_algorithm",
        description="Format a method as a LaTeX algorithm block.",
        inputSchema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}, "description": "Algorithm steps"},
                "caption": {"type": "string", "description": "Algorithm caption/title"},
                "label": {"type": "string", "default": "alg:main", "description": "LaTeX label"},
            },
            "required": ["steps", "caption"],
        },
    ),
    Tool(
        name="get_citations_for_topic",
        description="Get citations for papers related to a topic.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "max_citations": {"type": "integer", "default": 5},
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="list_conferences",
        description="List all supported conference formats.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_conference_requirements",
        description="Get formatting requirements for a specific conference.",
        inputSchema={
            "type": "object",
            "properties": {
                "conference": {"type": "string", "description": "Conference name (e.g., icml, neurips)"},
            },
            "required": ["conference"],
        },
    ),
    Tool(
        name="cast_to_format",
        description="Convert paper content to conference format LaTeX.",
        inputSchema={
            "type": "object",
            "properties": {
                "format_name": {"type": "string", "description": "Conference format (icml, neurips, cvpr, aaai, acl)"},
                "content": {
                    "type": "object",
                    "description": "Paper content dict with title, abstract, sections, etc.",
                },
                "output_dir": {"type": "string"},
            },
            "required": ["format_name", "content"],
        },
    ),
    Tool(
        name="check_paper_completeness",
        description="Check if paper has all required sections and meets targets.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="compile_paper",
        description="Compile LaTeX to PDF.",
        inputSchema={
            "type": "object",
            "properties": {
                "tex_path": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            "required": ["tex_path"],
        },
    ),
    Tool(
        name="extract_paper_context",
        description="Extract structure from a reference paper for style matching.",
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string"},
            },
            "required": ["arxiv_id"],
        },
    ),
    Tool(
        name="save_paper_draft",
        description="Save current paper draft to project.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "object"},
                "draft_name": {"type": "string"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="mark_complete",
        description="Mark paper as complete. Final step of the pipeline.",
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
    
    elif name == "get_verified_claims":
        return await get_verified_claims_for_writing()
    
    elif name == "get_project_writing_context":
        return await get_project_writing_context()
    
    elif name == "extract_style_from_context":
        return await extract_style_from_context()
    
    elif name == "extract_paper_metrics":
        return await _extract_paper_metrics(args["arxiv_id"])
    
    elif name == "extract_target_metrics":
        return await _extract_target_metrics()
    
    elif name == "extract_writing_style":
        return await _extract_writing_style(args.get("paper_ids", []))
    
    elif name == "get_full_writing_context":
        return await _get_full_writing_context(args.get("paper_ids", []))
    
    elif name == "estimate_paper_structure":
        return await estimate_paper_structure(
            args.get("target_pages", 9),
            args.get("conference", "icml"),
        )
    
    elif name == "create_paper_skeleton":
        return await create_paper_skeleton(
            args["title"],
            args.get("conference", "icml"),
            args.get("sections"),
        )
    
    elif name == "format_results_table":
        results = args["results"]
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                return json.dumps({"error": "results must be a valid JSON object"})
        return await format_results_table(
            results,
            args["metrics"],
            args.get("caption"),
        )
    
    elif name == "format_algorithm":
        return await format_algorithm(
            args["steps"],
            args.get("caption", "Algorithm"),
            args.get("label", "alg:main"),
        )
    
    elif name == "get_citations_for_topic":
        return await get_citations_for_topic(
            args["topic"],
            args.get("max_citations", 5),
        )
    
    elif name == "list_conferences":
        return await list_conferences()
    
    elif name == "get_conference_requirements":
        return await get_conference_requirements(args["conference"])
    
    elif name == "cast_to_format":
        content = args.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "content must be a valid JSON object, not a string",
                    "received": content[:100] if len(content) > 100 else content,
                })
        return await cast_to_format(
            content,
            args.get("format_name", "icml"),
            args.get("output_dir", "./output"),
        )
    
    elif name == "check_paper_completeness":
        current_project = await project_manager.get_current_project()
        if current_project:
            draft_path = Path(current_project.root_path) / "papers" / "drafts" / "latest.tex"
            if draft_path.exists():
                return await check_paper_completeness(latex_file=str(draft_path))
            for tex_file in (Path(current_project.root_path) / "papers" / "drafts").glob("*.tex"):
                return await check_paper_completeness(latex_file=str(tex_file))
        return await check_paper_completeness()
    
    elif name == "compile_paper":
        return await compile_paper(
            args["tex_path"],
            args.get("output_dir"),
        )
    
    elif name == "extract_paper_context":
        return await extract_paper_context(args["arxiv_id"])
    
    elif name == "save_paper_draft":
        return await _save_paper_draft(args["content"], args.get("draft_name"))
    
    elif name == "mark_complete":
        return await _mark_complete()
    
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


async def _extract_paper_metrics(arxiv_id: str) -> str:
    """Extract metrics from a single paper."""
    metrics = await extract_paper_metrics(arxiv_id)
    return json.dumps({
        "arxiv_id": arxiv_id,
        "metrics": metrics.to_dict(),
        "usage": "Use these as reference for your paper structure",
    }, indent=2)


async def _extract_target_metrics() -> str:
    """Extract and average metrics from all gathered papers."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    context_dir = current_project.context_dir
    if not context_dir.exists():
        return json.dumps({"error": "No papers gathered yet"})
    
    arxiv_ids = []
    for context_file in context_dir.glob("*.json"):
        try:
            paper_data = json.loads(context_file.read_text())
            arxiv_id = paper_data.get("arxiv_id")
            if arxiv_id:
                arxiv_ids.append(arxiv_id)
        except Exception:
            continue
    
    if not arxiv_ids:
        return json.dumps({
            "error": "No arXiv papers in context",
            "default": PaperMetrics.default().to_dict(),
        })
    
    metrics = await extract_metrics_from_papers(arxiv_ids)
    
    # Store in workflow for reference
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if workflow:
        workflow.target_metrics = metrics.to_dict()
        await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "papers_analyzed": len(arxiv_ids),
        "target_metrics": metrics.to_dict(),
        "recommendations": {
            "total_words": f"~{metrics.word_count} words",
            "figures": f"{metrics.figure_count} figures",
            "tables": f"{metrics.table_count} tables",
            "citations": f"~{metrics.citation_count} citations",
            "pages": f"~{metrics.page_count} pages",
        },
        "section_targets": metrics.section_lengths,
        "usage": "Match these metrics when writing your paper",
    }, indent=2)


async def _extract_writing_style(paper_ids: list[str]) -> str:
    """Extract writing style from papers."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    arxiv_ids = []
    
    if paper_ids:
        for pid in paper_ids:
            if pid.startswith("arxiv:"):
                arxiv_ids.append(pid[6:])
            else:
                arxiv_ids.append(pid)
    else:
        context_dir = current_project.context_dir
        if context_dir.exists():
            for context_file in context_dir.glob("*.json"):
                try:
                    paper_data = json.loads(context_file.read_text())
                    arxiv_id = paper_data.get("arxiv_id")
                    if arxiv_id:
                        arxiv_ids.append(arxiv_id)
                except Exception:
                    continue
    
    if not arxiv_ids:
        return json.dumps({
            "error": "No papers to analyze",
            "default_style": {
                "avg_sentence_length": 20,
                "passive_voice_ratio": 0.3,
                "first_person": True,
                "formality": "high",
            },
        })
    
    style_context = await extract_writing_style_context(arxiv_ids[:5])
    
    return json.dumps({
        "style_context": style_context,
        "usage": (
            "MATCH THIS STYLE when writing:\n"
            "1. Use similar sentence length\n"
            "2. Match the voice (we/passive)\n"
            "3. Match formality level\n"
            "4. Study the example paragraphs below and COPY their style"
        ),
    }, indent=2, ensure_ascii=False)


async def _get_full_writing_context(paper_ids: list[str]) -> str:
    """Get complete writing context: metrics + style + examples."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    arxiv_ids = []
    
    if paper_ids:
        for pid in paper_ids:
            if pid.startswith("arxiv:"):
                arxiv_ids.append(pid[6:])
            else:
                arxiv_ids.append(pid)
    else:
        context_dir = current_project.context_dir
        if context_dir.exists():
            for context_file in context_dir.glob("*.json"):
                try:
                    paper_data = json.loads(context_file.read_text())
                    arxiv_id = paper_data.get("arxiv_id")
                    if arxiv_id:
                        arxiv_ids.append(arxiv_id)
                except Exception:
                    continue
    
    if not arxiv_ids:
        return json.dumps({
            "error": "No papers in context",
            "message": "Gather papers first using researcher-mcp",
        })
    
    context = await get_paper_context_for_writing(arxiv_ids[:5])
    
    # Store target metrics in workflow
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if workflow and context.get("target_metrics"):
        workflow.target_metrics = context["target_metrics"]
        await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "papers_analyzed": len(arxiv_ids),
        "full_context": context,
        "instructions": (
            "USE THIS CONTEXT TO WRITE YOUR PAPER:\n\n"
            "1. METRICS - Match these targets:\n"
            f"   - Word count: {context.get('target_metrics', {}).get('word_count', 5000)}\n"
            f"   - Figures: {context.get('target_metrics', {}).get('figure_count', 6)}\n"
            f"   - Tables: {context.get('target_metrics', {}).get('table_count', 3)}\n\n"
            "2. STYLE - Match the writing style:\n"
            "   - Study the example paragraphs\n"
            "   - Match sentence length and formality\n"
            "   - Use similar voice (we/passive)\n\n"
            "3. STRUCTURE - Follow section word counts from reference papers"
        ),
    }, indent=2, ensure_ascii=False)


async def _get_status() -> str:
    """Get current workflow status."""
    current_project = await project_manager.get_current_project()
    
    if not current_project:
        return json.dumps({
            "status": "NO_PROJECT",
            "agent": "writer-mcp",
            "message": "No active project.",
        }, indent=2)
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    
    if not workflow:
        return json.dumps({
            "status": "NO_WORKFLOW",
            "agent": "writer-mcp",
        }, indent=2)
    
    if not workflow.verified_hypotheses:
        return json.dumps({
            "status": "WRONG_PHASE",
            "agent": "writer-mcp",
            "message": (
                "No verified hypotheses found. Experiments incomplete.\n"
                ">>> Use experimenter-mcp first to verify results."
            ),
        }, indent=2)
    
    idea = await experiments_db.get_idea(workflow.approved_idea_id) if workflow.approved_idea_id else None
    
    # Check if paper is complete
    if workflow.stage == "completed":
        return json.dumps({
            "status": "COMPLETE",
            "agent": "writer-mcp",
            "project": current_project.project_id,
            "message": "Paper is complete!",
            "paper_sections": list(workflow.paper_sections.keys()),
        }, indent=2)
    
    return json.dumps({
        "status": "IN_PROGRESS",
        "agent": "writer-mcp",
        "project": current_project.project_id,
        "stage": workflow.stage,
        "approved_idea": {
            "title": idea.title if idea else "Unknown",
        },
        "verified_claims": list(workflow.verified_hypotheses.keys()),
        "figures_available": workflow.figures_generated,
        "sections_written": list(workflow.paper_sections.keys()),
        "target_conference": workflow.target_conference or "icml",
        "next_action": _suggest_next_action(workflow),
    }, indent=2)


def _suggest_next_action(workflow) -> str:
    """Suggest what to do next."""
    if not workflow.paper_sections:
        return "Get context: get_project_writing_context(), then start writing sections"
    
    required = {"introduction", "method", "experiments", "conclusion"}
    written = set(workflow.paper_sections.keys())
    remaining = required - written
    
    if remaining:
        return f"Write remaining sections: {remaining}"
    
    if not workflow.target_conference:
        return "Format paper: cast_to_format(format_name='icml', content=...)"
    
    return "Compile and complete: compile_paper(...), then mark_complete()"


async def _save_paper_draft(content: dict, draft_name: Optional[str] = None) -> str:
    """Save paper draft to project."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    drafts_dir = current_project.root_path / "papers" / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    name = draft_name or f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    draft_path = drafts_dir / f"{name}.json"
    
    draft_path.write_text(json.dumps(content, indent=2, ensure_ascii=False))
    
    # Update workflow
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if workflow:
        if "sections" in content:
            for section in content["sections"]:
                if isinstance(section, dict) and "name" in section:
                    workflow.paper_sections[section["name"]] = section.get("content", "")
            await workflow_db.save_workflow(workflow)
    
    return json.dumps({
        "success": True,
        "draft_saved": str(draft_path),
        "sections": list(content.get("sections", [])),
    }, indent=2)


async def _mark_complete() -> str:
    """Mark paper as complete with validation."""
    current_project = await project_manager.get_current_project()
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow"})
    
    # Validation: check minimum requirements
    warnings = []
    
    if len(workflow.verified_hypotheses) == 0:
        warnings.append("No verified hypotheses - paper has no validated claims")
    
    if len(workflow.figures_generated) == 0:
        warnings.append("No figures generated - paper may lack visual results")
    
    # Check for draft files
    draft_dir = Path(current_project.root_path) / "papers" / "drafts"
    has_draft = draft_dir.exists() and any(draft_dir.glob("*.tex"))
    if not has_draft:
        return json.dumps({
            "error": "VALIDATION_FAILED",
            "message": "Cannot mark complete: no paper draft found",
            "action_required": "Create paper draft using cast_to_format() or save_paper_draft()",
        }, indent=2)
    
    workflow.stage = "completed"
    await workflow_db.save_workflow(workflow)
    
    idea = await experiments_db.get_idea(workflow.approved_idea_id) if workflow.approved_idea_id else None
    
    result = {
        "status": "PIPELINE_COMPLETE",
        "message": (
            "Congratulations! Research pipeline complete.\n\n"
            "Summary:\n"
            f"- Project: {current_project.project_id}\n"
            f"- Idea: {idea.title if idea else 'Unknown'}\n"
            f"- Verified claims: {len(workflow.verified_hypotheses)}\n"
            f"- Figures: {len(workflow.figures_generated)}\n"
            f"- Sections: {list(workflow.paper_sections.keys())}\n"
            f"- Conference: {workflow.target_conference or 'Not set'}"
        ),
        "project_path": str(current_project.root_path),
        "papers_dir": str(Path(current_project.root_path) / "papers"),
    }
    
    if warnings:
        result["warnings"] = warnings
    
    return json.dumps(result, indent=2)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
