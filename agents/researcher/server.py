"""Researcher MCP Server - Paper discovery, idea generation, and approval.

This is a focused MCP with only research-phase tools.
Start a NEW Cursor chat with this MCP for the research phase.

Tools included:
- Project management (create, list, set current)
- Paper discovery (arXiv, HuggingFace, search)
- Context extraction
- Idea generation and approval
- Status checking

After approving an idea, switch to experimenter-mcp in a new chat.
"""

import json
import logging
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import from existing src/
from src.db.workflow import workflow_db
from src.db.papers_cache import papers_cache
from src.db.experiments_db import experiments_db
from src.project.manager import project_manager

# Import existing tools
from src.tools.aggregation import (
    fetch_arxiv_trending,
    fetch_hf_trending,
    search_papers,
    get_paper_details,
    clone_paper_code,
)
from src.tools.ideas import (
    generate_ideas,
    submit_idea,
    approve_idea,
    reject_idea,
    check_novelty,
    list_ideas,
)
from src.context.extractor import extract_paper_context


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("researcher-mcp")

server = Server("researcher-mcp")


TOOL_DEFINITIONS = [
    Tool(
        name="get_status",
        description="Get current workflow status and what to do next. ALWAYS call this first.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="create_project",
        description="Create a new research project. Required before any other actions.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Project name (use underscores, no spaces)"},
                "description": {"type": "string", "description": "Brief description of research focus"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="list_projects",
        description="List all existing research projects.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="set_current_project",
        description="Set the active project to work on.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project ID to activate"},
            },
            "required": ["project_id"],
        },
    ),
    Tool(
        name="fetch_arxiv_trending",
        description="Fetch trending papers from arXiv by category. Papers are saved to project context.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "arXiv category (e.g., cs.LG, cs.CL, cs.CV)"},
                "days": {"type": "integer", "description": "Look back N days", "default": 7},
                "max_results": {"type": "integer", "description": "Max papers to fetch", "default": 20},
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
                "topic": {"type": "string", "description": "Topic to filter by (optional)"},
                "max_results": {"type": "integer", "description": "Max papers", "default": 20},
            },
            "required": [],
        },
    ),
    Tool(
        name="search_papers",
        description="Search cached papers by query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_paper_details",
        description="Get full details of a specific paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper ID (arxiv ID or paper_id)"},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="extract_paper_context",
        description="Extract detailed context from a paper (sections, style, figures).",
        inputSchema={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID (e.g., 2401.12345)"},
            },
            "required": ["arxiv_id"],
        },
    ),
    Tool(
        name="generate_ideas",
        description="Get paper context for idea generation. Returns paper abstracts. YOU must then generate ideas and call submit_idea.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paper IDs to base ideas on",
                },
                "count": {"type": "integer", "description": "Number of ideas to generate", "default": 3},
                "focus": {"type": "string", "description": "Focus area (optional)"},
            },
            "required": ["paper_ids"],
        },
    ),
    Tool(
        name="submit_idea",
        description="Submit a research idea for user approval. Call after reading paper context.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Clear, specific idea title"},
                "description": {"type": "string", "description": "Detailed description (2-3 paragraphs)"},
                "motivation": {"type": "string", "description": "Why this is novel and important"},
                "source_papers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paper IDs this is based on",
                },
            },
            "required": ["title", "description", "motivation"],
        },
    ),
    Tool(
        name="list_ideas",
        description="List all generated ideas with their status.",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status (pending_approval, approved, rejected)"},
            },
            "required": [],
        },
    ),
    Tool(
        name="approve_idea",
        description="HUMAN ONLY - Approve an idea. Requires confirmation code shown only to user.",
        inputSchema={
            "type": "object",
            "properties": {
                "idea_id": {"type": "string"},
                "confirmation_code": {"type": "string", "description": "4-digit code from submit_idea"},
                "user_feedback": {"type": "string"},
            },
            "required": ["idea_id", "confirmation_code"],
        },
    ),
    Tool(
        name="reject_idea",
        description="Reject an idea with feedback.",
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
        name="check_novelty",
        description="Check if an idea is novel against existing literature.",
        inputSchema={
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "Idea description to check"},
            },
            "required": ["idea"],
        },
    ),
    Tool(
        name="handoff_to_experimenter",
        description="Mark research phase complete. Tells user to start new chat with experimenter-mcp.",
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
    
    elif name == "create_project":
        project = await project_manager.create_project(args["name"], args.get("description", ""))
        await workflow_db.create_workflow(project.project_id)
        return json.dumps({
            "success": True,
            "project_id": project.project_id,
            "path": str(project.root_path),
            "next": "Use fetch_arxiv_trending or fetch_hf_trending to gather papers",
        }, indent=2)
    
    elif name == "list_projects":
        projects = await project_manager.list_projects()
        return json.dumps({"projects": projects}, indent=2)
    
    elif name == "set_current_project":
        await project_manager.set_current_project(args["project_id"])
        return json.dumps({"success": True, "current_project": args["project_id"]})
    
    elif name == "fetch_arxiv_trending":
        return await fetch_arxiv_trending(
            args["category"],
            args.get("days", 7),
            args.get("max_results", 20),
        )
    
    elif name == "fetch_hf_trending":
        from src.tools.aggregation import fetch_hf_trending as hf_fetch
        return await hf_fetch(
            args.get("topic"),
            args.get("max_results", 20),
        )
    
    elif name == "search_papers":
        return await search_papers(args["query"], args.get("max_results", 10))
    
    elif name == "get_paper_details":
        return await get_paper_details(args["paper_id"])
    
    elif name == "extract_paper_context":
        return await extract_paper_context(args["arxiv_id"])
    
    elif name == "generate_ideas":
        return await generate_ideas(
            args["paper_ids"],
            args.get("count", 3),
            args.get("focus"),
        )
    
    elif name == "submit_idea":
        return await submit_idea(
            args["title"],
            args["description"],
            args["motivation"],
            args.get("source_papers"),
        )
    
    elif name == "list_ideas":
        return await list_ideas(args.get("status"))
    
    elif name == "approve_idea":
        return await approve_idea(
            args["idea_id"],
            args["confirmation_code"],
            args.get("user_feedback"),
        )
    
    elif name == "reject_idea":
        return await reject_idea(args["idea_id"], args.get("reason"))
    
    elif name == "check_novelty":
        return await check_novelty(args["idea"])
    
    elif name == "handoff_to_experimenter":
        return await _handoff_to_experimenter()
    
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


async def _get_status() -> str:
    """Get current workflow status."""
    current_project = await project_manager.get_current_project()
    
    if not current_project:
        return json.dumps({
            "status": "NO_PROJECT",
            "message": "No active project. Call create_project(name) first.",
            "agent": "researcher-mcp",
        }, indent=2)
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    
    if not workflow:
        return json.dumps({
            "status": "NO_WORKFLOW",
            "project": current_project.project_id,
            "message": "Project exists but no workflow. This shouldn't happen.",
        }, indent=2)
    
    # Check if research phase is complete
    if workflow.approved_idea_id:
        idea = await experiments_db.get_idea(workflow.approved_idea_id)
        return json.dumps({
            "status": "RESEARCH_COMPLETE",
            "agent": "researcher-mcp",
            "project": current_project.project_id,
            "approved_idea": {
                "id": workflow.approved_idea_id,
                "title": idea.title if idea else "Unknown",
            },
            "message": (
                "Research phase is COMPLETE. An idea has been approved.\n"
                ">>> START NEW CURSOR CHAT with experimenter-mcp to continue.\n"
                ">>> In the new chat, the experimenter will implement and run experiments."
            ),
            "next_agent": "experimenter-mcp",
        }, indent=2)
    
    # Research in progress
    progress = workflow.get_progress_summary()
    ideas = await experiments_db.list_ideas(status="pending_approval")
    
    return json.dumps({
        "status": "IN_PROGRESS",
        "agent": "researcher-mcp",
        "project": current_project.project_id,
        "stage": workflow.stage,
        "progress": {
            "papers_gathered": progress["papers_gathered"],
            "ideas_generated": progress["ideas_generated"],
            "pending_approval": len(ideas),
        },
        "pending_ideas": [
            {"id": i.idea_id, "title": i.title, "novelty": i.novelty_score}
            for i in ideas[:5]
        ],
        "next_action": _suggest_next_action(workflow, ideas),
    }, indent=2)


def _suggest_next_action(workflow, pending_ideas) -> str:
    """Suggest what to do next."""
    if not workflow.gathered_papers:
        return "Gather papers using fetch_arxiv_trending(category='cs.LG') or fetch_hf_trending()"
    
    if not workflow.generated_ideas:
        paper_ids = workflow.gathered_papers[:5]
        return f"Generate ideas using generate_ideas(paper_ids={paper_ids})"
    
    if pending_ideas:
        return (
            "Ideas await USER approval. Present them and wait for user to type:\n"
            "APPROVE <idea_id> CODE <confirmation_code>"
        )
    
    return "Generate more ideas or wait for user approval"


async def _handoff_to_experimenter() -> str:
    """Mark research complete and instruct to switch to experimenter."""
    current_project = await project_manager.get_current_project()
    
    if not current_project:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project.project_id)
    
    if not workflow or not workflow.approved_idea_id:
        return json.dumps({
            "error": "Cannot handoff - no approved idea",
            "message": "An idea must be approved before moving to experiments",
        })
    
    # Update workflow stage
    workflow.stage = "experiment_setup"
    await workflow_db.save_workflow(workflow)
    
    idea = await experiments_db.get_idea(workflow.approved_idea_id)
    
    return json.dumps({
        "status": "HANDOFF_READY",
        "message": (
            "Research phase complete!\n\n"
            ">>> CLOSE THIS CHAT\n"
            ">>> START NEW CURSOR CHAT\n"
            ">>> Use experimenter-mcp in the new chat\n\n"
            "The experimenter agent will:\n"
            "1. Read the approved idea from the database\n"
            "2. Implement experiment code\n"
            "3. Run experiments on GPU\n"
            "4. Verify results statistically"
        ),
        "approved_idea": {
            "id": workflow.approved_idea_id,
            "title": idea.title if idea else "Unknown",
        },
        "next_agent": "experimenter-mcp",
    }, indent=2)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
