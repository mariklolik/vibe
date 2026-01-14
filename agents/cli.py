#!/usr/bin/env python3
"""CLI for multi-agent research pipeline.

This CLI helps orchestrate the three-agent system:
1. researcher-mcp - Paper discovery, idea generation
2. experimenter-mcp - Experiments, verification
3. writer-mcp - Paper writing, formatting

Usage:
    python -m agents.cli status     # Show current status and next agent
    python -m agents.cli projects   # List all projects
    python -m agents.cli set <id>   # Set current project
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.workflow import workflow_db
from src.db.experiments_db import experiments_db
from src.project.manager import project_manager


async def status():
    """Show current pipeline status."""
    current = await project_manager.get_current_project()
    
    if not current:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  NO ACTIVE PROJECT                                           ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  Start a new Cursor chat with researcher-mcp               ║")
        print("║  Create a project: create_project(name='my_research')       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return
    
    workflow = await workflow_db.get_project_workflow(current.project_id)
    
    if not workflow:
        print(f"\nProject: {current.project_id}")
        print("Status: No workflow (create one with researcher-mcp)")
        return
    
    progress = workflow.get_progress_summary()
    
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Project: {current.project_id:<50} ║")
    print(f"║  Stage: {workflow.stage:<52} ║")
    print(f"║  Progress: {progress['progress_percent']}%{' ' * 47}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    if not workflow.approved_idea_id:
        agent = "researcher-mcp"
        next_action = "Gather papers, generate ideas, get approval"
        print(f"║  Current Agent: {agent:<44} ║")
        print(f"║  Papers: {progress['papers_gathered']:<4} Ideas: {progress['ideas_generated']:<4} Approved: ✗{' ' * 24}║")
    elif not workflow.verified_hypotheses:
        agent = "experimenter-mcp"
        next_action = "Run experiments, verify hypotheses"
        print(f"║  Current Agent: {agent:<44} ║")
        print(f"║  Experiments: {len(workflow.completed_experiments):<4} Verified: {len(workflow.verified_hypotheses):<4}{' ' * 24}║")
    elif workflow.stage != "completed":
        agent = "writer-mcp"
        next_action = "Write paper sections, format for conference"
        print(f"║  Current Agent: {agent:<44} ║")
        print(f"║  Sections: {len(workflow.paper_sections):<4} Figures: {len(workflow.figures_generated):<4}{' ' * 26}║")
    else:
        agent = "COMPLETE"
        next_action = "Pipeline complete!"
        print(f"║  Status: COMPLETE{' ' * 43}║")
    
    print("╠══════════════════════════════════════════════════════════════╣")
    
    if workflow.approved_idea_id:
        idea = await experiments_db.get_idea(workflow.approved_idea_id)
        if idea:
            title = idea.title[:50] + "..." if len(idea.title) > 50 else idea.title
            print(f"║  Idea: {title:<54}║")
    
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Next: {next_action:<54}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    if agent != "COMPLETE":
        print(f"║  >>> Start Cursor chat with {agent:<31} ║")
    
    print("╚══════════════════════════════════════════════════════════════╝")


async def list_projects():
    """List all projects."""
    projects = await project_manager.list_projects()
    current = await project_manager.get_current_project()
    current_id = current.project_id if current else None
    
    print("\nProjects:")
    print("-" * 60)
    
    if not projects:
        print("  No projects found.")
        print("  Create one with researcher-mcp: create_project(name='...')")
        return
    
    for p in projects:
        marker = " [ACTIVE]" if p.get("id") == current_id else ""
        print(f"  {p.get('id', 'unknown')}{marker}")
        if p.get("description"):
            print(f"    {p['description']}")


async def set_project(project_id: str):
    """Set the current project."""
    try:
        await project_manager.set_current_project(project_id)
        print(f"✓ Set current project to: {project_id}")
        await status()
    except Exception as e:
        print(f"✗ Failed to set project: {e}")


def main():
    parser = argparse.ArgumentParser(description="Research pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    subparsers.add_parser("status", help="Show current status")
    subparsers.add_parser("projects", help="List all projects")
    
    set_parser = subparsers.add_parser("set", help="Set current project")
    set_parser.add_argument("project_id", help="Project ID to activate")
    
    args = parser.parse_args()
    
    if args.command == "status" or args.command is None:
        asyncio.run(status())
    elif args.command == "projects":
        asyncio.run(list_projects())
    elif args.command == "set":
        asyncio.run(set_project(args.project_id))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
