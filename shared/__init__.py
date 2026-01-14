"""Shared infrastructure for all agent MCPs.

This module re-exports from src/ to avoid code duplication.
All three MCPs (researcher, experimenter, writer) use these shared components.
"""

from src.db.workflow import WorkflowDB, WorkflowState, workflow_db
from src.db.papers_cache import PapersCache, CachedPaper, papers_cache
from src.db.experiments_db import ExperimentsDB, Experiment, Idea, experiments_db
from src.project.manager import project_manager, ProjectManager, Project
from src.apis.arxiv import ArxivClient, ArxivPaper
from src.apis.huggingface import HuggingFaceClient

__all__ = [
    "WorkflowDB",
    "WorkflowState",
    "workflow_db",
    "PapersCache",
    "CachedPaper",
    "papers_cache",
    "ExperimentsDB",
    "Experiment",
    "Idea",
    "experiments_db",
    "project_manager",
    "ProjectManager",
    "Project",
    "ArxivClient",
    "ArxivPaper",
    "HuggingFaceClient",
]
