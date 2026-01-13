"""Project management module for research workflow."""

from src.project.manager import ProjectManager, Project, project_manager
from src.project.git_ops import GitOps

__all__ = [
    "ProjectManager",
    "Project",
    "project_manager",
    "GitOps",
]
