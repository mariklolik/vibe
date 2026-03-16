"""Progress file management for durable state across agent iterations.

Implements the rom4ik pattern: append-only progress.txt as the single
source of truth for cross-iteration state. Human-readable, git-trackable.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_progress_path(project_dir: str) -> Path:
    return Path(project_dir) / "progress.txt"


def read_progress(project_dir: str) -> str:
    """Read the full progress file. Returns empty string if not exists."""
    path = get_progress_path(project_dir)
    if path.exists():
        return path.read_text()
    return ""


def append_progress(project_dir: str, entry: str, stage: Optional[str] = None):
    """Append a timestamped entry to progress.txt.

    Args:
        project_dir: Project directory path
        entry: Text to append
        stage: Optional stage tag (e.g., "research", "experiment", "writing")
    """
    path = get_progress_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f" [{stage}]" if stage else ""

    with open(path, "a") as f:
        f.write(f"\n--- {timestamp}{tag} ---\n")
        f.write(entry.strip())
        f.write("\n")


def get_last_stage(project_dir: str) -> Optional[str]:
    """Get the most recent stage from progress.txt."""
    content = read_progress(project_dir)
    if not content:
        return None

    import re
    stages = re.findall(r'\[(\w+)\]', content)
    return stages[-1] if stages else None


def save_project_config(project_dir: str, config: dict):
    """Save project configuration to project.json."""
    path = Path(project_dir) / "project.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def load_project_config(project_dir: str) -> dict:
    """Load project configuration from project.json."""
    path = Path(project_dir) / "project.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def ensure_project_structure(project_dir: str):
    """Create standard project directory structure."""
    dirs = [
        "context",          # Extracted paper analyses
        "ideas",            # Generated ideas
        "src",              # Core method implementation (importable classes)
        "scripts",          # Experiment runners that import from src/
        "configs",          # YAML/JSON configs for reproducibility
        "experiments",      # Experiment outputs and logs
        "experiments/logs",
        "experiments/results",
        "figures",          # Generated visualizations
        "paper",            # Final paper outputs
        "data",             # Datasets
        "verification",     # Statistical verification results
        "baselines",        # Baseline implementations
    ]
    for d in dirs:
        (Path(project_dir) / d).mkdir(parents=True, exist_ok=True)

    # Initialize git if not already
    git_dir = Path(project_dir) / ".git"
    if not git_dir.exists():
        os.system(f"cd {project_dir} && git init -q")
        gitignore = Path(project_dir) / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                ".venv/\n__pycache__/\n*.pyc\n*.ckpt\n*.pt\n*.pth\n"
                "*.egg-info/\ndata/\nwandb/\n.DS_Store\n"
            )
