"""Project directory management for research workflows."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite


@dataclass
class Project:
    """A research project with all its components."""
    project_id: str
    name: str
    description: str
    
    # Paths
    root_path: Path
    
    # State
    status: str  # "created", "idea_approved", "experimenting", "writing", "completed"
    current_idea_id: Optional[str]
    current_experiment_id: Optional[str]
    
    # Metadata
    created_at: str
    updated_at: str
    
    # Tracking
    paper_ids: list[str] = field(default_factory=list)  # Source papers
    experiment_ids: list[str] = field(default_factory=list)
    
    @property
    def context_dir(self) -> Path:
        return self.root_path / "context"
    
    @property
    def ideas_dir(self) -> Path:
        return self.root_path / "ideas"
    
    @property
    def experiments_dir(self) -> Path:
        return self.root_path / "experiments"
    
    @property
    def papers_dir(self) -> Path:
        return self.root_path / "papers"
    
    @property
    def drafts_dir(self) -> Path:
        return self.papers_dir / "drafts"
    
    @property
    def final_dir(self) -> Path:
        return self.papers_dir / "final"
    
    @property
    def data_dir(self) -> Path:
        return self.root_path / "data"
    
    @property
    def figures_dir(self) -> Path:
        return self.root_path / "figures"
    
    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "root_path": str(self.root_path),
            "status": self.status,
            "current_idea_id": self.current_idea_id,
            "current_experiment_id": self.current_experiment_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "paper_ids": self.paper_ids,
            "experiment_ids": self.experiment_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            description=data["description"],
            root_path=Path(data["root_path"]),
            status=data["status"],
            current_idea_id=data.get("current_idea_id"),
            current_experiment_id=data.get("current_experiment_id"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            paper_ids=data.get("paper_ids", []),
            experiment_ids=data.get("experiment_ids", []),
        )


class ProjectManager:
    """Manages research projects and their directory structures."""
    
    def __init__(self, projects_root: Optional[str] = None, db_path: Optional[str] = None):
        if projects_root is None:
            projects_root = Path.cwd() / "projects"
        self.projects_root = Path(projects_root)
        
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "research-mcp"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "projects.db")
        self.db_path = db_path
        self._initialized = False
        self._current_project: Optional[Project] = None
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS current_project (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    project_id TEXT
                )
            """)
            await db.commit()
        
        self._initialized = True
    
    async def create_project(
        self,
        name: str,
        description: str = "",
        project_id: Optional[str] = None,
    ) -> Project:
        """Create a new research project with full directory structure."""
        await self._ensure_initialized()
        
        if project_id is None:
            project_id = name.lower().replace(" ", "_").replace("-", "_")
            project_id = "".join(c for c in project_id if c.isalnum() or c == "_")
        
        now = datetime.now().isoformat()
        root_path = self.projects_root / project_id
        
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            root_path=root_path,
            status="created",
            current_idea_id=None,
            current_experiment_id=None,
            created_at=now,
            updated_at=now,
        )
        
        # Create directory structure
        for directory in [
            project.root_path,
            project.context_dir,
            project.ideas_dir,
            project.experiments_dir,
            project.papers_dir,
            project.drafts_dir,
            project.final_dir,
            project.data_dir,
            project.figures_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create project config file
        config_path = project.root_path / "project.json"
        config_path.write_text(json.dumps(project.to_dict(), indent=2))
        
        # Create .gitignore
        gitignore_path = project.root_path / ".gitignore"
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# Data files (too large for git)
data/*.csv
data/*.parquet
data/*.pkl
data/*.h5

# Checkpoints (too large)
experiments/*/checkpoints/*.pt
experiments/*/checkpoints/*.pth

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary
tmp/
temp/
"""
        gitignore_path.write_text(gitignore_content)
        
        # Create README
        readme_path = project.root_path / "README.md"
        readme_content = f"""# {name}

{description}

## Project Structure

```
{project_id}/
├── context/           # Extracted paper contexts and profiles
├── ideas/             # Generated and approved research ideas
├── experiments/       # Experiment runs with configs and results
├── papers/
│   ├── drafts/        # Paper drafts in progress
│   └── final/         # Final paper versions
├── data/              # Datasets
├── figures/           # Generated figures
└── project.json       # Project configuration
```

## Status

- Created: {now[:10]}
- Status: {project.status}

## Usage

This project is managed by research-mcp. Use the MCP tools to:
1. Generate and approve research ideas
2. Run experiments
3. Generate paper content
"""
        readme_path.write_text(readme_content)
        
        # Save to database
        await self._save_project(project)
        
        # Set as current project
        await self.set_current_project(project.project_id)
        
        return project
    
    async def _save_project(self, project: Project) -> None:
        """Save project to database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO projects (project_id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                project.project_id,
                json.dumps(project.to_dict()),
                project.created_at,
                project.updated_at,
            ))
            await db.commit()
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM projects WHERE project_id = ?",
                (project_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Project.from_dict(json.loads(row[0]))
        return None
    
    async def list_projects(self) -> list[dict]:
        """List all projects."""
        await self._ensure_initialized()
        
        projects = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM projects ORDER BY updated_at DESC"
            ) as cursor:
                async for row in cursor:
                    data = json.loads(row[0])
                    projects.append({
                        "project_id": data["project_id"],
                        "name": data["name"],
                        "status": data["status"],
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                    })
        return projects
    
    async def set_current_project(self, project_id: str) -> None:
        """Set the current active project."""
        await self._ensure_initialized()
        
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO current_project (id, project_id) VALUES (1, ?)
            """, (project_id,))
            await db.commit()
        
        self._current_project = project
    
    async def get_current_project(self) -> Optional[Project]:
        """Get the current active project."""
        await self._ensure_initialized()
        
        if self._current_project:
            return self._current_project
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT project_id FROM current_project WHERE id = 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    self._current_project = await self.get_project(row[0])
                    return self._current_project
        return None
    
    async def update_project(self, project: Project) -> None:
        """Update a project."""
        project.updated_at = datetime.now().isoformat()
        await self._save_project(project)
        
        # Update config file
        config_path = project.root_path / "project.json"
        config_path.write_text(json.dumps(project.to_dict(), indent=2))
        
        if self._current_project and self._current_project.project_id == project.project_id:
            self._current_project = project
    
    async def create_experiment(self, project: Project, experiment_name: str) -> Path:
        """Create a new experiment directory within a project."""
        exp_num = len(list(project.experiments_dir.iterdir())) + 1
        exp_id = f"exp_{exp_num:03d}_{experiment_name}"
        exp_path = project.experiments_dir / exp_id
        
        for subdir in ["configs", "results", "checkpoints", "logs"]:
            (exp_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create experiment config
        config = {
            "experiment_id": exp_id,
            "name": experiment_name,
            "project_id": project.project_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
        }
        (exp_path / "config.json").write_text(json.dumps(config, indent=2))
        
        project.experiment_ids.append(exp_id)
        project.current_experiment_id = exp_id
        await self.update_project(project)
        
        return exp_path
    
    async def save_idea(self, project: Project, idea: dict) -> Path:
        """Save an idea to the project's ideas directory."""
        idea_id = idea.get("idea_id", f"idea_{len(list(project.ideas_dir.iterdir())) + 1}")
        idea_path = project.ideas_dir / f"{idea_id}.json"
        idea_path.write_text(json.dumps(idea, indent=2))
        return idea_path
    
    async def save_context(self, project: Project, context: dict, paper_id: str) -> Path:
        """Save extracted context to the project."""
        context_path = project.context_dir / f"{paper_id.replace(':', '_').replace('/', '_')}.json"
        context_path.write_text(json.dumps(context, indent=2))
        
        if paper_id not in project.paper_ids:
            project.paper_ids.append(paper_id)
            await self.update_project(project)
        
        return context_path
    
    async def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its files."""
        project = await self.get_project(project_id)
        if not project:
            return False
        
        if project.root_path.exists():
            shutil.rmtree(project.root_path)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            await db.execute(
                "UPDATE current_project SET project_id = NULL WHERE project_id = ?",
                (project_id,)
            )
            await db.commit()
        
        if self._current_project and self._current_project.project_id == project_id:
            self._current_project = None
        
        return True


project_manager = ProjectManager()
