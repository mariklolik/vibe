"""Workflow state persistence for resumable sessions."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite


@dataclass
class WorkflowState:
    """Current state of a research workflow."""
    workflow_id: str
    project_id: str
    
    # Current stage
    stage: str  # "context_gathering", "idea_generation", "idea_approval", "experiment_setup", "experimenting", "analysis", "writing", "formatting"
    
    # Progress tracking
    completed_steps: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    next_steps: list[str] = field(default_factory=list)
    
    # Context
    gathered_papers: list[str] = field(default_factory=list)
    extracted_contexts: list[str] = field(default_factory=list)
    
    # Target metrics from reference papers
    target_metrics: Optional[dict] = None  # PaperMetrics.to_dict() format
    
    # Ideas
    generated_ideas: list[str] = field(default_factory=list)
    approved_idea_id: Optional[str] = None
    hypotheses: list[str] = field(default_factory=list)
    
    # Experiments
    experiment_configs: list[dict] = field(default_factory=list)
    completed_experiments: list[str] = field(default_factory=list)
    experiment_results: dict = field(default_factory=dict)
    experiment_runs: list[str] = field(default_factory=list)  # Tracked run IDs
    
    # Writing
    paper_sections: dict = field(default_factory=dict)  # {section_name: content}
    figures_generated: list[str] = field(default_factory=list)
    target_conference: Optional[str] = None
    paper_iterations: int = 0  # Number of paper expansion iterations
    
    # GitHub
    github_url: Optional[str] = None
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    last_action: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "project_id": self.project_id,
            "stage": self.stage,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "next_steps": self.next_steps,
            "gathered_papers": self.gathered_papers,
            "extracted_contexts": self.extracted_contexts,
            "target_metrics": self.target_metrics,
            "generated_ideas": self.generated_ideas,
            "approved_idea_id": self.approved_idea_id,
            "hypotheses": self.hypotheses,
            "experiment_configs": self.experiment_configs,
            "completed_experiments": self.completed_experiments,
            "experiment_results": self.experiment_results,
            "experiment_runs": self.experiment_runs,
            "paper_sections": self.paper_sections,
            "figures_generated": self.figures_generated,
            "target_conference": self.target_conference,
            "paper_iterations": self.paper_iterations,
            "github_url": self.github_url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_action": self.last_action,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        return cls(
            workflow_id=data["workflow_id"],
            project_id=data["project_id"],
            stage=data["stage"],
            completed_steps=data.get("completed_steps", []),
            current_step=data.get("current_step"),
            next_steps=data.get("next_steps", []),
            gathered_papers=data.get("gathered_papers", []),
            extracted_contexts=data.get("extracted_contexts", []),
            target_metrics=data.get("target_metrics"),
            generated_ideas=data.get("generated_ideas", []),
            approved_idea_id=data.get("approved_idea_id"),
            hypotheses=data.get("hypotheses", []),
            experiment_configs=data.get("experiment_configs", []),
            completed_experiments=data.get("completed_experiments", []),
            experiment_results=data.get("experiment_results", {}),
            experiment_runs=data.get("experiment_runs", []),
            paper_sections=data.get("paper_sections", {}),
            figures_generated=data.get("figures_generated", []),
            target_conference=data.get("target_conference"),
            paper_iterations=data.get("paper_iterations", 0),
            github_url=data.get("github_url"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            last_action=data.get("last_action"),
        )
    
    def get_progress_summary(self) -> dict:
        """Get a summary of workflow progress."""
        stages = [
            "context_gathering",
            "idea_generation", 
            "idea_approval",
            "experiment_setup",
            "experimenting",
            "analysis",
            "writing",
            "formatting",
        ]
        
        current_index = stages.index(self.stage) if self.stage in stages else 0
        
        target_words = self.target_metrics.get("word_count", 5000) if self.target_metrics else 5000
        target_figures = self.target_metrics.get("figure_count", 6) if self.target_metrics else 6
        
        return {
            "current_stage": self.stage,
            "progress_percent": int((current_index + 1) / len(stages) * 100),
            "papers_gathered": len(self.gathered_papers),
            "contexts_extracted": len(self.extracted_contexts),
            "target_metrics_set": self.target_metrics is not None,
            "ideas_generated": len(self.generated_ideas),
            "idea_approved": self.approved_idea_id is not None,
            "experiments_completed": len(self.completed_experiments),
            "tracked_runs": len(self.experiment_runs),
            "sections_written": len(self.paper_sections),
            "figures_generated": len(self.figures_generated),
            "target_figures": target_figures,
            "target_words": target_words,
            "paper_iterations": self.paper_iterations,
            "github_linked": self.github_url is not None,
            "github_url": self.github_url,
            "completed_steps": len(self.completed_steps),
            "next_steps": self.next_steps[:3],
        }
    
    def advance_stage(self, new_stage: str, action: str = None) -> None:
        """Advance to the next stage."""
        if self.current_step:
            self.completed_steps.append(self.current_step)
        self.stage = new_stage
        self.updated_at = datetime.now().isoformat()
        self.last_action = action


class WorkflowDB:
    """Database for workflow state persistence."""
    
    WORKFLOW_STAGES = [
        "context_gathering",
        "idea_generation",
        "idea_approval", 
        "experiment_setup",
        "experimenting",
        "analysis",
        "writing",
        "formatting",
        "completed",
    ]
    
    # Stages that require user approval to proceed
    APPROVAL_REQUIRED_STAGES = ["idea_approval"]
    
    # Actions that require specific prior stages to have been completed
    STAGE_REQUIREMENTS = {
        "run_experiment": "experiment_setup",
        "run_baseline": "experiment_setup",
        "run_ablation": "experimenting",
        "write_section": "analysis",
        "cast_to_format": "writing",
        "generate_poster": "writing",
    }
    
    # Actions that require specific workflow conditions
    ACTION_PREREQUISITES = {
        "run_experiment": {
            "requires_approved_idea": True,
            "requires_steps": ["env_created"],
            "error": "Must have approved idea and created environment first",
        },
        "run_baseline": {
            "requires_approved_idea": True,
            "requires_steps": ["env_created"],
            "error": "Must have approved idea and created environment first",
        },
        "cast_to_format": {
            "requires_figures": True,
            "min_figures": 1,
            "error": "Must generate at least 1 figure before formatting paper",
        },
        "generate_poster": {
            "requires_figures": True,
            "min_figures": 1,
            "error": "Must generate figures before creating poster",
        },
        "compile_paper": {
            "requires_sections": True,
            "error": "Must write paper sections before compiling",
        },
    }
    
    def validate_action(self, workflow: WorkflowState, action: str) -> tuple[bool, str]:
        """Check if an action can be performed given current workflow state.
        
        Returns (is_valid, error_message).
        """
        # Check if idea approval is required
        if not workflow.approved_idea_id:
            blocked_actions = [
                "run_experiment", "run_baseline", "run_ablation",
                "create_experiment_env", "setup_datasets", "define_hypotheses",
            ]
            if action in blocked_actions:
                return False, (
                    f"BLOCKED: Cannot {action} before idea approval. "
                    f"Generated ideas: {len(workflow.generated_ideas)}. "
                    f"User must type: APPROVE <idea_id> CODE <confirmation_code>"
                )
        
        # Check stage requirements
        if action in self.STAGE_REQUIREMENTS:
            required_stage = self.STAGE_REQUIREMENTS[action]
            required_index = self.WORKFLOW_STAGES.index(required_stage)
            current_index = self.WORKFLOW_STAGES.index(workflow.stage)
            
            if current_index < required_index:
                return False, (
                    f"BLOCKED: Cannot {action} at stage '{workflow.stage}'. "
                    f"Required stage: '{required_stage}' or later. "
                    f"Call get_next_action() to see what to do next."
                )
        
        # Check action prerequisites
        if action in self.ACTION_PREREQUISITES:
            prereqs = self.ACTION_PREREQUISITES[action]
            
            if prereqs.get("requires_approved_idea") and not workflow.approved_idea_id:
                return False, prereqs["error"]
            
            required_steps = prereqs.get("requires_steps", [])
            for step in required_steps:
                if step not in workflow.completed_steps:
                    return False, (
                        f"BLOCKED: {prereqs['error']}. "
                        f"Missing step: {step}. "
                        f"Call get_next_action() to see required steps."
                    )
            
            if prereqs.get("requires_figures"):
                min_figs = prereqs.get("min_figures", 1)
                if len(workflow.figures_generated) < min_figs:
                    return False, (
                        f"BLOCKED: {prereqs['error']}. "
                        f"You have {len(workflow.figures_generated)} figures, need {min_figs}. "
                        f"Use plot_comparison_bar() or plot_training_curves() first."
                    )
            
            if prereqs.get("requires_sections"):
                if not workflow.paper_sections:
                    return False, (
                        f"BLOCKED: {prereqs['error']}. "
                        f"Use create_paper_skeleton() and write paper content first."
                    )
        
        return True, ""
    
    def get_missing_prerequisites(self, workflow: WorkflowState, action: str) -> list[str]:
        """Get list of missing prerequisites for an action."""
        missing = []
        
        if action in self.ACTION_PREREQUISITES:
            prereqs = self.ACTION_PREREQUISITES[action]
            
            if prereqs.get("requires_approved_idea") and not workflow.approved_idea_id:
                missing.append("approve_idea (requires user)")
            
            for step in prereqs.get("requires_steps", []):
                if step not in workflow.completed_steps:
                    if step == "env_created":
                        missing.append("create_experiment_env")
                    elif step == "datasets_setup":
                        missing.append("setup_datasets")
                    else:
                        missing.append(step)
            
            if prereqs.get("requires_figures"):
                if len(workflow.figures_generated) < prereqs.get("min_figures", 1):
                    missing.append("plot_comparison_bar or plot_training_curves")
        
        return missing
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "research-mcp"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "workflows.db")
        self.db_path = db_path
        self._initialized = False
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_project ON workflows(project_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_stage ON workflows(stage)
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflow_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    data TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            await db.commit()
        
        self._initialized = True
    
    async def create_workflow(self, project_id: str, workflow_id: Optional[str] = None) -> WorkflowState:
        """Create a new workflow for a project."""
        await self._ensure_initialized()
        
        if workflow_id is None:
            workflow_id = f"wf_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        now = datetime.now().isoformat()
        
        workflow = WorkflowState(
            workflow_id=workflow_id,
            project_id=project_id,
            stage="context_gathering",
            next_steps=[
                "Gather relevant papers using fetch_arxiv_trending or search_papers",
                "Extract context from papers using extract_paper_context",
                "Generate research ideas using generate_ideas",
            ],
            created_at=now,
            updated_at=now,
        )
        
        await self.save_workflow(workflow)
        await self._log_action(workflow_id, "workflow_created", "context_gathering", {})
        
        return workflow
    
    async def save_workflow(self, workflow: WorkflowState) -> None:
        """Save workflow state."""
        await self._ensure_initialized()
        
        workflow.updated_at = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, project_id, data, stage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                workflow.workflow_id,
                workflow.project_id,
                json.dumps(workflow.to_dict()),
                workflow.stage,
                workflow.created_at,
                workflow.updated_at,
            ))
            await db.commit()
    
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get a workflow by ID."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM workflows WHERE workflow_id = ?",
                (workflow_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return WorkflowState.from_dict(json.loads(row[0]))
        return None
    
    async def get_project_workflow(self, project_id: str) -> Optional[WorkflowState]:
        """Get the most recent workflow for a project."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM workflows WHERE project_id = ? ORDER BY updated_at DESC LIMIT 1",
                (project_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return WorkflowState.from_dict(json.loads(row[0]))
        return None
    
    async def update_workflow(
        self,
        workflow: WorkflowState,
        action: str,
        data: Optional[dict] = None,
    ) -> None:
        """Update workflow state with action logging."""
        workflow.last_action = action
        await self.save_workflow(workflow)
        await self._log_action(workflow.workflow_id, action, workflow.stage, data or {})
    
    async def _log_action(
        self,
        workflow_id: str,
        action: str,
        stage: str,
        data: dict,
    ) -> None:
        """Log a workflow action for history."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO workflow_history (workflow_id, action, stage, data, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                workflow_id,
                action,
                stage,
                json.dumps(data),
                datetime.now().isoformat(),
            ))
            await db.commit()
    
    async def get_workflow_history(self, workflow_id: str, limit: int = 50) -> list[dict]:
        """Get action history for a workflow."""
        await self._ensure_initialized()
        
        history = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT action, stage, data, timestamp FROM workflow_history "
                "WHERE workflow_id = ? ORDER BY timestamp DESC LIMIT ?",
                (workflow_id, limit)
            ) as cursor:
                async for row in cursor:
                    history.append({
                        "action": row[0],
                        "stage": row[1],
                        "data": json.loads(row[2]) if row[2] else {},
                        "timestamp": row[3],
                    })
        return history
    
    async def add_paper(self, workflow: WorkflowState, paper_id: str) -> None:
        """Add a gathered paper to the workflow."""
        if paper_id not in workflow.gathered_papers:
            workflow.gathered_papers.append(paper_id)
        await self.update_workflow(workflow, "paper_gathered", {"paper_id": paper_id})
    
    async def add_context(self, workflow: WorkflowState, context_id: str) -> None:
        """Add an extracted context to the workflow."""
        if context_id not in workflow.extracted_contexts:
            workflow.extracted_contexts.append(context_id)
        
        # Suggest moving to next stage if enough context gathered
        if len(workflow.extracted_contexts) >= 3:
            workflow.next_steps = [
                "Extract paper metrics using extract_paper_metrics to set targets",
                "Generate research ideas using generate_ideas",
                "Or gather more papers for additional context",
            ]
        
        await self.update_workflow(workflow, "context_extracted", {"context_id": context_id})
    
    async def set_target_metrics(self, workflow: WorkflowState, metrics: dict) -> None:
        """Set target metrics from reference papers."""
        workflow.target_metrics = metrics
        await self.update_workflow(workflow, "target_metrics_set", metrics)
    
    async def add_experiment_run(self, workflow: WorkflowState, run_id: str) -> None:
        """Track an experiment run."""
        if run_id not in workflow.experiment_runs:
            workflow.experiment_runs.append(run_id)
        await self.save_workflow(workflow)
    
    async def increment_paper_iterations(self, workflow: WorkflowState) -> None:
        """Track paper expansion iterations."""
        workflow.paper_iterations += 1
        await self.save_workflow(workflow)
    
    async def set_github_url(self, workflow: WorkflowState, url: str) -> None:
        """Set the GitHub repository URL."""
        workflow.github_url = url
        await self.update_workflow(workflow, "github_linked", {"url": url})
    
    async def add_idea(self, workflow: WorkflowState, idea_id: str) -> None:
        """Add a generated idea to the workflow."""
        if idea_id not in workflow.generated_ideas:
            workflow.generated_ideas.append(idea_id)
        
        if workflow.stage == "context_gathering":
            workflow.stage = "idea_generation"
        
        workflow.next_steps = [
            "Review generated ideas and approve one",
            "Define hypotheses for the approved idea",
        ]
        
        await self.update_workflow(workflow, "idea_generated", {"idea_id": idea_id})
    
    async def approve_idea(self, workflow: WorkflowState, idea_id: str, hypotheses: list[str]) -> None:
        """Approve an idea and move to experiment setup."""
        workflow.approved_idea_id = idea_id
        workflow.hypotheses = hypotheses
        workflow.stage = "experiment_setup"
        workflow.next_steps = [
            "Set up experiment environment",
            "Configure experiment parameters",
            "Run experiments to test hypotheses",
        ]
        
        await self.update_workflow(workflow, "idea_approved", {
            "idea_id": idea_id,
            "hypotheses": hypotheses,
        })
    
    async def add_experiment_result(
        self,
        workflow: WorkflowState,
        experiment_id: str,
        results: dict,
    ) -> None:
        """Add experiment results."""
        workflow.experiment_results[experiment_id] = results
        if experiment_id not in workflow.completed_experiments:
            workflow.completed_experiments.append(experiment_id)
        
        if workflow.stage == "experiment_setup":
            workflow.stage = "experimenting"
        
        # Suggest analysis when enough experiments done
        if len(workflow.completed_experiments) >= 3:
            workflow.next_steps = [
                "Analyze results and verify hypotheses",
                "Generate figures from experiment data",
                "Start writing paper sections",
            ]
            workflow.stage = "analysis"
        
        await self.update_workflow(workflow, "experiment_completed", {
            "experiment_id": experiment_id,
            "results_summary": str(results)[:200],
        })
    
    async def add_section(self, workflow: WorkflowState, section_name: str, content: str) -> None:
        """Add a written paper section."""
        workflow.paper_sections[section_name] = content
        
        if workflow.stage in ["analysis", "experimenting"]:
            workflow.stage = "writing"
        
        required_sections = {"introduction", "method", "experiments", "conclusion"}
        written_sections = set(workflow.paper_sections.keys())
        
        remaining = required_sections - written_sections
        if remaining:
            workflow.next_steps = [f"Write {s} section" for s in remaining]
        else:
            workflow.next_steps = [
                "Format paper for target conference",
                "Generate supplementary materials",
            ]
            workflow.stage = "formatting"
        
        await self.update_workflow(workflow, "section_written", {
            "section": section_name,
            "word_count": len(content.split()),
        })
    
    async def complete_workflow(self, workflow: WorkflowState, output_path: str) -> None:
        """Mark workflow as completed."""
        workflow.stage = "completed"
        workflow.next_steps = []
        
        await self.update_workflow(workflow, "workflow_completed", {
            "output_path": output_path,
            "sections_count": len(workflow.paper_sections),
            "experiments_count": len(workflow.completed_experiments),
        })


workflow_db = WorkflowDB()
