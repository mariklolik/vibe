"""Experiment tracking and logging system."""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import aiosqlite


@dataclass
class ExperimentRun:
    """A single experiment run with config and results."""
    run_id: str
    name: str
    config: dict
    metrics: dict
    status: str  # "running", "completed", "failed"
    started_at: str
    completed_at: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        return cls(
            run_id=data["run_id"],
            name=data["name"],
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            status=data.get("status", "completed"),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            artifacts=data.get("artifacts", []),
            notes=data.get("notes", ""),
        )


class ExperimentTracker:
    """Track experiments with metrics, parameters, and artifacts."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.log_dir = self.project_dir / "experiments" / "logs"
        self.metrics_db = self.project_dir / "experiments" / "metrics.db"
        self._initialized = False
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    artifacts TEXT,
                    notes TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            await db.commit()
        
        self._initialized = True
    
    async def start_run(
        self,
        name: str,
        config: dict,
        run_id: Optional[str] = None,
    ) -> ExperimentRun:
        """Start a new experiment run."""
        await self._ensure_initialized()
        
        if run_id is None:
            import uuid
            run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        now = datetime.now().isoformat()
        
        run = ExperimentRun(
            run_id=run_id,
            name=name,
            config=config,
            metrics={},
            status="running",
            started_at=now,
        )
        
        await self._save_run(run)
        
        run_log_dir = self.log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = run_log_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        
        return run
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for an experiment run."""
        await self._ensure_initialized()
        
        now = datetime.now().isoformat()
        
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            for name, value in metrics.items():
                await db.execute("""
                    INSERT INTO metric_history (run_id, metric_name, metric_value, step, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (run_id, name, float(value), step or 0, now))
            await db.commit()
        
        run = await self.get_run(run_id)
        if run:
            run.metrics.update(metrics)
            await self._save_run(run)
    
    async def complete_run(
        self,
        run_id: str,
        final_metrics: Optional[dict] = None,
        status: str = "completed",
        notes: str = "",
    ) -> ExperimentRun:
        """Complete an experiment run."""
        await self._ensure_initialized()
        
        run = await self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        
        run.status = status
        run.completed_at = datetime.now().isoformat()
        run.notes = notes
        
        if final_metrics:
            run.metrics.update(final_metrics)
        
        await self._save_run(run)
        
        run_log_dir = self.log_dir / run_id
        results_path = run_log_dir / "results.json"
        results_path.write_text(json.dumps(run.to_dict(), indent=2))
        
        return run
    
    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_type: str = "file",
    ) -> None:
        """Log an artifact (file) for an experiment run."""
        run = await self.get_run(run_id)
        if run:
            run.artifacts.append(artifact_path)
            await self._save_run(run)
    
    async def _save_run(self, run: ExperimentRun) -> None:
        """Save a run to the database."""
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            await db.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, name, config, metrics, status, started_at, completed_at, artifacts, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.name,
                json.dumps(run.config),
                json.dumps(run.metrics),
                run.status,
                run.started_at,
                run.completed_at,
                json.dumps(run.artifacts),
                run.notes,
            ))
            await db.commit()
    
    async def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            async with db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ExperimentRun(
                        run_id=row[0],
                        name=row[1],
                        config=json.loads(row[2]),
                        metrics=json.loads(row[3]),
                        status=row[4],
                        started_at=row[5],
                        completed_at=row[6],
                        artifacts=json.loads(row[7]) if row[7] else [],
                        notes=row[8] or "",
                    )
        return None
    
    async def get_all_runs(self) -> list[ExperimentRun]:
        """Get all experiment runs."""
        await self._ensure_initialized()
        
        runs = []
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            async with db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC"
            ) as cursor:
                async for row in cursor:
                    runs.append(ExperimentRun(
                        run_id=row[0],
                        name=row[1],
                        config=json.loads(row[2]),
                        metrics=json.loads(row[3]),
                        status=row[4],
                        started_at=row[5],
                        completed_at=row[6],
                        artifacts=json.loads(row[7]) if row[7] else [],
                        notes=row[8] or "",
                    ))
        return runs
    
    async def get_metric_history(
        self,
        run_id: str,
        metric_name: Optional[str] = None,
    ) -> list[dict]:
        """Get metric history for a run."""
        await self._ensure_initialized()
        
        history = []
        async with aiosqlite.connect(str(self.metrics_db)) as db:
            if metric_name:
                query = """
                    SELECT metric_name, metric_value, step, timestamp 
                    FROM metric_history 
                    WHERE run_id = ? AND metric_name = ?
                    ORDER BY step
                """
                params = (run_id, metric_name)
            else:
                query = """
                    SELECT metric_name, metric_value, step, timestamp 
                    FROM metric_history 
                    WHERE run_id = ?
                    ORDER BY step
                """
                params = (run_id,)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    history.append({
                        "metric_name": row[0],
                        "value": row[1],
                        "step": row[2],
                        "timestamp": row[3],
                    })
        
        return history
    
    async def get_summary(self) -> dict:
        """Get summary of all experiments."""
        runs = await self.get_all_runs()
        
        completed = [r for r in runs if r.status == "completed"]
        failed = [r for r in runs if r.status == "failed"]
        running = [r for r in runs if r.status == "running"]
        
        best_metrics = {}
        for run in completed:
            for metric, value in run.metrics.items():
                if metric not in best_metrics or value > best_metrics[metric]["value"]:
                    best_metrics[metric] = {
                        "value": value,
                        "run_id": run.run_id,
                        "run_name": run.name,
                    }
        
        return {
            "total_runs": len(runs),
            "completed": len(completed),
            "failed": len(failed),
            "running": len(running),
            "best_metrics": best_metrics,
            "latest_run": runs[0].to_dict() if runs else None,
        }


async def log_experiment(
    project_dir: str,
    name: str,
    config: dict,
    metrics: dict,
) -> str:
    """Log a completed experiment.
    
    Args:
        project_dir: Path to the project directory
        name: Experiment name
        config: Configuration dictionary
        metrics: Results metrics dictionary
    
    Returns:
        JSON string with run details
    """
    tracker = ExperimentTracker(Path(project_dir))
    
    run = await tracker.start_run(name, config)
    run = await tracker.complete_run(run.run_id, final_metrics=metrics)
    
    return json.dumps({
        "success": True,
        "run_id": run.run_id,
        "name": run.name,
        "metrics": run.metrics,
        "log_path": str(tracker.log_dir / run.run_id),
    }, indent=2)


async def get_experiment_history(project_dir: str) -> str:
    """Get all experiment runs for a project.
    
    Args:
        project_dir: Path to the project directory
    
    Returns:
        JSON string with all runs
    """
    tracker = ExperimentTracker(Path(project_dir))
    runs = await tracker.get_all_runs()
    summary = await tracker.get_summary()
    
    return json.dumps({
        "summary": summary,
        "runs": [r.to_dict() for r in runs],
    }, indent=2)
