"""Experiment tracking and logging system."""

import json
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import aiosqlite


def _compute_log_signature(log_path: Path) -> str:
    """Compute SHA256 hash of log file to prove real execution."""
    if not log_path.exists():
        return ""
    content = log_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


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
    experiment_signature: str = ""  # Hash of stdout log proving real execution
    log_file_path: str = ""  # Path to stdout.log
    is_verified: bool = False  # Whether metrics came from actual logs
    parsed_metrics: dict = field(default_factory=dict)  # Metrics parsed from logs
    
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
            "experiment_signature": self.experiment_signature,
            "log_file_path": self.log_file_path,
            "is_verified": self.is_verified,
            "parsed_metrics": self.parsed_metrics,
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
            experiment_signature=data.get("experiment_signature", ""),
            log_file_path=data.get("log_file_path", ""),
            is_verified=data.get("is_verified", False),
            parsed_metrics=data.get("parsed_metrics", {}),
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
                    notes TEXT,
                    experiment_signature TEXT DEFAULT '',
                    log_file_path TEXT DEFAULT '',
                    is_verified INTEGER DEFAULT 0,
                    parsed_metrics TEXT DEFAULT '{}'
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
        
        run_log_dir = self.log_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file_path = str(run_log_dir / "stdout.log")
        
        run = ExperimentRun(
            run_id=run_id,
            name=name,
            config=config,
            metrics={},
            status="running",
            started_at=now,
            log_file_path=log_file_path,
        )
        
        await self._save_run(run)
        
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
        
        # Compute signature from log file to prove real execution
        run_log_dir = self.log_dir / run_id
        stdout_log = run_log_dir / "stdout.log"
        
        if stdout_log.exists():
            run.experiment_signature = _compute_log_signature(stdout_log)
            run.log_file_path = str(stdout_log)
            # Parse metrics from actual log file
            run.parsed_metrics = self._parse_metrics_from_log(stdout_log)
            run.is_verified = len(run.experiment_signature) > 0
        
        await self._save_run(run)
        
        results_path = run_log_dir / "results.json"
        results_path.write_text(json.dumps(run.to_dict(), indent=2))
        
        return run
    
    def _parse_metrics_from_log(self, log_path: Path) -> dict:
        """Parse metrics from stdout log file."""
        import re
        
        if not log_path.exists():
            return {}
        
        content = log_path.read_text()
        
        patterns = {
            "loss": r"(?:loss|Loss)[:=]\s*([\d.]+)",
            "accuracy": r"(?:accuracy|Accuracy|acc|Acc)[:=]\s*([\d.]+)",
            "f1": r"(?:f1|F1)[:=]\s*([\d.]+)",
            "precision": r"(?:precision|Precision)[:=]\s*([\d.]+)",
            "recall": r"(?:recall|Recall)[:=]\s*([\d.]+)",
            "val_loss": r"(?:val_loss|val loss)[:=]\s*([\d.]+)",
            "val_accuracy": r"(?:val_accuracy|val_acc|val accuracy)[:=]\s*([\d.]+)",
            "test_accuracy": r"(?:test_accuracy|test accuracy|test_acc)[:=]\s*([\d.]+)",
            "rmse": r"(?:rmse|RMSE)[:=]\s*([\d.]+)",
            "mae": r"(?:mae|MAE)[:=]\s*([\d.]+)",
            "auc": r"(?:auc|AUC)[:=]\s*([\d.]+)",
            "bleu": r"(?:bleu|BLEU)[:=]\s*([\d.]+)",
            "perplexity": r"(?:perplexity|ppl|PPL)[:=]\s*([\d.]+)",
        }
        
        result = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                values = [float(m) for m in matches]
                result[name] = {
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        return result
    
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
                (run_id, name, config, metrics, status, started_at, completed_at, artifacts, notes,
                 experiment_signature, log_file_path, is_verified, parsed_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                run.experiment_signature,
                run.log_file_path,
                1 if run.is_verified else 0,
                json.dumps(run.parsed_metrics),
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
                        experiment_signature=row[9] if len(row) > 9 else "",
                        log_file_path=row[10] if len(row) > 10 else "",
                        is_verified=bool(row[11]) if len(row) > 11 else False,
                        parsed_metrics=json.loads(row[12]) if len(row) > 12 and row[12] else {},
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
                        experiment_signature=row[9] if len(row) > 9 else "",
                        log_file_path=row[10] if len(row) > 10 else "",
                        is_verified=bool(row[11]) if len(row) > 11 else False,
                        parsed_metrics=json.loads(row[12]) if len(row) > 12 and row[12] else {},
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
    run_id: str,
    additional_notes: str = "",
) -> str:
    """Log and finalize a completed experiment.
    
    IMPORTANT: This function REQUIRES a valid run_id from run_experiment().
    You cannot log arbitrary metrics - they must come from real experiment logs.
    
    Args:
        project_dir: Path to the project directory
        run_id: The run_id returned by run_experiment() - REQUIRED
        additional_notes: Optional notes about the experiment
    
    Returns:
        JSON string with run details including parsed metrics from logs
    """
    tracker = ExperimentTracker(Path(project_dir))
    
    # Validate run exists
    run = await tracker.get_run(run_id)
    if not run:
        return json.dumps({
            "success": False,
            "error": "INVALID_RUN_ID",
            "message": (
                f"Run '{run_id}' not found. You must use a run_id from run_experiment(). "
                "Cannot log arbitrary metrics without a real experiment run."
            ),
        }, indent=2)
    
    # Check if already completed
    if run.status == "completed" and run.is_verified:
        return json.dumps({
            "success": True,
            "message": "Experiment already logged and verified",
            "run_id": run.run_id,
            "name": run.name,
            "metrics": run.parsed_metrics,
            "is_verified": run.is_verified,
            "experiment_signature": run.experiment_signature,
        }, indent=2)
    
    # Complete the run and parse metrics from actual logs
    run = await tracker.complete_run(run_id, notes=additional_notes)
    
    if not run.is_verified:
        return json.dumps({
            "success": False,
            "error": "NO_LOG_FILE",
            "message": (
                f"No log file found for run '{run_id}'. "
                "The experiment may not have generated output or may still be running. "
                "Check the log directory: " + str(tracker.log_dir / run_id)
            ),
        }, indent=2)
    
    return json.dumps({
        "success": True,
        "run_id": run.run_id,
        "name": run.name,
        "metrics": run.parsed_metrics,
        "is_verified": run.is_verified,
        "experiment_signature": run.experiment_signature,
        "log_path": str(tracker.log_dir / run_id),
        "message": "Metrics parsed from actual experiment logs",
    }, indent=2)


async def get_real_metrics(project_dir: str, run_id: str) -> str:
    """Get only the metrics parsed from actual log files.
    
    This function returns ONLY metrics that were extracted from real
    experiment stdout logs. It will NOT return user-provided metrics.
    
    Args:
        project_dir: Path to the project directory
        run_id: The run_id from run_experiment()
    
    Returns:
        JSON with parsed metrics or error if run not verified
    """
    tracker = ExperimentTracker(Path(project_dir))
    
    run = await tracker.get_run(run_id)
    if not run:
        return json.dumps({
            "success": False,
            "error": "Run not found",
            "run_id": run_id,
        })
    
    if not run.is_verified:
        return json.dumps({
            "success": False,
            "error": "RUN_NOT_VERIFIED",
            "message": (
                "This run has no verified metrics. Either:\n"
                "1. The experiment has not completed\n"
                "2. No log file was generated\n"
                "3. The log file could not be parsed\n"
                "Use log_experiment(run_id) to finalize and verify the run."
            ),
            "run_id": run_id,
            "status": run.status,
        })
    
    return json.dumps({
        "success": True,
        "run_id": run_id,
        "name": run.name,
        "is_verified": True,
        "experiment_signature": run.experiment_signature,
        "parsed_metrics": run.parsed_metrics,
        "log_file": run.log_file_path,
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
