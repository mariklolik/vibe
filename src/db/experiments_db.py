"""SQLite-based experiment results database."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite


@dataclass
class Experiment:
    experiment_id: str
    name: str
    status: str
    config: dict
    metrics: dict
    created_at: str
    updated_at: str
    logs_dir: Optional[str]
    checkpoint_path: Optional[str]
    extra_data: dict

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "logs_dir": self.logs_dir,
            "checkpoint_path": self.checkpoint_path,
            "extra_data": self.extra_data,
        }


@dataclass
class Idea:
    idea_id: str
    title: str
    description: str
    source_papers: list[str]
    hypotheses: list[str]
    research_plan: dict
    novelty_score: Optional[float]
    created_at: str
    status: str  # "pending_approval", "approved", "rejected", "planned", "completed"
    
    # Extended fields for user verification
    motivation: str = ""
    themes: list[str] = field(default_factory=list)
    similar_works: list[dict] = field(default_factory=list)
    approved_at: Optional[str] = None
    user_feedback: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # Confirmation code for user approval (prevents auto-approval by AI)
    confirmation_code: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "idea_id": self.idea_id,
            "title": self.title,
            "description": self.description,
            "source_papers": self.source_papers,
            "hypotheses": self.hypotheses,
            "research_plan": self.research_plan,
            "novelty_score": self.novelty_score,
            "created_at": self.created_at,
            "status": self.status,
            "motivation": self.motivation,
            "themes": self.themes,
            "approved_at": self.approved_at,
            "user_feedback": self.user_feedback,
            "rejection_reason": self.rejection_reason,
        }


class ExperimentsDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "research-mcp"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "experiments.db")
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT,
                    metrics TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    logs_dir TEXT,
                    checkpoint_path TEXT,
                    extra_data TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ideas (
                    idea_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    source_papers TEXT,
                    hypotheses TEXT,
                    research_plan TEXT,
                    novelty_score REAL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    motivation TEXT,
                    themes TEXT,
                    approved_at TEXT,
                    user_feedback TEXT,
                    rejection_reason TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metric_history(experiment_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status)
            """)
            
            await db.commit()
        
        self._initialized = True

    async def save_experiment(self, exp: Experiment) -> None:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO experiments 
                (experiment_id, name, status, config, metrics, created_at, 
                 updated_at, logs_dir, checkpoint_path, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.experiment_id,
                exp.name,
                exp.status,
                json.dumps(exp.config),
                json.dumps(exp.metrics),
                exp.created_at,
                exp.updated_at,
                exp.logs_dir,
                exp.checkpoint_path,
                json.dumps(exp.extra_data),
            ))
            await db.commit()

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE experiment_id = ? OR name = ?",
                (experiment_id, experiment_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_experiment(row)
        return None

    async def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[Experiment]:
        await self._ensure_initialized()
        
        experiments = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if status:
                query = "SELECT * FROM experiments WHERE status = ? ORDER BY updated_at DESC LIMIT ?"
                params = (status, limit)
            else:
                query = "SELECT * FROM experiments ORDER BY updated_at DESC LIMIT ?"
                params = (limit,)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    experiments.append(self._row_to_experiment(row))
        
        return experiments

    async def update_metrics(
        self,
        experiment_id: str,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        await self._ensure_initialized()
        
        timestamp = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            for name, value in metrics.items():
                await db.execute("""
                    INSERT INTO metric_history 
                    (experiment_id, metric_name, metric_value, step, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (experiment_id, name, value, step, timestamp))
            
            exp = await self.get_experiment(experiment_id)
            if exp:
                exp.metrics.update(metrics)
                exp.updated_at = timestamp
                await self.save_experiment(exp)
            
            await db.commit()

    async def get_metric_history(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> list[dict]:
        await self._ensure_initialized()
        
        history = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if metric_name:
                query = """
                    SELECT * FROM metric_history 
                    WHERE experiment_id = ? AND metric_name = ?
                    ORDER BY step, timestamp
                """
                params = (experiment_id, metric_name)
            else:
                query = """
                    SELECT * FROM metric_history 
                    WHERE experiment_id = ?
                    ORDER BY step, timestamp
                """
                params = (experiment_id,)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    history.append({
                        "metric_name": row["metric_name"],
                        "metric_value": row["metric_value"],
                        "step": row["step"],
                        "timestamp": row["timestamp"],
                    })
        
        return history

    async def save_idea(self, idea: Idea) -> None:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO ideas 
                (idea_id, title, description, source_papers, hypotheses, 
                 research_plan, novelty_score, created_at, status,
                 motivation, themes, approved_at, user_feedback, rejection_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idea.idea_id,
                idea.title,
                idea.description,
                json.dumps(idea.source_papers),
                json.dumps(idea.hypotheses),
                json.dumps(idea.research_plan),
                idea.novelty_score,
                idea.created_at,
                idea.status,
                idea.motivation,
                json.dumps(idea.themes),
                idea.approved_at,
                idea.user_feedback,
                idea.rejection_reason,
            ))
            await db.commit()

    async def get_idea(self, idea_id: str) -> Optional[Idea]:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ideas WHERE idea_id = ?",
                (idea_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Idea(
                        idea_id=row["idea_id"],
                        title=row["title"],
                        description=row["description"] or "",
                        source_papers=json.loads(row["source_papers"] or "[]"),
                        hypotheses=json.loads(row["hypotheses"] or "[]"),
                        research_plan=json.loads(row["research_plan"] or "{}"),
                        novelty_score=row["novelty_score"],
                        created_at=row["created_at"],
                        status=row["status"],
                        motivation=row["motivation"] or "" if "motivation" in row.keys() else "",
                        themes=json.loads(row["themes"] or "[]") if "themes" in row.keys() else [],
                        approved_at=row["approved_at"] if "approved_at" in row.keys() else None,
                        user_feedback=row["user_feedback"] if "user_feedback" in row.keys() else None,
                        rejection_reason=row["rejection_reason"] if "rejection_reason" in row.keys() else None,
                    )
        return None

    async def list_ideas(self, status: Optional[str] = None, limit: int = 50) -> list[Idea]:
        await self._ensure_initialized()
        
        ideas = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if status:
                query = "SELECT * FROM ideas WHERE status = ? ORDER BY created_at DESC LIMIT ?"
                params = (status, limit)
            else:
                query = "SELECT * FROM ideas ORDER BY created_at DESC LIMIT ?"
                params = (limit,)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    ideas.append(Idea(
                        idea_id=row["idea_id"],
                        title=row["title"],
                        description=row["description"] or "",
                        source_papers=json.loads(row["source_papers"] or "[]"),
                        hypotheses=json.loads(row["hypotheses"] or "[]"),
                        research_plan=json.loads(row["research_plan"] or "{}"),
                        novelty_score=row["novelty_score"],
                        created_at=row["created_at"],
                        status=row["status"],
                        motivation=row["motivation"] or "" if "motivation" in row.keys() else "",
                        themes=json.loads(row["themes"] or "[]") if "themes" in row.keys() else [],
                    ))
        
        return ideas

    def _row_to_experiment(self, row: aiosqlite.Row) -> Experiment:
        return Experiment(
            experiment_id=row["experiment_id"],
            name=row["name"],
            status=row["status"],
            config=json.loads(row["config"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            logs_dir=row["logs_dir"],
            checkpoint_path=row["checkpoint_path"],
            extra_data=json.loads(row["extra_data"] or "{}"),
        )


experiments_db = ExperimentsDB()
