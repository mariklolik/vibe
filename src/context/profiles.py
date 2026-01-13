"""Context profiles database for storing and retrieving paper analysis."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite


@dataclass
class ContextProfile:
    """A context profile from analyzed papers."""
    profile_id: str
    paper_ids: list[str]
    conference: Optional[str]
    
    # Aggregated section statistics
    section_structure: dict  # {section_name: {avg_words, avg_paragraphs, ...}}
    
    # Figure/table patterns
    figure_patterns: dict  # {type: {count, avg_caption_length, positions}}
    
    # Citation patterns
    citation_style: str
    citations_per_paragraph: float
    total_citations_range: tuple[int, int]
    
    # Writing style
    writing_style: dict
    
    # Target metrics
    target_word_count: int
    target_pages: int
    
    created_at: str
    updated_at: str
    
    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "paper_ids": self.paper_ids,
            "conference": self.conference,
            "section_structure": self.section_structure,
            "figure_patterns": self.figure_patterns,
            "citation_style": self.citation_style,
            "citations_per_paragraph": self.citations_per_paragraph,
            "total_citations_range": list(self.total_citations_range),
            "writing_style": self.writing_style,
            "target_word_count": self.target_word_count,
            "target_pages": self.target_pages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContextProfile":
        return cls(
            profile_id=data["profile_id"],
            paper_ids=data["paper_ids"],
            conference=data.get("conference"),
            section_structure=data["section_structure"],
            figure_patterns=data["figure_patterns"],
            citation_style=data["citation_style"],
            citations_per_paragraph=data["citations_per_paragraph"],
            total_citations_range=tuple(data["total_citations_range"]),
            writing_style=data["writing_style"],
            target_word_count=data["target_word_count"],
            target_pages=data["target_pages"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
    
    @classmethod
    def create_from_papers(
        cls,
        profile_id: str,
        paper_structures: list[dict],
        conference: Optional[str] = None,
    ) -> "ContextProfile":
        """Create a profile by aggregating multiple paper structures."""
        if not paper_structures:
            return cls.create_default(profile_id, conference)
        
        # Aggregate section statistics
        section_stats = {}
        for paper in paper_structures:
            for section in paper.get("sections", []):
                name = section["name"].lower()
                if name not in section_stats:
                    section_stats[name] = {
                        "word_counts": [],
                        "paragraph_counts": [],
                        "has_equations": [],
                        "has_figures": [],
                        "has_tables": [],
                        "citation_counts": [],
                    }
                section_stats[name]["word_counts"].append(section.get("word_count", 0))
                section_stats[name]["paragraph_counts"].append(section.get("paragraph_count", 0))
                section_stats[name]["has_equations"].append(section.get("has_equations", False))
                section_stats[name]["has_figures"].append(section.get("has_figures", False))
                section_stats[name]["has_tables"].append(section.get("has_tables", False))
                section_stats[name]["citation_counts"].append(section.get("citation_count", 0))
        
        section_structure = {}
        for name, stats in section_stats.items():
            section_structure[name] = {
                "avg_words": int(sum(stats["word_counts"]) / len(stats["word_counts"])),
                "avg_paragraphs": int(sum(stats["paragraph_counts"]) / len(stats["paragraph_counts"])),
                "equation_probability": sum(stats["has_equations"]) / len(stats["has_equations"]),
                "figure_probability": sum(stats["has_figures"]) / len(stats["has_figures"]),
                "table_probability": sum(stats["has_tables"]) / len(stats["has_tables"]),
                "avg_citations": int(sum(stats["citation_counts"]) / len(stats["citation_counts"])),
            }
        
        # Aggregate figure patterns
        figure_stats = {"figure": [], "table": [], "algorithm": []}
        for paper in paper_structures:
            for fig in paper.get("figures", []):
                fig_type = fig.get("type", "figure")
                if fig_type in figure_stats:
                    figure_stats[fig_type].append(fig)
        
        figure_patterns = {}
        for fig_type, figs in figure_stats.items():
            if figs:
                figure_patterns[fig_type] = {
                    "avg_count": len(figs) / len(paper_structures),
                    "avg_caption_length": sum(f.get("caption_length", 20) for f in figs) / len(figs),
                    "positions": list(set(f.get("position", "top") for f in figs)),
                }
        
        # Aggregate citation patterns
        citation_styles = [p.get("citation_pattern", {}).get("style", "numeric") for p in paper_structures]
        citation_style = max(set(citation_styles), key=citation_styles.count) if citation_styles else "numeric"
        
        citations_per_para = [
            p.get("citation_pattern", {}).get("avg_per_paragraph", 1.5) 
            for p in paper_structures
        ]
        avg_citations_per_para = sum(citations_per_para) / len(citations_per_para) if citations_per_para else 1.5
        
        total_citations = [p.get("citation_pattern", {}).get("total_citations", 30) for p in paper_structures]
        citations_range = (min(total_citations) if total_citations else 20, max(total_citations) if total_citations else 50)
        
        # Aggregate writing style
        writing_styles = [p.get("writing_style", {}) for p in paper_structures]
        avg_writing_style = {
            "avg_sentence_length": sum(ws.get("avg_sentence_length", 20) for ws in writing_styles) / len(writing_styles),
            "passive_voice_ratio": sum(ws.get("passive_voice_ratio", 0.3) for ws in writing_styles) / len(writing_styles),
            "first_person": any(ws.get("first_person", True) for ws in writing_styles),
            "avg_paragraph_length": sum(ws.get("avg_paragraph_length", 80) for ws in writing_styles) / len(writing_styles),
            "formality_score": sum(ws.get("formality_score", 0.85) for ws in writing_styles) / len(writing_styles),
        }
        
        # Calculate targets
        word_counts = [p.get("total_word_count", 4500) for p in paper_structures]
        page_counts = [p.get("total_pages", 9) for p in paper_structures]
        
        now = datetime.now().isoformat()
        
        return cls(
            profile_id=profile_id,
            paper_ids=[],  # Will be set by caller
            conference=conference,
            section_structure=section_structure,
            figure_patterns=figure_patterns,
            citation_style=citation_style,
            citations_per_paragraph=round(avg_citations_per_para, 2),
            total_citations_range=citations_range,
            writing_style=avg_writing_style,
            target_word_count=int(sum(word_counts) / len(word_counts)),
            target_pages=int(sum(page_counts) / len(page_counts)),
            created_at=now,
            updated_at=now,
        )
    
    @classmethod
    def create_default(cls, profile_id: str, conference: Optional[str] = None) -> "ContextProfile":
        """Create a default profile for ML papers."""
        now = datetime.now().isoformat()
        
        return cls(
            profile_id=profile_id,
            paper_ids=[],
            conference=conference,
            section_structure={
                "introduction": {"avg_words": 800, "avg_paragraphs": 5, "equation_probability": 0.1, "figure_probability": 0.3, "table_probability": 0.0, "avg_citations": 15},
                "related work": {"avg_words": 700, "avg_paragraphs": 4, "equation_probability": 0.0, "figure_probability": 0.1, "table_probability": 0.1, "avg_citations": 30},
                "method": {"avg_words": 1500, "avg_paragraphs": 10, "equation_probability": 0.8, "figure_probability": 0.9, "table_probability": 0.2, "avg_citations": 10},
                "experiments": {"avg_words": 1800, "avg_paragraphs": 12, "equation_probability": 0.3, "figure_probability": 0.8, "table_probability": 0.9, "avg_citations": 15},
                "conclusion": {"avg_words": 350, "avg_paragraphs": 2, "equation_probability": 0.0, "figure_probability": 0.0, "table_probability": 0.0, "avg_citations": 3},
            },
            figure_patterns={
                "figure": {"avg_count": 5, "avg_caption_length": 35, "positions": ["top", "bottom"]},
                "table": {"avg_count": 3, "avg_caption_length": 25, "positions": ["top"]},
                "algorithm": {"avg_count": 1, "avg_caption_length": 15, "positions": ["top"]},
            },
            citation_style="numeric",
            citations_per_paragraph=1.8,
            total_citations_range=(25, 60),
            writing_style={
                "avg_sentence_length": 22,
                "passive_voice_ratio": 0.25,
                "first_person": True,
                "avg_paragraph_length": 85,
                "formality_score": 0.85,
            },
            target_word_count=5000,
            target_pages=9,
            created_at=now,
            updated_at=now,
        )


class ContextProfilesDB:
    """Database for storing context profiles."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "research-mcp"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "context_profiles.db")
        self.db_path = db_path
        self._initialized = False
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    conference TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_conference ON profiles(conference)
            """)
            await db.commit()
        
        self._initialized = True
    
    async def save_profile(self, profile: ContextProfile) -> None:
        """Save a context profile."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO profiles (profile_id, data, conference, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                profile.profile_id,
                json.dumps(profile.to_dict()),
                profile.conference,
                profile.created_at,
                profile.updated_at,
            ))
            await db.commit()
    
    async def get_profile(self, profile_id: str) -> Optional[ContextProfile]:
        """Get a profile by ID."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM profiles WHERE profile_id = ?",
                (profile_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ContextProfile.from_dict(json.loads(row[0]))
        return None
    
    async def get_by_conference(self, conference: str) -> Optional[ContextProfile]:
        """Get a profile for a specific conference."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM profiles WHERE conference = ? ORDER BY updated_at DESC LIMIT 1",
                (conference.lower(),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ContextProfile.from_dict(json.loads(row[0]))
        return None
    
    async def list_profiles(self) -> list[dict]:
        """List all profiles."""
        await self._ensure_initialized()
        
        profiles = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT profile_id, conference, created_at, updated_at FROM profiles ORDER BY updated_at DESC"
            ) as cursor:
                async for row in cursor:
                    profiles.append({
                        "profile_id": row[0],
                        "conference": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                    })
        return profiles


context_profiles_db = ContextProfilesDB()
