"""SQLite-based paper caching system."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite


@dataclass
class CachedPaper:
    paper_id: str
    source: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    arxiv_id: Optional[str]
    doi: Optional[str]
    pdf_url: Optional[str]
    code_url: Optional[str]
    citation_count: int
    cached_at: str
    extra_data: dict

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "source": self.source,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "code_url": self.code_url,
            "citation_count": self.citation_count,
            "cached_at": self.cached_at,
            "extra_data": self.extra_data,
        }


class PapersCache:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "research-mcp"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "papers.db")
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self):
        # Check if db file exists - if deleted, need to reinitialize
        if self._initialized and Path(self.db_path).exists():
            return
        self._initialized = False
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors TEXT,
                    categories TEXT,
                    published TEXT,
                    arxiv_id TEXT,
                    doi TEXT,
                    pdf_url TEXT,
                    code_url TEXT,
                    citation_count INTEGER DEFAULT 0,
                    cached_at TEXT NOT NULL,
                    extra_data TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published)
            """)
            
            # Standalone FTS5 table (not content-linked to avoid sync issues)
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    paper_id,
                    title,
                    abstract
                )
            """)
            
            await db.commit()
        
        self._initialized = True

    async def cache_paper(self, paper: CachedPaper) -> None:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO papers 
                (paper_id, source, title, abstract, authors, categories, published,
                 arxiv_id, doi, pdf_url, code_url, citation_count, cached_at, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.paper_id,
                paper.source,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                json.dumps(paper.categories),
                paper.published,
                paper.arxiv_id,
                paper.doi,
                paper.pdf_url,
                paper.code_url,
                paper.citation_count,
                paper.cached_at,
                json.dumps(paper.extra_data),
            ))
            
            await db.execute("""
                INSERT OR REPLACE INTO papers_fts(paper_id, title, abstract)
                VALUES (?, ?, ?)
            """, (paper.paper_id, paper.title, paper.abstract))
            
            await db.commit()

    async def get_paper(self, paper_id: str) -> Optional[CachedPaper]:
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM papers WHERE paper_id = ? OR arxiv_id = ?",
                (paper_id, paper_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_paper(row)
        return None

    async def search(self, query: str, max_results: int = 10) -> list[CachedPaper]:
        await self._ensure_initialized()
        
        papers = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT p.* FROM papers p
                JOIN papers_fts fts ON p.paper_id = fts.paper_id
                WHERE papers_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, max_results)) as cursor:
                async for row in cursor:
                    papers.append(self._row_to_paper(row))
        
        return papers

    async def get_recent(
        self,
        source: Optional[str] = None,
        days: int = 7,
        limit: int = 50,
    ) -> list[CachedPaper]:
        await self._ensure_initialized()
        
        cutoff = datetime.now().isoformat()[:10]
        
        papers = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if source:
                query = """
                    SELECT * FROM papers 
                    WHERE source = ? AND cached_at >= ?
                    ORDER BY published DESC
                    LIMIT ?
                """
                params = (source, cutoff, limit)
            else:
                query = """
                    SELECT * FROM papers 
                    WHERE cached_at >= ?
                    ORDER BY published DESC
                    LIMIT ?
                """
                params = (cutoff, limit)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    papers.append(self._row_to_paper(row))
        
        return papers

    async def get_by_category(
        self,
        category: str,
        limit: int = 50,
    ) -> list[CachedPaper]:
        await self._ensure_initialized()
        
        papers = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM papers 
                WHERE categories LIKE ?
                ORDER BY published DESC
                LIMIT ?
            """, (f'%"{category}"%', limit)) as cursor:
                async for row in cursor:
                    papers.append(self._row_to_paper(row))
        
        return papers

    def _row_to_paper(self, row: aiosqlite.Row) -> CachedPaper:
        return CachedPaper(
            paper_id=row["paper_id"],
            source=row["source"],
            title=row["title"],
            abstract=row["abstract"] or "",
            authors=json.loads(row["authors"] or "[]"),
            categories=json.loads(row["categories"] or "[]"),
            published=row["published"] or "",
            arxiv_id=row["arxiv_id"],
            doi=row["doi"],
            pdf_url=row["pdf_url"],
            code_url=row["code_url"],
            citation_count=row["citation_count"] or 0,
            cached_at=row["cached_at"],
            extra_data=json.loads(row["extra_data"] or "{}"),
        )

    async def clear(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM papers")
            await db.execute("DELETE FROM papers_fts")
            await db.commit()


papers_cache = PapersCache()
