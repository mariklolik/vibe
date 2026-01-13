"""arXiv API client."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import arxiv
import httpx


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    html_url: Optional[str]
    comment: Optional[str]
    doi: Optional[str]
    primary_category: str

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "html_url": self.html_url,
            "comment": self.comment,
            "doi": self.doi,
            "primary_category": self.primary_category,
        }


class ArxivClient:
    CATEGORY_MAP = {
        "ml": "cs.LG",
        "machine_learning": "cs.LG",
        "nlp": "cs.CL",
        "cv": "cs.CV",
        "vision": "cs.CV",
        "ai": "cs.AI",
        "robotics": "cs.RO",
        "ir": "cs.IR",
        "neural": "cs.NE",
        "stat_ml": "stat.ML",
    }

    def __init__(self):
        # Rate limiting: 3 second delay between requests to avoid 403 errors
        self.client = arxiv.Client(
            delay_seconds=3.0,
            num_retries=3,
        )

    def _normalize_category(self, category: str) -> str:
        return self.CATEGORY_MAP.get(category.lower(), category)

    async def fetch_trending(
        self,
        category: str,
        days: int = 7,
        max_results: int = 20,
    ) -> list[ArxivPaper]:
        category = self._normalize_category(category)
        
        query = f"cat:{category}"
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in await asyncio.to_thread(lambda: list(self.client.results(search))):
            paper = ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published=result.published,
                updated=result.updated,
                pdf_url=result.pdf_url,
                html_url=f"https://arxiv.org/html/{result.entry_id.split('/')[-1]}",
                comment=result.comment,
                doi=result.doi,
                primary_category=result.primary_category,
            )
            papers.append(paper)

        return papers

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[ArxivPaper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in await asyncio.to_thread(lambda: list(self.client.results(search))):
            paper = ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published=result.published,
                updated=result.updated,
                pdf_url=result.pdf_url,
                html_url=f"https://arxiv.org/html/{result.entry_id.split('/')[-1]}",
                comment=result.comment,
                doi=result.doi,
                primary_category=result.primary_category,
            )
            papers.append(paper)

        return papers

    async def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        
        search = arxiv.Search(id_list=[arxiv_id])
        results = await asyncio.to_thread(lambda: list(self.client.results(search)))
        
        if not results:
            return None

        result = results[0]
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            abstract=result.summary,
            authors=[str(a) for a in result.authors],
            categories=result.categories,
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
            html_url=f"https://arxiv.org/html/{result.entry_id.split('/')[-1]}",
            comment=result.comment,
            doi=result.doi,
            primary_category=result.primary_category,
        )

    async def get_code_url(self, arxiv_id: str) -> Optional[str]:
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        paper = data["results"][0]
                        if paper.get("proceeding"):
                            repo_response = await client.get(
                                f"https://paperswithcode.com/api/v1/papers/{paper['id']}/repositories/"
                            )
                            if repo_response.status_code == 200:
                                repos = repo_response.json().get("results", [])
                                if repos:
                                    return repos[0].get("url")
            except Exception:
                pass
        
        return None


arxiv_client = ArxivClient()
