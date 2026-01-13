"""Semantic Scholar API client."""

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class S2Paper:
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: list[dict]
    year: Optional[int]
    venue: Optional[str]
    citation_count: int
    reference_count: int
    influential_citation_count: int
    arxiv_id: Optional[str]
    doi: Optional[str]
    url: str
    open_access_pdf: Optional[str]
    fields_of_study: list[str]

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "influential_citation_count": self.influential_citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "url": self.url,
            "open_access_pdf": self.open_access_pdf,
            "fields_of_study": self.fields_of_study,
        }


@dataclass
class S2Citation:
    citing_paper: S2Paper
    is_influential: bool
    contexts: list[str]
    intents: list[str]


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "externalIds",
        "url",
        "openAccessPdf",
        "fieldsOfStudy",
    ]

    def __init__(self, api_key: Optional[str] = None):
        import asyncio
        self.api_key = api_key
        self._last_request = 0.0
        self._rate_limit_delay = 1.0  # 1 second between requests (100 req/100s limit)
        headers = {
            "User-Agent": "research-mcp/1.0 (academic research tool)"
        }
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
    
    async def _rate_limit(self):
        """Ensure rate limiting between requests."""
        import asyncio
        import time
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _parse_paper(self, data: dict) -> S2Paper:
        external_ids = data.get("externalIds", {}) or {}
        open_access = data.get("openAccessPdf", {}) or {}
        
        return S2Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            authors=[{"name": a.get("name", ""), "authorId": a.get("authorId")} 
                    for a in (data.get("authors") or [])],
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0),
            reference_count=data.get("referenceCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            arxiv_id=external_ids.get("ArXiv"),
            doi=external_ids.get("DOI"),
            url=data.get("url", ""),
            open_access_pdf=open_access.get("url"),
            fields_of_study=data.get("fieldsOfStudy") or [],
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        year_range: Optional[tuple[int, int]] = None,
        fields_of_study: Optional[list[str]] = None,
    ) -> list[S2Paper]:
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(self.PAPER_FIELDS),
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        try:
            await self._rate_limit()
            response = await self.client.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
            )
            
            if response.status_code == 200:
                data = response.json()
                return [self._parse_paper(p) for p in data.get("data", [])]
            elif response.status_code == 429:
                # Rate limited - wait and retry once
                import asyncio
                await asyncio.sleep(5)
                response = await self.client.get(f"{self.BASE_URL}/paper/search", params=params)
                if response.status_code == 200:
                    return [self._parse_paper(p) for p in response.json().get("data", [])]
                return []
            elif response.status_code == 403:
                # API may be blocking - return empty rather than error
                return []
            else:
                raise RuntimeError(f"S2 API error: {response.status_code}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to search Semantic Scholar: {e}")

    async def get_paper(self, paper_id: str) -> Optional[S2Paper]:
        if paper_id.startswith("arxiv:") or paper_id.startswith("arXiv:"):
            paper_id = f"ARXIV:{paper_id.split(':')[1]}"
        elif "/" not in paper_id and not paper_id.startswith("ARXIV:"):
            if paper_id[0].isdigit():
                paper_id = f"ARXIV:{paper_id}"
        
        try:
            await self._rate_limit()
            response = await self.client.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": ",".join(self.PAPER_FIELDS)},
            )
            
            if response.status_code == 200:
                return self._parse_paper(response.json())
            elif response.status_code == 404:
                return None
            elif response.status_code in (403, 429):
                return None  # Rate limited or blocked
            else:
                raise RuntimeError(f"S2 API error: {response.status_code}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to get paper from Semantic Scholar: {e}")

    async def get_citations(
        self,
        paper_id: str,
        max_results: int = 100,
    ) -> list[S2Paper]:
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": max_results,
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                return [self._parse_paper(c.get("citingPaper", {})) 
                       for c in data.get("data", []) 
                       if c.get("citingPaper")]
            else:
                return []
        except httpx.HTTPError:
            return []

    async def get_references(
        self,
        paper_id: str,
        max_results: int = 100,
    ) -> list[S2Paper]:
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": max_results,
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                return [self._parse_paper(r.get("citedPaper", {})) 
                       for r in data.get("data", []) 
                       if r.get("citedPaper")]
            else:
                return []
        except httpx.HTTPError:
            return []

    async def get_recommendations(
        self,
        paper_ids: list[str],
        max_results: int = 10,
    ) -> list[S2Paper]:
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/recommendations",
                json={"positivePaperIds": paper_ids},
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": max_results,
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                return [self._parse_paper(p) for p in data.get("recommendedPapers", [])]
            else:
                return []
        except httpx.HTTPError:
            return []

    async def close(self):
        await self.client.aclose()


s2_client = SemanticScholarClient()
