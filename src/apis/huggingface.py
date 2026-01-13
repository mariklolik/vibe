"""HuggingFace API client for trending papers."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx


@dataclass
class HFPaper:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    published_at: datetime
    upvotes: int
    arxiv_id: Optional[str]
    github_url: Optional[str]

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "summary": self.summary,
            "authors": self.authors,
            "published_at": self.published_at.isoformat(),
            "upvotes": self.upvotes,
            "arxiv_id": self.arxiv_id,
            "github_url": self.github_url,
        }


class HuggingFaceClient:
    BASE_URL = "https://huggingface.co/api"
    PAPERS_URL = "https://huggingface.co/papers"
    TRENDING_URL = "https://huggingface.co/papers/trending"

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "research-mcp/1.0"}
        )
    
    async def fetch_trending_with_query(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[HFPaper]:
        """Fetch trending papers with a search query.
        
        Parses https://huggingface.co/papers/trending?q=<query>
        Extracts searchResults JSON from the page.
        """
        papers = []
        
        try:
            response = await self.client.get(
                self.TRENDING_URL,
                params={"q": query},
            )
            
            if response.status_code != 200:
                return papers
            
            html = response.text
            
            # Extract searchResults JSON embedded in the page
            papers = self._parse_search_results_json(html, max_results)
            
            # Fallback to HTML parsing if JSON extraction fails
            if not papers:
                papers = self._parse_trending_html(html, max_results)
            
        except Exception:
            pass
        
        return papers
    
    def _parse_search_results_json(self, html: str, max_results: int) -> list[HFPaper]:
        """Extract searchResults from the page using regex on unescaped content."""
        import html as html_module
        
        papers = []
        
        # Unescape HTML entities first
        page = html_module.unescape(html)
        
        # Find searchResults section
        idx = page.find('"searchResults":')
        if idx == -1:
            return papers
        
        try:
            # Get section of page containing search results
            section = page[idx:idx + 100000]
            
            # Extract paper IDs and titles using regex
            # Pattern matches: "id":"XXXX.XXXXX"... "title":"..."
            pattern = r'"id":"(\d{4}\.\d{4,5})".*?"title":"([^"]+)"'
            matches = re.findall(pattern, section)
            
            seen_ids = set()
            for arxiv_id, title in matches:
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)
                
                # Also try to extract summary/abstract
                summary = ""
                summary_pattern = rf'"id":"{arxiv_id}".*?"summary":"([^"]*)"'
                summary_match = re.search(summary_pattern, section, re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1)[:500]
                
                paper = HFPaper(
                    paper_id=arxiv_id,
                    title=title,
                    summary=summary,
                    authors=[],
                    published_at=datetime.now(),
                    upvotes=0,
                    arxiv_id=arxiv_id,
                    github_url=None,
                )
                papers.append(paper)
                
                if len(papers) >= max_results:
                    break
                    
        except Exception:
            pass
        
        return papers
    
    def _parse_trending_html(self, html: str, max_results: int) -> list[HFPaper]:
        """Parse the HuggingFace trending papers HTML page."""
        papers = []
        
        # Pattern: h3 or anchor with href to /papers/XXXX.XXXXX and title
        # <a href="/papers/2407.08250" ...>Title</a>
        pattern = r'<a[^>]*href="/papers/(\d{4}\.\d{4,5})"[^>]*>\s*([^<]+?)\s*</a>'
        matches = re.findall(pattern, html)
        
        seen_ids = set()
        for arxiv_id, title in matches:
            if arxiv_id in seen_ids:
                continue
            
            # Clean title
            title = " ".join(title.split()).strip()
            if not title or len(title) < 10:
                continue
            
            seen_ids.add(arxiv_id)
            
            paper = HFPaper(
                paper_id=arxiv_id,
                title=title,
                summary="",
                authors=[],
                published_at=datetime.now(),
                upvotes=0,
                arxiv_id=arxiv_id,
                github_url=None,
            )
            papers.append(paper)
            
            if len(papers) >= max_results:
                break
        
        return papers

    async def fetch_trending(
        self,
        topic: Optional[str] = None,
        days: int = 365,  # Default to 1 year for topic search
        max_results: int = 20,
    ) -> list[HFPaper]:
        """Fetch trending papers.
        
        If topic is provided, uses the search-based trending page for relevant results.
        Otherwise fetches general daily papers.
        """
        # If topic provided, use search-based trending for better relevance
        if topic:
            papers = await self.fetch_trending_with_query(topic, max_results * 2)
            
            # Enrich papers with details while preserving original data
            enriched = []
            for paper in papers[:max_results]:
                detailed = await self.get_paper(paper.paper_id)
                if detailed:
                    # Preserve arxiv_id from original if detailed doesn't have it
                    if not detailed.arxiv_id and paper.arxiv_id:
                        detailed.arxiv_id = paper.arxiv_id
                    # Preserve title if detailed is empty
                    if not detailed.title and paper.title:
                        detailed.title = paper.title
                    enriched.append(detailed)
                else:
                    enriched.append(paper)
            
            return enriched[:max_results]
        
        # No topic - fetch general daily papers
        papers = []
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/daily_papers",
                params={"limit": max_results * 2},
            )
            
            if response.status_code != 200:
                response = await self.client.get(
                    "https://huggingface.co/api/papers",
                    params={"limit": max_results * 2},
                )
            
            if response.status_code == 200:
                data = response.json()
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for item in data:
                    paper_data = item.get("paper", item)
                    
                    published_str = paper_data.get("publishedAt", paper_data.get("published_at", ""))
                    if published_str:
                        try:
                            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        except ValueError:
                            published = datetime.now()
                    else:
                        published = datetime.now()
                    
                    if published.replace(tzinfo=None) < cutoff_date:
                        continue
                    
                    title = paper_data.get("title", "")
                    summary = paper_data.get("summary", paper_data.get("abstract", ""))
                    
                    arxiv_id = paper_data.get("arxiv_id") or paper_data.get("id", "")
                    if arxiv_id.startswith("http"):
                        arxiv_id = arxiv_id.split("/")[-1]
                    
                    paper = HFPaper(
                        paper_id=paper_data.get("id", arxiv_id),
                        title=title,
                        summary=summary,
                        authors=[a.get("name", str(a)) if isinstance(a, dict) else str(a) 
                                for a in paper_data.get("authors", [])],
                        published_at=published,
                        upvotes=item.get("upvotes", paper_data.get("upvotes", 0)),
                        arxiv_id=arxiv_id if arxiv_id else None,
                        github_url=paper_data.get("github_url"),
                    )
                    papers.append(paper)
                    
                    if len(papers) >= max_results:
                        break
                
                papers.sort(key=lambda p: p.upvotes, reverse=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to fetch HuggingFace papers: {e}")
        
        return papers[:max_results]

    async def get_paper(self, paper_id: str) -> Optional[HFPaper]:
        try:
            response = await self.client.get(f"{self.BASE_URL}/papers/{paper_id}")
            
            if response.status_code == 200:
                paper_data = response.json()
                
                published_str = paper_data.get("publishedAt", "")
                if published_str:
                    try:
                        published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    except ValueError:
                        published = datetime.now()
                else:
                    published = datetime.now()
                
                return HFPaper(
                    paper_id=paper_data.get("id", paper_id),
                    title=paper_data.get("title", ""),
                    summary=paper_data.get("summary", ""),
                    authors=[a.get("name", str(a)) if isinstance(a, dict) else str(a) 
                            for a in paper_data.get("authors", [])],
                    published_at=published,
                    upvotes=paper_data.get("upvotes", 0),
                    arxiv_id=paper_data.get("arxiv_id"),
                    github_url=paper_data.get("github_url"),
                )
        except Exception:
            pass
        
        return None

    async def close(self):
        await self.client.aclose()


hf_client = HuggingFaceClient()
