"""Paper aggregation tools - fetch and search papers from arXiv, HuggingFace."""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.apis.arxiv import arxiv_client, ArxivPaper
from src.apis.huggingface import hf_client, HFPaper
from src.apis.semantic_scholar import s2_client
from src.db.papers_cache import papers_cache, CachedPaper


def compute_relevance_score(query: str, title: str, abstract: str) -> float:
    """Compute relevance score based on keyword matching in title/abstract.
    
    Returns a score from 0.0 to 1.0 based on how many query terms appear.
    """
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    title_terms = set(re.findall(r'\b\w+\b', title.lower()))
    abstract_terms = set(re.findall(r'\b\w+\b', abstract.lower()))
    
    if not query_terms:
        return 0.0
    
    # Title matches are worth more
    title_matches = len(query_terms & title_terms)
    abstract_matches = len(query_terms & abstract_terms)
    
    # Score: title match = 2 points, abstract match = 1 point
    max_score = len(query_terms) * 3  # All terms in both title and abstract
    actual_score = (title_matches * 2) + abstract_matches
    
    return min(1.0, actual_score / max_score) if max_score > 0 else 0.0


async def fetch_arxiv_trending(
    category: str,
    days: int = 7,
    max_results: int = 20,
) -> str:
    papers = await arxiv_client.fetch_trending(category, days, max_results)
    
    for paper in papers:
        cached = CachedPaper(
            paper_id=f"arxiv:{paper.arxiv_id}",
            source="arxiv",
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            categories=paper.categories,
            published=paper.published.isoformat(),
            arxiv_id=paper.arxiv_id,
            doi=paper.doi,
            pdf_url=paper.pdf_url,
            code_url=None,
            citation_count=0,
            cached_at=datetime.now().isoformat(),
            extra_data={"comment": paper.comment, "html_url": paper.html_url},
        )
        await papers_cache.cache_paper(cached)
    
    result = {
        "count": len(papers),
        "category": category,
        "days": days,
        "papers": [
            {
                "id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:3],
                "abstract": p.abstract[:300] + "..." if len(p.abstract) > 300 else p.abstract,
                "published": p.published.strftime("%Y-%m-%d"),
                "categories": p.categories,
                "pdf": p.pdf_url,
            }
            for p in papers
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def fetch_hf_trending(
    topic: Optional[str] = None,
    days: int = 7,
    max_results: int = 20,
) -> str:
    papers = await hf_client.fetch_trending(topic, days, max_results)
    
    for paper in papers:
        cached = CachedPaper(
            paper_id=f"hf:{paper.paper_id}",
            source="huggingface",
            title=paper.title,
            abstract=paper.summary,
            authors=paper.authors,
            categories=[],
            published=paper.published_at.isoformat(),
            arxiv_id=paper.arxiv_id,
            doi=None,
            pdf_url=None,
            code_url=paper.github_url,
            citation_count=paper.upvotes,
            cached_at=datetime.now().isoformat(),
            extra_data={"upvotes": paper.upvotes},
        )
        await papers_cache.cache_paper(cached)
    
    result = {
        "count": len(papers),
        "topic": topic,
        "days": days,
        "papers": [
            {
                "id": p.paper_id,
                "title": p.title,
                "authors": p.authors[:3],
                "summary": p.summary[:300] + "..." if len(p.summary) > 300 else p.summary,
                "published": p.published_at.strftime("%Y-%m-%d"),
                "upvotes": p.upvotes,
                "arxiv_id": p.arxiv_id,
                "github": p.github_url,
            }
            for p in papers
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def search_papers(
    query: str,
    max_results: int = 10,
    min_relevance: float = 0.0,
) -> str:
    """Search papers with relevance scoring.
    
    Args:
        query: Search query
        max_results: Maximum papers to return
        min_relevance: Minimum relevance score (0.0-1.0) to include paper
    """
    cached_results = await papers_cache.search(query, max_results * 3)  # Fetch more for filtering
    
    if len(cached_results) < max_results * 2:
        try:
            arxiv_results = await arxiv_client.search(query, max_results * 2)
            for paper in arxiv_results:
                cached = CachedPaper(
                    paper_id=f"arxiv:{paper.arxiv_id}",
                    source="arxiv",
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    categories=paper.categories,
                    published=paper.published.isoformat(),
                    arxiv_id=paper.arxiv_id,
                    doi=paper.doi,
                    pdf_url=paper.pdf_url,
                    code_url=None,
                    citation_count=0,
                    cached_at=datetime.now().isoformat(),
                    extra_data={"comment": paper.comment},
                )
                await papers_cache.cache_paper(cached)
                cached_results.append(cached)
        except Exception:
            pass
    
    if len(cached_results) < max_results * 2:
        try:
            s2_results = await s2_client.search(query, max_results * 2)
            for paper in s2_results:
                cached = CachedPaper(
                    paper_id=f"s2:{paper.paper_id}",
                    source="semantic_scholar",
                    title=paper.title,
                    abstract=paper.abstract or "",
                    authors=[a["name"] for a in paper.authors],
                    categories=paper.fields_of_study,
                    published=str(paper.year) if paper.year else "",
                    arxiv_id=paper.arxiv_id,
                    doi=paper.doi,
                    pdf_url=paper.open_access_pdf,
                    code_url=None,
                    citation_count=paper.citation_count,
                    cached_at=datetime.now().isoformat(),
                    extra_data={
                        "venue": paper.venue,
                        "influential_citations": paper.influential_citation_count,
                    },
                )
                await papers_cache.cache_paper(cached)
                cached_results.append(cached)
        except Exception:
            pass
    
    # Score and sort by relevance
    scored_results = []
    for p in cached_results:
        score = compute_relevance_score(query, p.title, p.abstract)
        if score >= min_relevance:
            scored_results.append((score, p))
    
    # Sort by relevance score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    result = {
        "query": query,
        "count": len(scored_results[:max_results]),
        "min_relevance": min_relevance,
        "papers": [
            {
                "id": p.paper_id,
                "title": p.title,
                "authors": p.authors[:3],
                "abstract": p.abstract[:300] + "..." if len(p.abstract) > 300 else p.abstract,
                "source": p.source,
                "citations": p.citation_count,
                "relevance_score": round(score, 3),
            }
            for score, p in scored_results[:max_results]
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def get_paper_details(paper_id: str) -> str:
    cached = await papers_cache.get_paper(paper_id)
    
    if cached:
        if not cached.code_url and cached.arxiv_id:
            code_url = await arxiv_client.get_code_url(cached.arxiv_id)
            if code_url:
                cached.code_url = code_url
                await papers_cache.cache_paper(cached)
        
        s2_paper = await s2_client.get_paper(cached.arxiv_id or paper_id)
        if s2_paper:
            cached.citation_count = s2_paper.citation_count
            await papers_cache.cache_paper(cached)
        
        result = cached.to_dict()
        
        if s2_paper:
            references = await s2_client.get_references(s2_paper.paper_id, max_results=10)
            citations = await s2_client.get_citations(s2_paper.paper_id, max_results=10)
            
            result["references"] = [
                {"title": r.title, "year": r.year, "citations": r.citation_count}
                for r in references
            ]
            result["recent_citations"] = [
                {"title": c.title, "year": c.year, "citations": c.citation_count}
                for c in citations
            ]
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    arxiv_paper = await arxiv_client.get_paper(paper_id)
    if arxiv_paper:
        code_url = await arxiv_client.get_code_url(paper_id)
        
        cached = CachedPaper(
            paper_id=f"arxiv:{arxiv_paper.arxiv_id}",
            source="arxiv",
            title=arxiv_paper.title,
            abstract=arxiv_paper.abstract,
            authors=arxiv_paper.authors,
            categories=arxiv_paper.categories,
            published=arxiv_paper.published.isoformat(),
            arxiv_id=arxiv_paper.arxiv_id,
            doi=arxiv_paper.doi,
            pdf_url=arxiv_paper.pdf_url,
            code_url=code_url,
            citation_count=0,
            cached_at=datetime.now().isoformat(),
            extra_data={"comment": arxiv_paper.comment, "html_url": arxiv_paper.html_url},
        )
        await papers_cache.cache_paper(cached)
        
        return json.dumps(cached.to_dict(), indent=2, ensure_ascii=False)
    
    return json.dumps({"error": f"Paper not found: {paper_id}"})


async def clone_paper_code(
    paper_id: str,
    target_dir: Optional[str] = None,
) -> str:
    cached = await papers_cache.get_paper(paper_id)
    
    code_url = None
    if cached and cached.code_url:
        code_url = cached.code_url
    elif cached and cached.arxiv_id:
        code_url = await arxiv_client.get_code_url(cached.arxiv_id)
    else:
        code_url = await arxiv_client.get_code_url(paper_id)
    
    if not code_url:
        return json.dumps({
            "success": False,
            "error": f"No code repository found for paper: {paper_id}",
        })
    
    if target_dir is None:
        target_dir = "./paper_code"
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    repo_name = code_url.rstrip("/").split("/")[-1]
    clone_path = target_path / repo_name
    
    if clone_path.exists():
        return json.dumps({
            "success": True,
            "path": str(clone_path),
            "message": "Repository already exists",
            "url": code_url,
        })
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", code_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0:
            return json.dumps({
                "success": True,
                "path": str(clone_path),
                "url": code_url,
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.stderr,
                "url": code_url,
            })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Clone operation timed out",
            "url": code_url,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "url": code_url,
        })
