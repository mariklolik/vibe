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
from src.db.workflow import workflow_db
from src.project.manager import project_manager


async def _require_active_project() -> tuple[bool, str, any]:
    """Check that an active project exists before paper fetching.
    
    Returns:
        (is_valid, error_message, project)
    """
    current_project = await project_manager.get_current_project()
    if not current_project:
        error_json = json.dumps({
            "success": False,
            "error": "NO_ACTIVE_PROJECT",
            "message": (
                "BLOCKED: You must create a project before gathering papers. "
                "Papers need to be saved to the project context folder for later use."
            ),
            "action_required": "Call create_project(name='your_research_topic') first",
        }, indent=2)
        return False, error_json, None
    
    return True, "", current_project


async def _save_paper_to_project_context(project, paper_data: dict, paper_id: str) -> None:
    """Save a paper's data to the project context folder."""
    context_dir = project.context_dir
    context_dir.mkdir(parents=True, exist_ok=True)
    
    safe_id = paper_id.replace(":", "_").replace("/", "_")
    context_file = context_dir / f"{safe_id}.json"
    
    context_file.write_text(json.dumps(paper_data, indent=2, ensure_ascii=False))
    
    workflow = await workflow_db.get_project_workflow(project.project_id)
    if workflow:
        if paper_id not in workflow.gathered_papers:
            workflow.gathered_papers.append(paper_id)
            await workflow_db.save_workflow(workflow)


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
    """Fetch trending papers from arXiv and save to project context.
    
    REQUIRES: Active project must exist. Papers are saved to project/context/ folder.
    """
    is_valid, error_msg, project = await _require_active_project()
    if not is_valid:
        return error_msg
    
    papers = await arxiv_client.fetch_trending(category, days, max_results)
    
    for paper in papers:
        paper_id = f"arxiv:{paper.arxiv_id}"
        
        cached = CachedPaper(
            paper_id=paper_id,
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
        
        paper_data = {
            "paper_id": paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "arxiv_id": paper.arxiv_id,
            "pdf_url": paper.pdf_url,
            "source": "arxiv",
            "gathered_at": datetime.now().isoformat(),
        }
        await _save_paper_to_project_context(project, paper_data, paper_id)
    
    result = {
        "count": len(papers),
        "category": category,
        "days": days,
        "project": project.project_id,
        "saved_to": str(project.context_dir),
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
    """Fetch trending papers from HuggingFace and save to project context.
    
    REQUIRES: Active project must exist. Papers are saved to project/context/ folder.
    """
    is_valid, error_msg, project = await _require_active_project()
    if not is_valid:
        return error_msg
    
    papers = await hf_client.fetch_trending(topic, days, max_results)
    
    for paper in papers:
        paper_id = f"hf:{paper.paper_id}"
        
        extra_data = {"upvotes": paper.upvotes}
        if paper.metrics:
            extra_data["metrics"] = paper.metrics.to_dict()
        
        cached = CachedPaper(
            paper_id=paper_id,
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
            extra_data=extra_data,
        )
        await papers_cache.cache_paper(cached)
        
        paper_data = {
            "paper_id": paper_id,
            "title": paper.title,
            "abstract": paper.summary,
            "authors": paper.authors,
            "categories": [],
            "published": paper.published_at.isoformat(),
            "arxiv_id": paper.arxiv_id,
            "github_url": paper.github_url,
            "upvotes": paper.upvotes,
            "source": "huggingface",
            "gathered_at": datetime.now().isoformat(),
        }
        await _save_paper_to_project_context(project, paper_data, paper_id)
    
    result = {
        "count": len(papers),
        "topic": topic,
        "days": days,
        "project": project.project_id,
        "saved_to": str(project.context_dir),
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
                "metrics": p.metrics.to_dict() if p.metrics else None,
            }
            for p in papers
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def fetch_hf_trending_with_metrics(
    topic: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """
    Fetch trending papers from HuggingFace with paper metrics (word count, figures, etc).
    
    This is specifically designed for setting target metrics for paper writing.
    It extracts metrics from each paper's PDF/HTML to get realistic targets.
    
    Args:
        topic: Topic to filter papers by (e.g., "attention mechanisms")
        max_results: Maximum papers to fetch and analyze
    
    Returns:
        JSON with papers and their metrics, plus average metrics
    """
    papers = await hf_client.fetch_trending_with_metrics(topic=topic, max_results=max_results)
    
    papers_with_valid_metrics = [
        p for p in papers 
        if p.metrics and p.metrics.word_count > 0
    ]
    
    if papers_with_valid_metrics:
        n = len(papers_with_valid_metrics)
        avg_metrics = {
            "word_count": sum(p.metrics.word_count for p in papers_with_valid_metrics) // n,
            "figure_count": sum(p.metrics.figure_count for p in papers_with_valid_metrics) // n,
            "table_count": sum(p.metrics.table_count for p in papers_with_valid_metrics) // n,
            "page_count": sum(p.metrics.page_count for p in papers_with_valid_metrics) // n,
        }
    else:
        avg_metrics = {
            "word_count": 5000,
            "figure_count": 6,
            "table_count": 3,
            "page_count": 9,
        }
    
    result = {
        "count": len(papers),
        "topic": topic,
        "average_metrics": avg_metrics,
        "papers": [
            {
                "id": p.paper_id,
                "title": p.title,
                "arxiv_id": p.arxiv_id,
                "upvotes": p.upvotes,
                "metrics": p.metrics.to_dict() if p.metrics else None,
            }
            for p in papers
        ],
        "recommendation": (
            f"Based on {len(papers_with_valid_metrics)} analyzed papers, "
            f"target {avg_metrics['word_count']} words, {avg_metrics['figure_count']} figures, "
            f"{avg_metrics['table_count']} tables for your paper."
        ),
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def search_papers(
    query: str,
    max_results: int = 10,
    min_relevance: float = 0.0,
) -> str:
    """Search papers with relevance scoring and save to project context.
    
    REQUIRES: Active project must exist. Papers are saved to project/context/ folder.
    
    Args:
        query: Search query
        max_results: Maximum papers to return
        min_relevance: Minimum relevance score (0.0-1.0) to include paper
    """
    is_valid, error_msg, project = await _require_active_project()
    if not is_valid:
        return error_msg
    
    cached_results = await papers_cache.search(query, max_results * 3)
    
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
    
    scored_results = []
    for p in cached_results:
        score = compute_relevance_score(query, p.title, p.abstract)
        if score >= min_relevance:
            scored_results.append((score, p))
    
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    for score, p in scored_results[:max_results]:
        paper_data = {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": p.authors,
            "categories": p.categories,
            "published": p.published,
            "source": p.source,
            "citation_count": p.citation_count,
            "relevance_score": score,
            "gathered_at": datetime.now().isoformat(),
        }
        await _save_paper_to_project_context(project, paper_data, p.paper_id)
    
    result = {
        "query": query,
        "count": len(scored_results[:max_results]),
        "min_relevance": min_relevance,
        "project": project.project_id,
        "saved_to": str(project.context_dir),
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
