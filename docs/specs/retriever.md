# Spec: Retriever (Literature Search)

**Module:** `src/apis/` + `src/agents/research.py` (search phase)

## Purpose

Fetch, deduplicate, and rank academic papers from multiple sources to provide the research agent with relevant prior work.

## Sources

| Source | API | Auth | Rate limit | Fallback |
|--------|-----|------|-----------|---------|
| arXiv | `http://export.arxiv.org/api/query` | None | 3 s between requests | Skip if timeout |
| HuggingFace Papers | `https://huggingface.co/api/daily_papers` | None | No strict limit | Skip if 5xx |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/search` | Optional API key | 1 req/s | Skip on 429 |

## Index

No persistent vector index. Each pipeline run fetches fresh results. Papers are stored in `context/papers.json` and expire when the project is deleted.

## Search

1. Query each API with the topic string.
2. Collect raw results (title, abstract, authors, year, citation count, URL).
3. Deduplicate by arXiv ID (or title similarity if no arXiv ID).
4. Score each paper: `score = log(1 + citations) × recency_decay(year)` where `recency_decay = 1.0` for current year, decreasing by 0.15 per year.
5. Return top-N by score (default N = 15 before passing to LLM; LLM selects top 5 for deep context extraction).

## Reranking

The research agent re-ranks candidates based on semantic relevance to the specific idea, not just the topic. This is an LLM call (sonnet, medium effort) that returns a ranked list with relevance justification.

## Constraints

| Constraint | Value |
|-----------|-------|
| Max papers returned per source | 20 |
| Min citation count | 5 (configurable) |
| Max paper age | 3 years (configurable) |
| Request timeout | 10 s per source |
| Total retrieval time target | ≤ 60 s |
