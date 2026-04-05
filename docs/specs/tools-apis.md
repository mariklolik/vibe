# Spec: Tools / APIs

**Module:** `src/agents/client.py`, `src/apis/`

## Anthropic API (via proxy)

### Contract

All LLM calls go through `AgentClient.call()`. The client never calls the Anthropic API directly — all requests route through `localhost:3456`.

```python
response = await client.call(
    model="sonnet",          # short name; proxy normalises to full ID
    system=system_prompt,
    messages=[{"role": "user", "content": content}],
    effort="high",           # proxy feature: low / medium / high
    max_budget_usd=0.50,     # proxy feature: per-call cap (optional)
    fallback=["haiku"],      # proxy feature: fallback chain on rate limit
)
```

### Errors and handling

| Error | HTTP code | Handling |
|-------|-----------|---------|
| Rate limit | 429 | Exponential backoff (1 s, 2 s, 4 s); then fallback model |
| Budget exceeded | 402 (proxy) | Raise `BudgetExceededError`; pipeline aborted |
| Server error | 500/503 | Retry × 3 with 2 s delay |
| Timeout (> 30 s sonnet / 60 s opus) | — | `asyncio.TimeoutError`; treated as server error |
| Invalid response (no content) | 200 but empty | Retry × 1; if still empty, raise `EmptyResponseError` |

### Side effects

- Token usage accumulated in `client.total_tokens` and logged to `progress.txt` after each call.
- No state written to disk by the client itself.

### Timeouts

| Model | Timeout |
|-------|---------|
| haiku | 15 s |
| sonnet | 30 s |
| opus | 60 s |

## Paper APIs

### arXiv (`src/apis/arxiv.py`)

- **Endpoint:** `http://export.arxiv.org/api/query?search_query=...&max_results=N`
- **Output:** Atom XML parsed to list of `{id, title, abstract, authors, published}`
- **Error handling:** HTTP error → log warning, return empty list
- **Rate limit:** 3 s `asyncio.sleep` between consecutive calls

### HuggingFace (`src/apis/huggingface.py`)

- **Endpoint:** `https://huggingface.co/api/daily_papers?date=...`
- **Output:** JSON list of `{title, abstract, authors, upvotes}`
- **Error handling:** 5xx → return empty list; no retry
- **Note:** No arXiv ID guaranteed; matched by title for deduplication

### Semantic Scholar (`src/apis/semantic_scholar.py`)

- **Endpoint:** `https://api.semanticscholar.org/graph/v1/paper/search`
- **Query params:** `query`, `fields=title,abstract,authors,year,citationCount`, `limit=N`
- **Output:** List of `{paperId, title, abstract, year, citationCount}`
- **Error handling:** 429 → log warning, return empty list (graceful fallback)
- **Rate limit:** 1 s sleep between requests; optional `X-API-KEY` header
