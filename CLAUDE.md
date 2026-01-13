# CLAUDE.md - AI Agent Instructions

This file provides instructions for AI agents (Claude, GPT, etc.) working with this codebase.

## Project Overview

ResearchMCP is an end-to-end AI research automation pipeline designed as an MCP server for Cursor IDE. It enables "vibe-coding" style research workflows where the human provides high-level direction and the AI handles implementation details.

## Architecture

```
research-mcp/
├── src/
│   ├── server.py          # MCP server entry point (70 tools)
│   ├── tools/             # Tool implementations
│   │   ├── aggregation.py # Paper fetching (arXiv, HuggingFace, S2)
│   │   ├── ideas.py       # Idea generation + user approval
│   │   ├── experiments.py # Experiment execution
│   │   ├── visualization.py # Figure generation
│   │   ├── verification.py  # Statistical tests
│   │   ├── writing.py     # LaTeX utilities
│   │   └── formatting.py  # Conference formatting + PDF
│   ├── apis/              # External API clients
│   ├── db/                # SQLite databases
│   ├── context/           # Paper context extraction
│   └── project/           # Project management + git
├── projects/              # Created research projects
├── templates/             # LaTeX templates
└── data/                  # Conference metadata
```

## Key Design Decisions

### 1. LLM Generates Content, Tools Format

The MCP provides formatting utilities, NOT content generation. The LLM in Cursor chat:
- Reads paper abstracts
- Generates creative ideas
- Writes paper sections
- The tools just format to LaTeX/PDF

### 2. User Approval Required for Ideas

Ideas require explicit user approval with confirmation codes:
```
generate_ideas() → returns paper context
LLM calls submit_idea() for each idea
User types: APPROVE idea_xxx CODE 1234
```

This prevents auto-approval of low-quality ideas.

### 3. Workflow Enforcement

Tools check prerequisites before running:
- `run_experiment` requires approved idea
- `cast_to_format` requires generated figures
- `compile_paper` requires paper content

Use `get_next_action()` to see what's needed next.

## Working with This Codebase

### Adding New Tools

1. Implement async function in `src/tools/`
2. Add import in `src/server.py`
3. Add to `TOOL_HANDLERS` dict
4. Add `Tool()` definition with schema
5. Add tests in `tests/`

### API Rate Limiting

- arXiv: 3s delay between requests
- Semantic Scholar: 1s delay, graceful fallback
- HuggingFace: No strict limit

### Database Schema

Papers cached in SQLite with FTS5 for search:
```sql
CREATE TABLE papers (id, title, abstract, authors, ...);
CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract);
```

Ideas stored with status tracking:
```sql
CREATE TABLE ideas (id, title, description, status, confirmation_code, ...);
```

## Common Tasks

### Debug MCP Connection
```bash
# Check if server starts
cd research-mcp
python run_server.py
# Should print "Research MCP Server running..."
```

### Test Tool Manually
```python
import asyncio
from src.tools.aggregation import search_papers

result = asyncio.run(search_papers("attention mechanisms", max_results=5))
print(result)
```

### Clear Caches
```bash
rm -f *.db  # Clears paper cache, experiments db
```

## Error Handling

- API failures: Return graceful error JSON, don't crash
- Missing prerequisites: Return `WORKFLOW_BLOCKED` status
- User approval needed: Return `AWAITING_USER_APPROVAL` status

## Style Guide

- Async functions for all tools
- Type hints on all parameters
- Return JSON strings from tools
- No hardcoded content - LLM generates everything
- Tests for all new functionality
