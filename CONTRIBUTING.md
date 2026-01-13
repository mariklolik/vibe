# Contributing to ResearchMCP

Thank you for your interest in contributing to ResearchMCP!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/mariklolik/vibe.git
cd vibe/research-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
├── server.py           # MCP server, tool registration
├── tools/              # Tool implementations (add new tools here)
├── apis/               # External API clients
├── db/                 # Database models and queries
├── context/            # Paper context extraction
└── project/            # Project management
```

## Adding a New Tool

### 1. Implement the Tool Function

Create or edit a file in `src/tools/`:

```python
# src/tools/my_tools.py
import json
from typing import Optional

async def my_new_tool(
    param1: str,
    param2: Optional[int] = 10,
) -> str:
    """
    Tool description for the AI agent.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        JSON string with results
    """
    # Implementation
    result = {"success": True, "data": ...}
    return json.dumps(result, indent=2)
```

### 2. Register in server.py

```python
# Import
from src.tools.my_tools import my_new_tool

# Add to TOOL_HANDLERS
TOOL_HANDLERS = {
    ...
    "my_new_tool": my_new_tool,
}

# Add Tool definition
Tool(
    name="my_new_tool",
    description="Description for AI agent",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer", "default": 10},
        },
        "required": ["param1"],
    },
),
```

### 3. Add Tests

```python
# tests/test_my_tools.py
import pytest
from src.tools.my_tools import my_new_tool

@pytest.mark.asyncio
async def test_my_new_tool():
    result = await my_new_tool("test_input")
    data = json.loads(result)
    assert data["success"] is True
```

## Code Style

- **Python 3.10+** with type hints
- **Async functions** for all tools
- **JSON returns** from all tools
- **No hardcoded content** - LLM generates text
- **Docstrings** on all public functions

### Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ideas.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit tests**: Individual tool functions
- **Integration tests**: Multi-tool workflows
- **E2E tests**: Full research pipeline

## Pull Request Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/ -v`
5. **Commit**: `git commit -m "Add my feature"`
6. **Push**: `git push origin feature/my-feature`
7. **Open PR** with description

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Docstrings added
- [ ] CHANGELOG.md updated
- [ ] No hardcoded content

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Describe:
- Use case
- Proposed solution
- Alternatives considered

## Questions?

Open an issue or start a discussion on GitHub.
