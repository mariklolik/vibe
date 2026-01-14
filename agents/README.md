# Multi-Agent Research Pipeline

Three separate MCP servers, each handling a distinct phase of the research pipeline.
Each agent has a focused set of tools to prevent context overload.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Shared State                                │
│  ~/.cache/research-mcp/                                             │
│  ├── workflows.db   (workflow state, progress)                      │
│  ├── papers.db      (cached papers)                                 │
│  └── experiments.db (ideas, experiments, metrics)                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  researcher   │────▶│  experimenter │────▶│    writer     │
│     -mcp      │     │      -mcp     │     │     -mcp      │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ 16 tools      │     │ 22 tools      │     │ 17 tools      │
│               │     │               │     │               │
│ • Paper fetch │     │ • Env setup   │     │ • Context     │
│ • Idea gen    │     │ • Experiments │     │ • Formatting  │
│ • Approval    │     │ • Verification│     │ • LaTeX       │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Setup

1. Add MCP configs to Cursor settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "researcher-mcp": {
      "command": "python",
      "args": ["-m", "agents.researcher.server"],
      "cwd": "/path/to/research-mcp"
    },
    "experimenter-mcp": {
      "command": "python", 
      "args": ["-m", "agents.experimenter.server"],
      "cwd": "/path/to/research-mcp"
    },
    "writer-mcp": {
      "command": "python",
      "args": ["-m", "agents.writer.server"],
      "cwd": "/path/to/research-mcp"
    }
  }
}
```

2. Enable only ONE agent per chat session.

## Workflow

### Phase 1: Research (researcher-mcp)

Start a new Cursor chat. The researcher agent will:

1. Create a project
2. Fetch trending papers (arXiv, HuggingFace)
3. Extract context from papers
4. Generate research ideas based on papers
5. Wait for user approval of an idea

**User action required:** Type `APPROVE <idea_id> CODE <confirmation_code>`

After approval, the agent will prompt you to switch to experimenter-mcp.

### Phase 2: Experiments (experimenter-mcp)

Start a NEW Cursor chat with experimenter-mcp. This agent will:

1. Read the approved idea from the database
2. Define testable hypotheses
3. Set up experiment environment
4. Implement experiment code
5. Run experiments on GPU
6. Verify results with statistical tests

After verification, switch to writer-mcp.

### Phase 3: Writing (writer-mcp)

Start a NEW Cursor chat with writer-mcp. This agent will:

1. Read verified claims from the database
2. Gather writing context from papers
3. Write paper sections
4. Format for target conference (ICML, NeurIPS, etc.)
5. Compile to PDF

## CLI Status

Check current pipeline status:

```bash
python -m agents.cli status
```

List projects:

```bash
python -m agents.cli projects
```

## Key Constraints

1. **All LLM processing is done by Cursor chat** - no external API keys
2. **One agent per chat** - prevents context overload
3. **State persists in SQLite** - allows resuming across sessions
4. **Ideas require human approval** - confirmation codes prevent auto-approval
5. **Claims must be verified** - statistical tests before paper writing

## Tool Counts

| Agent | Tools | Focus |
|-------|-------|-------|
| researcher-mcp | 16 | Paper discovery, ideas |
| experimenter-mcp | 22 | Experiments, verification |
| writer-mcp | 17 | Writing, formatting |

Each agent is under 25 tools for manageable context.
