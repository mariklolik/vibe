"""Multi-agent MCP system for research automation.

This package contains three separate MCP servers:
- researcher: Paper discovery, idea generation, approval
- experimenter: Code implementation, experiments, verification  
- writer: Paper writing and formatting

Each agent has its own focused set of tools and runs as a separate MCP server.
State is shared via SQLite databases in ~/.cache/research-mcp/
"""
