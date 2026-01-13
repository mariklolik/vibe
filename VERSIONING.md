# Versioning Policy

ResearchMCP follows [Semantic Versioning 2.0.0](https://semver.org/).

## Version Format

```
MAJOR.MINOR.PATCH
```

### MAJOR Version

Incremented for:
- Breaking changes to MCP tool interfaces
- Removal of existing tools
- Changes to required parameters
- Database schema changes requiring migration

Example: Changing `generate_ideas()` to require different parameters.

### MINOR Version

Incremented for:
- New tools added
- New optional parameters
- New features that don't break existing usage
- New conference support

Example: Adding `submit_idea()` tool.

### PATCH Version

Incremented for:
- Bug fixes
- Performance improvements
- Documentation updates
- Internal refactoring

Example: Fixing arXiv API rate limiting.

## Current Version

**v0.3.0** (2026-01-14)

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Create git tag: `git tag v0.3.0`
4. Push tag: `git push origin v0.3.0`

## Compatibility

### MCP Protocol

This project targets MCP protocol version 1.0. Tool schemas follow the MCP specification.

### Python Version

- **Minimum**: Python 3.10
- **Tested**: Python 3.10, 3.11, 3.12

### Dependencies

Major dependencies pinned in `requirements.txt`:
- `mcp` - MCP server library
- `aiosqlite` - Async SQLite
- `matplotlib` - Visualization
- `scipy` - Statistical tests

## Deprecation Policy

- Deprecated features announced one MINOR version before removal
- Migration guides provided in CHANGELOG
- Breaking changes only in MAJOR versions
