# Changelog

All notable changes to ResearchMCP are documented here.

## [0.3.0] - 2026-01-14

### Added
- **LLM-driven idea generation**: `generate_ideas()` now returns paper context for LLM to create ideas
- **`submit_idea()` tool**: LLM submits creative ideas based on paper content
- **Confirmation codes**: Ideas require user-typed codes for approval (prevents auto-approval)
- **Workflow orchestration**: `get_next_action()` and `get_workflow_checklist()` tools
- **Prerequisite enforcement**: Tools check workflow stage before executing
- **HuggingFace topic search**: `fetch_hf_trending(topic="...")` filters by query

### Changed
- Removed hardcoded idea templates - LLM generates all content
- Improved method extraction from papers
- Better error messages for blocked actions

### Fixed
- arXiv 403 errors with rate limiting (3s delay)
- Semantic Scholar fallback on API failures
- FTS5 index corruption on paper cache

## [0.2.0] - 2026-01-13

### Added
- **Project management**: `create_project()`, `set_current_project()`
- **Git integration**: Auto-commit experiment results
- **PDF compilation**: `compile_paper()` with pdflatex
- **Conference templates**: NeurIPS, ICML, ICLR, CVPR support
- **Visualization tools**: `plot_comparison_bar()`, `plot_training_curves()`
- **Statistical verification**: `check_significance()`, `compare_to_baselines()`

### Changed
- Simplified writing tools - formatting only, no content generation
- Project isolation with dedicated directories

### Fixed
- LaTeX compilation errors with missing packages
- Jinja2 template syntax conflicts with LaTeX braces

## [0.1.0] - 2026-01-12

### Added
- Initial MCP server implementation
- Paper aggregation from arXiv, HuggingFace, Semantic Scholar
- Basic idea generation with novelty scoring
- Experiment environment setup (venv, conda, docker)
- Experiment execution and monitoring
- Results collection from logs
- Basic LaTeX paper generation
- Conference formatting (9 A* conferences)

### Infrastructure
- SQLite caching for papers and experiments
- Async tool implementations
- MCP protocol compliance

---

## Version Format

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes to tool interfaces
- **MINOR**: New tools or features
- **PATCH**: Bug fixes and improvements
