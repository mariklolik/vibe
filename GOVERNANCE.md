# Governance

## Overview

ResearchMCP is a founder-led open source project. The goal of this document is to make decision-making transparent so contributors know what to expect when they submit code, open issues, or propose changes.

---

## Roles

### Maintainer

Currently: [@mariklolik](https://github.com/mariklolik)

The maintainer has final say on all technical decisions, roadmap priorities, and releases. This is a BDFL (Benevolent Dictator for Life) model — appropriate for the current size and stage of the project. If the project grows to the point where shared maintainership makes sense, this document will be updated.

Responsibilities:
- Reviewing and merging pull requests
- Triaging issues
- Cutting releases and maintaining the changelog
- Deciding which proposals move forward (see [PROPOSAL.md](PROPOSAL.md))
- Maintaining the quality and integrity of the research pipeline

### Contributor

Anyone who has had a pull request merged. Contributors are listed in the GitHub contributor graph. There is no formal commit access beyond the maintainer at this time.

### Community member

Anyone who opens issues, participates in discussions, or uses the project. All community members are expected to follow the [Code of Conduct](#code-of-conduct).

---

## Decision making

### Everyday changes

Bug fixes, documentation improvements, and small enhancements that don't affect the public API or workflow semantics can be submitted as a pull request directly. The maintainer will review and merge without a formal proposal process.

### Significant changes

Changes that affect any of the following require a proposal (see [PROPOSAL.md](PROPOSAL.md)) before a pull request is opened:

- The core workflow stages or their enforcement logic (HARD_BLOCK, claim verification, expansion loop)
- The MCP tool interface — adding, removing, or renaming tools
- The project directory structure or file formats (`paper_ir.json`, `metrics.db`, etc.)
- Supported conference templates
- New external service integrations (new APIs, new experiment tracking backends)
- Breaking changes to the Python API or CLI

Open a GitHub Issue using the proposal template, tag it `proposal`, and wait for maintainer feedback before investing time in implementation.

### Conflict resolution

If a contributor disagrees with a decision, they are welcome to open an issue to discuss it. The maintainer will explain the reasoning. If consensus cannot be reached, the maintainer has final say. This is by design for a project at this stage — predictability and consistency matter more than committee consensus.

---

## Pull request process

1. Fork the repository and create a branch from `main`.
2. For significant changes, open a proposal issue first.
3. Write or update tests for any changed behaviour.
4. Ensure `python3 verify_setup.py` passes.
5. Open a pull request with a clear description of what changed and why.
6. The maintainer will review within a reasonable time (typically within a week for small PRs).
7. Address review comments. Once approved, the maintainer will merge.

Squash commits are preferred for cleanliness. The maintainer may squash on merge.

---

## Releases

Releases follow [Semantic Versioning](https://semver.org/):

- **Patch** (`x.y.Z`): bug fixes, documentation, no behaviour change
- **Minor** (`x.Y.0`): new tools, new conference templates, backward-compatible additions
- **Major** (`X.0.0`): breaking changes to the tool interface, workflow semantics, or file formats

Release notes are maintained in [CHANGELOG.md](CHANGELOG.md). The maintainer cuts releases manually by tagging `main`.

---

## Code of conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). In short: be respectful, assume good intent, and keep discussion focused on the work. Issues or PRs that become hostile will be closed. Repeated violations result in a block.

---

## Changing this document

Changes to `GOVERNANCE.md` itself are decided by the maintainer. If you have a suggestion, open an issue with the label `governance`.
