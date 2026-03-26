# Proposals (RFC process)

This document explains when and how to submit a proposal for a significant change to ResearchMCP. The process is intentionally lightweight — it exists to surface design issues early, not to create bureaucracy.

---

## When do you need a proposal?

You need a proposal before opening a pull request if your change affects any of the following:

| Area | Examples |
|------|---------|
| Workflow enforcement | Changing HARD_BLOCK logic, claim verification rules, expansion loop thresholds |
| MCP tool interface | Adding, removing, or renaming tools; changing required parameters |
| File formats | `paper_ir.json` schema, `metrics.db` structure, `project.json` fields |
| Conference templates | Adding a new venue, changing page/column/abstract limits |
| External integrations | New paper APIs, new experiment trackers (W&B, MLflow), new LLM backends |
| Persona system | Adding a persona, changing which tools are visible per stage |
| Breaking changes | Anything that would require existing users to update their config or workflow |

You do **not** need a proposal for:
- Bug fixes
- Documentation improvements
- New unit or integration tests
- Refactors that don't change observable behaviour
- Adding a new conference template that doesn't touch existing ones

When in doubt, open a small issue first and ask.

---

## How to submit a proposal

1. Open a GitHub Issue in this repository.
2. Use the template below as the issue body.
3. Add the label `proposal`.
4. The maintainer will respond with one of: `proposal: accepted`, `proposal: needs-discussion`, or `proposal: declined` — along with a comment explaining the reasoning.
5. Once accepted, open a pull request that links back to the issue (`Closes #NNN`).

You are welcome to start prototyping before the proposal is formally accepted — early prototypes often reveal design issues that improve the final proposal. Just don't open a large PR expecting a merge before the design is agreed on.

---

## Proposal template

Copy this into your GitHub Issue:

```markdown
## Summary

<!-- One paragraph. What are you proposing and why? -->

## Motivation

<!-- What problem does this solve? Who is affected?
     Link to any relevant issues, discussions, or prior art. -->

## Proposed change

<!-- Describe the change in enough detail that a reviewer can understand
     what you intend to build. You don't need a full spec — focus on:
     - What new behaviour or interface is introduced
     - What existing behaviour changes or is removed
     - How the change fits into the existing workflow stages and persona system -->

## Impact on existing users

<!-- Will existing projects, configs, or experiment logs break?
     What is the migration path if so? -->

## Alternatives considered

<!-- What other approaches did you think about and why did you rule them out? -->

## Open questions

<!-- Anything you're unsure about and would like input on. -->

## Out of scope

<!-- What are you explicitly NOT proposing in this change? -->
```

---

## Proposal lifecycle

```
Open issue (label: proposal)
        │
        ▼
Maintainer reviews
        │
   ┌────┴────────────────────┐
   │                         │
proposal: needs-discussion   │
   │                         │
   ▼                         ▼
Discussion / iteration    proposal: accepted
   │                         │
   ▼                         ▼
proposal: accepted        Open pull request
   │                      (links to issue)
   ▼                         │
Open pull request             ▼
                          Code review
                              │
                              ▼
                           Merged
```

A proposal marked `proposal: declined` is closed. The issue stays open for reference. The same idea can be re-proposed later if circumstances change — include a link to the original proposal and explain what has changed.

---

## Examples of what a good proposal looks like

**New conference template (ACL 2026 format)**
> Simple — one paragraph motivation ("the current ACL template is outdated"), a note on which fields change, and confirmation that no existing templates are touched. This would be accepted quickly.

**New tool: `compare_reviewers()`**
> Medium complexity — needs to explain how it fits into the Writer persona, what the tool inputs/outputs are, whether it requires a new external API, and how verified claims interact with the output. Expect a round of discussion.

**Replacing the SQLite metrics store with a pluggable backend**
> High complexity — touches `metrics.db`, experiment tracking, the `log_experiment` and `collect_metrics` tools, and potentially the project directory structure. Needs a detailed proposal with migration notes before any code is written.

---

## Questions?

Open an issue with the label `question` or `governance`.
