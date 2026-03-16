"""Verification reviewer agent (rom4ik pattern).

A SEPARATE fresh Claude instance that critically reviews completed work
with no sunk-cost bias. Two consecutive PASS verdicts required.

The reviewer controls the revision loop via three verdicts:
- PASS: Paper is submittable (2 consecutive required)
- REVISE: Issues are fixable, send back for revision
- REJECT: Fundamental issues that cannot be fixed by revision

Proxy optimization:
- Uses opus model for deepest critical analysis
- effort="high" — review quality is paramount
- Fallback to sonnet if opus unavailable
"""

import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

from src.agents.base import BaseAgent
from src.agents.client import DEFAULT_MODEL, EFFORT_HIGH
from src.state.progress import load_project_config, save_project_config

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a critical peer reviewer for top ML conferences (NeurIPS, ICML, ICLR, CVPR).

You are reviewing a paper that was written by an AI agent. Your job is to find ALL issues, no matter how small. You have NO prior context about this work — you are seeing it fresh.

## Review Criteria

1. **Technical Correctness**: Are claims supported by evidence? Are statistical tests proper?
2. **Novelty**: Is this genuinely new or incremental?
3. **Experimental Rigor**: Multiple seeds? Proper baselines? Statistical significance?
4. **Citation Accuracy**: Are ALL cited papers real? Are authors/titles/venues correct?
5. **Writing Quality**: Clear? Well-organized? Proper LaTeX?
6. **Figures/Tables**: Real data? Proper labels? Referenced in text?
7. **Reproducibility**: Enough detail to reproduce? Hyperparameters listed?
8. **Limitations**: Honestly discussed?

## Output Format

```json
{
  "review": {
    "summary": "2-3 sentence summary of the paper",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "issues": [
      {
        "severity": "CRITICAL",
        "category": "technical_correctness",
        "description": "Claim X not supported by evidence",
        "location": "Section 4, paragraph 2",
        "suggestion": "Add statistical test or remove claim"
      }
    ],
    "citation_check": {
      "total_citations": 25,
      "verified_real": 20,
      "suspicious": ["author2024fake — cannot find this paper"],
      "missing_important": ["Should cite Smith2023 which is highly relevant"]
    },
    "scores": {
      "technical_correctness": 7,
      "novelty": 6,
      "experimental_rigor": 7,
      "writing_quality": 8,
      "reproducibility": 6,
      "overall": 6
    },
    "verdict": "PASS",
    "verdict_reasoning": "Paper meets minimum bar for workshop submission but needs work for main conference",
    "revision_possible": true
  }
}
```

## Severity Levels
- **CRITICAL**: Must fix before submission (factual errors, fabricated results, missing baselines)
- **IMPORTANT**: Should fix (weak claims, missing ablations, unclear writing)
- **MINOR**: Nice to fix (typos, formatting, style)

## Verdict Rules
- **PASS**: No CRITICAL issues, <= 3 IMPORTANT issues, paper is submittable
- **REVISE**: Has fixable CRITICAL or IMPORTANT issues — revision can address them. Set `revision_possible: true`
- **REJECT**: Fundamental flaws that NO amount of revision can fix (e.g., the entire idea is not novel, the approach is fundamentally wrong, no salvageable contribution). Set `revision_possible: false`

Use REVISE (not FAIL) when issues exist but are fixable. Use REJECT only when the work is unsalvageable.
Be HARSH but FAIR. Real reviewers at top venues are tough. Better to catch issues now.
"""


class ReviewerAgent(BaseAgent):
    """Independent verification reviewer (rom4ik pattern).

    Key design: this agent has NO access to the builder's conversation
    history. It reads the paper and project files fresh, evaluating
    with no sunk-cost bias.

    Uses opus model with effort="high" for deepest critical analysis.
    Falls back to sonnet if opus is unavailable.

    Controls the revision loop via verdict:
    - PASS → accept paper
    - REVISE → send back for fixes (reviewer decides, not hardcoded limit)
    - REJECT → stop, unfixable
    """

    name = "reviewer"
    system_prompt = REVIEWER_SYSTEM_PROMPT
    default_effort = EFFORT_HIGH
    default_fallback_model = "sonnet"

    def __init__(
        self,
        client: anthropic.Anthropic,
        project_dir: str,
        model: str = "opus",
    ):
        super().__init__(client, project_dir, model)

    def _load_review_context(self) -> str:
        """Load everything needed for review."""
        parts = []

        paper_json = Path(self.project_dir) / "paper" / "paper.json"
        if paper_json.exists():
            paper = json.loads(paper_json.read_text())
            parts.append(f"## Paper Content\n{json.dumps(paper, indent=2, default=str)}")

        tex_path = Path(self.project_dir) / "paper" / "main.tex"
        if tex_path.exists():
            latex = tex_path.read_text()
            parts.append(f"## LaTeX Source\n```latex\n{latex[:10000]}\n```")

        results_path = Path(self.project_dir) / "experiments" / "all_results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            parts.append(f"## Experiment Results\n{json.dumps(results, indent=2, default=str)}")

        # Show method code (src/) if available
        src_dir = Path(self.project_dir) / "src"
        if src_dir.exists():
            src_files = []
            for f in sorted(src_dir.glob("*.py")):
                content = f.read_text()
                if content.strip():
                    src_files.append(f"### {f.name}\n```python\n{content[:3000]}\n```")
            if src_files:
                parts.append(f"## Method Implementation (src/)\n" + "\n\n".join(src_files))

        verif_dir = Path(self.project_dir) / "verification"
        if verif_dir.exists():
            verif_results = []
            for f in sorted(verif_dir.glob("*.json")):
                verif_results.append(json.loads(f.read_text()))
            if verif_results:
                parts.append(f"## Statistical Verification\n{json.dumps(verif_results, indent=2)}")

        idea_path = Path(self.project_dir) / "ideas" / "selected_idea.json"
        if idea_path.exists():
            idea = json.loads(idea_path.read_text())
            parts.append(f"## Original Research Idea\n{json.dumps(idea, indent=2)}")

        # Show previous review if this is a revision
        prev_reviews = sorted(Path(self.project_dir).glob("review_*.json"))
        if prev_reviews:
            latest = json.loads(prev_reviews[-1].read_text())
            parts.append(f"## Previous Review\n{json.dumps(latest, indent=2, default=str)}")

        return "\n\n".join(parts)

    def review(self) -> tuple[Optional[dict], str]:
        """Perform a fresh critical review of the project.

        Now also reviews src/ code and previous review context.
        """
        context = self._load_review_context()

        task = (
            "Perform a thorough peer review of this research paper.\n\n"
            "You are a fresh reviewer with NO prior context. Be critical.\n\n"
            "Check:\n"
            "1. ALL citations — are they real papers with correct authors?\n"
            "2. ALL claims — are they supported by statistical evidence (p<0.05)?\n"
            "3. Method implementation (src/) — does it match what the paper describes?\n"
            "4. Figures and tables — do they contain real data?\n"
            "5. Method description — is it complete and reproducible?\n"
            "6. Related work — comprehensive? Fair comparison?\n"
            "7. Limitations — honestly discussed?\n\n"
            "Rate 1-10 on each criterion.\n\n"
            "Give verdict:\n"
            "- PASS: submittable quality\n"
            "- REVISE: has fixable issues (specify what to fix)\n"
            "- REJECT: fundamental unfixable flaws (only if truly unsalvageable)\n\n"
            "If previous review exists, check whether those issues were addressed.\n"
        )

        return self.call_structured(
            task=task,
            extra_context=context,
            max_tokens=8192,
            temperature=0.3,
            effort=EFFORT_HIGH,
        )

    async def run(self, required_passes: int = 2) -> dict:
        """Run verification loop.

        Instead of hardcoded max attempts, the reviewer controls the loop:
        - PASS: count toward required consecutive passes
        - REVISE: return to orchestrator for revision (don't count as pass)
        - REJECT: stop immediately, unfixable

        The orchestrator handles the actual revision and re-calls this.
        For the initial review, we do up to required_passes + 2 attempts
        to get 2 consecutive PASSes.
        """
        self.log_progress("Starting verification review (opus model, effort=high)")

        reviews = []
        consecutive_passes = 0

        for i in range(required_passes + 2):
            review, raw = self.review()

            if not review:
                self.log_progress(f"Review {i+1}: PARSE_ERROR\nRaw: {raw[:500]}")
                reviews.append({"attempt": i + 1, "error": "parse_failed", "raw": raw[:500]})
                consecutive_passes = 0
                continue

            review_data = review.get("review", review)
            verdict = review_data.get("verdict", "REVISE").upper()
            # Normalize old FAIL → REVISE for backwards compat
            if verdict == "FAIL":
                revision_possible = review_data.get("revision_possible", True)
                verdict = "REJECT" if not revision_possible else "REVISE"

            issues = review_data.get("issues", [])
            critical_count = sum(1 for iss in issues if iss.get("severity") == "CRITICAL")
            important_count = sum(1 for iss in issues if iss.get("severity") == "IMPORTANT")

            self.log_progress(
                f"Review {i+1}: {verdict}\n"
                f"  Scores: {review_data.get('scores', {})}\n"
                f"  Issues: {critical_count} CRITICAL, {important_count} IMPORTANT, "
                f"{len(issues) - critical_count - important_count} MINOR\n"
                f"  Reasoning: {review_data.get('verdict_reasoning', 'N/A')}"
            )

            reviews.append({"attempt": i + 1, "review": review_data, "verdict": verdict})

            review_path = Path(self.project_dir) / f"review_{i+1}.json"
            review_path.write_text(json.dumps(review_data, indent=2, default=str))

            if verdict == "PASS":
                consecutive_passes += 1
                if consecutive_passes >= required_passes:
                    self.log_progress(f"VERIFICATION PASSED ({consecutive_passes} consecutive passes)")
                    break
            elif verdict == "REJECT":
                self.log_progress("REJECTED — reviewer determined issues are unfixable")
                consecutive_passes = 0
                break
            else:
                # REVISE — return to orchestrator for revision
                consecutive_passes = 0
                # Don't keep reviewing here — let orchestrator fix and come back
                break

        if consecutive_passes >= required_passes:
            final_verdict = "PASS"
        elif any(r.get("verdict") == "REJECT" for r in reviews):
            final_verdict = "REJECT"
        else:
            final_verdict = "REVISE"

        config = load_project_config(self.project_dir)
        config["stage"] = {
            "PASS": "complete",
            "REJECT": "rejected",
            "REVISE": "revision_needed",
        }.get(final_verdict, "revision_needed")
        config["review_verdict"] = final_verdict
        save_project_config(self.project_dir, config)

        self.log_progress(f"FINAL VERDICT: {final_verdict}")

        return {
            "final_verdict": final_verdict,
            "consecutive_passes": consecutive_passes,
            "reviews": reviews,
        }
