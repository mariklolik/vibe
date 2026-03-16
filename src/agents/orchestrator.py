"""Main pipeline orchestrator with closed revision loop.

Coordinates the full research pipeline:
research -> experiment -> writing -> review -> [revision loop] -> done

The revision loop is the key differentiator:
- Reviewer (opus) identifies issues with severity levels
- Orchestrator classifies issues → routes to correct agent
- CRITICAL experiment issues → back to ExperimentAgent with review context
- CRITICAL/IMPORTANT writing issues → back to WriterAgent with review context
- Re-review after fixes until 2x consecutive PASS

No other framework has this closed loop — rom4ik flags for humans,
research_claude_agents has no reviewer at all.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from src.agents.client import (
    create_client,
    call_agent_structured,
    check_proxy_health,
    check_proxy_metrics,
    get_token_usage,
    DEFAULT_MODEL,
    EFFORT_HIGH,
    EFFORT_MEDIUM,
)
from src.agents.research import ResearchAgent
from src.agents.experiment import ExperimentAgent
from src.agents.writer import WriterAgent
from src.agents.reviewer import ReviewerAgent
from src.state.progress import (
    ensure_project_structure,
    save_project_config,
    load_project_config,
    append_progress,
    read_progress,
)

logger = logging.getLogger(__name__)

DEFAULT_PROJECTS_DIR = os.path.expanduser("~/research-projects")

# Categories for routing review issues back to the right agent
EXPERIMENT_CATEGORIES = {
    "technical_correctness", "experimental_rigor", "reproducibility",
    "fabricated_results", "missing_baselines", "statistical_issues",
}
WRITING_CATEGORIES = {
    "writing_quality", "citation_accuracy", "figures_tables",
    "novelty", "clarity", "formatting", "limitations",
}


def classify_review_issues(review_data: dict) -> dict:
    """Classify review issues into experiment vs writing problems.

    Returns dict with:
      - needs_experiment_revision: bool
      - needs_writing_revision: bool
      - experiment_issues: list of issues to fix in experiments
      - writing_issues: list of issues to fix in paper
    """
    issues = review_data.get("issues", [])

    experiment_issues = []
    writing_issues = []

    for issue in issues:
        severity = issue.get("severity", "MINOR")
        category = issue.get("category", "").lower()

        if severity not in ("CRITICAL", "IMPORTANT"):
            continue

        # Route by category
        if category in EXPERIMENT_CATEGORIES:
            experiment_issues.append(issue)
        elif category in WRITING_CATEGORIES:
            writing_issues.append(issue)
        else:
            # Heuristic: check description keywords
            desc = issue.get("description", "").lower()
            if any(kw in desc for kw in [
                "experiment", "baseline", "result", "metric", "statistic",
                "fabricat", "p-value", "significance", "reproduce", "code",
                "implementation", "ablation", "evaluation",
            ]):
                experiment_issues.append(issue)
            else:
                writing_issues.append(issue)

    needs_exp = any(i.get("severity") == "CRITICAL" for i in experiment_issues)
    needs_writing = len(writing_issues) > 0

    return {
        "needs_experiment_revision": needs_exp,
        "needs_writing_revision": needs_writing,
        "experiment_issues": experiment_issues,
        "writing_issues": writing_issues,
        "total_critical": sum(1 for i in issues if i.get("severity") == "CRITICAL"),
        "total_important": sum(1 for i in issues if i.get("severity") == "IMPORTANT"),
    }


class Orchestrator:
    """Main pipeline orchestrator with closed revision loop.

    Lifecycle:
    1. Health check proxy
    2. Create project with topic
    3. Research: search papers, generate ideas, auto-select
    4. Experiment: implement method (src/), design scripts, execute, verify
    5. Writing: section-by-section draft, expansion loop, compile PDF
    6. Review: opus model, identify all issues
    7. REVISION LOOP (max 2 iterations):
       - Classify issues → experiment vs writing
       - CRITICAL experiment issues → re-run experiment fixes
       - Writing issues → re-write affected sections
       - Re-review until 2x PASS or max iterations
    8. Report metrics

    Each agent call is fresh (no accumulated context).
    State persists via progress.txt and project files.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "x",
        model: str = DEFAULT_MODEL,
        projects_dir: str = DEFAULT_PROJECTS_DIR,
    ):
        self.client = create_client(base_url=base_url, api_key=api_key)
        self.model = model
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def check_health(self) -> dict:
        """Check proxy health before starting pipeline."""
        health = check_proxy_health(str(self.client.base_url).rstrip("/"))
        logger.info(f"Proxy health: {health}")
        return health

    def get_metrics(self) -> dict:
        """Get proxy metrics (rate limits, queue status)."""
        return check_proxy_metrics(str(self.client.base_url).rstrip("/"))

    def create_project(self, topic: str, project_name: Optional[str] = None) -> str:
        """Create a new research project."""
        if project_name is None:
            safe_name = topic.lower().replace(" ", "_")[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"research_{safe_name}_{timestamp}"

        project_dir = str(self.projects_dir / project_name)
        ensure_project_structure(project_dir)

        config = {
            "id": project_name,
            "topic": topic,
            "stage": "research",
            "created": datetime.now().isoformat(),
            "model": self.model,
        }
        save_project_config(project_dir, config)
        append_progress(project_dir, f"Project created for topic: {topic}", stage="init")

        logger.info(f"Created project: {project_dir}")
        return project_dir

    async def run_research(self, project_dir: str, topic: str, min_novelty: float = 0.7) -> Optional[dict]:
        """Run the research phase."""
        agent = ResearchAgent(self.client, project_dir, self.model)
        return await agent.run(topic, min_novelty=min_novelty)

    async def run_experiments(self, project_dir: str, idea: dict) -> dict:
        """Run the experiment phase (implement src/ + scripts/ + execute)."""
        agent = ExperimentAgent(self.client, project_dir, self.model)
        return await agent.run(idea)

    async def run_writing(self, project_dir: str, conference: str = "icml") -> dict:
        """Run the paper writing phase."""
        agent = WriterAgent(self.client, project_dir, self.model)
        return await agent.run(conference=conference)

    async def run_review(self, project_dir: str) -> dict:
        """Run the verification review phase (opus model)."""
        agent = ReviewerAgent(self.client, project_dir)
        return await agent.run(required_passes=2)

    async def run_single_review(self, project_dir: str) -> tuple[Optional[dict], str]:
        """Run a single review pass (for revision loop)."""
        agent = ReviewerAgent(self.client, project_dir)
        return agent.review()

    async def fix_experiment_issues(
        self,
        project_dir: str,
        idea: dict,
        issues: list[dict],
    ) -> dict:
        """Fix experiment-level issues identified by reviewer.

        Feeds review issues as fresh context to the experiment agent.
        The agent re-implements/fixes the method in src/ and re-runs experiments.
        """
        agent = ExperimentAgent(self.client, project_dir, self.model)

        issues_text = json.dumps(issues, indent=2)
        append_progress(
            project_dir,
            f"REVISION: Fixing {len(issues)} experiment issues:\n{issues_text}",
            stage="revision",
        )

        # Re-implement method with review feedback as context
        agent.log_progress(
            f"Re-implementing method to address reviewer issues:\n{issues_text}"
        )

        # Use the experiment agent's implement_method with review context injected
        # We modify the idea to include the review feedback
        revised_idea = dict(idea)
        revision_guidance = "\n\n## REVIEWER ISSUES TO FIX\n"
        for iss in issues:
            revision_guidance += (
                f"- [{iss.get('severity')}] {iss.get('description')}\n"
                f"  Location: {iss.get('location', 'N/A')}\n"
                f"  Suggestion: {iss.get('suggestion', 'N/A')}\n"
            )
        revised_idea["description"] = (
            idea.get("description", "") + revision_guidance
        )

        return await agent.run(revised_idea)

    async def fix_writing_issues(
        self,
        project_dir: str,
        issues: list[dict],
        conference: str = "icml",
    ) -> dict:
        """Fix writing-level issues identified by reviewer.

        Re-writes affected sections with review issues as context.
        """
        agent = WriterAgent(self.client, project_dir, self.model)

        issues_text = json.dumps(issues, indent=2)
        append_progress(
            project_dir,
            f"REVISION: Fixing {len(issues)} writing issues:\n{issues_text}",
            stage="revision",
        )

        # Load current paper
        paper_json_path = Path(project_dir) / "paper" / "paper.json"
        if not paper_json_path.exists():
            agent.log_progress("ERROR: No paper.json found for revision")
            return {"success": False, "error": "No paper to revise"}

        paper = json.loads(paper_json_path.read_text())

        # Identify which sections need revision based on issue locations
        sections_to_fix = set()
        for issue in issues:
            location = issue.get("location", "").lower()
            for section_name in ["abstract", "introduction", "related_work", "method",
                                  "experiments", "discussion", "conclusion"]:
                if section_name.replace("_", " ") in location or section_name in location:
                    sections_to_fix.add(section_name)

        # If no specific sections identified, fix all sections mentioned in issues
        if not sections_to_fix:
            # Default to re-writing the sections most commonly flagged
            sections_to_fix = {"introduction", "experiments", "related_work"}

        agent.log_progress(f"Re-writing sections: {sections_to_fix}")

        # Re-write each affected section with review context
        sections = paper.get("paper", {}).get("sections", [])
        written_sections = []
        for section in sections:
            if section.get("name") in sections_to_fix:
                # Build revision guidance from relevant issues
                section_issues = [
                    iss for iss in issues
                    if section["name"].replace("_", " ") in iss.get("location", "").lower()
                    or section["name"] in iss.get("location", "").lower()
                    or not iss.get("location")  # Include unlocated issues
                ]

                guidance = f"REVISE this section. Reviewer issues to address:\n"
                for iss in section_issues:
                    guidance += f"- [{iss.get('severity')}] {iss.get('description')}\n"
                    if iss.get("suggestion"):
                        guidance += f"  Fix: {iss['suggestion']}\n"

                result, raw = agent.write_section(
                    section_name=section["name"],
                    section_guidance=guidance,
                    target_words=section.get("word_count", 800),
                    previous_sections=written_sections,
                    conference=conference,
                )

                if result and result.get("content"):
                    section["content"] = result["content"]
                    section["word_count"] = result.get("word_count", len(result["content"].split()))
                    agent.log_progress(f"Revised {section['name']}: ~{section['word_count']} words")
                else:
                    agent.log_progress(f"WARNING: Failed to revise {section['name']}")

            written_sections.append(section)

        # Save revised paper
        tex_path = agent.save_paper(paper, conference)
        pdf_path = agent.compile_pdf(tex_path)

        return {
            "success": True,
            "revised_sections": list(sections_to_fix),
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
        }

    def _try_generate_figures(self, project_dir: str):
        """Try to run generate_figures.py if it exists, even after partial experiment success."""
        import subprocess
        fig_script = Path(project_dir) / "scripts" / "generate_figures.py"
        if not fig_script.exists():
            return

        try:
            result = subprocess.run(
                ["python", str(fig_script)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_dir,
                env={**os.environ},
            )
            figures_dir = Path(project_dir) / "figures"
            fig_count = len(list(figures_dir.glob("*.png"))) + len(list(figures_dir.glob("*.pdf")))
            if result.returncode == 0 and fig_count > 0:
                append_progress(project_dir, f"Generated {fig_count} figures", stage="orchestrator")
                logger.info(f"Figure generation: {fig_count} figures created")
            else:
                logger.info(f"Figure generation: script returned {result.returncode}, {fig_count} figures")
                if result.stderr:
                    logger.debug(f"Figure generation stderr: {result.stderr[:300]}")
        except Exception as e:
            logger.info(f"Figure generation skipped: {e}")

    async def run_full_pipeline(
        self,
        topic: str,
        conference: str = "icml",
        min_novelty: float = 0.7,
        project_name: Optional[str] = None,
        max_revisions: int = 5,
    ) -> dict:
        """Run the complete research pipeline with reviewer-controlled revision loop.

        The REVIEWER controls the loop via verdict:
        - PASS (2 consecutive) → accept and finish
        - REVISE → classify issues, route to correct agent, re-review
        - REJECT → stop immediately, work is unsalvageable

        max_revisions is a safety cap (default 5), not the primary control.
        The reviewer decides when to stop, not a counter.
        """
        start_time = time.time()

        # Phase 0: Health check
        health = self.check_health()
        if health.get("status") == "unreachable":
            logger.error("Proxy is unreachable! Start it first.")
            return {"final_status": "PROXY_UNREACHABLE", "error": "Proxy not running"}
        logger.info(f"Proxy OK: {health}")

        initial_tokens = get_token_usage()

        # Create project
        project_dir = self.create_project(topic, project_name)
        logger.info(f"=== STARTING PIPELINE: {topic} ===")
        logger.info(f"Project: {project_dir}")

        result = {
            "project_dir": project_dir,
            "topic": topic,
            "conference": conference,
            "phases": {},
            "revisions": [],
            "token_usage": {},
        }

        # Phase 1: Research
        logger.info("=== PHASE 1: RESEARCH ===")
        phase_start = time.time()
        idea = await self.run_research(project_dir, topic, min_novelty)
        phase_tokens = get_token_usage()
        result["phases"]["research"] = {
            "success": idea is not None,
            "idea": idea,
            "elapsed_seconds": round(time.time() - phase_start, 1),
            "tokens_used": {k: phase_tokens[k] - initial_tokens[k] for k in phase_tokens},
        }
        if not idea:
            result["final_status"] = "FAILED_AT_RESEARCH"
            result["elapsed_seconds"] = round(time.time() - start_time, 1)
            result["token_usage"] = get_token_usage()
            return result

        # Phase 2: Experiments (implement src/ + scripts/ + execute)
        logger.info("=== PHASE 2: EXPERIMENTS (implement src/ → scripts/ → execute) ===")
        phase_start = time.time()
        pre_tokens = get_token_usage()
        exp_result = await self.run_experiments(project_dir, idea)
        post_tokens = get_token_usage()
        result["phases"]["experiments"] = {
            **exp_result,
            "elapsed_seconds": round(time.time() - phase_start, 1),
            "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
        }
        if not exp_result.get("success"):
            logger.warning("Experiments had issues, continuing to writing with available results")

        # Phase 2.5: Generate figures from whatever results we have
        self._try_generate_figures(project_dir)

        # Phase 3: Paper Writing
        logger.info("=== PHASE 3: WRITING ===")
        phase_start = time.time()
        pre_tokens = get_token_usage()
        write_result = await self.run_writing(project_dir, conference)
        post_tokens = get_token_usage()
        result["phases"]["writing"] = {
            **write_result,
            "elapsed_seconds": round(time.time() - phase_start, 1),
            "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
        }
        if not write_result.get("success"):
            result["final_status"] = "FAILED_AT_WRITING"
            result["elapsed_seconds"] = round(time.time() - start_time, 1)
            result["token_usage"] = get_token_usage()
            return result

        # Phase 4: Reviewer-controlled revision loop
        # The reviewer decides: PASS / REVISE / REJECT
        # max_revisions is just a safety cap
        logger.info("=== PHASE 4: REVIEW + REVISION LOOP (reviewer-controlled) ===")
        final_verdict = "REVISE"

        for revision_round in range(max_revisions + 1):
            round_label = "initial" if revision_round == 0 else f"revision_{revision_round}"
            logger.info(f"--- Review round: {round_label} ---")

            phase_start = time.time()
            pre_tokens = get_token_usage()

            # Run full review (attempts 2x PASS internally)
            review_result = await self.run_review(project_dir)

            post_tokens = get_token_usage()
            review_phase_key = f"review_{round_label}"
            result["phases"][review_phase_key] = {
                **review_result,
                "elapsed_seconds": round(time.time() - phase_start, 1),
                "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
            }

            final_verdict = review_result.get("final_verdict", "REVISE")

            # PASS → done
            if final_verdict == "PASS":
                logger.info(f"REVIEW PASSED after {revision_round} revision(s)")
                append_progress(
                    project_dir,
                    f"REVIEW PASSED after {revision_round} revision round(s)",
                    stage="orchestrator",
                )
                break

            # REJECT → stop, unfixable
            if final_verdict == "REJECT":
                logger.info("REVIEWER REJECTED — issues deemed unfixable")
                append_progress(
                    project_dir,
                    "REVIEWER REJECTED work as unsalvageable",
                    stage="orchestrator",
                )
                break

            # REVISE → classify issues and route to correct agent
            # Safety cap check
            if revision_round >= max_revisions:
                logger.info(f"Safety cap ({max_revisions} revisions) reached")
                append_progress(
                    project_dir,
                    f"Safety cap reached after {max_revisions} revisions, stopping",
                    stage="orchestrator",
                )
                break

            # Extract issues from latest review
            last_review = None
            for rev in reversed(review_result.get("reviews", [])):
                if rev.get("review"):
                    last_review = rev["review"]
                    break

            if not last_review:
                logger.warning("No parseable review to extract issues from, stopping")
                break

            classification = classify_review_issues(last_review)
            logger.info(
                f"Issue classification: "
                f"{classification['total_critical']} CRITICAL, "
                f"{classification['total_important']} IMPORTANT | "
                f"experiment_issues={len(classification['experiment_issues'])}, "
                f"writing_issues={len(classification['writing_issues'])}"
            )

            revision_entry = {
                "round": revision_round + 1,
                "classification": classification,
                "fixes": [],
            }

            # Fix experiment issues (CRITICAL → back to experiment agent)
            if classification["needs_experiment_revision"]:
                logger.info(f"=== REVISION {revision_round+1}: FIXING EXPERIMENT ISSUES ===")
                phase_start = time.time()
                pre_tokens = get_token_usage()

                exp_fix = await self.fix_experiment_issues(
                    project_dir, idea, classification["experiment_issues"]
                )
                post_tokens = get_token_usage()

                revision_entry["fixes"].append({
                    "type": "experiment",
                    "issues_fixed": len(classification["experiment_issues"]),
                    "success": exp_fix.get("success", False),
                    "elapsed_seconds": round(time.time() - phase_start, 1),
                    "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
                })

                # Re-generate figures after experiment fixes
                self._try_generate_figures(project_dir)

                # Re-write paper after experiment fixes (new results → new paper)
                logger.info("=== REVISION: RE-WRITING PAPER AFTER EXPERIMENT FIXES ===")
                phase_start = time.time()
                pre_tokens = get_token_usage()
                write_result = await self.run_writing(project_dir, conference)
                post_tokens = get_token_usage()

                revision_entry["fixes"].append({
                    "type": "rewrite_after_experiment_fix",
                    "success": write_result.get("success", False),
                    "elapsed_seconds": round(time.time() - phase_start, 1),
                    "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
                })

            # Fix writing issues only
            elif classification["needs_writing_revision"]:
                logger.info(f"=== REVISION {revision_round+1}: FIXING WRITING ISSUES ===")
                phase_start = time.time()
                pre_tokens = get_token_usage()

                write_fix = await self.fix_writing_issues(
                    project_dir, classification["writing_issues"], conference
                )
                post_tokens = get_token_usage()

                revision_entry["fixes"].append({
                    "type": "writing",
                    "sections_revised": write_fix.get("revised_sections", []),
                    "success": write_fix.get("success", False),
                    "elapsed_seconds": round(time.time() - phase_start, 1),
                    "tokens_used": {k: post_tokens[k] - pre_tokens[k] for k in post_tokens},
                })
            else:
                logger.info("No actionable issues identified, stopping revision loop")
                break

            result["revisions"].append(revision_entry)

        # Final status
        result["final_status"] = f"COMPLETE_{final_verdict}"
        result["elapsed_seconds"] = round(time.time() - start_time, 1)
        result["token_usage"] = get_token_usage()
        result["revision_rounds"] = len(result["revisions"])

        # Log final metrics
        final_metrics = self.get_metrics()
        result["proxy_metrics"] = final_metrics

        # Git commit final state
        try:
            import subprocess
            subprocess.run(["git", "add", "-A"], cwd=project_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"Pipeline complete: {final_verdict} ({topic}, "
                 f"{len(result['revisions'])} revisions)"],
                cwd=project_dir, capture_output=True,
            )
        except Exception:
            pass

        # Update project config
        config = load_project_config(project_dir)
        config["stage"] = "complete" if final_verdict == "PASS" else "revision_needed"
        config["final_verdict"] = final_verdict
        config["revision_rounds"] = len(result["revisions"])
        save_project_config(project_dir, config)

        logger.info(
            f"=== PIPELINE COMPLETE: {final_verdict} "
            f"({len(result['revisions'])} revisions, "
            f"elapsed: {result['elapsed_seconds']}s, "
            f"tokens: {result['token_usage']}) ==="
        )

        return result

    async def run_batch(
        self,
        topics: list[str],
        conference: str = "icml",
        parallel: int = 1,
    ) -> list[dict]:
        """Run pipeline for multiple topics (sequential by default)."""
        if parallel <= 1:
            results = []
            for topic in topics:
                result = await self.run_full_pipeline(topic, conference)
                results.append(result)
            return results
        else:
            semaphore = asyncio.Semaphore(parallel)

            async def run_with_semaphore(topic):
                async with semaphore:
                    return await self.run_full_pipeline(topic, conference)

            return await asyncio.gather(
                *(run_with_semaphore(t) for t in topics),
                return_exceptions=True,
            )
