"""Idea generation tools - generate, validate, and plan research ideas with user verification."""

import json
import random
import uuid
from datetime import datetime
from typing import Optional

from src.apis.semantic_scholar import s2_client
from src.db.papers_cache import papers_cache
from src.db.experiments_db import experiments_db, Idea
from src.db.workflow import workflow_db
from src.project.manager import project_manager


def _generate_confirmation_code() -> str:
    """Generate a 4-digit confirmation code for user approval."""
    return str(random.randint(1000, 9999))


async def compute_idea_novelty(idea_description: str) -> tuple[float, list[dict]]:
    """Compute novelty score for an idea by searching existing literature.
    
    Returns (novelty_score, similar_works)
    """
    try:
        similar_papers = await s2_client.search(idea_description[:200], max_results=10)
    except Exception:
        return 0.8, []  # Default score if search fails
    
    novelty_score = 1.0
    similar_works = []
    
    idea_words = set(idea_description.lower().split())
    
    for paper in similar_papers:
        if paper.title:
            title_words = set(paper.title.lower().split())
            abstract_words = set((paper.abstract or "").lower().split())
            
            title_overlap = len(idea_words & title_words) / max(len(idea_words), 1)
            abstract_overlap = len(idea_words & abstract_words) / max(len(idea_words), 1)
            
            overlap = max(title_overlap, abstract_overlap * 0.5)
            
            if overlap > 0.5:
                novelty_score -= 0.12
                similar_works.append({
                    "title": paper.title[:80],
                    "overlap": round(overlap, 2),
                })
            elif overlap > 0.3:
                novelty_score -= 0.06
    
    return max(0.1, novelty_score), similar_works[:3]


async def generate_ideas(
    paper_ids: list[str],
    count: int = 3,
    focus: Optional[str] = None,
) -> str:
    """
    Provide paper context for idea generation.
    
    This tool returns the full context from gathered papers. The LLM should then:
    1. Read the paper abstracts carefully
    2. Generate {count} creative research ideas based on the content
    3. Call submit_idea() for each idea
    4. Wait for user approval
    
    DO NOT generate generic template ideas. Use the actual paper content.
    """
    papers_context = []
    
    for paper_id in paper_ids:
        cached = await papers_cache.get_paper(paper_id)
        if cached:
            papers_context.append({
                "id": paper_id,
                "title": cached.title,
                "abstract": cached.abstract,  # Full abstract
                "categories": cached.categories,
            })
    
    if not papers_context:
        return json.dumps({"error": "No papers found with provided IDs"})
    
    # Extract themes for guidance
    all_text = " ".join(p["title"] + " " + p["abstract"] for p in papers_context)
    themes = _extract_themes(all_text)
    
    result = {
        "status": "CONTEXT_PROVIDED",
        "message": (
            "Paper context loaded. Now YOU (the LLM) must generate ideas.\n"
            "Read the abstracts below and create novel research ideas.\n"
            "For each idea, call: submit_idea(title, description, motivation)"
        ),
        "papers": papers_context,
        "detected_themes": themes,
        "focus": focus,
        "requested_count": count,
        "instructions": {
            "step_1": "Read all paper abstracts carefully",
            "step_2": f"Generate {count} novel research ideas that combine or extend these works",
            "step_3": "For each idea call: submit_idea(title='...', description='...', motivation='...')",
            "step_4": "After submitting, wait for user to approve one idea",
        },
        "idea_guidelines": [
            "Ideas should be specific to the paper content, not generic",
            "Combine insights from multiple papers",
            "Identify gaps or extensions the papers don't address",
            "Be concrete about the proposed method/approach",
            "Include what makes this novel compared to existing work",
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def submit_idea(
    title: str,
    description: str,
    motivation: str,
    source_papers: Optional[list[str]] = None,
) -> str:
    """
    Submit a research idea for user approval.
    
    Call this after generate_ideas() returns paper context.
    The LLM should generate creative ideas based on the papers and submit each one.
    
    Args:
        title: Clear, specific title for the idea
        description: Detailed description of the proposed research (2-3 paragraphs)
        motivation: Why this is novel and worth pursuing
        source_papers: List of paper IDs this idea is based on
    """
    idea_id = f"idea_{uuid.uuid4().hex[:8]}"
    confirmation_code = _generate_confirmation_code()
    
    # Compute novelty score
    novelty_score, similar_works = await compute_idea_novelty(description)
    
    idea = Idea(
        idea_id=idea_id,
        title=title,
        description=description,
        source_papers=source_papers or [],
        hypotheses=[],
        research_plan={},
        novelty_score=novelty_score,
        created_at=datetime.now().isoformat(),
        status="pending_approval",
        motivation=motivation,
        themes=_extract_themes(title + " " + description),
        similar_works=similar_works,
        confirmation_code=confirmation_code,
    )
    
    await experiments_db.save_idea(idea)
    
    star_rating = "★" * int(novelty_score * 5) + "☆" * (5 - int(novelty_score * 5))
    
    return json.dumps({
        "success": True,
        "idea_submitted": {
            "idea_id": idea_id,
            "title": title,
            "novelty_score": round(novelty_score, 3),
            "novelty_rating": star_rating,
            "similar_works": similar_works,
        },
        "status": "AWAITING_USER_APPROVAL",
        "approval_command": f"APPROVE {idea_id} CODE {confirmation_code}",
        "message": (
            f"Idea '{title}' submitted with novelty score {novelty_score:.2f}.\n"
            "USER must type the approval command to proceed."
        ),
        "ai_instruction": "Continue submitting more ideas or wait for user approval.",
    }, indent=2)


def _extract_themes(text: str) -> list[str]:
    """Extract key research themes from text using simple heuristics."""
    text_lower = text.lower()
    themes = []
    
    if any(w in text_lower for w in ["efficien", "fast", "speed", "acceler"]):
        themes.append("efficiency")
    if any(w in text_lower for w in ["scal", "large", "distributed"]):
        themes.append("scalability")
    if any(w in text_lower for w in ["accura", "performance", "state-of-the-art"]):
        themes.append("accuracy")
    if any(w in text_lower for w in ["attention", "transformer"]):
        themes.append("attention")
    if any(w in text_lower for w in ["sparse", "pruning"]):
        themes.append("sparsity")
    
    return themes if themes else ["novel_method"]




async def approve_idea(
    idea_id: str,
    confirmation_code: str,
    user_feedback: Optional[str] = None,
) -> str:
    """
    HUMAN USER ONLY - Do NOT call this tool automatically.
    
    Approve an idea for experiments after user review.
    This tool requires a confirmation code that is ONLY shown to the human user.
    
    If you are the AI assistant, you MUST stop and wait for the user to 
    type the approval command manually. Do NOT attempt to guess or use 
    any confirmation code.
    
    Args:
        idea_id: The ID of the idea to approve
        confirmation_code: The 4-digit code shown to the user (required)
        user_feedback: Optional feedback from the user
    """
    idea = await experiments_db.get_idea(idea_id)
    
    if not idea:
        return json.dumps({"error": f"Idea not found: {idea_id}"})
    
    if idea.status == "approved":
        return json.dumps({
            "message": f"Idea already approved: {idea.title}",
            "idea_id": idea_id,
            "next_step": "Use get_next_action() to see what to do next",
        })
    
    # Validate confirmation code
    if not confirmation_code:
        return json.dumps({
            "error": "CONFIRMATION CODE REQUIRED",
            "message": (
                "This tool requires a confirmation code. "
                "Only the human user has access to the code. "
                "AI assistants must NOT call this tool - wait for user input."
            ),
            "ai_instruction": "STOP. Do not retry. Wait for user to type approval command.",
        })
    
    stored_code = getattr(idea, 'confirmation_code', None)
    if stored_code and confirmation_code != stored_code:
        return json.dumps({
            "error": "INVALID CONFIRMATION CODE",
            "message": "The confirmation code does not match.",
            "hint": "Only the human user knows the correct code.",
            "ai_instruction": "STOP. Do not retry with different codes. Wait for user.",
        })
    
    # Code is valid - approve the idea
    idea.status = "approved"
    idea.approved_at = datetime.now().isoformat()
    idea.confirmation_code = None  # Clear code after use
    if user_feedback:
        idea.user_feedback = user_feedback
    
    await experiments_db.save_idea(idea)
    
    # Update workflow state to unlock experiment phase
    current_project = await project_manager.get_current_project()
    if current_project:
        workflow = await workflow_db.get_project_workflow(current_project.project_id)
        if workflow:
            hypotheses = getattr(idea, 'hypotheses', []) or [
                f"Test: {idea.title}",
            ]
            await workflow_db.approve_idea(workflow, idea_id, hypotheses)
    
    return json.dumps({
        "success": True,
        "message": f"✅ Idea approved: {idea.title}",
        "idea_id": idea_id,
        "status": "approved",
        "workflow_unlocked": True,
        "workflow_stage": "experiment_setup",
        "next_steps": [
            "1. Call get_next_action() to see required workflow steps",
            "2. Use create_experiment_env() to set up environment",
            "3. Use setup_datasets() to prepare data",
            "4. Use define_hypotheses(idea_id) to formalize experiments",
        ],
    }, indent=2)


async def reject_idea(idea_id: str, reason: Optional[str] = None) -> str:
    """Reject an idea and optionally provide reason for improvement."""
    idea = await experiments_db.get_idea(idea_id)
    
    if not idea:
        return json.dumps({"error": f"Idea not found: {idea_id}"})
    
    idea.status = "rejected"
    idea.rejection_reason = reason
    
    await experiments_db.save_idea(idea)
    
    return json.dumps({
        "success": True,
        "message": f"Idea rejected: {idea.title}",
        "reason": reason,
        "suggestion": "Use generate_ideas() with different focus to get new ideas",
    })


async def check_novelty(idea: str) -> str:
    """Check the novelty of a research idea against existing literature."""
    try:
        similar_papers = await s2_client.search(idea, max_results=15)
    except Exception:
        similar_papers = []
    
    novelty_score = 1.0
    similar_works = []
    
    idea_words = set(idea.lower().split())
    
    for paper in similar_papers:
        if paper.title:
            title_words = set(paper.title.lower().split())
            abstract_words = set((paper.abstract or "").lower().split())
            
            title_overlap = len(idea_words & title_words) / max(len(idea_words), 1)
            abstract_overlap = len(idea_words & abstract_words) / max(len(idea_words), 1)
            
            overlap = max(title_overlap, abstract_overlap * 0.5)
            
            if overlap > 0.5:
                novelty_score -= 0.15
                similar_works.append({
                    "title": paper.title,
                    "year": paper.year,
                    "citations": paper.citation_count,
                    "overlap": round(overlap, 2),
                    "venue": paper.venue,
                })
            elif overlap > 0.3:
                novelty_score -= 0.08
                similar_works.append({
                    "title": paper.title,
                    "year": paper.year,
                    "overlap": round(overlap, 2),
                })
    
    novelty_score = max(0.0, novelty_score)
    
    if novelty_score >= 0.7:
        recommendation = "accept"
        assessment = "High novelty - idea appears distinct from existing work"
    elif novelty_score >= 0.4:
        recommendation = "revise"
        assessment = "Moderate novelty - consider differentiating from similar works"
    else:
        recommendation = "reject"
        assessment = "Low novelty - significant overlap with existing literature"
    
    result = {
        "novelty_score": round(novelty_score, 2),
        "assessment": assessment,
        "recommendation": recommendation,
        "similar_works": similar_works[:5],
        "papers_checked": len(similar_papers),
        "differentiation_suggestions": [
            "Consider a different application domain",
            "Add a unique technical contribution",
            "Focus on a specific underexplored aspect",
        ] if novelty_score < 0.7 else [],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def create_research_plan(idea_id: str) -> str:
    """Create a detailed research plan for an approved idea."""
    idea = await experiments_db.get_idea(idea_id)
    
    if not idea:
        return json.dumps({"error": f"Idea not found: {idea_id}"})
    
    if idea.status != "approved":
        return json.dumps({
            "error": "Idea must be approved before creating a research plan",
            "current_status": idea.status,
            "action_required": "Use approve_idea(idea_id) first",
        })
    
    research_plan = {
        "phase_1_literature": {
            "duration": "1-2 weeks",
            "tasks": [
                "Conduct comprehensive literature review on related methods",
                "Identify key baselines for comparison",
                "Define evaluation metrics and protocols",
                "Document gaps in existing approaches",
            ],
            "deliverables": ["Literature survey document", "Baseline list", "Evaluation protocol"],
        },
        "phase_2_methodology": {
            "duration": "2-3 weeks",
            "tasks": [
                "Formalize the problem mathematically",
                "Design the proposed method in detail",
                "Implement core components",
                "Set up experiment infrastructure",
            ],
            "deliverables": ["Method description", "Implementation", "Experiment configs"],
        },
        "phase_3_experiments": {
            "duration": "3-4 weeks",
            "tasks": [
                "Run baseline experiments for comparison",
                "Execute main experiments with proposed method",
                "Conduct ablation studies",
                "Analyze failure cases",
            ],
            "deliverables": ["Main results table", "Ablation results", "Analysis document"],
        },
        "phase_4_analysis": {
            "duration": "1-2 weeks",
            "tasks": [
                "Statistical analysis of results",
                "Hypothesis verification",
                "Generate figures and tables",
                "Write analysis section",
            ],
            "deliverables": ["Statistical tests", "Figures", "Analysis draft"],
        },
        "phase_5_writing": {
            "duration": "2 weeks",
            "tasks": [
                "Write all paper sections",
                "Internal review and revision",
                "Format for target conference",
                "Prepare supplementary materials",
            ],
            "deliverables": ["Complete paper draft", "Supplementary materials"],
        },
        "estimated_total": "9-13 weeks",
        "resources_needed": {
            "compute": "GPU cluster or cloud instances",
            "datasets": "To be determined based on methodology",
            "baselines": idea.source_papers,
        },
    }
    
    idea.research_plan = research_plan
    idea.status = "planned"
    await experiments_db.save_idea(idea)
    
    result = {
        "idea_id": idea_id,
        "title": idea.title,
        "research_plan": research_plan,
        "next_step": "Use define_hypotheses(idea_id) to create testable hypotheses",
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def define_hypotheses(idea_id: str) -> str:
    """Define testable hypotheses for an approved idea."""
    idea = await experiments_db.get_idea(idea_id)
    
    if not idea:
        return json.dumps({"error": f"Idea not found: {idea_id}"})
    
    if idea.status not in ["approved", "planned"]:
        return json.dumps({
            "error": "Idea must be approved before defining hypotheses",
            "current_status": idea.status,
            "action_required": "Use approve_idea(idea_id) first",
        })
    
    # Generate hypotheses based on idea themes
    hypotheses = [
        {
            "id": "H1",
            "statement": f"The proposed method will outperform existing baselines on standard benchmarks by at least 5%",
            "type": "primary",
            "test": "Compare mean performance with paired t-test, α=0.05",
            "metrics": ["accuracy", "F1-score"],
            "expected_outcome": "Significant improvement over all baselines",
        },
        {
            "id": "H2",
            "statement": "Each component of the proposed method contributes positively to overall performance",
            "type": "ablation",
            "test": "Ablation study removing each component individually",
            "metrics": ["relative_improvement", "performance_drop"],
            "expected_outcome": "Removing any component degrades performance",
        },
        {
            "id": "H3",
            "statement": "The method maintains competitive performance while reducing computational cost",
            "type": "efficiency",
            "test": "Measure FLOPs, memory usage, and inference time",
            "metrics": ["FLOPs", "memory_mb", "latency_ms"],
            "expected_outcome": "At least 2x speedup with <5% accuracy drop",
        },
        {
            "id": "H4",
            "statement": "The method generalizes across different data distributions and domains",
            "type": "generalization",
            "test": "Cross-domain evaluation on held-out datasets",
            "metrics": ["transfer_accuracy", "domain_gap"],
            "expected_outcome": "Consistent performance across domains",
        },
    ]
    
    idea.hypotheses = [h["statement"] for h in hypotheses]
    await experiments_db.save_idea(idea)
    
    result = {
        "idea_id": idea_id,
        "title": idea.title,
        "hypotheses": hypotheses,
        "experimental_design": {
            "independent_variables": ["method_variant", "dataset", "hyperparameters"],
            "dependent_variables": ["accuracy", "efficiency", "generalization"],
            "controls": ["random_seed", "hardware", "data_splits"],
            "replication": "3 runs with different seeds",
        },
        "next_steps": [
            "1. Set up experiment environment with create_experiment_env()",
            "2. Run baseline experiments first",
            "3. Run proposed method experiments",
            "4. Conduct ablation studies",
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


async def list_ideas(status: Optional[str] = None) -> str:
    """List all generated ideas, optionally filtered by status."""
    ideas = await experiments_db.list_ideas(status=status)
    
    result = {
        "count": len(ideas),
        "filter": status,
        "ideas": [
            {
                "idea_id": i.idea_id,
                "title": i.title,
                "status": i.status,
                "created_at": i.created_at,
                "themes": getattr(i, "themes", []),
            }
            for i in ideas
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)
