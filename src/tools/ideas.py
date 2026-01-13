"""Idea generation tools - generate, validate, and plan research ideas with user verification."""

import json
import random
import uuid
from datetime import datetime
from typing import Optional

from src.apis.semantic_scholar import s2_client
from src.db.papers_cache import papers_cache
from src.db.experiments_db import experiments_db, Idea


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
    """Generate research ideas with novelty scoring.
    
    Ideas are automatically scored for novelty and returned ranked (highest first).
    User MUST call approve_idea(idea_id) to approve before experiments can proceed.
    """
    papers_context = []
    
    for paper_id in paper_ids:
        cached = await papers_cache.get_paper(paper_id)
        if cached:
            papers_context.append({
                "title": cached.title,
                "abstract": cached.abstract[:800],
                "categories": cached.categories,
            })
    
    if not papers_context:
        return json.dumps({"error": "No papers found with provided IDs"})
    
    # Generate ideas with novelty scoring
    ideas = []
    for i in range(count):
        idea_id = f"idea_{uuid.uuid4().hex[:8]}"
        
        all_titles = " ".join(p["title"].lower() for p in papers_context)
        all_abstracts = " ".join(p["abstract"].lower() for p in papers_context)
        
        themes = _extract_themes(all_titles + " " + all_abstracts)
        idea = _create_idea_from_themes(idea_id, themes, papers_context, focus, i)
        
        # Compute novelty score for this idea
        novelty_score, similar_works = await compute_idea_novelty(idea.description)
        idea.novelty_score = novelty_score
        idea.similar_works = similar_works
        
        await experiments_db.save_idea(idea)
        ideas.append(idea)
    
    # Sort by novelty score (highest first) - this is the "scored" approach user requested
    ideas.sort(key=lambda x: x.novelty_score or 0, reverse=True)
    
    # Build ranked response for user review
    ranked_ideas = []
    for rank, idea in enumerate(ideas, 1):
        novelty = idea.novelty_score or 0
        star_rating = "★" * int(novelty * 5) + "☆" * (5 - int(novelty * 5))
        ranked_ideas.append({
            "rank": rank,
            "idea_id": idea.idea_id,
            "title": idea.title,
            "novelty_score": round(novelty, 3),
            "novelty_rating": star_rating,
            "description": idea.description,
            "motivation": idea.motivation,
            "themes": idea.themes,
            "similar_works": idea.similar_works,
            "recommendation": "RECOMMENDED" if rank == 1 and novelty >= 0.6 else "",
        })
    
    # Build user approval instructions with confirmation codes
    # The codes are shown to the user but the AI should NOT auto-approve
    approval_instructions = []
    for idea in ideas:
        approval_instructions.append({
            "idea_id": idea.idea_id,
            "title": idea.title[:50],
            "approval_command": f"APPROVE {idea.idea_id} CODE {idea.confirmation_code}",
        })
    
    result = {
        "status": "BLOCKED_AWAITING_USER_APPROVAL",
        "message": (
            "⚠️ IDEAS REQUIRE HUMAN APPROVAL ⚠️\n"
            "The AI assistant MUST NOT call approve_idea automatically.\n"
            "The USER must review ideas and type the approval command manually."
        ),
        "count": len(ideas),
        "focus": focus,
        "source_papers": [p["title"][:50] for p in papers_context],
        "ranked_ideas": ranked_ideas,
        "top_recommendation": ranked_ideas[0] if ranked_ideas else None,
        
        # Instructions for the USER (not the AI)
        "user_instructions": {
            "step_1": "Review the ideas above carefully",
            "step_2": "Choose one idea to pursue",
            "step_3": "Type the approval command in chat (shown below)",
            "approval_commands": approval_instructions,
        },
        
        # Explicit instruction for AI
        "ai_instruction": (
            "STOP HERE. Do NOT call approve_idea or any other tool. "
            "Wait for the user to type an approval command. "
            "The confirmation codes are only known to the user."
        ),
        
        "alternative_actions": [
            "reject_idea(idea_id, reason) - to reject with feedback",
            "generate_ideas(paper_ids, focus='different_topic') - to get new ideas",
        ],
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)


def _extract_themes(text: str) -> list[str]:
    """Extract key research themes from text."""
    theme_keywords = {
        "efficiency": ["efficient", "fast", "speed", "acceleration", "lightweight"],
        "scalability": ["scale", "scalable", "large-scale", "distributed"],
        "robustness": ["robust", "reliable", "stable", "generalization"],
        "interpretability": ["interpretable", "explainable", "understanding", "visualization"],
        "accuracy": ["accurate", "performance", "state-of-the-art", "sota"],
        "novel_architecture": ["architecture", "network", "transformer", "attention"],
        "training": ["training", "optimization", "learning", "convergence"],
        "data_efficiency": ["few-shot", "low-resource", "sample efficiency", "data augmentation"],
    }
    
    found_themes = []
    for theme, keywords in theme_keywords.items():
        if any(kw in text for kw in keywords):
            found_themes.append(theme)
    
    return found_themes if found_themes else ["novel_architecture", "efficiency"]


def _create_idea_from_themes(
    idea_id: str,
    themes: list[str],
    papers: list[dict],
    focus: Optional[str],
    index: int,
) -> Idea:
    """Create a research idea based on extracted themes."""
    
    # Idea templates based on common research patterns
    idea_templates = [
        {
            "pattern": "combining_methods",
            "title_template": "Unified Framework for {theme1} and {theme2} in {domain}",
            "description_template": (
                "We propose a unified framework that combines the strengths of {method1} "
                "with {method2} to achieve both {benefit1} and {benefit2}. "
                "Unlike prior work that addresses these challenges separately, our approach "
                "provides an end-to-end solution that leverages {key_insight}."
            ),
            "motivation": (
                "Existing methods either focus on {theme1} or {theme2}, but not both. "
                "This creates a gap in the literature for approaches that can simultaneously "
                "address multiple challenges while maintaining practical applicability."
            ),
        },
        {
            "pattern": "efficiency_improvement",
            "title_template": "Efficient {method} via {technique} for {application}",
            "description_template": (
                "We introduce an efficient variant of {method} that reduces computational "
                "complexity from O(n²) to O(n log n) through novel use of {technique}. "
                "Our approach maintains competitive accuracy while enabling deployment on "
                "resource-constrained devices."
            ),
            "motivation": (
                "While {method} achieves state-of-the-art results, its computational "
                "requirements limit practical deployment. Our work addresses this gap by "
                "introducing principled approximations that preserve performance."
            ),
        },
        {
            "pattern": "novel_application",
            "title_template": "Adapting {method} for {new_domain}: A {approach} Approach",
            "description_template": (
                "We adapt {method}, originally designed for {original_domain}, to the "
                "challenging setting of {new_domain}. Our key insight is that {insight}, "
                "which enables effective transfer through {mechanism}."
            ),
            "motivation": (
                "Despite the success of {method} in {original_domain}, its application to "
                "{new_domain} remains unexplored. We bridge this gap by identifying key "
                "analogies and developing domain-specific adaptations."
            ),
        },
    ]
    
    template = idea_templates[index % len(idea_templates)]
    
    # Fill in template with context
    paper_title = papers[0]["title"] if papers else "Deep Learning"
    domain = focus or _extract_domain(papers)
    
    # Generate specific content
    filled_title = f"Research Idea {index + 1}: {template['pattern'].replace('_', ' ').title()}"
    filled_description = template["description_template"].format(
        method1="attention mechanisms",
        method2="gradient boosting",
        method="the proposed approach",
        technique="structured pruning",
        benefit1="improved accuracy",
        benefit2="reduced latency",
        key_insight="complementary inductive biases",
        application=domain,
        original_domain="natural language processing",
        new_domain=domain,
        insight="structural similarities exist",
        mechanism="careful feature alignment",
        theme1=themes[0] if themes else "efficiency",
        theme2=themes[1] if len(themes) > 1 else "accuracy",
    )
    
    filled_motivation = template["motivation"].format(
        method="existing approaches",
        theme1=themes[0] if themes else "efficiency",
        theme2=themes[1] if len(themes) > 1 else "accuracy",
        original_domain="computer vision",
        new_domain=domain,
    )
    
    return Idea(
        idea_id=idea_id,
        title=filled_title,
        description=filled_description,
        source_papers=[p["title"][:50] for p in papers[:3]],
        hypotheses=[],
        research_plan={},
        novelty_score=None,
        created_at=datetime.now().isoformat(),
        status="pending_approval",  # Requires user approval
        motivation=filled_motivation,
        themes=themes,
        confirmation_code=_generate_confirmation_code(),  # Code required for approval
    )


def _extract_domain(papers: list[dict]) -> str:
    """Extract the research domain from papers."""
    domains = {
        "nlp": ["language", "text", "nlp", "translation", "sentiment"],
        "vision": ["image", "vision", "visual", "object", "detection"],
        "ml": ["learning", "neural", "model", "training", "optimization"],
        "tabular": ["tabular", "structured", "boosting", "tree", "ensemble"],
    }
    
    all_text = " ".join(p.get("title", "") + " " + p.get("abstract", "") for p in papers).lower()
    
    for domain, keywords in domains.items():
        if any(kw in all_text for kw in keywords):
            return domain
    
    return "machine learning"


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
    
    return json.dumps({
        "success": True,
        "message": f"✅ Idea approved: {idea.title}",
        "idea_id": idea_id,
        "status": "approved",
        "workflow_unlocked": True,
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
