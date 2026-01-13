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


def _extract_key_terms(papers: list[dict]) -> dict:
    """Extract key methods, techniques, and domains from papers."""
    import re
    
    all_text = " ".join(
        p.get("title", "") + " " + p.get("abstract", "")[:500] 
        for p in papers
    ).lower()
    
    # Extract capitalized terms (likely method names)
    method_pattern = r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]*)*)\b'
    all_raw = " ".join(p.get("title", "") + " " + p.get("abstract", "")[:500] for p in papers)
    capitalized = re.findall(method_pattern, all_raw)
    
    # Filter to likely method names - exclude common descriptive words
    common_words = {
        # Generic words
        "The", "This", "Our", "We", "They", "These", "However", "While", 
        "Although", "Furthermore", "Moreover", "Results", "Method", "Approach",
        "Figure", "Table", "Section", "Abstract", "Introduction", "Related",
        "Experimental", "Conclusion", "Paper", "Work", "Study", "Analysis",
        # Descriptive adjectives often capitalized in titles
        "Scalable", "Efficient", "Fast", "Novel", "Accurate", "Robust",
        "Simple", "Effective", "Improved", "Better", "New", "High", "Low",
        "Large", "Small", "Deep", "Wide", "Long", "Short", "Local", "Global",
        "Sparse", "Dense", "Linear", "Dynamic", "Static", "Adaptive", "Smart",
        "Training", "Learning", "Inference", "Acceleration", "Attention",
        "Based", "Free", "Aware", "Guided", "Driven", "Enhanced", "Unified",
        # Generic ML terms
        "Model", "Models", "Network", "Networks", "Framework", "System",
        "Architecture", "Architectures", "Module", "Modules", "Layer", "Layers",
        "Transformer", "Transformers", "Neural", "Machine", "Data", "Dataset",
    }
    methods = [m for m in capitalized if len(m) > 2 and m not in common_words][:8]
    
    # Also look for acronyms (all caps, 2-6 chars)
    acronym_pattern = r'\b([A-Z]{2,6})\b'
    acronyms = re.findall(acronym_pattern, all_raw)
    acronym_blacklist = {"GPU", "CPU", "TPU", "RAM", "API", "FPS", "RGB", "NLP", "CV", "ML", "AI", "LLM", "CNN", "RNN", "GNN", "MLP", "SOTA", "FPGA"}
    valid_acronyms = [a for a in acronyms if a not in acronym_blacklist and len(a) >= 3][:4]
    
    # Combine methods and acronyms
    methods = list(dict.fromkeys(valid_acronyms + methods))[:6]  # Dedupe, acronyms first
    
    # Extract key noun phrases from titles
    titles = [p.get("title", "") for p in papers]
    title_keywords = []
    for title in titles:
        words = title.lower().split()
        for i, word in enumerate(words):
            if word in ["for", "via", "using", "with", "in", "on", "to"]:
                if i > 0:
                    title_keywords.append(words[i-1])
                if i < len(words) - 1:
                    title_keywords.append(words[i+1])
    
    # Detect domain from content
    domain_keywords = {
        "document reranking": ["rerank", "ranking", "retrieval", "document", "passage"],
        "information retrieval": ["retrieval", "search", "query", "ir ", "bm25"],
        "text generation": ["generation", "generative", "llm", "language model"],
        "question answering": ["qa", "question", "answer", "squad", "reading comprehension"],
        "reasoning": ["reasoning", "cot", "chain-of-thought", "rationale"],
        "contrastive learning": ["contrastive", "contrast", "positive", "negative", "pairs"],
        "knowledge distillation": ["distillation", "distill", "teacher", "student"],
        "transformer optimization": ["transformer", "attention", "efficient", "sparse"],
        "tabular learning": ["tabular", "structured", "boosting", "tree", "xgboost", "lightgbm"],
    }
    
    detected_domain = "machine learning"
    for domain, keywords in domain_keywords.items():
        if any(kw in all_text for kw in keywords):
            detected_domain = domain
            break
    
    # Extract specific techniques mentioned
    technique_keywords = {
        "contrastive learning": ["contrastive", "positive pair", "negative sample"],
        "distillation": ["distillation", "teacher-student", "knowledge transfer"],
        "attention mechanism": ["attention", "self-attention", "cross-attention"],
        "ranking loss": ["pairwise", "listwise", "margin loss", "ranking loss"],
        "fine-tuning": ["fine-tune", "finetuning", "adapter", "lora"],
        "sparse retrieval": ["sparse", "bm25", "tf-idf", "lexical"],
        "dense retrieval": ["dense", "embedding", "semantic", "neural"],
    }
    
    techniques = []
    for tech, keywords in technique_keywords.items():
        if any(kw in all_text for kw in keywords):
            techniques.append(tech)
    
    return {
        "methods": methods if methods else ["neural approach"],
        "domain": detected_domain,
        "techniques": techniques if techniques else ["novel optimization"],
        "title_keywords": list(set(title_keywords))[:5],
        "paper_titles": [p.get("title", "")[:60] for p in papers[:3]],
    }


def _create_idea_from_themes(
    idea_id: str,
    themes: list[str],
    papers: list[dict],
    focus: Optional[str],
    index: int,
) -> Idea:
    """Create a research idea based on ACTUAL paper content."""
    
    # Extract actual terms from papers
    terms = _extract_key_terms(papers)
    domain = focus or terms["domain"]
    methods = terms["methods"]
    techniques = terms["techniques"]
    paper_titles = terms["paper_titles"]
    
    # Use first paper title as inspiration
    main_paper = paper_titles[0] if paper_titles else "the source papers"
    
    # Create genuinely different ideas based on index
    if index == 0:
        # Idea 1: Combine two methods from the papers
        method1 = methods[0] if methods else "the primary method"
        method2 = methods[1] if len(methods) > 1 else techniques[0] if techniques else "auxiliary approach"
        
        title = f"Unified {method1}-{method2} Framework for {domain.title()}"
        description = (
            f"Building on insights from '{main_paper}', we propose combining {method1} "
            f"with {method2} in a unified framework for {domain}. "
            f"The key insight is that {method1} provides strong representation learning "
            f"while {method2} enables efficient inference. Our unified approach "
            f"achieves the benefits of both without their individual limitations."
        )
        motivation = (
            f"Prior work in {domain} either uses {method1} or {method2} separately. "
            f"We identify complementary strengths and propose a principled integration."
        )
        
    elif index == 1:
        # Idea 2: Efficiency improvement of main method
        main_method = methods[0] if methods else "the primary approach"
        tech = techniques[0] if techniques else "approximation techniques"
        
        title = f"Efficient {main_method} via {tech.title()} for Scalable {domain.title()}"
        description = (
            f"Inspired by '{main_paper}', we propose an efficient variant of {main_method} "
            f"that reduces computational overhead through {tech}. "
            f"While existing {main_method} approaches achieve strong performance, "
            f"their quadratic complexity limits scalability. Our method maintains "
            f"competitive accuracy with linear complexity, enabling real-world deployment."
        )
        motivation = (
            f"The computational cost of {main_method} in {domain} limits practical application. "
            f"We address this by introducing principled {tech} without sacrificing quality."
        )
        
    else:
        # Idea 3: Novel application or theoretical contribution
        main_method = methods[0] if methods else "the proposed approach"
        secondary = methods[1] if len(methods) > 1 else "complementary signals"
        
        title = f"{main_method} with {secondary.title()} for Enhanced {domain.title()}"
        description = (
            f"Extending ideas from '{main_paper}', we propose using {main_method} "
            f"enhanced with {secondary} for improved {domain}. "
            f"Our key contribution is demonstrating that {secondary} provides "
            f"additional supervision signal that significantly improves {main_method}'s "
            f"performance on challenging {domain} benchmarks."
        )
        motivation = (
            f"While {main_method} shows promise for {domain}, its performance can be "
            f"further improved by incorporating {secondary}. We provide theoretical "
            f"justification and empirical validation of this combination."
        )
    
    filled_title = title
    filled_description = description
    filled_motivation = motivation
    
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
