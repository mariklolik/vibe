"""Research agent: literature search + idea generation.

Combines arxiv-search and idea-generator skills from research_claude_agents.
Handles: paper discovery, context extraction, idea generation, auto-selection.

Proxy optimization:
- Paper analysis: effort="medium", sonnet (fast, cheap)
- Idea generation: effort="high", sonnet with opus fallback (creative quality)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

from src.agents.base import BaseAgent
from src.agents.client import DEFAULT_MODEL, EFFORT_MEDIUM, EFFORT_HIGH
from src.apis.arxiv import arxiv_client
from src.apis.semantic_scholar import s2_client
from src.apis.huggingface import hf_client
from src.state.progress import append_progress, save_project_config, load_project_config

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are an expert AI research scientist specializing in finding novel, publishable research ideas.

Your role is to:
1. Analyze recent papers in a given research area
2. Identify gaps, limitations, and opportunities
3. Generate creative, feasible research ideas with clear novelty
4. Assess each idea's novelty score (0.0-1.0) and feasibility

## Output Format
Always respond with a JSON block containing your analysis:

```json
{
  "paper_analysis": {
    "key_trends": ["trend1", "trend2"],
    "gaps_identified": ["gap1", "gap2"],
    "promising_directions": ["dir1", "dir2"]
  },
  "ideas": [
    {
      "title": "Short descriptive title",
      "description": "2-3 sentence description of the idea",
      "motivation": "Why this is important and timely",
      "novelty_score": 0.85,
      "feasibility_score": 0.8,
      "method_summary": "Brief technical approach",
      "datasets": ["dataset1", "dataset2"],
      "baselines": ["baseline1", "baseline2"],
      "target_conference": "ICML",
      "estimated_compute": "4x A100 for 24h"
    }
  ],
  "recommended_idea_index": 0
}
```

## Guidelines
- Generate 3-5 ideas per request
- Novelty score reflects how different the idea is from existing work
- Feasibility score reflects practical implementability
- Focus on ideas that can produce real experimental results
- Prefer ideas with clear evaluation metrics
- Target top ML venues (NeurIPS, ICML, ICLR, CVPR, ACL)
- Be specific about datasets and baselines — no vague claims
"""


class ResearchAgent(BaseAgent):
    """Handles literature search and idea generation.

    Uses effort="medium" for paper analysis (fast/cheap) and
    effort="high" with opus fallback for idea generation (quality).
    """

    name = "research"
    system_prompt = RESEARCH_SYSTEM_PROMPT
    default_effort = EFFORT_MEDIUM
    default_fallback_model = "haiku"

    def __init__(
        self,
        client: anthropic.Anthropic,
        project_dir: str,
        model: str = DEFAULT_MODEL,
    ):
        super().__init__(client, project_dir, model)
        self.context_dir = Path(project_dir) / "context"
        self.ideas_dir = Path(project_dir) / "ideas"
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.ideas_dir.mkdir(parents=True, exist_ok=True)

    async def search_papers(
        self,
        topic: str,
        max_papers: int = 15,
    ) -> list[dict]:
        """Search for papers across all sources.

        Uses vibe's existing API clients (arXiv, S2, HuggingFace).
        Returns deduplicated list of paper dicts.
        """
        logger.info(f"Searching papers for topic: {topic}")
        all_papers = []
        seen_titles = set()

        # Search arXiv
        try:
            arxiv_papers = await arxiv_client.search(topic, max_results=max_papers)
            for p in arxiv_papers:
                if p.title.lower() not in seen_titles:
                    seen_titles.add(p.title.lower())
                    all_papers.append(p.to_dict())
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

        # Search Semantic Scholar
        try:
            s2_papers = await s2_client.search(topic, max_results=max_papers)
            for p in s2_papers:
                if p.title and p.title.lower() not in seen_titles:
                    seen_titles.add(p.title.lower())
                    all_papers.append(p.to_dict())
        except Exception as e:
            logger.warning(f"S2 search failed: {e}")

        # Search HuggingFace trending
        try:
            hf_papers = await hf_client.fetch_trending(topic=topic, max_results=max_papers // 2)
            for p in hf_papers:
                if p.title and p.title.lower() not in seen_titles:
                    seen_titles.add(p.title.lower())
                    all_papers.append(p.to_dict())
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {e}")

        logger.info(f"Found {len(all_papers)} unique papers")

        # Save to context directory
        context_file = self.context_dir / "papers.json"
        context_file.write_text(json.dumps(all_papers, indent=2, default=str))

        return all_papers[:max_papers]

    def generate_ideas(
        self,
        topic: str,
        papers: list[dict],
        num_ideas: int = 5,
    ) -> tuple[Optional[dict], str]:
        """Use Claude to analyze papers and generate research ideas.

        Uses effort="high" with opus fallback for creative quality.
        """
        # Build paper summaries for context
        paper_summaries = []
        for i, p in enumerate(papers[:15], 1):
            title = p.get("title", "Unknown")
            abstract = p.get("abstract", p.get("summary", ""))
            if abstract:
                abstract = abstract[:500]
            citations = p.get("citation_count", "N/A")
            year = p.get("year", p.get("published", ""))
            paper_summaries.append(
                f"[{i}] {title} ({year})\n"
                f"    Citations: {citations}\n"
                f"    Abstract: {abstract}"
            )

        papers_text = "\n\n".join(paper_summaries)

        task = (
            f"Analyze these {len(papers)} recent papers on '{topic}' and generate "
            f"{num_ideas} novel research ideas.\n\n"
            f"## Papers\n{papers_text}\n\n"
            f"Generate ideas that are:\n"
            f"- Novel (not a direct copy of any listed paper)\n"
            f"- Feasible (can be implemented and evaluated)\n"
            f"- Publishable (clear contribution for a top ML venue)\n"
            f"- Specific (concrete method, datasets, and baselines)\n"
        )

        # Higher effort for idea generation — this is the creative step
        return self.call_structured(
            task=task,
            max_tokens=8192,
            temperature=0.8,
            effort=EFFORT_HIGH,
            fallback_model="haiku",
        )

    def select_best_idea(
        self,
        ideas: dict,
        min_novelty: float = 0.7,
    ) -> Optional[dict]:
        """Auto-select the best idea (research_claude_agents pattern).

        Selects the highest-scoring idea with novelty >= threshold.
        Returns None if no idea meets the threshold.
        """
        idea_list = ideas.get("ideas", [])
        if not idea_list:
            return None

        # Sort by combined score (novelty * 0.6 + feasibility * 0.4)
        def score(idea):
            n = idea.get("novelty_score", 0)
            f = idea.get("feasibility_score", 0)
            return n * 0.6 + f * 0.4

        sorted_ideas = sorted(idea_list, key=score, reverse=True)

        for idea in sorted_ideas:
            if idea.get("novelty_score", 0) >= min_novelty:
                logger.info(
                    f"Auto-selected idea: {idea['title']} "
                    f"(novelty={idea['novelty_score']}, "
                    f"feasibility={idea.get('feasibility_score', 'N/A')})"
                )
                return idea

        logger.warning(f"No ideas met novelty threshold {min_novelty}")
        return None

    async def run(
        self,
        topic: str,
        min_novelty: float = 0.7,
    ) -> Optional[dict]:
        """Full research pipeline: search -> analyze -> generate -> select.

        This is the main entry point for the research agent.
        Returns the selected idea dict, or None if nothing qualified.
        """
        self.log_progress(f"Starting research on topic: {topic}")

        # Step 1: Search papers
        papers = await self.search_papers(topic)
        if not papers:
            self.log_progress("ERROR: No papers found. Check topic and API connectivity.")
            return None

        self.log_progress(
            f"Found {len(papers)} papers. "
            f"Top papers: {', '.join(p.get('title', '')[:60] for p in papers[:3])}"
        )

        # Step 2: Generate ideas using Claude (effort="high" for creativity)
        ideas_dict, raw_text = self.generate_ideas(topic, papers)
        if not ideas_dict:
            self.log_progress(f"ERROR: Failed to parse ideas from Claude response.\nRaw: {raw_text[:500]}")
            return None

        # Save all ideas
        ideas_file = self.ideas_dir / "all_ideas.json"
        ideas_file.write_text(json.dumps(ideas_dict, indent=2))

        idea_list = ideas_dict.get("ideas", [])
        self.log_progress(
            f"Generated {len(idea_list)} ideas:\n" +
            "\n".join(
                f"  - {i.get('title', '?')} (novelty={i.get('novelty_score', '?')})"
                for i in idea_list
            )
        )

        # Step 3: Auto-select best idea
        selected = self.select_best_idea(ideas_dict, min_novelty=min_novelty)
        if not selected:
            self.log_progress(
                f"No ideas met novelty threshold {min_novelty}. "
                f"Best was {idea_list[0].get('novelty_score', 0) if idea_list else 'N/A'}"
            )
            return None

        # Save selected idea
        selected_file = self.ideas_dir / "selected_idea.json"
        selected_file.write_text(json.dumps(selected, indent=2))

        # Update project config
        config = load_project_config(self.project_dir)
        config["selected_idea"] = selected["title"]
        config["stage"] = "experiment"
        save_project_config(self.project_dir, config)

        self.log_progress(
            f"SELECTED IDEA: {selected['title']}\n"
            f"Novelty: {selected.get('novelty_score')}\n"
            f"Feasibility: {selected.get('feasibility_score')}\n"
            f"Description: {selected.get('description')}\n"
            f"Method: {selected.get('method_summary')}\n"
            f"Datasets: {selected.get('datasets')}\n"
            f"Baselines: {selected.get('baselines')}\n"
            f"Target: {selected.get('target_conference')}"
        )

        return selected
