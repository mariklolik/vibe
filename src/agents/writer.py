"""Paper writing agent: drafts publication-ready papers.

Maps to paper-writer skill from research_claude_agents.
Handles: style extraction, section-by-section writing, LaTeX formatting, expansion loop.

Proxy optimization:
- Writing: effort="high" (quality matters for publication)
- Expansion: effort="medium" (additive, less critical)
- Fallback to haiku if rate-limited
- max_tokens=4096 per section (proxy stability)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

from src.agents.base import BaseAgent
from src.agents.client import DEFAULT_MODEL, EFFORT_HIGH, EFFORT_MEDIUM
from src.state.progress import load_project_config, save_project_config

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are an expert academic paper writer specializing in ML/AI research papers for top venues (NeurIPS, ICML, ICLR, CVPR, ACL).

Your role is to write publication-ready paper sections with:
- Precise technical language matching the target venue's style
- Proper LaTeX formatting (equations, tables, figures, algorithms)
- Only verified claims backed by experimental results (p < 0.05)
- Statistical notation: mean +/- std for all reported numbers
- Honest discussion of limitations

## Output Format

Always respond with a JSON code block:

```json
{
  "section": "section_name",
  "content": "LaTeX content here...",
  "word_count": 950,
  "citations_used": ["author2024method"]
}
```

## Style Guidelines
- Write in third person or "we" (first person plural)
- Use active voice where possible
- Keep sentences concise (15-25 words average)
- Every claim must reference supporting evidence
- Abstract: 150-250 words, structured (motivation, method, results, conclusion)
- Introduction: clear gap -> our contribution -> brief results
- Related Work: organized by theme, not chronologically
- Method: formal notation, algorithm blocks where helpful
- Experiments: setup first, then main results, then ablations
- Discussion: honest limitations, broader impact
- All references must be REAL papers with correct authors/titles/venues
"""

CONFERENCE_TARGETS = {
    "neurips": {"pages": 9, "words": 6000, "figures": 7, "tables": 4, "abstract_words": 250},
    "icml": {"pages": 9, "words": 6000, "figures": 7, "tables": 4, "abstract_words": 250},
    "iclr": {"pages": 10, "words": 6500, "figures": 7, "tables": 4, "abstract_words": 250},
    "cvpr": {"pages": 8, "words": 5500, "figures": 8, "tables": 3, "abstract_words": 200},
    "acl": {"pages": 8, "words": 5000, "figures": 5, "tables": 5, "abstract_words": 200},
    "aaai": {"pages": 8, "words": 5500, "figures": 6, "tables": 4, "abstract_words": 200},
}


class WriterAgent(BaseAgent):
    """Handles paper writing with section-by-section approach and expansion loop.

    Uses effort="high" for initial writing (quality matters).
    Uses effort="medium" for expansion (additive content).
    """

    name = "writer"
    system_prompt = WRITER_SYSTEM_PROMPT
    default_effort = EFFORT_HIGH
    default_fallback_model = "haiku"

    def __init__(
        self,
        client: anthropic.Anthropic,
        project_dir: str,
        model: str = DEFAULT_MODEL,
    ):
        super().__init__(client, project_dir, model)
        self.paper_dir = Path(project_dir) / "paper"
        self.paper_dir.mkdir(parents=True, exist_ok=True)

    def _load_writing_context(self) -> str:
        """Load all context needed for paper writing."""
        parts = []

        idea_path = Path(self.project_dir) / "ideas" / "selected_idea.json"
        if idea_path.exists():
            idea = json.loads(idea_path.read_text())
            parts.append(f"## Research Idea\n{json.dumps(idea, indent=2)}")

        results_path = Path(self.project_dir) / "experiments" / "all_results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            parts.append(f"## Experiment Results\n{json.dumps(results, indent=2, default=str)}")

        verif_dir = Path(self.project_dir) / "verification"
        if verif_dir.exists():
            verif_results = []
            for f in verif_dir.glob("*.json"):
                verif_results.append(json.loads(f.read_text()))
            if verif_results:
                parts.append(f"## Verified Claims\n{json.dumps(verif_results, indent=2)}")

        context_dir = Path(self.project_dir) / "context"
        blueprint_path = context_dir / "writing_blueprint.json"
        if blueprint_path.exists():
            blueprint = json.loads(blueprint_path.read_text())
            parts.append(f"## Writing Style Blueprint\n{json.dumps(blueprint, indent=2)}")

        return "\n\n".join(parts)

    def _get_paper_title(self) -> str:
        idea_path = Path(self.project_dir) / "ideas" / "selected_idea.json"
        if idea_path.exists():
            idea = json.loads(idea_path.read_text())
            return idea.get("title", "Untitled Research Paper")
        return "Untitled Research Paper"

    def write_section(
        self,
        section_name: str,
        section_guidance: str,
        target_words: int,
        previous_sections: list[dict],
        conference: str = "icml",
    ) -> tuple[Optional[dict], str]:
        """Write a single paper section (smaller requests = more reliable with proxy).

        Uses effort="high" for quality writing.
        """
        context = self._load_writing_context()

        prev_text = ""
        if previous_sections:
            prev_text = "\n\n## Previously Written Sections\n"
            for s in previous_sections:
                prev_text += f"\n### {s.get('name', '')}\n{s.get('content', '')[:500]}...\n"

        task = (
            f"Write the '{section_name}' section for a {conference.upper()} 2026 paper.\n\n"
            f"Target: ~{target_words} words.\n"
            f"{section_guidance}\n\n"
            f"Requirements:\n"
            f"- ONLY verified claims (p < 0.05)\n"
            f"- Use mean +/- std notation\n"
            f"- Proper LaTeX formatting\n"
            f"- All citations must be REAL papers\n"
            f"{prev_text}\n\n"
            f"Respond with:\n"
            f'```json\n{{"section": "{section_name}", "content": "LaTeX content...", '
            f'"word_count": N, "citations_used": ["key1"]}}\n```'
        )

        return self.call_structured(
            task=task,
            extra_context=context,
            max_tokens=4096,
            temperature=0.6,
            effort=EFFORT_HIGH,
        )

    def write_full_paper(
        self,
        conference: str = "icml",
    ) -> tuple[Optional[dict], str]:
        """Generate the full paper by writing section by section.

        Smaller requests are more reliable with the proxy and produce
        better papers (focused attention per section).
        """
        SECTIONS = [
            ("abstract", "Write a concise abstract: motivation, method, key results, conclusion.", 200),
            ("introduction", "Motivate the problem, state contributions, preview results.", 950),
            ("related_work", "Organize by theme. Compare to our approach. Be fair.", 850),
            ("method", "Formal notation, algorithm blocks. Be precise and reproducible.", 1800),
            ("experiments", "Setup first, then main results with tables, then ablations.", 1500),
            ("discussion", "Honest limitations, broader impact, future work.", 500),
            ("conclusion", "Summarize contributions and key findings.", 400),
        ]

        written_sections = []
        all_citations = []
        raw_texts = []

        for name, guidance, target_words in SECTIONS:
            logger.info(f"Writing section: {name}")
            result, raw = self.write_section(
                section_name=name,
                section_guidance=guidance,
                target_words=target_words,
                previous_sections=written_sections,
                conference=conference,
            )
            raw_texts.append(raw)

            if result:
                content = result.get("content", raw)
                written_sections.append({
                    "name": name,
                    "content": content,
                    "word_count": result.get("word_count", len(content.split())),
                })
                all_citations.extend(result.get("citations_used", []))
                self.log_progress(f"Wrote {name}: ~{result.get('word_count', '?')} words")
            else:
                written_sections.append({
                    "name": name,
                    "content": raw if raw else f"% TODO: {name} section",
                    "word_count": len(raw.split()) if raw else 0,
                })
                self.log_progress(f"Wrote {name}: (raw fallback, ~{len(raw.split()) if raw else 0} words)")

        # Enrich references with real metadata (not just citation keys)
        unique_citations = list(set(all_citations))
        enriched_refs = self._enrich_references(unique_citations)

        paper = {
            "paper": {
                "title": self._get_paper_title(),
                "abstract": next(
                    (s["content"] for s in written_sections if s["name"] == "abstract"),
                    "Abstract placeholder.",
                ),
                "sections": [s for s in written_sections if s["name"] != "abstract"],
                "references": enriched_refs,
            }
        }

        return paper, "\n\n".join(raw_texts)

    def _enrich_references(self, citation_keys: list[str]) -> list[dict]:
        """Enrich citation keys with real bibliographic metadata via Claude.

        Without this, references are just stubs like {key: "brown2020language", title: "brown2020language"}.
        This calls Claude to populate authors, title, venue, year for each citation.
        """
        if not citation_keys:
            return []

        # Batch all keys in one call for efficiency
        keys_text = "\n".join(f"- {k}" for k in citation_keys)

        task = (
            f"Provide real bibliographic metadata for these citation keys.\n\n"
            f"Citation keys:\n{keys_text}\n\n"
            f"For EACH key, provide the REAL paper's: authors, title, venue, year.\n"
            f"If a key doesn't map to a real paper, provide your best match.\n\n"
            f"Return a JSON block:\n"
            f'```json\n{{"references": [\n'
            f'  {{"key": "brown2020language", "authors": "Tom Brown et al.", '
            f'"title": "Language Models are Few-Shot Learners", '
            f'"venue": "NeurIPS", "year": "2020"}}\n'
            f']}}\n```\n'
        )

        result, raw = self.call_structured(
            task=task,
            max_tokens=4096,
            temperature=0.2,
            effort="medium",
        )

        if result and "references" in result:
            refs = result["references"]
            # Validate: each ref must have key, authors, title
            valid_refs = []
            for ref in refs:
                if ref.get("key") and ref.get("title") and ref.get("title") != ref.get("key"):
                    valid_refs.append(ref)
                else:
                    # Fallback: keep as stub
                    valid_refs.append({"key": ref.get("key", "unknown"), "title": ref.get("key", "unknown")})
            self.log_progress(f"Enriched {len(valid_refs)}/{len(citation_keys)} references with real metadata")
            return valid_refs

        # Fallback: return stubs (same as before)
        logger.warning("Reference enrichment failed, using citation key stubs")
        return [{"key": k, "title": k} for k in citation_keys]

    def check_completeness(self, paper: dict, conference: str = "icml") -> dict:
        """Check if paper meets target metrics (expansion loop trigger)."""
        targets = CONFERENCE_TARGETS.get(conference.lower(), CONFERENCE_TARGETS["icml"])

        sections = paper.get("paper", {}).get("sections", [])
        total_words = sum(len(s.get("content", "").split()) for s in sections)
        abstract = paper.get("paper", {}).get("abstract", "")
        total_words += len(abstract.split())

        num_refs = len(paper.get("paper", {}).get("references", []))
        word_ratio = total_words / targets["words"] if targets["words"] > 0 else 0
        min_ratio = 0.85

        return {
            "status": "NEEDS_EXPANSION" if word_ratio < min_ratio else "COMPLETE",
            "total_words": total_words,
            "target_words": targets["words"],
            "word_ratio": round(word_ratio, 2),
            "deficit_words": max(0, int(targets["words"] * min_ratio) - total_words),
            "num_references": num_refs,
            "num_sections": len(sections),
        }

    def expand_section(
        self,
        section: dict,
        extra_words_needed: int,
        conference: str = "icml",
    ) -> tuple[Optional[dict], str]:
        """Expand a single section to meet word targets.

        Uses effort="medium" — expansion is additive, less critical than initial draft.
        """
        context = self._load_writing_context()

        task = (
            f"Expand the '{section['name']}' section by ~{extra_words_needed} words.\n\n"
            f"## Current Content\n{section['content'][:2000]}\n\n"
            f"Add more detail: deeper analysis, additional comparisons, "
            f"more technical explanation. Keep all existing content.\n\n"
            f"Respond with:\n"
            f'```json\n{{"section": "{section["name"]}", "content": "EXPANDED LaTeX...", '
            f'"word_count": N}}\n```'
        )

        return self.call_structured(
            task=task,
            extra_context=context,
            max_tokens=4096,
            temperature=0.6,
            effort=EFFORT_MEDIUM,
        )

    def generate_latex(self, paper: dict, conference: str = "icml") -> str:
        """Convert paper dict to LaTeX source."""
        paper_data = paper.get("paper", paper)
        title = paper_data.get("title", "Untitled")
        abstract = paper_data.get("abstract", "")
        sections = paper_data.get("sections", [])
        references = paper_data.get("references", [])

        latex_parts = [
            "\\documentclass{article}",
            "\\usepackage{booktabs,amsmath,amssymb,graphicx,hyperref,algorithm,algorithmic}",
            "\\usepackage{natbib}",
            "",
            f"\\title{{{title}}}",
            "\\author{Anonymous}",
            "\\date{}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\begin{abstract}",
            abstract,
            "\\end{abstract}",
            "",
        ]

        for section in sections:
            content = section.get("content", "")
            if "\\section" in content:
                latex_parts.append(content)
            else:
                name = section.get("name", "").replace("_", " ").title()
                latex_parts.append(f"\\section{{{name}}}")
                latex_parts.append(content)
            latex_parts.append("")

        if references:
            latex_parts.append("\\bibliographystyle{plainnat}")
            latex_parts.append("\\begin{thebibliography}{99}")
            for ref in references:
                key = ref.get("key", "unknown")
                title = ref.get("title", key)
                authors = ref.get("authors", "Unknown")
                year = ref.get("year", "2024")
                venue = ref.get("venue", "")
                latex_parts.append(
                    f"\\bibitem{{{key}}} {authors}. {title}. "
                    f"\\textit{{{venue}}}, {year}."
                )
            latex_parts.append("\\end{thebibliography}")

        latex_parts.append("\\end{document}")
        return "\n".join(latex_parts)

    def save_paper(self, paper: dict, conference: str = "icml") -> Path:
        """Save paper as LaTeX source file."""
        latex = self.generate_latex(paper, conference)
        tex_path = self.paper_dir / "main.tex"
        tex_path.write_text(latex)

        json_path = self.paper_dir / "paper.json"
        json_path.write_text(json.dumps(paper, indent=2, default=str))

        return tex_path

    def compile_pdf(self, tex_path: Optional[Path] = None) -> Optional[Path]:
        """Compile LaTeX to PDF."""
        import subprocess

        if tex_path is None:
            tex_path = self.paper_dir / "main.tex"

        if not tex_path.exists():
            logger.error(f"LaTeX file not found: {tex_path}")
            return None

        try:
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-halt-on-error", "-interaction=nonstopmode",
                     str(tex_path.name)],
                    cwd=str(self.paper_dir),
                    capture_output=True,
                    timeout=60,
                )

            pdf_path = self.paper_dir / "main.pdf"
            if pdf_path.exists():
                logger.info(f"PDF compiled: {pdf_path}")
                return pdf_path
            else:
                logger.error("PDF compilation failed")
                return None
        except Exception as e:
            logger.error(f"PDF compilation error: {e}")
            return None

    async def run(
        self,
        conference: str = "icml",
        max_expansions: int = 2,
    ) -> dict:
        """Full writing pipeline: section-by-section draft -> check -> expand -> compile."""
        self.log_progress(f"Starting paper writing for {conference.upper()}")

        # Step 1: Write section by section (effort="high")
        paper, raw = self.write_full_paper(conference)
        if not paper:
            self.log_progress(f"ERROR: Failed to generate paper.\nRaw: {raw[:500]}")
            return {"success": False, "error": "Writing failed"}

        self.log_progress("Initial draft generated (section by section)")

        # Step 2: Expansion loop (effort="medium" — additive content)
        for i in range(max_expansions):
            completeness = self.check_completeness(paper, conference)
            self.log_progress(
                f"Completeness check {i+1}: {completeness['status']} "
                f"({completeness['total_words']}/{completeness['target_words']} words, "
                f"ratio={completeness['word_ratio']})"
            )

            if completeness["status"] == "COMPLETE":
                break

            sections = paper.get("paper", {}).get("sections", [])
            sections.sort(key=lambda s: s.get("word_count", 0))
            deficit = completeness["deficit_words"]
            expanded_any = False

            for section in sections[:2]:
                words_to_add = deficit // 2
                if words_to_add < 50:
                    break
                result, _ = self.expand_section(section, words_to_add, conference)
                if result and result.get("content"):
                    section["content"] = result["content"]
                    section["word_count"] = result.get("word_count", len(result["content"].split()))
                    expanded_any = True
                    self.log_progress(f"Expanded {section['name']}: +{words_to_add} words target")

            if not expanded_any:
                break

        # Step 3: Save and compile
        tex_path = self.save_paper(paper, conference)
        self.log_progress(f"LaTeX saved to {tex_path}")

        pdf_path = self.compile_pdf(tex_path)
        if pdf_path:
            self.log_progress(f"PDF compiled: {pdf_path}")
        else:
            self.log_progress("PDF compilation failed (may need LaTeX installation)")

        config = load_project_config(self.project_dir)
        config["stage"] = "review"
        save_project_config(self.project_dir, config)

        final_check = self.check_completeness(paper, conference)

        return {
            "success": True,
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "completeness": final_check,
            "paper": paper,
        }
