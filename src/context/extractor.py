"""Paper structure and style extraction from PDF/HTML sources."""

import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import aiohttp


@dataclass
class SectionInfo:
    """Information about a paper section."""
    name: str
    level: int  # 1=section, 2=subsection, 3=subsubsection
    word_count: int
    paragraph_count: int
    has_equations: bool
    has_figures: bool
    has_tables: bool
    has_algorithms: bool
    citation_count: int
    subsections: list["SectionInfo"] = field(default_factory=list)


@dataclass
class FigureInfo:
    """Information about a figure or table."""
    fig_type: str  # "figure", "table", "algorithm"
    caption_length: int
    position: str  # "top", "bottom", "inline"
    width: str  # "full", "half", "column"
    section: str


@dataclass
class CitationPattern:
    """Citation usage patterns."""
    style: str  # "numeric", "author-year", "footnote"
    avg_per_paragraph: float
    inline_ratio: float  # ratio of inline vs parenthetical
    total_citations: int
    self_citations: int


@dataclass
class WritingStyle:
    """Writing style metrics."""
    avg_sentence_length: float
    passive_voice_ratio: float
    technical_term_density: float
    first_person_usage: bool  # "we" vs impersonal
    avg_paragraph_length: float
    formality_score: float  # 0-1, higher = more formal


@dataclass
class PaperStructure:
    """Complete paper structure analysis."""
    title: str
    abstract_word_count: int
    total_word_count: int
    total_pages: int
    sections: list[SectionInfo]
    figures: list[FigureInfo]
    citation_pattern: CitationPattern
    writing_style: WritingStyle
    conference: Optional[str]
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract_word_count": self.abstract_word_count,
            "total_word_count": self.total_word_count,
            "total_pages": self.total_pages,
            "sections": [self._section_to_dict(s) for s in self.sections],
            "figures": [
                {
                    "type": f.fig_type,
                    "caption_length": f.caption_length,
                    "position": f.position,
                    "width": f.width,
                    "section": f.section,
                }
                for f in self.figures
            ],
            "citation_pattern": {
                "style": self.citation_pattern.style,
                "avg_per_paragraph": self.citation_pattern.avg_per_paragraph,
                "inline_ratio": self.citation_pattern.inline_ratio,
                "total_citations": self.citation_pattern.total_citations,
            },
            "writing_style": {
                "avg_sentence_length": self.writing_style.avg_sentence_length,
                "passive_voice_ratio": self.writing_style.passive_voice_ratio,
                "first_person": self.writing_style.first_person_usage,
                "avg_paragraph_length": self.writing_style.avg_paragraph_length,
                "formality_score": self.writing_style.formality_score,
            },
            "conference": self.conference,
        }
    
    def _section_to_dict(self, section: SectionInfo) -> dict:
        return {
            "name": section.name,
            "level": section.level,
            "word_count": section.word_count,
            "paragraph_count": section.paragraph_count,
            "has_equations": section.has_equations,
            "has_figures": section.has_figures,
            "has_tables": section.has_tables,
            "citation_count": section.citation_count,
            "subsections": [self._section_to_dict(s) for s in section.subsections],
        }


class PaperExtractor:
    """Extract structure and style from papers."""
    
    SECTION_PATTERNS = [
        (r"^#+\s*(\d+\.?\s*)?(.+)$", "markdown"),
        (r"^(\d+\.?\s+)?([A-Z][^.]+)$", "heading"),
        (r"\\section\{([^}]+)\}", "latex_section"),
        (r"\\subsection\{([^}]+)\}", "latex_subsection"),
    ]
    
    STANDARD_SECTIONS = [
        "Abstract", "Introduction", "Related Work", "Background",
        "Method", "Methodology", "Approach", "Model",
        "Experiments", "Evaluation", "Results",
        "Discussion", "Analysis",
        "Conclusion", "Conclusions", "Future Work",
        "Acknowledgments", "References", "Appendix",
    ]
    
    async def extract_from_arxiv_html(self, arxiv_id: str) -> Optional[PaperStructure]:
        """Extract structure from arXiv HTML version."""
        url = f"https://arxiv.org/html/{arxiv_id}v1"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        return None
                    html = await response.text()
        except Exception:
            return None
        
        return self._parse_html(html, arxiv_id)
    
    def _parse_html(self, html: str, paper_id: str) -> PaperStructure:
        """Parse HTML content to extract structure."""
        title = self._extract_title(html)
        abstract = self._extract_abstract(html)
        sections = self._extract_sections(html)
        figures = self._extract_figures(html)
        citations = self._analyze_citations(html)
        style = self._analyze_writing_style(html)
        
        total_words = sum(s.word_count for s in sections)
        total_pages = max(1, total_words // 500)  # Estimate
        
        return PaperStructure(
            title=title,
            abstract_word_count=len(abstract.split()),
            total_word_count=total_words,
            total_pages=total_pages,
            sections=sections,
            figures=figures,
            citation_pattern=citations,
            writing_style=style,
            conference=self._detect_conference(html),
        )
    
    def _extract_title(self, html: str) -> str:
        """Extract paper title from HTML."""
        title_match = re.search(r"<h1[^>]*class=\"ltx_title[^\"]*\"[^>]*>([^<]+)</h1>", html)
        if title_match:
            return title_match.group(1).strip()
        
        title_match = re.search(r"<title>([^<]+)</title>", html)
        if title_match:
            return title_match.group(1).split(" - ")[0].strip()
        
        return "Unknown Title"
    
    def _extract_abstract(self, html: str) -> str:
        """Extract abstract text."""
        abstract_match = re.search(
            r"<div[^>]*class=\"ltx_abstract\"[^>]*>(.*?)</div>",
            html, re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            text = re.sub(r"<[^>]+>", " ", abstract_match.group(1))
            return " ".join(text.split())
        
        abstract_match = re.search(
            r"Abstract[:\s]*(.*?)(?:Introduction|1\.|$)",
            html, re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            text = re.sub(r"<[^>]+>", " ", abstract_match.group(1))
            return " ".join(text.split())[:1500]
        
        return ""
    
    def _extract_sections(self, html: str) -> list[SectionInfo]:
        """Extract section structure."""
        sections = []
        
        section_pattern = re.compile(
            r"<(h[1-6]|section)[^>]*(?:class=\"[^\"]*ltx_(?:section|subsection)[^\"]*\")?[^>]*>"
            r"(?:<[^>]+>)*(\d+\.?\d*\.?\d*\.?\s*)?([A-Za-z][^<]{2,60})",
            re.IGNORECASE
        )
        
        content_blocks = re.split(r"<h[1-6][^>]*>", html)
        
        current_level = 0
        for i, match in enumerate(section_pattern.finditer(html)):
            tag = match.group(1).lower()
            section_num = match.group(2) or ""
            section_name = match.group(3).strip()
            
            section_name = re.sub(r"<[^>]+>", "", section_name)
            
            if any(std.lower() in section_name.lower() for std in self.STANDARD_SECTIONS):
                level = 1
                if tag in ["h3", "h4"] or "subsection" in match.group(0).lower():
                    level = 2
                elif tag in ["h5", "h6"]:
                    level = 3
                
                content_start = match.end()
                next_section = section_pattern.search(html, content_start)
                content_end = next_section.start() if next_section else len(html)
                content = html[content_start:content_end]
                
                text_content = re.sub(r"<[^>]+>", " ", content)
                words = text_content.split()
                paragraphs = len(re.findall(r"<p[^>]*>", content))
                
                section = SectionInfo(
                    name=section_name,
                    level=level,
                    word_count=len(words),
                    paragraph_count=max(1, paragraphs),
                    has_equations="\\(" in content or "equation" in content.lower(),
                    has_figures="<figure" in content.lower() or "<img" in content.lower(),
                    has_tables="<table" in content.lower(),
                    has_algorithms="algorithm" in content.lower(),
                    citation_count=len(re.findall(r"\[[\d,\s]+\]|\(\w+\s+et\s+al\.", content)),
                )
                sections.append(section)
        
        if not sections:
            sections = self._create_default_sections()
        
        return sections
    
    def _create_default_sections(self) -> list[SectionInfo]:
        """Create default section structure for ML papers."""
        return [
            SectionInfo("Introduction", 1, 800, 4, False, False, False, False, 15),
            SectionInfo("Related Work", 1, 600, 3, False, False, False, False, 25),
            SectionInfo("Method", 1, 1200, 6, True, True, False, True, 10),
            SectionInfo("Experiments", 1, 1500, 8, True, True, True, False, 20),
            SectionInfo("Conclusion", 1, 300, 2, False, False, False, False, 5),
        ]
    
    def _extract_figures(self, html: str) -> list[FigureInfo]:
        """Extract figure and table information."""
        figures = []
        
        fig_pattern = re.compile(
            r"<figure[^>]*>(.*?)</figure>",
            re.DOTALL | re.IGNORECASE
        )
        
        for match in fig_pattern.finditer(html):
            content = match.group(1)
            caption_match = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", content, re.DOTALL)
            caption = ""
            if caption_match:
                caption = re.sub(r"<[^>]+>", "", caption_match.group(1))
            
            fig_type = "figure"
            if "<table" in content.lower():
                fig_type = "table"
            
            figures.append(FigureInfo(
                fig_type=fig_type,
                caption_length=len(caption.split()),
                position="top",
                width="full" if "width" in content and "100%" in content else "column",
                section="unknown",
            ))
        
        table_pattern = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
        for match in table_pattern.finditer(html):
            if not any(f.fig_type == "table" for f in figures):
                figures.append(FigureInfo(
                    fig_type="table",
                    caption_length=20,
                    position="top",
                    width="column",
                    section="experiments",
                ))
        
        return figures
    
    def _analyze_citations(self, html: str) -> CitationPattern:
        """Analyze citation patterns."""
        numeric_cites = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", html)
        author_cites = re.findall(r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?[,;]?\s*\d{4})\)", html)
        
        total = len(numeric_cites) + len(author_cites)
        
        style = "numeric" if len(numeric_cites) > len(author_cites) else "author-year"
        
        paragraphs = len(re.findall(r"<p[^>]*>", html))
        avg_per_para = total / max(1, paragraphs)
        
        return CitationPattern(
            style=style,
            avg_per_paragraph=round(avg_per_para, 2),
            inline_ratio=0.3,
            total_citations=total,
            self_citations=0,
        )
    
    def _analyze_writing_style(self, html: str) -> WritingStyle:
        """Analyze writing style metrics."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        passive_patterns = [
            r"\b(is|are|was|were|been|being)\s+\w+ed\b",
            r"\b(is|are|was|were)\s+\w+en\b",
        ]
        passive_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in passive_patterns)
        passive_ratio = passive_count / max(1, len(sentences))
        
        we_count = len(re.findall(r"\bwe\b", text, re.IGNORECASE))
        first_person = we_count > 5
        
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
        avg_para_len = sum(len(re.sub(r"<[^>]+>", "", p).split()) for p in paragraphs) / max(1, len(paragraphs))
        
        return WritingStyle(
            avg_sentence_length=round(avg_sentence_len, 1),
            passive_voice_ratio=round(min(1.0, passive_ratio), 2),
            technical_term_density=0.15,
            first_person_usage=first_person,
            avg_paragraph_length=round(avg_para_len, 1),
            formality_score=0.85,
        )
    
    def _detect_conference(self, html: str) -> Optional[str]:
        """Detect which conference the paper was submitted to."""
        conferences = {
            "neurips": ["neurips", "nips", "neural information processing"],
            "icml": ["icml", "international conference on machine learning"],
            "iclr": ["iclr", "international conference on learning representations"],
            "cvpr": ["cvpr", "computer vision and pattern recognition"],
            "acl": ["acl ", "association for computational linguistics"],
            "emnlp": ["emnlp", "empirical methods in natural language"],
            "aaai": ["aaai", "association for the advancement of artificial intelligence"],
        }
        
        html_lower = html.lower()
        for conf, patterns in conferences.items():
            if any(p in html_lower for p in patterns):
                return conf
        
        return None


async def extract_paper_context(arxiv_id: str) -> Optional[dict]:
    """Extract context from an arXiv paper.
    
    Args:
        arxiv_id: The arXiv ID (e.g., "2502.14678")
    
    Returns:
        Dictionary with paper structure and style analysis
    """
    extractor = PaperExtractor()
    structure = await extractor.extract_from_arxiv_html(arxiv_id)
    
    if structure:
        return structure.to_dict()
    
    return None
