"""Paper Intermediate Representation - universal format for conference-agnostic paper content."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class Author:
    """Author information."""
    name: str
    affiliation: str = ""
    email: str = ""
    is_corresponding: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "affiliation": self.affiliation,
            "email": self.email,
            "is_corresponding": self.is_corresponding,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Author":
        return cls(
            name=data.get("name", ""),
            affiliation=data.get("affiliation", ""),
            email=data.get("email", ""),
            is_corresponding=data.get("is_corresponding", False),
        )


@dataclass
class FigureRef:
    """Reference to a figure in the paper."""
    label: str
    path: str
    caption: str
    width: str = "0.8\\textwidth"
    position: str = "t"
    subfigures: list["FigureRef"] = field(default_factory=list)
    source_experiment: Optional[str] = None
    verified: bool = False
    
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "path": self.path,
            "caption": self.caption,
            "width": self.width,
            "position": self.position,
            "subfigures": [sf.to_dict() for sf in self.subfigures],
            "source_experiment": self.source_experiment,
            "verified": self.verified,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FigureRef":
        return cls(
            label=data.get("label", ""),
            path=data.get("path", ""),
            caption=data.get("caption", ""),
            width=data.get("width", "0.8\\textwidth"),
            position=data.get("position", "t"),
            subfigures=[cls.from_dict(sf) for sf in data.get("subfigures", [])],
            source_experiment=data.get("source_experiment"),
            verified=data.get("verified", False),
        )


@dataclass
class TableData:
    """Structured table data."""
    label: str
    caption: str
    headers: list[str]
    rows: list[list[str]]
    position: str = "t"
    bold_best: bool = True
    source_experiment: Optional[str] = None
    verified: bool = False
    
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "position": self.position,
            "bold_best": self.bold_best,
            "source_experiment": self.source_experiment,
            "verified": self.verified,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TableData":
        return cls(
            label=data.get("label", ""),
            caption=data.get("caption", ""),
            headers=data.get("headers", []),
            rows=data.get("rows", []),
            position=data.get("position", "t"),
            bold_best=data.get("bold_best", True),
            source_experiment=data.get("source_experiment"),
            verified=data.get("verified", False),
        )


@dataclass
class Algorithm:
    """Algorithm pseudocode."""
    label: str
    caption: str
    steps: list[str]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "caption": self.caption,
            "steps": self.steps,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Algorithm":
        return cls(
            label=data.get("label", ""),
            caption=data.get("caption", ""),
            steps=data.get("steps", []),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
        )


@dataclass
class Equation:
    """Mathematical equation."""
    label: str
    latex: str
    inline: bool = False
    
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "latex": self.latex,
            "inline": self.inline,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Equation":
        return cls(
            label=data.get("label", ""),
            latex=data.get("latex", ""),
            inline=data.get("inline", False),
        )


@dataclass
class Citation:
    """BibTeX citation entry."""
    key: str
    entry_type: str
    title: str
    authors: list[str]
    year: int
    venue: str = ""
    url: str = ""
    doi: str = ""
    extra_fields: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "entry_type": self.entry_type,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "extra_fields": self.extra_fields,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            key=data.get("key", ""),
            entry_type=data.get("entry_type", "article"),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", 2024),
            venue=data.get("venue", ""),
            url=data.get("url", ""),
            doi=data.get("doi", ""),
            extra_fields=data.get("extra_fields", {}),
        )
    
    def to_bibtex(self) -> str:
        """Generate BibTeX entry."""
        authors_str = " and ".join(self.authors)
        lines = [f"@{self.entry_type}{{{self.key},"]
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  author = {{{authors_str}}},")
        lines.append(f"  year = {{{self.year}}},")
        
        if self.venue:
            if self.entry_type == "inproceedings":
                lines.append(f"  booktitle = {{{self.venue}}},")
            else:
                lines.append(f"  journal = {{{self.venue}}},")
        
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        
        for key, value in self.extra_fields.items():
            lines.append(f"  {key} = {{{value}}},")
        
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Section:
    """Paper section with content."""
    name: str
    level: int
    content: str
    figure_refs: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    equation_refs: list[str] = field(default_factory=list)
    algorithm_refs: list[str] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "content": self.content,
            "figure_refs": self.figure_refs,
            "table_refs": self.table_refs,
            "equation_refs": self.equation_refs,
            "algorithm_refs": self.algorithm_refs,
            "subsections": [s.to_dict() for s in self.subsections],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        return cls(
            name=data.get("name", ""),
            level=data.get("level", 1),
            content=data.get("content", ""),
            figure_refs=data.get("figure_refs", []),
            table_refs=data.get("table_refs", []),
            equation_refs=data.get("equation_refs", []),
            algorithm_refs=data.get("algorithm_refs", []),
            subsections=[cls.from_dict(s) for s in data.get("subsections", [])],
        )
    
    def word_count(self) -> int:
        """Count words in this section and subsections."""
        count = len(self.content.split())
        for sub in self.subsections:
            count += sub.word_count()
        return count


@dataclass
class VerifiedClaim:
    """A claim that has been statistically verified."""
    claim_id: str
    statement: str
    hypothesis_id: str
    experiment_id: str
    p_value: float
    effect_size: float
    is_significant: bool
    verified_at: str
    
    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "is_significant": self.is_significant,
            "verified_at": self.verified_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerifiedClaim":
        return cls(
            claim_id=data.get("claim_id", ""),
            statement=data.get("statement", ""),
            hypothesis_id=data.get("hypothesis_id", ""),
            experiment_id=data.get("experiment_id", ""),
            p_value=data.get("p_value", 1.0),
            effect_size=data.get("effect_size", 0.0),
            is_significant=data.get("is_significant", False),
            verified_at=data.get("verified_at", ""),
        )


@dataclass
class PaperMetadata:
    """Paper metadata."""
    title: str
    authors: list[Author]
    abstract: str
    keywords: list[str] = field(default_factory=list)
    method_name: str = "ProposedMethod"
    short_title: str = ""
    github_url: str = ""
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "abstract": self.abstract,
            "keywords": self.keywords,
            "method_name": self.method_name,
            "short_title": self.short_title or self.title[:50],
            "github_url": self.github_url,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PaperMetadata":
        return cls(
            title=data.get("title", ""),
            authors=[Author.from_dict(a) for a in data.get("authors", [])],
            abstract=data.get("abstract", ""),
            keywords=data.get("keywords", []),
            method_name=data.get("method_name", "ProposedMethod"),
            short_title=data.get("short_title", ""),
            github_url=data.get("github_url", ""),
        )


@dataclass
class PaperIR:
    """
    Paper Intermediate Representation.
    
    Universal format that can be rendered to any conference's LaTeX template.
    All content is structured, figures/tables reference verified experiments.
    """
    metadata: PaperMetadata
    sections: list[Section] = field(default_factory=list)
    figures: list[FigureRef] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    algorithms: list[Algorithm] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    verified_claims: list[VerifiedClaim] = field(default_factory=list)
    appendix_sections: list[Section] = field(default_factory=list)
    
    # Metadata for tracking
    created_at: str = ""
    updated_at: str = ""
    target_conference: str = ""
    target_metrics: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "figures": [f.to_dict() for f in self.figures],
            "tables": [t.to_dict() for t in self.tables],
            "algorithms": [a.to_dict() for a in self.algorithms],
            "equations": [e.to_dict() for e in self.equations],
            "citations": [c.to_dict() for c in self.citations],
            "verified_claims": [v.to_dict() for v in self.verified_claims],
            "appendix_sections": [s.to_dict() for s in self.appendix_sections],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target_conference": self.target_conference,
            "target_metrics": self.target_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PaperIR":
        return cls(
            metadata=PaperMetadata.from_dict(data.get("metadata", {})),
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
            figures=[FigureRef.from_dict(f) for f in data.get("figures", [])],
            tables=[TableData.from_dict(t) for t in data.get("tables", [])],
            algorithms=[Algorithm.from_dict(a) for a in data.get("algorithms", [])],
            equations=[Equation.from_dict(e) for e in data.get("equations", [])],
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            verified_claims=[VerifiedClaim.from_dict(v) for v in data.get("verified_claims", [])],
            appendix_sections=[Section.from_dict(s) for s in data.get("appendix_sections", [])],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            target_conference=data.get("target_conference", ""),
            target_metrics=data.get("target_metrics", {}),
        )
    
    def save(self, path: Path) -> None:
        """Save PaperIR to JSON file."""
        self.updated_at = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
    
    @classmethod
    def load(cls, path: Path) -> "PaperIR":
        """Load PaperIR from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)
    
    def total_word_count(self) -> int:
        """Calculate total word count across all sections."""
        count = len(self.metadata.abstract.split())
        for section in self.sections:
            count += section.word_count()
        return count
    
    def figure_count(self) -> int:
        """Count total figures including subfigures."""
        count = len(self.figures)
        for fig in self.figures:
            count += len(fig.subfigures)
        return count
    
    def table_count(self) -> int:
        """Count tables."""
        return len(self.tables)
    
    def get_figure(self, label: str) -> Optional[FigureRef]:
        """Get figure by label."""
        for fig in self.figures:
            if fig.label == label:
                return fig
        return None
    
    def get_table(self, label: str) -> Optional[TableData]:
        """Get table by label."""
        for table in self.tables:
            if table.label == label:
                return table
        return None
    
    def get_section(self, name: str) -> Optional[Section]:
        """Get section by name."""
        for section in self.sections:
            if section.name.lower() == name.lower():
                return section
        return None
    
    def add_section(self, section: Section) -> None:
        """Add or update a section."""
        for i, existing in enumerate(self.sections):
            if existing.name.lower() == section.name.lower():
                self.sections[i] = section
                return
        self.sections.append(section)
    
    def add_figure(self, figure: FigureRef) -> None:
        """Add a figure."""
        for i, existing in enumerate(self.figures):
            if existing.label == figure.label:
                self.figures[i] = figure
                return
        self.figures.append(figure)
    
    def add_table(self, table: TableData) -> None:
        """Add a table."""
        for i, existing in enumerate(self.tables):
            if existing.label == table.label:
                self.tables[i] = table
                return
        self.tables.append(table)
    
    def add_verified_claim(self, claim: VerifiedClaim) -> None:
        """Add a verified claim."""
        self.verified_claims.append(claim)
    
    def generate_bibliography(self) -> str:
        """Generate BibTeX file content."""
        entries = [c.to_bibtex() for c in self.citations]
        return "\n\n".join(entries)
    
    def check_completeness(self) -> dict:
        """Check if paper meets target metrics."""
        target_words = self.target_metrics.get("word_count", 5000)
        target_figures = self.target_metrics.get("figure_count", 6)
        target_tables = self.target_metrics.get("table_count", 3)
        
        current_words = self.total_word_count()
        current_figures = self.figure_count()
        current_tables = self.table_count()
        
        word_ratio = current_words / max(1, target_words)
        
        issues = []
        if word_ratio < 0.85:
            issues.append(f"Word count too low: {current_words}/{target_words} ({word_ratio:.0%})")
        if current_figures < target_figures:
            issues.append(f"Need more figures: {current_figures}/{target_figures}")
        if current_tables < target_tables:
            issues.append(f"Need more tables: {current_tables}/{target_tables}")
        
        # Check for unverified figures/tables
        unverified_figs = [f.label for f in self.figures if not f.verified]
        unverified_tables = [t.label for t in self.tables if not t.verified]
        
        if unverified_figs:
            issues.append(f"Unverified figures: {unverified_figs}")
        if unverified_tables:
            issues.append(f"Unverified tables: {unverified_tables}")
        
        return {
            "sufficient": len(issues) == 0,
            "current_words": current_words,
            "target_words": target_words,
            "word_ratio": round(word_ratio, 2),
            "current_figures": current_figures,
            "target_figures": target_figures,
            "current_tables": current_tables,
            "target_tables": target_tables,
            "issues": issues,
        }


def create_empty_paper(title: str, abstract: str = "") -> PaperIR:
    """Create an empty paper with default structure."""
    metadata = PaperMetadata(
        title=title,
        authors=[Author(name="Anonymous Author", affiliation="Anonymous Institution")],
        abstract=abstract,
    )
    
    sections = [
        Section(name="Introduction", level=1, content=""),
        Section(name="Related Work", level=1, content=""),
        Section(name="Method", level=1, content=""),
        Section(name="Experiments", level=1, content=""),
        Section(name="Conclusion", level=1, content=""),
    ]
    
    return PaperIR(
        metadata=metadata,
        sections=sections,
    )
