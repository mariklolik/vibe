"""Conference-specific style guidelines and requirements."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConferenceStyle:
    """Style guidelines for a specific conference."""
    name: str
    full_name: str
    
    # Page layout
    page_limit: int
    columns: int
    font_size: int
    
    # Abstract
    abstract_word_limit: int
    abstract_style: str  # "block" or "indented"
    
    # Sections
    required_sections: list[str]
    optional_sections: list[str]
    section_numbering: bool
    
    # Figures
    figure_position: str  # "top", "bottom", "inline"
    caption_position: str  # "below", "above"
    max_figures: int
    
    # Citations
    citation_style: str  # "numeric", "author-year"
    bibliography_style: str  # "unsrtnat", "plainnat", etc.
    
    # Writing guidelines
    first_person_allowed: bool
    passive_preferred: bool
    anonymous_submission: bool
    
    # LaTeX specifics
    document_class: str
    style_package: str
    required_packages: list[str]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "page_limit": self.page_limit,
            "columns": self.columns,
            "font_size": self.font_size,
            "abstract_word_limit": self.abstract_word_limit,
            "required_sections": self.required_sections,
            "optional_sections": self.optional_sections,
            "citation_style": self.citation_style,
            "first_person_allowed": self.first_person_allowed,
            "anonymous_submission": self.anonymous_submission,
            "document_class": self.document_class,
            "style_package": self.style_package,
        }


CONFERENCE_STYLES = {
    "neurips": ConferenceStyle(
        name="neurips",
        full_name="Conference on Neural Information Processing Systems",
        page_limit=9,
        columns=1,
        font_size=10,
        abstract_word_limit=250,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Discussion", "Broader Impact", "Limitations"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=8,
        citation_style="numeric",
        bibliography_style="unsrtnat",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="neurips_2024",
        required_packages=["inputenc", "fontenc", "hyperref", "url", "booktabs", "amsfonts", "nicefrac", "microtype", "graphicx", "xcolor"],
    ),
    "icml": ConferenceStyle(
        name="icml",
        full_name="International Conference on Machine Learning",
        page_limit=9,
        columns=2,
        font_size=10,
        abstract_word_limit=200,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Preliminaries", "Discussion"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=8,
        citation_style="author-year",
        bibliography_style="icml2024",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="icml2024",
        style_package="icml2024",
        required_packages=["amsmath", "amssymb", "mathtools", "amsthm", "cleveref", "booktabs", "graphicx", "hyperref"],
    ),
    "iclr": ConferenceStyle(
        name="iclr",
        full_name="International Conference on Learning Representations",
        page_limit=9,
        columns=1,
        font_size=10,
        abstract_word_limit=250,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Discussion", "Reproducibility Statement"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=8,
        citation_style="author-year",
        bibliography_style="iclr2024_conference",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="iclr2024_conference",
        required_packages=["inputenc", "fontenc", "hyperref", "url", "booktabs", "amsfonts", "nicefrac", "microtype", "graphicx"],
    ),
    "cvpr": ConferenceStyle(
        name="cvpr",
        full_name="IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        page_limit=8,
        columns=2,
        font_size=10,
        abstract_word_limit=300,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Preliminaries", "Discussion", "Ablation Study"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=10,
        citation_style="numeric",
        bibliography_style="ieee_fullname",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="cvpr",
        required_packages=["times", "epsfig", "graphicx", "amsmath", "amssymb", "booktabs", "hyperref"],
    ),
    "acl": ConferenceStyle(
        name="acl",
        full_name="Annual Meeting of the Association for Computational Linguistics",
        page_limit=8,
        columns=2,
        font_size=11,
        abstract_word_limit=200,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Analysis", "Limitations", "Ethics Statement"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=6,
        citation_style="author-year",
        bibliography_style="acl_natbib",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="acl2024",
        required_packages=["times", "latexsym", "graphicx", "booktabs", "amsmath", "amssymb"],
    ),
    "emnlp": ConferenceStyle(
        name="emnlp",
        full_name="Conference on Empirical Methods in Natural Language Processing",
        page_limit=9,
        columns=2,
        font_size=11,
        abstract_word_limit=200,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Analysis", "Limitations"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=6,
        citation_style="author-year",
        bibliography_style="acl_natbib",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="emnlp2024",
        required_packages=["times", "latexsym", "graphicx", "booktabs", "amsmath", "amssymb"],
    ),
    "aaai": ConferenceStyle(
        name="aaai",
        full_name="AAAI Conference on Artificial Intelligence",
        page_limit=7,
        columns=2,
        font_size=10,
        abstract_word_limit=150,
        abstract_style="block",
        required_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        optional_sections=["Background", "Discussion"],
        section_numbering=True,
        figure_position="top",
        caption_position="below",
        max_figures=6,
        citation_style="author-year",
        bibliography_style="aaai24",
        first_person_allowed=True,
        passive_preferred=False,
        anonymous_submission=True,
        document_class="article",
        style_package="aaai24",
        required_packages=["times", "helvet", "courier", "graphicx", "amsmath", "amssymb"],
    ),
}


def get_conference_style(conference: str) -> Optional[ConferenceStyle]:
    """Get style guidelines for a conference."""
    return CONFERENCE_STYLES.get(conference.lower())


def list_conference_styles() -> list[str]:
    """List all available conference styles."""
    return list(CONFERENCE_STYLES.keys())


def get_section_guidelines(conference: str, section: str) -> dict:
    """Get specific guidelines for a section in a conference."""
    style = get_conference_style(conference)
    if not style:
        style = CONFERENCE_STYLES["neurips"]
    
    # Section-specific word count targets based on page limits
    section_lower = section.lower()
    total_words = style.page_limit * 550  # Approx words per page
    
    section_ratios = {
        "abstract": 0.04,
        "introduction": 0.15,
        "related work": 0.12,
        "background": 0.08,
        "method": 0.25,
        "methodology": 0.25,
        "approach": 0.25,
        "experiments": 0.30,
        "results": 0.30,
        "evaluation": 0.30,
        "discussion": 0.08,
        "analysis": 0.10,
        "conclusion": 0.06,
        "limitations": 0.04,
    }
    
    ratio = section_ratios.get(section_lower, 0.10)
    target_words = int(total_words * ratio)
    
    # Section-specific content guidelines
    content_focus = {
        "introduction": {
            "paragraphs": ["motivation", "problem_statement", "limitations_of_prior_work", "our_approach", "contributions", "paper_organization"],
            "must_include": ["clear problem statement", "main contribution", "paper outline"],
            "citations_expected": 8,
        },
        "related work": {
            "paragraphs": ["area_1", "area_2", "area_3", "comparison_to_ours"],
            "must_include": ["comprehensive coverage", "comparison to our approach", "gap identification"],
            "citations_expected": 20,
        },
        "method": {
            "paragraphs": ["overview", "formulation", "component_1", "component_2", "implementation"],
            "must_include": ["clear notation", "algorithm or equations", "intuition"],
            "citations_expected": 5,
        },
        "experiments": {
            "paragraphs": ["setup", "datasets", "baselines", "main_results", "ablations", "analysis"],
            "must_include": ["experimental setup", "dataset description", "baseline comparison", "ablation study"],
            "citations_expected": 10,
        },
        "conclusion": {
            "paragraphs": ["summary", "limitations", "future_work"],
            "must_include": ["summary of contributions", "key findings"],
            "citations_expected": 2,
        },
    }
    
    guidelines = content_focus.get(section_lower, {
        "paragraphs": ["main_content"],
        "must_include": [],
        "citations_expected": 5,
    })
    
    return {
        "section": section,
        "conference": conference,
        "target_words": target_words,
        "target_paragraphs": len(guidelines["paragraphs"]),
        "paragraph_structure": guidelines["paragraphs"],
        "must_include": guidelines["must_include"],
        "expected_citations": guidelines["citations_expected"],
        "first_person": style.first_person_allowed,
        "has_equations": section_lower in ["method", "methodology", "approach", "background"],
        "has_figures": section_lower in ["method", "experiments", "results", "introduction"],
        "has_tables": section_lower in ["experiments", "results", "ablation"],
    }
