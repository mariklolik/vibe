"""Paper intermediate representation and rendering."""

from src.paper.ir import (
    PaperIR,
    PaperMetadata,
    Section,
    FigureRef,
    TableData,
    Algorithm,
    Equation,
    Citation,
    Author,
    VerifiedClaim,
    create_empty_paper,
)

from src.paper.renderer import (
    PaperRenderer,
    renderer,
    CONFERENCE_CONFIGS,
    render_section,
    render_figure,
    render_table,
    render_algorithm,
)

__all__ = [
    "PaperIR",
    "PaperMetadata",
    "Section",
    "FigureRef",
    "TableData",
    "Algorithm",
    "Equation",
    "Citation",
    "Author",
    "VerifiedClaim",
    "create_empty_paper",
    "PaperRenderer",
    "renderer",
    "CONFERENCE_CONFIGS",
    "render_section",
    "render_figure",
    "render_table",
    "render_algorithm",
]
