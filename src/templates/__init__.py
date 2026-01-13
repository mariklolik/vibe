"""LaTeX template management for conference papers."""

from pathlib import Path
from typing import Optional

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def get_template_path(conference: str) -> Optional[Path]:
    """Get path to conference template directory."""
    conf_lower = conference.lower()
    
    # Map conference names to template directories
    template_map = {
        "neurips": TEMPLATES_DIR / "Styles",
        "icml": TEMPLATES_DIR / "icml",
        "iclr": TEMPLATES_DIR / "iclr",
        "cvpr": TEMPLATES_DIR / "cvpr",
        "acl": TEMPLATES_DIR / "acl",
    }
    
    path = template_map.get(conf_lower)
    if path and path.exists():
        return path
    
    return TEMPLATES_DIR / "Styles"  # Default to NeurIPS


def get_style_file(conference: str) -> Optional[Path]:
    """Get the .sty file for a conference."""
    template_path = get_template_path(conference)
    if not template_path:
        return None
    
    # Look for .sty files
    sty_files = list(template_path.glob("*.sty"))
    if sty_files:
        return sty_files[0]
    
    return None


def list_available_templates() -> list[str]:
    """List all available conference templates."""
    if not TEMPLATES_DIR.exists():
        return []
    
    templates = []
    for path in TEMPLATES_DIR.iterdir():
        if path.is_dir() and any(path.glob("*.sty")):
            templates.append(path.name)
    
    return templates
