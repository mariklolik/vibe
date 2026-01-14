"""Conference-specific LaTeX renderer using Jinja2 templates."""

import re
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.paper.ir import PaperIR, Section, FigureRef, TableData, Algorithm


TEMPLATES_DIR = Path(__file__).parent / "templates"


CONFERENCE_CONFIGS = {
    "neurips": {
        "template": "neurips.tex.j2",
        "style_package": "neurips_2024",
        "columns": 1,
        "font_size": 10,
        "page_limit": 9,
        "abstract_limit": 200,
        "anonymous": True,
    },
    "icml": {
        "template": "icml.tex.j2",
        "style_package": "icml2024",
        "columns": 2,
        "font_size": 10,
        "page_limit": 9,
        "abstract_limit": 150,
        "anonymous": True,
    },
    "iclr": {
        "template": "neurips.tex.j2",
        "style_package": "iclr2025_conference",
        "columns": 1,
        "font_size": 10,
        "page_limit": 9,
        "abstract_limit": 200,
        "anonymous": True,
    },
    "acl": {
        "template": "acl.tex.j2",
        "style_package": "acl2024",
        "columns": 2,
        "font_size": 11,
        "page_limit": 8,
        "abstract_limit": 150,
        "anonymous": True,
        "requires_limitations": True,
        "requires_ethics": True,
    },
    "emnlp": {
        "template": "acl.tex.j2",
        "style_package": "emnlp2024",
        "columns": 2,
        "font_size": 11,
        "page_limit": 8,
        "abstract_limit": 150,
        "anonymous": True,
        "requires_limitations": True,
        "requires_ethics": True,
    },
    "cvpr": {
        "template": "cvpr.tex.j2",
        "style_package": "cvpr",
        "columns": 2,
        "font_size": 10,
        "page_limit": 8,
        "abstract_limit": 200,
        "anonymous": True,
    },
    "iccv": {
        "template": "cvpr.tex.j2",
        "style_package": "iccv",
        "columns": 2,
        "font_size": 10,
        "page_limit": 8,
        "abstract_limit": 200,
        "anonymous": True,
    },
    "aaai": {
        "template": "aaai.tex.j2",
        "style_package": "aaai24",
        "columns": 2,
        "font_size": 10,
        "page_limit": 7,
        "abstract_limit": 150,
        "anonymous": False,
    },
}


def markdown_to_latex(text: str) -> str:
    """Convert markdown-style content to LaTeX."""
    if not text:
        return ""
    
    result = text
    
    result = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', result)
    result = re.sub(r'\*(.+?)\*', r'\\textit{\1}', result)
    result = re.sub(r'`(.+?)`', r'\\texttt{\1}', result)
    
    result = re.sub(r'\$\$(.+?)\$\$', r'\\[\1\\]', result, flags=re.DOTALL)
    
    result = result.replace('&', '\\&')
    result = result.replace('%', '\\%')
    result = result.replace('#', '\\#')
    result = result.replace('_', '\\_')
    
    result = result.replace('\\textbf{', '\\textbf{')
    result = result.replace('\\textit{', '\\textit{')
    result = result.replace('\\texttt{', '\\texttt{')
    result = result.replace('\\[', '\\[')
    result = result.replace('\\]', '\\]')
    result = result.replace('\\ref{', '\\ref{')
    result = result.replace('\\cite{', '\\cite{')
    
    return result


def render_section(section: Section, level: int = 1) -> str:
    """Render a section to LaTeX."""
    commands = {1: "section", 2: "subsection", 3: "subsubsection"}
    cmd = commands.get(level, "paragraph")
    
    content = markdown_to_latex(section.content)
    
    latex = f"\\{cmd}{{{section.name}}}\n"
    if section.content.strip():
        latex += f"{content}\n"
    
    for subsection in section.subsections:
        latex += "\n" + render_section(subsection, level + 1)
    
    return latex


def render_figure(figure: FigureRef) -> str:
    """Render a figure to LaTeX."""
    if figure.subfigures:
        subfigs = []
        width = f"{0.9 / len(figure.subfigures):.2f}\\textwidth"
        for sf in figure.subfigures:
            subfigs.append(f"""\\begin{{subfigure}}{{{width}}}
    \\centering
    \\includegraphics[width=\\textwidth]{{{sf.path}}}
    \\caption{{{sf.caption}}}
    \\label{{{sf.label}}}
\\end{{subfigure}}""")
        
        return f"""\\begin{{figure}}[{figure.position}]
    \\centering
    {chr(10).join(subfigs)}
    \\caption{{{figure.caption}}}
    \\label{{{figure.label}}}
\\end{{figure}}"""
    
    return f"""\\begin{{figure}}[{figure.position}]
    \\centering
    \\includegraphics[width={figure.width}]{{{figure.path}}}
    \\caption{{{figure.caption}}}
    \\label{{{figure.label}}}
\\end{{figure}}"""


def render_table(table: TableData) -> str:
    """Render a table to LaTeX."""
    cols = "l" + "c" * (len(table.headers) - 1) if table.headers else "c"
    
    header_row = " & ".join(f"\\textbf{{{h}}}" for h in table.headers)
    
    rows_latex = []
    for row in table.rows:
        row_cells = []
        for i, cell in enumerate(row):
            if table.bold_best and i > 0:
                try:
                    val = float(cell.replace("\\textbf{", "").replace("}", ""))
                    col_vals = []
                    for r in table.rows:
                        try:
                            col_vals.append(float(r[i].replace("\\textbf{", "").replace("}", "")))
                        except (ValueError, IndexError):
                            pass
                    if col_vals and val == max(col_vals):
                        cell = f"\\textbf{{{cell}}}"
                except (ValueError, IndexError):
                    pass
            row_cells.append(cell)
        rows_latex.append(" & ".join(row_cells))
    
    rows_str = " \\\\\n        ".join(rows_latex)
    
    return f"""\\begin{{table}}[{table.position}]
    \\centering
    \\caption{{{table.caption}}}
    \\label{{{table.label}}}
    \\begin{{tabular}}{{{cols}}}
        \\toprule
        {header_row} \\\\
        \\midrule
        {rows_str} \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}"""


def render_algorithm(algorithm: Algorithm) -> str:
    """Render an algorithm to LaTeX."""
    steps_latex = []
    for step in algorithm.steps:
        if step.startswith("for ") or step.startswith("while "):
            steps_latex.append(f"\\FOR{{{step[4:]}}}")
        elif step.startswith("if "):
            steps_latex.append(f"\\IF{{{step[3:]}}}")
        elif step.startswith("else"):
            steps_latex.append("\\ELSE")
        elif step.startswith("end"):
            if "for" in step.lower():
                steps_latex.append("\\ENDFOR")
            elif "if" in step.lower():
                steps_latex.append("\\ENDIF")
            else:
                steps_latex.append("\\ENDWHILE")
        elif step.startswith("return "):
            steps_latex.append(f"\\RETURN {step[7:]}")
        else:
            steps_latex.append(f"\\STATE {step}")
    
    steps_str = "\n        ".join(steps_latex)
    
    inputs_str = ""
    if algorithm.inputs:
        inputs_str = f"\\REQUIRE {', '.join(algorithm.inputs)}\n        "
    
    outputs_str = ""
    if algorithm.outputs:
        outputs_str = f"\\ENSURE {', '.join(algorithm.outputs)}\n        "
    
    return f"""\\begin{{algorithm}}[t]
    \\caption{{{algorithm.caption}}}
    \\label{{{algorithm.label}}}
    \\begin{{algorithmic}}[1]
        {inputs_str}{outputs_str}{steps_str}
    \\end{{algorithmic}}
\\end{{algorithm}}"""


class PaperRenderer:
    """Renders PaperIR to conference-specific LaTeX."""
    
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        self.env.filters["markdown_to_latex"] = markdown_to_latex
        self.env.filters["render_section"] = render_section
        self.env.filters["render_figure"] = render_figure
        self.env.filters["render_table"] = render_table
        self.env.filters["render_algorithm"] = render_algorithm
    
    def render(
        self,
        paper: PaperIR,
        conference: str,
        anonymous: Optional[bool] = None,
    ) -> str:
        """Render paper to LaTeX for a specific conference."""
        conf_lower = conference.lower()
        
        if conf_lower not in CONFERENCE_CONFIGS:
            conf_lower = "neurips"
        
        config = CONFERENCE_CONFIGS[conf_lower]
        
        if anonymous is None:
            anonymous = config.get("anonymous", True)
        
        template_name = config["template"]
        
        try:
            template = self.env.get_template(template_name)
        except Exception:
            return self._render_fallback(paper, config, anonymous)
        
        sections_latex = []
        for section in paper.sections:
            sections_latex.append(render_section(section))
        
        figures_latex = [render_figure(f) for f in paper.figures]
        tables_latex = [render_table(t) for t in paper.tables]
        algorithms_latex = [render_algorithm(a) for a in paper.algorithms]
        
        appendix_latex = []
        for section in paper.appendix_sections:
            appendix_latex.append(render_section(section))
        
        context = {
            "paper": paper,
            "config": config,
            "anonymous": anonymous,
            "sections_latex": "\n\n".join(sections_latex),
            "figures_latex": "\n\n".join(figures_latex),
            "tables_latex": "\n\n".join(tables_latex),
            "algorithms_latex": "\n\n".join(algorithms_latex),
            "appendix_latex": "\n\n".join(appendix_latex),
            "bibliography": paper.generate_bibliography(),
        }
        
        return template.render(**context)
    
    def _render_fallback(
        self,
        paper: PaperIR,
        config: dict,
        anonymous: bool,
    ) -> str:
        """Fallback rendering without templates."""
        style_pkg = config.get("style_package", "article")
        
        authors_tex = ""
        if not anonymous:
            for i, author in enumerate(paper.metadata.authors):
                if i > 0:
                    authors_tex += " \\And\n"
                authors_tex += f"  {author.name}"
                if author.affiliation:
                    authors_tex += f" \\\\ {author.affiliation}"
                if author.email:
                    authors_tex += f" \\\\ \\texttt{{{author.email}}}"
        else:
            authors_tex = "Anonymous Author(s)"
        
        sections_latex = []
        for section in paper.sections:
            sections_latex.append(render_section(section))
        
        figures_latex = [render_figure(f) for f in paper.figures]
        tables_latex = [render_table(t) for t in paper.tables]
        
        content_parts = []
        for i, section_tex in enumerate(sections_latex):
            content_parts.append(section_tex)
            if i < len(figures_latex):
                content_parts.append(figures_latex[i])
            if i < len(tables_latex):
                content_parts.append(tables_latex[i])
        
        content = "\n\n".join(content_parts)
        
        appendix = ""
        if paper.appendix_sections:
            appendix = "\\appendix\n"
            for section in paper.appendix_sections:
                appendix += render_section(section) + "\n"
        
        return f"""\\documentclass{{article}}

\\usepackage[preprint]{{{style_pkg}}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{hyperref}}
\\usepackage{{url}}
\\usepackage{{booktabs}}
\\usepackage{{amsfonts}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{nicefrac}}
\\usepackage{{microtype}}
\\usepackage{{graphicx}}
\\usepackage{{xcolor}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{subcaption}}
\\usepackage{{multirow}}

\\newcommand{{\\method}}{{{paper.metadata.method_name}}}
\\newcommand{{\\ie}}{{\\textit{{i.e.}}}}
\\newcommand{{\\eg}}{{\\textit{{e.g.}}}}
\\newcommand{{\\etal}}{{\\textit{{et al.}}}}

\\title{{{paper.metadata.title}}}

\\author{{
{authors_tex}
}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{paper.metadata.abstract}
\\end{{abstract}}

{content}

\\section*{{Acknowledgments}}
We thank the anonymous reviewers for their helpful feedback.

\\bibliographystyle{{unsrtnat}}
\\bibliography{{references}}

{appendix}

\\end{{document}}
"""
    
    def save_to_directory(
        self,
        paper: PaperIR,
        conference: str,
        output_dir: Path,
        anonymous: Optional[bool] = None,
    ) -> dict:
        """Render and save paper to a directory with all assets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        latex = self.render(paper, conference, anonymous)
        tex_file = output_dir / f"paper_{conference}.tex"
        tex_file.write_text(latex)
        
        bib_content = paper.generate_bibliography()
        bib_file = output_dir / "references.bib"
        bib_file.write_text(bib_content)
        
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        paper_ir_file = output_dir / "paper_ir.json"
        paper.save(paper_ir_file)
        
        return {
            "tex_file": str(tex_file),
            "bib_file": str(bib_file),
            "figures_dir": str(figures_dir),
            "paper_ir_file": str(paper_ir_file),
            "conference": conference,
            "word_count": paper.total_word_count(),
            "figure_count": paper.figure_count(),
            "table_count": paper.table_count(),
        }


renderer = PaperRenderer()
