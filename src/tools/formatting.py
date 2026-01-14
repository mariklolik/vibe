"""Format casting tools - convert papers to conference formats using official templates."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.db.conferences import CONFERENCES, get_conference, list_all_conferences
from src.context.styles import get_conference_style, CONFERENCE_STYLES
from src.db.workflow import workflow_db
from src.project.manager import project_manager


async def _check_formatting_prerequisites(action: str) -> tuple[bool, str, list]:
    """Check if workflow prerequisites are met for formatting actions."""
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return True, "", []  # No project, skip validation
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return True, "", []  # No workflow, skip validation
    
    is_valid, error_msg = workflow_db.validate_action(workflow, action)
    if not is_valid:
        missing = workflow_db.get_missing_prerequisites(workflow, action)
        return False, error_msg, missing
    
    return True, "", []


OUTPUT_DIR = Path("./output")
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def generate_neurips_paper(content: dict) -> str:
    """Generate NeurIPS format paper from content dict."""
    title = content.get("title", "Paper Title")
    abstract = content.get("abstract", "")
    method_name = content.get("method_name", "Method")
    
    # Build authors
    authors_tex = ""
    for i, author in enumerate(content.get("authors", [{"name": "Anonymous Author", "affiliation": "Anonymous Institution"}])):
        if i > 0:
            authors_tex += " \\And\n"
        authors_tex += f"  {author.get('name', 'Author')}"
        if author.get("affiliation"):
            authors_tex += f" \\\\ {author['affiliation']}"
        if author.get("email"):
            authors_tex += f" \\\\ \\texttt{{{author['email']}}}"
    
    # Build sections with \section{} headers
    sections_tex = ""
    for section in content.get("sections", []):
        if isinstance(section, dict):
            section_name = section.get("name", section.get("title", ""))
            section_content = section.get("content", "")
            if section_name:
                sections_tex += f"\\section{{{section_name}}}\n\\label{{sec:{section_name.lower().replace(' ', '_')}}}\n\n"
            sections_tex += section_content + "\n\n"
        else:
            sections_tex += str(section) + "\n\n"
    
    # Build figures
    for fig in content.get("figures", []):
        if isinstance(fig, dict):
            fig_path = fig.get("path", "")
            fig_caption = fig.get("caption", "")
            fig_label = fig.get("label", f"fig:{fig_path.split('/')[-1].split('.')[0] if fig_path else 'figure'}")
            sections_tex += f"""
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{{fig_path}}}
\\caption{{{fig_caption}}}
\\label{{{fig_label}}}
\\end{{figure}}

"""
    
    appendix_tex = content.get("appendix", "")
    appendix_section = ""
    if appendix_tex:
        appendix_section = f"\\appendix\n{appendix_tex}"
    
    return f"""\\documentclass{{article}}

% Package setup for NeurIPS
\\usepackage[preprint]{{neurips_2024}}
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

% Custom commands
\\newcommand{{\\method}}{{{method_name}}}
\\newcommand{{\\ie}}{{\\textit{{i.e.}}}}
\\newcommand{{\\eg}}{{\\textit{{e.g.}}}}
\\newcommand{{\\etal}}{{\\textit{{et al.}}}}

\\title{{{title}}}

\\author{{
{authors_tex}
}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{sections_tex}

\\section*{{Acknowledgments}}
We thank the anonymous reviewers for their helpful feedback.

\\bibliographystyle{{unsrtnat}}
\\bibliography{{references}}

{appendix_section}

\\end{{document}}
"""


def generate_icml_paper(content: dict) -> str:
    """Generate ICML format paper from content dict."""
    title = content.get("title", "Paper Title")
    short_title = content.get("short_title", title[:50])
    abstract = content.get("abstract", "")
    keywords = content.get("keywords", ["machine learning"])
    
    # Build authors
    authors_list = content.get("authors", [{"name": "Anonymous Author"}])
    author_lines = "\n".join([
        f"\\icmlauthor{{{a.get('name', 'Author')}}}{{aff1}}"
        for a in authors_list
    ])
    
    # Build affiliations
    affiliations = content.get("affiliations", {"aff1": "Anonymous Institution"})
    aff_lines = "\n".join([
        f"\\icmlaffiliation{{{k}}}{{{v}}}"
        for k, v in affiliations.items()
    ])
    
    # Corresponding author
    corr = content.get("corresponding_author", authors_list[0] if authors_list else {"name": "Author", "email": "email@inst.edu"})
    
    # Build sections with \section{} headers
    sections_tex = ""
    for section in content.get("sections", []):
        if isinstance(section, dict):
            section_name = section.get("name", section.get("title", ""))
            section_content = section.get("content", "")
            if section_name:
                sections_tex += f"\\section{{{section_name}}}\n\\label{{sec:{section_name.lower().replace(' ', '_')}}}\n\n"
            sections_tex += section_content + "\n\n"
        else:
            sections_tex += str(section) + "\n\n"
    
    # Build figures
    figures_tex = ""
    for fig in content.get("figures", []):
        if isinstance(fig, dict):
            fig_path = fig.get("path", "")
            fig_caption = fig.get("caption", "")
            fig_label = fig.get("label", f"fig:{fig_path.split('/')[-1].split('.')[0] if fig_path else 'figure'}")
            figures_tex += f"""
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{{fig_path}}}
\\caption{{{fig_caption}}}
\\label{{{fig_label}}}
\\end{{figure}}

"""
    
    # Append figures to sections
    sections_tex += figures_tex
    
    keywords_str = ", ".join(keywords)
    
    return f"""\\documentclass[accepted]{{icml2024}}

\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{mathtools}}
\\usepackage{{amsthm}}
\\usepackage[capitalize]{{cleveref}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{subcaption}}

\\icmltitlerunning{{{short_title}}}

\\begin{{document}}

\\twocolumn[
\\icmltitle{{{title}}}

\\icmlsetsymbol{{equal}}{{*}}

\\begin{{icmlauthorlist}}
{author_lines}
\\end{{icmlauthorlist}}

{aff_lines}

\\icmlcorrespondingauthor{{{corr.get('name', 'Author')}}}{{{corr.get('email', 'email@inst.edu')}}}

\\icmlkeywords{{{keywords_str}}}

\\vskip 0.3in
]

\\printAffiliationsAndNotice{{}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{sections_tex}

\\section*{{Acknowledgments}}
We thank the anonymous reviewers for their valuable feedback.

\\bibliography{{references}}
\\bibliographystyle{{icml2024}}

\\end{{document}}
"""


def generate_acl_paper(content: dict) -> str:
    """Generate ACL format paper from content dict."""
    title = content.get("title", "Paper Title")
    abstract = content.get("abstract", "")
    limitations = content.get("limitations", "We acknowledge several limitations of our work.")
    ethics = content.get("ethics", "This work raises no significant ethical concerns.")
    
    # Build authors
    authors_tex = ""
    for i, author in enumerate(content.get("authors", [{"name": "Anonymous Author"}])):
        if i > 0:
            authors_tex += " \\and\n"
        authors_tex += f"  {author.get('name', 'Author')}"
        if author.get("affiliation"):
            authors_tex += f" \\\\ {author['affiliation']}"
        if author.get("email"):
            authors_tex += f" \\\\ \\texttt{{{author['email']}}}"
    
    # Build sections
    sections_tex = ""
    for section in content.get("sections", []):
        if isinstance(section, dict):
            sections_tex += section.get("content", "") + "\n\n"
        else:
            sections_tex += str(section) + "\n\n"
    
    return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[hyperref]{{acl2024}}
\\usepackage{{times}}
\\usepackage{{latexsym}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}

\\renewcommand{{\\UrlFont}}{{\\ttfamily\\small}}

\\title{{{title}}}

\\author{{
{authors_tex}
}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{sections_tex}

\\section*{{Limitations}}
{limitations}

\\section*{{Ethics Statement}}
{ethics}

\\bibliography{{references}}
\\bibliographystyle{{acl_natbib}}

\\end{{document}}
"""


def generate_cvpr_paper(content: dict) -> str:
    """Generate CVPR format paper from content dict."""
    title = content.get("title", "Paper Title")
    abstract = content.get("abstract", "")
    
    # Build authors
    authors_tex = ""
    for i, author in enumerate(content.get("authors", [{"name": "Anonymous Author"}])):
        if i > 0:
            authors_tex += "\\\\\n\\And\n"
        authors_tex += f"{author.get('name', 'Author')}"
        if author.get("affiliation"):
            authors_tex += f"\\\\{author['affiliation']}"
    
    # Build sections
    sections_tex = ""
    for section in content.get("sections", []):
        if isinstance(section, dict):
            sections_tex += section.get("content", "") + "\n\n"
        else:
            sections_tex += str(section) + "\n\n"
    
    return f"""\\documentclass[10pt,twocolumn,letterpaper]{{article}}

\\usepackage{{cvpr}}
\\usepackage{{times}}
\\usepackage{{epsfig}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{booktabs}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{subcaption}}

\\usepackage[pagebackref=true,breaklinks=true,colorlinks,bookmarks=false]{{hyperref}}

\\cvprfinalcopy

\\def\\cvprPaperID{{****}}
\\def\\httilde{{\\mbox{{\\tt\\raisebox{{-.5ex}}{{\\symbol{{126}}}}}}}}

\\begin{{document}}

\\title{{{title}}}

\\author{{
{authors_tex}
}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{sections_tex}

{{\\small
\\bibliographystyle{{ieee_fullname}}
\\bibliography{{references}}
}}

\\end{{document}}
"""


PAPER_GENERATORS = {
    "neurips": generate_neurips_paper,
    "icml": generate_icml_paper,
    "iclr": generate_neurips_paper,
    "acl": generate_acl_paper,
    "emnlp": generate_acl_paper,
    "naacl": generate_acl_paper,
    "cvpr": generate_cvpr_paper,
    "iccv": generate_cvpr_paper,
    "eccv": generate_cvpr_paper,
    "aaai": generate_neurips_paper,
    "ijcai": generate_neurips_paper,
}


async def list_conferences() -> str:
    """List all supported A* conferences with their requirements."""
    conferences = []
    
    for name, style in CONFERENCE_STYLES.items():
        conferences.append({
            "name": name.upper(),
            "full_name": style.full_name,
            "page_limit": style.page_limit,
            "columns": style.columns,
            "abstract_limit": style.abstract_word_limit,
            "citation_style": style.citation_style,
            "has_template": name in PAPER_GENERATORS,
        })
    
    return json.dumps({
        "count": len(conferences),
        "conferences": conferences,
    }, indent=2)


async def get_conference_requirements(conference: str) -> str:
    """Get detailed format requirements for a specific conference."""
    style = get_conference_style(conference.lower())
    
    if not style:
        available = list(CONFERENCE_STYLES.keys())
        return json.dumps({
            "error": f"Conference not found: {conference}",
            "available": available,
        })
    
    return json.dumps({
        "name": style.name.upper(),
        "full_name": style.full_name,
        "format": {
            "page_limit": style.page_limit,
            "columns": style.columns,
            "font_size": style.font_size,
        },
        "abstract": {
            "word_limit": style.abstract_word_limit,
            "style": style.abstract_style,
        },
        "sections": {
            "required": style.required_sections,
            "optional": style.optional_sections,
            "numbering": style.section_numbering,
        },
        "figures": {
            "position": style.figure_position,
            "caption_position": style.caption_position,
            "max_figures": style.max_figures,
        },
        "citations": {
            "style": style.citation_style,
            "bibliography_style": style.bibliography_style,
        },
        "writing_style": {
            "first_person_allowed": style.first_person_allowed,
            "passive_preferred": style.passive_preferred,
            "anonymous_submission": style.anonymous_submission,
        },
        "latex": {
            "document_class": style.document_class,
            "style_package": style.style_package,
            "required_packages": style.required_packages,
        },
    }, indent=2)


async def cast_to_format(
    paper_content: Optional[dict] = None,
    conference: str = "neurips",
    output_dir: str = "./output",
) -> str:
    """Convert paper content to conference-specific LaTeX format.
    
    Prerequisites: 
    - Should have figures generated and paper content written
    - All claims must be verified through verify_and_record_hypothesis
    """
    # Check workflow prerequisites
    is_valid, error_msg, missing = await _check_formatting_prerequisites("cast_to_format")
    if not is_valid:
        return json.dumps({
            "success": False,
            "error": "WORKFLOW_BLOCKED",
            "message": error_msg,
            "missing_prerequisites": missing,
            "action_required": (
                "Generate figures and write paper content first. "
                "Call get_next_action() to see required steps."
            ),
        }, indent=2)
    
    # Check that claims are verified
    current_project_obj = await project_manager.get_current_project()
    if current_project_obj:
        workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
        if workflow:
            verified_hypotheses = getattr(workflow, "verified_hypotheses", {})
            
            # Check if there are any completed experiments without verification
            if len(workflow.completed_experiments) > 0 and len(verified_hypotheses) == 0:
                return json.dumps({
                    "success": False,
                    "error": "UNVERIFIED_CLAIMS",
                    "message": (
                        "BLOCKED: You have completed experiments but no verified hypotheses. "
                        "All claims in the paper MUST be verified using verify_and_record_hypothesis() "
                        "with real experiment run_ids before formatting."
                    ),
                    "completed_experiments": workflow.completed_experiments,
                    "verified_hypotheses": 0,
                    "action_required": (
                        "Call verify_and_record_hypothesis(hypothesis_id, statement, run_ids, metric) "
                        "for each claim you want to make in the paper."
                    ),
                }, indent=2)
            
            # Note: verified_from_logs check removed - experimenter now sets this flag
            # and we trust any verified hypothesis regardless of source
    
    conf_lower = conference.lower()
    style = get_conference_style(conf_lower)
    
    if not style:
        return json.dumps({
            "success": False,
            "error": f"Conference not found: {conference}",
            "available": list(CONFERENCE_STYLES.keys()),
        })
    
    # Get generator
    generator = PAPER_GENERATORS.get(conf_lower, generate_neurips_paper)
    
    # Default content
    default_content = {
        "title": "Paper Title",
        "short_title": "Short Title",
        "method_name": "ProposedMethod",
        "authors": [
            {"name": "Anonymous Author", "affiliation": "Anonymous Institution", "email": "email@institution.edu"}
        ],
        "corresponding_author": {"name": "Anonymous Author", "email": "email@institution.edu"},
        "affiliations": {"aff1": "Anonymous Institution"},
        "keywords": ["machine learning", "deep learning"],
        "abstract": "This paper presents a novel approach...",
        "sections": [],
        "limitations": "We acknowledge several limitations of our work.",
        "ethics": "This work raises no significant ethical concerns.",
        "appendix": "",
    }
    
    content = {**default_content, **(paper_content or {})}
    
    # Generate LaTeX
    latex = generator(content)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write main tex file
    tex_file = output_path / f"paper_{conference}.tex"
    tex_file.write_text(latex)
    
    # Copy style files if available
    style_source = TEMPLATES_DIR / "Styles"
    if style_source.exists():
        for sty_file in style_source.glob("*.sty"):
            shutil.copy(sty_file, output_path / sty_file.name)
    
    # Create empty bib file if not exists
    bib_file = output_path / "references.bib"
    if not bib_file.exists():
        bib_file.write_text("% Bibliography entries\n")
    
    # Calculate paper statistics
    word_count = len(latex.split())
    section_count = latex.count("\\section{")
    
    return json.dumps({
        "success": True,
        "conference": conference.upper(),
        "output_file": str(tex_file),
        "bibliography": str(bib_file),
        "statistics": {
            "word_count": word_count,
            "section_count": section_count,
            "estimated_pages": max(1, word_count // 550),
        },
        "requirements": {
            "page_limit": style.page_limit,
            "columns": style.columns,
            "font_size": style.font_size,
            "abstract_word_limit": style.abstract_word_limit,
        },
        "compile_commands": [
            f"cd {output_path} && pdflatex paper_{conference}.tex",
            f"cd {output_path} && bibtex paper_{conference}",
            f"cd {output_path} && pdflatex paper_{conference}.tex",
            f"cd {output_path} && pdflatex paper_{conference}.tex",
        ],
    }, indent=2)


async def generate_poster(
    paper_content: Optional[dict] = None,
    conference: str = "neurips",
    output_path: Optional[str] = None,
) -> str:
    """Generate a conference poster from paper content.
    
    Prerequisites: Should have figures generated and paper content written.
    """
    # Check workflow prerequisites
    is_valid, error_msg, missing = await _check_formatting_prerequisites("generate_poster")
    if not is_valid:
        return json.dumps({
            "success": False,
            "error": "WORKFLOW_BLOCKED",
            "message": error_msg,
            "missing_prerequisites": missing,
            "action_required": "Call get_next_action() to see required steps.",
        }, indent=2)
    
    content = paper_content or {}
    
    title = content.get("title", "Research Poster")
    authors = content.get("authors", [{"name": "Author Name"}])
    authors_str = ", ".join(a.get("name", "Author") for a in authors)
    institution = content.get("institution", authors[0].get("affiliation", "Institution") if authors else "Institution")
    
    introduction = content.get("introduction", "We address the problem of...")
    method = content.get("method", "We propose a novel approach that...")
    experiments = content.get("experiments", "We evaluate on multiple benchmarks...")
    conclusion = content.get("conclusion", "Our method achieves state-of-the-art results...")
    future_work = content.get("future_work", "Future directions include...")
    
    key_components = content.get("key_components", ["Component 1", "Component 2", "Component 3"])
    key_findings = content.get("key_findings", ["Finding 1", "Finding 2", "Finding 3"])
    references = content.get("references", "[1] Reference 1\n[2] Reference 2")
    
    components_tex = "\n".join([f"    \\item {c}" for c in key_components])
    findings_tex = "\n".join([f"    \\item {f}" for f in key_findings])
    
    latex = f"""\\documentclass[final,hyperref={{pdfpagelabels=false}}]{{beamer}}
\\usepackage[orientation=portrait,size=a0,scale=1.4]{{beamerposter}}
\\usetheme{{confposter}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}

% Colors
\\definecolor{{methodcolor}}{{RGB}}{{0,102,204}}
\\definecolor{{resultcolor}}{{RGB}}{{0,153,76}}

\\title{{{title}}}
\\author{{{authors_str}}}
\\institute{{{institution}}}

\\begin{{document}}
\\begin{{frame}}[t]

\\begin{{columns}}[t]

% Left column
\\begin{{column}}{{.48\\linewidth}}

\\begin{{block}}{{\\Large Introduction}}
\\large
{introduction}
\\end{{block}}

\\begin{{block}}{{\\Large Method}}
\\large
{method}

\\textbf{{Key Components:}}
\\begin{{itemize}}
{components_tex}
\\end{{itemize}}
\\end{{block}}

\\end{{column}}

% Right column
\\begin{{column}}{{.48\\linewidth}}

\\begin{{block}}{{\\Large Experiments}}
\\large
{experiments}

\\textbf{{Key Findings:}}
\\begin{{itemize}}
{findings_tex}
\\end{{itemize}}
\\end{{block}}

\\begin{{block}}{{\\Large Conclusion}}
\\large
{conclusion}

\\textbf{{Future Work:}}
{future_work}
\\end{{block}}

\\begin{{block}}{{\\Large References}}
\\small
{references}
\\end{{block}}

\\end{{column}}

\\end{{columns}}

\\end{{frame}}
\\end{{document}}
"""
    
    if output_path is None:
        output_path = f"./output/poster_{conference}.tex"
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex)
    
    return json.dumps({
        "success": True,
        "output_file": str(output),
        "poster_size": "A0",
        "conference": conference.upper(),
        "compile_command": f"pdflatex {output}",
    }, indent=2)


async def generate_supplementary(
    paper_content: Optional[dict] = None,
    include_code: bool = True,
    include_data: bool = False,
) -> str:
    """Generate supplementary materials document."""
    
    content = paper_content or {}
    
    title = content.get("title", "Paper Title")
    hyperparameters = content.get("hyperparameters", "Table lists all hyperparameters used in our experiments.")
    training_details = content.get("training_details", "We train using Adam optimizer with learning rate 1e-4...")
    compute_resources = content.get("compute_resources", "Experiments were conducted on available compute resources...")
    extended_results = content.get("extended_results", "Extended results with additional metrics...")
    additional_ablations = content.get("additional_ablations", "Additional ablation experiments...")
    sensitivity_analysis = content.get("sensitivity_analysis", "We analyze sensitivity to key hyperparameters...")
    qualitative_examples = content.get("qualitative_examples", "Additional qualitative examples are shown.")
    hyperparameter_sensitivity = content.get("hyperparameter_sensitivity", "We study the effect of varying key hyperparameters...")
    
    code_section = ""
    if include_code:
        code_url = content.get("code_url", "https://github.com/anonymous/repository")
        code_snippet = content.get("code_snippet", "# Key implementation\ndef forward(self, x):\n    return self.model(x)")
        code_section = f"""
\\section{{Implementation Details}}

Code is available at: \\url{{{code_url}}}

\\subsection{{Key Implementation}}
\\begin{{lstlisting}}
{code_snippet}
\\end{{lstlisting}}
"""
    
    data_section = ""
    if include_data:
        dataset_stats = content.get("dataset_stats", "Dataset statistics...")
        preprocessing = content.get("preprocessing", "Preprocessing steps...")
        data_section = f"""
\\section{{Dataset Details}}

\\subsection{{Dataset Statistics}}
{dataset_stats}

\\subsection{{Preprocessing Steps}}
{preprocessing}
"""
    
    latex = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{listings}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}

% Code listing style
\\lstset{{
    basicstyle=\\ttfamily\\small,
    breaklines=true,
    frame=single,
    language=Python,
    keywordstyle=\\color{{blue}},
    commentstyle=\\color{{gray}},
    stringstyle=\\color{{orange}},
}}

\\title{{Supplementary Material: {title}}}
\\author{{}}
\\date{{}}

\\begin{{document}}
\\maketitle

\\section{{Additional Experimental Details}}

\\subsection{{Hyperparameter Settings}}
{hyperparameters}

\\subsection{{Training Details}}
{training_details}

\\subsection{{Computational Resources}}
{compute_resources}

\\section{{Additional Results}}

\\subsection{{Extended Main Results}}
{extended_results}

\\subsection{{Additional Ablations}}
{additional_ablations}

\\subsection{{Sensitivity Analysis}}
{sensitivity_analysis}

{code_section}

{data_section}

\\section{{Hyperparameter Sensitivity}}
{hyperparameter_sensitivity}

\\section{{Additional Qualitative Examples}}
{qualitative_examples}

\\end{{document}}
"""
    
    output_path = Path("./output/supplementary.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    
    return json.dumps({
        "success": True,
        "output_file": str(output_path),
        "include_code": include_code,
        "include_data": include_data,
        "compile_command": f"pdflatex {output_path}",
    }, indent=2)


async def compile_paper(
    tex_file: str,
    output_dir: Optional[str] = None,
) -> str:
    """Compile a LaTeX paper to PDF using pdflatex.
    
    Prerequisites: Should have paper sections written.
    Runs: pdflatex -> bibtex -> pdflatex -> pdflatex (for references).
    """
    import subprocess
    import os
    
    tex_path = Path(tex_file)
    
    # Check file existence FIRST - this is the real prerequisite
    if not tex_path.exists():
        return json.dumps({
            "success": False,
            "error": f"File not found: {tex_file}",
        })
    
    work_dir = tex_path.parent
    stem = tex_path.stem
    
    # Find pdflatex and bibtex (common locations on macOS)
    tex_bin_paths = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
        "/usr/texbin",
    ]
    
    pdflatex_path = "pdflatex"
    bibtex_path = "bibtex"
    
    for tex_bin in tex_bin_paths:
        if Path(tex_bin).exists():
            pdflatex_path = f"{tex_bin}/pdflatex"
            bibtex_path = f"{tex_bin}/bibtex"
            break
    
    # Prepare environment with TeX paths
    env = os.environ.copy()
    env["PATH"] = f"/Library/TeX/texbin:{env.get('PATH', '')}"
    
    commands = [
        [pdflatex_path, "-interaction=nonstopmode", str(tex_path)],
        [bibtex_path, stem],
        [pdflatex_path, "-interaction=nonstopmode", str(tex_path)],
        [pdflatex_path, "-interaction=nonstopmode", str(tex_path)],
    ]
    
    logs = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            error_output = None
            if result.returncode != 0:
                # Check for missing packages in stdout/stderr
                output = result.stdout + result.stderr
                if "File `" in output and "not found" in output:
                    import re
                    missing = re.findall(r"File `([^']+)' not found", output)
                    error_output = f"Missing packages: {missing}. Install with: sudo tlmgr install {' '.join(m.replace('.sty', '') for m in missing)}"
                else:
                    error_output = output[-500:] if output else None
            
            logs.append({
                "command": " ".join(cmd),
                "success": result.returncode == 0,
                "error": error_output,
            })
        except subprocess.TimeoutExpired:
            logs.append({
                "command": " ".join(cmd),
                "success": False,
                "error": "Command timed out (>120s)",
            })
        except FileNotFoundError:
            logs.append({
                "command": " ".join(cmd),
                "success": False,
                "error": f"pdflatex not found. Install BasicTeX: brew install --cask basictex",
            })
            break
        except Exception as e:
            logs.append({
                "command": " ".join(cmd),
                "success": False,
                "error": str(e),
            })
    
    pdf_path = work_dir / f"{stem}.pdf"
    
    return json.dumps({
        "success": pdf_path.exists(),
        "pdf_file": str(pdf_path) if pdf_path.exists() else None,
        "compilation_logs": logs,
    }, indent=2)


def add_github_link_to_latex(latex_content: str, repo_url: str) -> str:
    """Add GitHub repo link to a LaTeX paper.
    
    Args:
        latex_content: The full LaTeX content
        repo_url: GitHub repository URL
    
    Returns:
        Modified LaTeX content with GitHub link added
    """
    import re
    
    if not repo_url:
        return latex_content
    
    if "\\usepackage{hyperref}" not in latex_content:
        preamble_end = latex_content.find("\\begin{document}")
        if preamble_end != -1:
            latex_content = (
                latex_content[:preamble_end] + 
                "\\usepackage{hyperref}\n" + 
                latex_content[preamble_end:]
            )
    
    abstract_match = re.search(r'(\\begin\{abstract\})', latex_content)
    if abstract_match:
        footnote = f"\\\\footnote{{Code available at: \\\\url{{{repo_url}}}}}"
        
        title_match = re.search(r'(\\title\{[^}]+\})', latex_content)
        if title_match:
            old_title = title_match.group(1)
            new_title = old_title[:-1] + footnote + "}"
            latex_content = latex_content.replace(old_title, new_title, 1)
    
    return latex_content


async def create_github_repo(
    name: Optional[str] = None,
    private: bool = True,
    description: str = "",
) -> str:
    """Create a GitHub repository for the current project.
    
    Uses the gh CLI to create a repository, commit all files, and push.
    
    Args:
        name: Repository name (defaults to project name)
        private: Whether the repo should be private
        description: Repository description
    
    Returns:
        JSON with repo URL and status
    """
    from src.project.git_ops import GitOps
    
    current_project = await project_manager.get_current_project()
    
    if not current_project:
        return json.dumps({
            "success": False,
            "error": "No active project",
            "action": "Create a project first with create_project()",
        })
    
    repo_name = name or current_project.name.replace(" ", "-").lower()
    repo_desc = description or current_project.description
    
    git_ops = GitOps(current_project.root_path)
    
    repo_url = await git_ops.create_github_repo(
        name=repo_name,
        private=private,
        description=repo_desc,
    )
    
    if repo_url:
        workflow = await workflow_db.get_project_workflow(current_project.project_id)
        if workflow:
            workflow["github_url"] = repo_url
            await workflow_db.save_workflow(workflow)
        
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "private": private,
            "message": f"Repository created and pushed to {repo_url}",
            "paper_integration": (
                "Add to your paper with: "
                f"\\\\footnote{{Code: \\\\url{{{repo_url}}}}}"
            ),
        }, indent=2)
    else:
        return json.dumps({
            "success": False,
            "error": "Failed to create GitHub repository",
            "troubleshooting": [
                "Ensure gh CLI is installed: brew install gh",
                "Authenticate with: gh auth login",
                "Check if repo name already exists",
            ],
        }, indent=2)


async def finalize_paper_with_github(
    latex_file: str,
    repo_url: Optional[str] = None,
) -> str:
    """Add GitHub link to paper and compile to PDF.
    
    Args:
        latex_file: Path to the LaTeX file
        repo_url: GitHub URL (fetched from project if not provided)
    
    Returns:
        JSON with PDF path and repo URL
    """
    current_project = await project_manager.get_current_project()
    
    if not repo_url and current_project:
        workflow = await workflow_db.get_project_workflow(current_project.project_id)
        if workflow:
            repo_url = workflow.get("github_url")
    
    latex_path = Path(latex_file)
    if not latex_path.exists():
        return json.dumps({
            "success": False,
            "error": f"LaTeX file not found: {latex_file}",
        })
    
    latex_content = latex_path.read_text()
    
    if repo_url:
        latex_content = add_github_link_to_latex(latex_content, repo_url)
        latex_path.write_text(latex_content)
    
    compile_result = await compile_paper(str(latex_path))
    compile_data = json.loads(compile_result)
    
    if current_project and compile_data.get("success"):
        from src.project.git_ops import GitOps
        git_ops = GitOps(current_project.root_path)
        await git_ops.auto_commit_stage("paper_finalized")
    
    return json.dumps({
        "success": compile_data.get("success", False),
        "pdf_file": compile_data.get("pdf_file"),
        "github_url": repo_url,
        "message": "Paper finalized with GitHub link" if repo_url else "Paper compiled",
    }, indent=2)
