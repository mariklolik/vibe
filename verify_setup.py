#!/usr/bin/env python3
"""Verify that all ResearchMCP dependencies are properly installed."""

import subprocess
import shutil
import sys
from pathlib import Path


REQUIRED_PYTHON_PACKAGES = [
    ("mcp", "mcp"),
    ("httpx", "httpx"),
    ("aiohttp", "aiohttp"),
    ("arxiv", "arxiv"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("jinja2", "jinja2"),
    ("yaml", "pyyaml"),
    ("aiosqlite", "aiosqlite"),
    ("rich", "rich"),
    ("pydantic", "pydantic"),
    ("git", "gitpython"),
]

REQUIRED_LATEX_PACKAGES = [
    "booktabs",
    "amsmath",
    "amssymb",
    "amsfonts",
    "graphicx",
    "xcolor",
    "hyperref",
    "algorithm",
    "algorithmic",
    "subcaption",
    "multirow",
    "microtype",
    "nicefrac",
    "natbib",
]

REQUIRED_COMMANDS = [
    ("python3", "Python 3.11+"),
    ("pdflatex", "LaTeX compiler"),
    ("bibtex", "BibTeX processor"),
    ("git", "Git version control"),
]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def check_python_version():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro}{Colors.RESET}")
        return True
    print(f"{Colors.RED}✗ Python 3.11+ required, found {version.major}.{version.minor}{Colors.RESET}")
    return False


def check_command(cmd, description):
    path = shutil.which(cmd)
    if path:
        print(f"{Colors.GREEN}✓ {description}: {path}{Colors.RESET}")
        return True
    
    common_paths = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
        "/usr/texbin",
        "/opt/texlive/2024/bin/x86_64-linux",
        "/opt/texlive/2023/bin/x86_64-linux",
    ]
    
    for base in common_paths:
        full_path = Path(base) / cmd
        if full_path.exists():
            print(f"{Colors.GREEN}✓ {description}: {full_path}{Colors.RESET}")
            print(f"  {Colors.YELLOW}Note: Add {base} to PATH for easier access{Colors.RESET}")
            return True
    
    print(f"{Colors.RED}✗ {description} ({cmd}) not found{Colors.RESET}")
    return False


def check_python_package(import_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_python_packages():
    missing = []
    for import_name, pip_name in REQUIRED_PYTHON_PACKAGES:
        if not check_python_package(import_name):
            missing.append(pip_name)
    
    if missing:
        print(f"{Colors.RED}✗ Missing Python packages: {', '.join(missing)}{Colors.RESET}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    
    print(f"{Colors.GREEN}✓ All {len(REQUIRED_PYTHON_PACKAGES)} Python packages installed{Colors.RESET}")
    return True


def find_pdflatex():
    path = shutil.which("pdflatex")
    if path:
        return path
    
    common_paths = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
        "/usr/texbin",
        "/opt/texlive/2024/bin/x86_64-linux",
        "/opt/texlive/2023/bin/x86_64-linux",
    ]
    
    for base in common_paths:
        full_path = Path(base) / "pdflatex"
        if full_path.exists():
            return str(full_path)
    
    return None


def check_latex_package(package, pdflatex_path):
    test_doc = f"""\\documentclass{{article}}
\\usepackage{{{package}}}
\\begin{{document}}
Test
\\end{{document}}
"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "test.tex"
        tex_file.write_text(test_doc)
        
        try:
            result = subprocess.run(
                [pdflatex_path, "-interaction=nonstopmode", "-halt-on-error", str(tex_file)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def check_latex_packages():
    pdflatex_path = find_pdflatex()
    if not pdflatex_path:
        print(f"{Colors.YELLOW}⚠ Cannot check LaTeX packages: pdflatex not found{Colors.RESET}")
        return False
    
    missing = []
    for pkg in REQUIRED_LATEX_PACKAGES:
        if not check_latex_package(pkg, pdflatex_path):
            missing.append(pkg)
    
    if missing:
        print(f"{Colors.RED}✗ Missing LaTeX packages: {', '.join(missing)}{Colors.RESET}")
        print(f"  Install with: sudo tlmgr install {' '.join(missing)}")
        return False
    
    print(f"{Colors.GREEN}✓ All {len(REQUIRED_LATEX_PACKAGES)} LaTeX packages available{Colors.RESET}")
    return True


def check_tex_paths():
    common_paths = [
        "/Library/TeX/texbin",
        "/usr/local/texlive/2024/bin/universal-darwin",
        "/usr/local/texlive/2023/bin/universal-darwin",
        "/usr/texbin",
        "/usr/bin",
        "/usr/local/bin",
    ]
    
    for path in common_paths:
        pdflatex = Path(path) / "pdflatex"
        if pdflatex.exists():
            print(f"{Colors.GREEN}✓ TeX binaries found at: {path}{Colors.RESET}")
            return True
    
    if shutil.which("pdflatex"):
        print(f"{Colors.GREEN}✓ pdflatex found in PATH{Colors.RESET}")
        return True
    
    print(f"{Colors.YELLOW}⚠ TeX binaries not in common locations{Colors.RESET}")
    return False


def check_conference_styles():
    styles_dir = Path(__file__).parent / "templates" / "Styles"
    if not styles_dir.exists():
        styles_dir = Path(__file__).parent / "src" / "paper" / "templates"
    
    required_templates = ["neurips.tex.j2", "acl.tex.j2", "icml.tex.j2"]
    templates_dir = Path(__file__).parent / "src" / "paper" / "templates"
    
    missing = []
    for tmpl in required_templates:
        if not (templates_dir / tmpl).exists():
            missing.append(tmpl)
    
    if missing:
        print(f"{Colors.YELLOW}⚠ Missing templates: {', '.join(missing)}{Colors.RESET}")
        return False
    
    print(f"{Colors.GREEN}✓ Conference templates available{Colors.RESET}")
    return True


def check_write_permissions():
    test_dirs = [
        Path.home() / "research-projects",
        Path(__file__).parent / "projects",
    ]
    
    for test_dir in test_dirs:
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            print(f"{Colors.GREEN}✓ Write access to: {test_dir}{Colors.RESET}")
            return True
        except (PermissionError, OSError):
            continue
    
    print(f"{Colors.RED}✗ No write access to project directories{Colors.RESET}")
    return False


def main():
    print("=" * 50)
    print(f"{Colors.BOLD}  ResearchMCP Dependency Verification{Colors.RESET}")
    print("=" * 50)
    print()
    
    errors = 0
    warnings = 0
    
    print(f"{Colors.BOLD}System Commands:{Colors.RESET}")
    if not check_python_version():
        errors += 1
    
    for cmd, desc in REQUIRED_COMMANDS[1:]:
        if not check_command(cmd, desc):
            errors += 1
    
    print()
    print(f"{Colors.BOLD}Python Packages:{Colors.RESET}")
    if not check_python_packages():
        errors += 1
    
    print()
    print(f"{Colors.BOLD}LaTeX Environment:{Colors.RESET}")
    check_tex_paths()
    if not check_latex_packages():
        warnings += 1
    
    print()
    print(f"{Colors.BOLD}Project Setup:{Colors.RESET}")
    check_conference_styles()
    if not check_write_permissions():
        errors += 1
    
    print()
    print("=" * 50)
    
    if errors == 0 and warnings == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All dependencies verified!{Colors.RESET}")
        print("  ResearchMCP is ready to use.")
        return 0
    elif errors == 0:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Setup complete with {warnings} warning(s){Colors.RESET}")
        print("  Some features may not work correctly.")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ {errors} critical error(s) found{Colors.RESET}")
        print("  Run ./setup_dependencies.sh to fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
