#!/bin/bash
set -e

# ResearchMCP - System Dependencies Setup Script
# Supports: macOS, Ubuntu/Debian, RHEL/CentOS/Fedora

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  ResearchMCP Dependencies Setup"
echo "=========================================="

detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "redhat"
    elif [[ -f /etc/arch-release ]]; then
        echo "arch"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "${GREEN}Detected OS: $OS${NC}"

install_macos() {
    echo -e "${YELLOW}Installing dependencies for macOS...${NC}"
    
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found. Installing...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "Installing system packages..."
    brew install git python@3.11 || true
    
    echo "Installing LaTeX (MacTeX - this may take a while)..."
    if ! command -v pdflatex &> /dev/null; then
        echo "Installing BasicTeX (smaller distribution)..."
        brew install --cask basictex
        
        eval "$(/usr/libexec/path_helper)"
        export PATH="/Library/TeX/texbin:$PATH"
        
        echo "Updating tlmgr and installing required packages..."
        sudo tlmgr update --self
        sudo tlmgr install \
            collection-fontsrecommended \
            collection-latexrecommended \
            collection-latexextra \
            booktabs \
            algorithms \
            algorithmicx \
            multirow \
            microtype \
            xcolor \
            hyperref \
            natbib \
            subcaption \
            nicefrac \
            preprint \
            comment \
            titlesec \
            fancyhdr \
            lastpage \
            geometry \
            caption \
            float \
            enumitem \
            wrapfig \
            placeins \
            pdfpages \
            etoolbox \
            xstring \
            environ \
            trimspaces \
            ncctools \
            latexmk
    else
        echo -e "${GREEN}pdflatex already installed${NC}"
    fi
    
    echo "Installing poppler for PDF processing..."
    brew install poppler || true
}

install_debian() {
    echo -e "${YELLOW}Installing dependencies for Ubuntu/Debian...${NC}"
    
    sudo apt-get update
    
    echo "Installing basic packages..."
    sudo apt-get install -y \
        git \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        wget
    
    echo "Installing LaTeX (TeX Live - this may take a while)..."
    sudo apt-get install -y \
        texlive-full || \
    sudo apt-get install -y \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-science \
        texlive-bibtex-extra \
        texlive-publishers \
        latexmk \
        biber
    
    echo "Installing additional tools..."
    sudo apt-get install -y \
        poppler-utils \
        ghostscript
}

install_redhat() {
    echo -e "${YELLOW}Installing dependencies for RHEL/CentOS/Fedora...${NC}"
    
    if command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
    else
        PKG_MGR="yum"
    fi
    
    echo "Installing basic packages..."
    sudo $PKG_MGR install -y \
        git \
        python3 \
        python3-pip \
        curl \
        wget
    
    echo "Installing LaTeX (TeX Live)..."
    sudo $PKG_MGR install -y \
        texlive-scheme-full || \
    sudo $PKG_MGR install -y \
        texlive-scheme-medium \
        texlive-collection-latexrecommended \
        texlive-collection-latexextra \
        texlive-collection-fontsrecommended \
        texlive-collection-science \
        texlive-collection-publishers \
        latexmk
    
    echo "Installing additional tools..."
    sudo $PKG_MGR install -y \
        poppler-utils \
        ghostscript
}

install_arch() {
    echo -e "${YELLOW}Installing dependencies for Arch Linux...${NC}"
    
    echo "Installing packages..."
    sudo pacman -S --noconfirm \
        git \
        python \
        python-pip \
        texlive-core \
        texlive-latexextra \
        texlive-fontsextra \
        texlive-science \
        texlive-publishers \
        texlive-bibtexextra \
        biber \
        poppler \
        ghostscript
}

install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
    fi
    
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    
    echo -e "${GREEN}Python dependencies installed in .venv${NC}"
}

verify_installation() {
    echo ""
    echo "=========================================="
    echo "  Verifying Installation"
    echo "=========================================="
    
    ERRORS=0
    
    if command -v pdflatex &> /dev/null; then
        echo -e "${GREEN}✓ pdflatex found: $(which pdflatex)${NC}"
    else
        echo -e "${RED}✗ pdflatex not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    if command -v bibtex &> /dev/null; then
        echo -e "${GREEN}✓ bibtex found: $(which bibtex)${NC}"
    else
        echo -e "${RED}✗ bibtex not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✓ git found: $(which git)${NC}"
    else
        echo -e "${RED}✗ git not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓ python3 found: $(python3 --version)${NC}"
    else
        echo -e "${RED}✗ python3 not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
        if python3 -c "import mcp" 2>/dev/null; then
            echo -e "${GREEN}✓ MCP package installed${NC}"
        else
            echo -e "${RED}✗ MCP package not installed${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    fi
    
    echo ""
    if [[ $ERRORS -eq 0 ]]; then
        echo -e "${GREEN}=========================================="
        echo "  All dependencies installed successfully!"
        echo "==========================================${NC}"
    else
        echo -e "${RED}=========================================="
        echo "  $ERRORS dependencies missing"
        echo "==========================================${NC}"
        exit 1
    fi
}

case $OS in
    macos)
        install_macos
        ;;
    debian)
        install_debian
        ;;
    redhat)
        install_redhat
        ;;
    arch)
        install_arch
        ;;
    *)
        echo -e "${RED}Unknown OS. Please install dependencies manually:${NC}"
        echo "  - Python 3.11+"
        echo "  - Git"
        echo "  - LaTeX distribution with pdflatex, bibtex"
        echo "  - LaTeX packages: booktabs, amsmath, algorithms, microtype, xcolor, hyperref, natbib"
        exit 1
        ;;
esac

install_python_deps
verify_installation

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Add to ~/.cursor/mcp.json:"
echo '     {
       "mcpServers": {
         "research-mcp": {
           "command": "python3",
           "args": ["run_server.py"],
           "cwd": "'$(pwd)'"
         }
       }
     }'
echo ""
echo "  3. Restart Cursor"
