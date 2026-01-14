# ResearchMCP Docker Image
# Full environment with LaTeX, Python, and all dependencies

FROM python:3.11-slim-bookworm

LABEL maintainer="ResearchMCP Team"
LABEL description="End-to-end AI research pipeline MCP server"

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    # LaTeX full installation
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    texlive-publishers \
    latexmk \
    biber \
    # PDF processing
    poppler-utils \
    ghostscript \
    # Build tools
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Create projects directory
RUN mkdir -p /root/research-projects

# Verify installation
RUN python3 verify_setup.py || echo "Some checks may fail in Docker build"

# Default command
CMD ["python3", "run_server.py"]
