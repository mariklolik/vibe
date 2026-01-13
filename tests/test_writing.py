"""Tests for paper writing utilities."""

import json
import pytest

from src.tools.writing import (
    format_results_table,
    format_equation,
    format_algorithm,
    format_figure,
    estimate_paper_structure,
    validate_latex,
)


class TestFormatResultsTable:
    """Tests for LaTeX results table generation."""
    
    @pytest.mark.asyncio
    async def test_generates_latex_table(self, sample_results):
        """Should generate valid LaTeX table."""
        result = await format_results_table(results=sample_results)
        
        data = json.loads(result)
        assert "latex" in data
        assert "\\begin{table}" in data["latex"]
        assert "\\end{table}" in data["latex"]
    
    @pytest.mark.asyncio
    async def test_bolds_best_result(self, sample_results):
        """Should bold the best result by default."""
        result = await format_results_table(results=sample_results, bold_best=True)
        
        data = json.loads(result)
        # Best accuracy is 0.92, should be bolded
        assert "\\textbf{0.92}" in data["latex"]
    
    @pytest.mark.asyncio
    async def test_custom_caption(self):
        """Should use custom caption."""
        results = {"method": {"metric": 0.5}}
        result = await format_results_table(
            results=results,
            caption="Custom Caption",
            label="tab:custom",
        )
        
        data = json.loads(result)
        assert "Custom Caption" in data["latex"]
        assert "tab:custom" in data["latex"]


class TestFormatEquation:
    """Tests for LaTeX equation formatting."""
    
    @pytest.mark.asyncio
    async def test_equation_environment(self):
        """Should wrap in equation environment."""
        result = await format_equation(
            equation="E = mc^2",
            label="eq:energy",
        )
        
        data = json.loads(result)
        assert "\\begin{equation}" in data["latex"]
        assert "E = mc^2" in data["latex"]
        assert "\\label{eq:energy}" in data["latex"]
    
    @pytest.mark.asyncio
    async def test_equation_without_label(self):
        """Should work without label."""
        result = await format_equation(equation="a^2 + b^2 = c^2")
        
        data = json.loads(result)
        assert "a^2 + b^2 = c^2" in data["latex"]


class TestFormatAlgorithm:
    """Tests for LaTeX algorithm formatting."""
    
    @pytest.mark.asyncio
    async def test_algorithm_environment(self):
        """Should generate algorithm environment."""
        steps = [
            "Initialize weights $w$",
            "For each iteration $t$:",
            "  Compute gradient $g_t$",
            "  Update $w \\leftarrow w - \\eta g_t$",
            "Return $w$",
        ]
        
        result = await format_algorithm(
            steps=steps,
            caption="Gradient Descent",
        )
        
        data = json.loads(result)
        assert "\\begin{algorithm}" in data["latex"]
        assert "Gradient Descent" in data["latex"]


class TestFormatFigure:
    """Tests for LaTeX figure formatting."""
    
    @pytest.mark.asyncio
    async def test_figure_environment(self):
        """Should generate figure environment."""
        result = await format_figure(
            figure_path="figures/comparison.pdf",
            caption="Comparison of methods",
            label="fig:comparison",
        )
        
        data = json.loads(result)
        assert "\\begin{figure}" in data["latex"]
        assert "figures/comparison.pdf" in data["latex"]
        assert "Comparison of methods" in data["latex"]
    
    @pytest.mark.asyncio
    async def test_custom_width(self):
        """Should use custom width."""
        result = await format_figure(
            figure_path="fig.pdf",
            caption="Test",
            label="fig:test",
            width="0.5\\textwidth",
        )
        
        data = json.loads(result)
        assert "0.5\\textwidth" in data["latex"]


class TestEstimatePaperStructure:
    """Tests for paper structure estimation."""
    
    @pytest.mark.asyncio
    async def test_neurips_estimation(self):
        """Should estimate structure for NeurIPS."""
        result = await estimate_paper_structure(
            conference="neurips",
            target_pages=9,
        )
        
        data = json.loads(result)
        assert data["total_words"] > 4000
        assert "sections" in data
        assert "figures" in data
    
    @pytest.mark.asyncio
    async def test_section_percentages(self):
        """Should provide section word counts."""
        result = await estimate_paper_structure(
            conference="icml",
            target_pages=9,
        )
        
        data = json.loads(result)
        sections = data["sections"]
        assert "introduction" in sections
        assert "method" in sections
        assert "experiments" in sections


class TestValidateLatex:
    """Tests for LaTeX validation."""
    
    @pytest.mark.asyncio
    async def test_valid_latex(self):
        """Should pass valid LaTeX."""
        latex = r"""
        \begin{document}
        \section{Introduction}
        This is valid \LaTeX.
        \end{document}
        """
        
        result = await validate_latex(latex_content=latex)
        
        data = json.loads(result)
        assert data["valid"] is True
    
    @pytest.mark.asyncio
    async def test_unmatched_braces(self):
        """Should detect unmatched braces."""
        latex = r"\textbf{unclosed"
        
        result = await validate_latex(latex_content=latex)
        
        data = json.loads(result)
        assert data["valid"] is False
        assert len(data["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_unmatched_environments(self):
        """Should detect unmatched environments."""
        latex = r"\begin{equation} x = 1"  # Missing \end{equation}
        
        result = await validate_latex(latex_content=latex)
        
        data = json.loads(result)
        assert data["valid"] is False
