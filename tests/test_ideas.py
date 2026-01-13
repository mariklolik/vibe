"""Tests for idea generation and approval workflow."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from src.tools.ideas import (
    submit_idea,
    approve_idea,
    reject_idea,
    _extract_themes,
    _generate_confirmation_code,
)


class TestConfirmationCode:
    """Tests for confirmation code generation."""
    
    def test_code_is_4_digits(self):
        """Confirmation code should be exactly 4 digits."""
        code = _generate_confirmation_code()
        assert len(code) == 4
        assert code.isdigit()
    
    def test_code_in_valid_range(self):
        """Code should be between 1000 and 9999."""
        for _ in range(100):
            code = int(_generate_confirmation_code())
            assert 1000 <= code <= 9999


class TestThemeExtraction:
    """Tests for theme extraction from text."""
    
    def test_efficiency_theme(self):
        """Should detect efficiency theme."""
        text = "We propose an efficient method for faster processing"
        themes = _extract_themes(text)
        assert "efficiency" in themes
    
    def test_attention_theme(self):
        """Should detect attention theme."""
        text = "Self-attention mechanism in transformer models"
        themes = _extract_themes(text)
        assert "attention" in themes
    
    def test_sparsity_theme(self):
        """Should detect sparsity theme."""
        text = "Sparse attention reduces computation through pruning"
        themes = _extract_themes(text)
        assert "sparsity" in themes
    
    def test_default_theme(self):
        """Should return default theme for unrecognized text."""
        text = "Random unrelated content here"
        themes = _extract_themes(text)
        assert "novel_method" in themes


class TestSubmitIdea:
    """Tests for idea submission."""
    
    @pytest.mark.asyncio
    async def test_submit_returns_json(self, sample_idea):
        """submit_idea should return valid JSON."""
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.save_idea = AsyncMock()
            with patch("src.tools.ideas.compute_idea_novelty") as mock_novelty:
                mock_novelty.return_value = (0.85, [])
                
                result = await submit_idea(
                    title=sample_idea["title"],
                    description=sample_idea["description"],
                    motivation=sample_idea["motivation"],
                )
        
        data = json.loads(result)
        assert data["success"] is True
        assert "idea_id" in data["idea_submitted"]
    
    @pytest.mark.asyncio
    async def test_submit_includes_approval_command(self, sample_idea):
        """Submitted idea should include approval command."""
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.save_idea = AsyncMock()
            with patch("src.tools.ideas.compute_idea_novelty") as mock_novelty:
                mock_novelty.return_value = (0.85, [])
                
                result = await submit_idea(
                    title=sample_idea["title"],
                    description=sample_idea["description"],
                    motivation=sample_idea["motivation"],
                )
        
        data = json.loads(result)
        assert "approval_command" in data
        assert "APPROVE" in data["approval_command"]
        assert "CODE" in data["approval_command"]


class TestApproveIdea:
    """Tests for idea approval."""
    
    @pytest.mark.asyncio
    async def test_approve_requires_code(self):
        """Approval should fail without confirmation code."""
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.get_idea = AsyncMock(return_value=None)
            
            result = await approve_idea("idea_123", "")
        
        data = json.loads(result)
        assert "error" in data or "CONFIRMATION CODE REQUIRED" in str(data)
    
    @pytest.mark.asyncio
    async def test_approve_validates_code(self):
        """Approval should fail with wrong code."""
        from src.db.experiments_db import Idea
        
        mock_idea = Idea(
            idea_id="idea_123",
            title="Test Idea",
            description="Test description",
            source_papers=[],
            hypotheses=[],
            research_plan={},
            novelty_score=0.8,
            created_at="2026-01-14",
            status="pending_approval",
            confirmation_code="1234",
        )
        
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.get_idea = AsyncMock(return_value=mock_idea)
            
            result = await approve_idea("idea_123", "9999")  # Wrong code
        
        data = json.loads(result)
        assert "error" in data
        assert "INVALID" in data["error"] or "does not match" in str(data)


class TestRejectIdea:
    """Tests for idea rejection."""
    
    @pytest.mark.asyncio
    async def test_reject_with_reason(self):
        """Rejection should record the reason."""
        from src.db.experiments_db import Idea
        
        mock_idea = Idea(
            idea_id="idea_123",
            title="Test Idea",
            description="Test description",
            source_papers=[],
            hypotheses=[],
            research_plan={},
            novelty_score=0.8,
            created_at="2026-01-14",
            status="pending_approval",
        )
        
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.get_idea = AsyncMock(return_value=mock_idea)
            mock_db.save_idea = AsyncMock()
            
            result = await reject_idea("idea_123", "Too similar to existing work")
        
        data = json.loads(result)
        assert data["success"] is True
