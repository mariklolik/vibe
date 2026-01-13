"""Integration tests for end-to-end workflows."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestIdeaSubmissionFlow:
    """Tests for the idea submission and approval flow."""
    
    @pytest.mark.asyncio
    async def test_full_idea_flow(self, sample_idea):
        """Test complete flow: submit → approve → proceed."""
        from src.tools.ideas import submit_idea, approve_idea
        from src.db.experiments_db import Idea
        
        # Mock the database
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.save_idea = AsyncMock()
            
            with patch("src.tools.ideas.compute_idea_novelty") as mock_novelty:
                mock_novelty.return_value = (0.85, [])
                
                # Step 1: Submit idea
                result = await submit_idea(
                    title=sample_idea["title"],
                    description=sample_idea["description"],
                    motivation=sample_idea["motivation"],
                )
        
        data = json.loads(result)
        assert data["success"] is True
        
        idea_id = data["idea_submitted"]["idea_id"]
        approval_command = data["approval_command"]
        
        # Extract confirmation code from command
        # Format: "APPROVE idea_xxx CODE 1234"
        code = approval_command.split("CODE ")[-1]
        
        # Step 2: Approve with correct code
        mock_idea = Idea(
            idea_id=idea_id,
            title=sample_idea["title"],
            description=sample_idea["description"],
            source_papers=[],
            hypotheses=[],
            research_plan={},
            novelty_score=0.85,
            created_at="2026-01-14",
            status="pending_approval",
            confirmation_code=code,
        )
        
        with patch("src.tools.ideas.experiments_db") as mock_db:
            mock_db.get_idea = AsyncMock(return_value=mock_idea)
            mock_db.save_idea = AsyncMock()
            
            with patch("src.tools.ideas.workflow_db") as mock_wf:
                mock_wf.update_idea_approval = AsyncMock()
                
                approval_result = await approve_idea(idea_id, code)
        
        approval_data = json.loads(approval_result)
        assert approval_data["success"] is True


class TestPaperGenerationFlow:
    """Tests for paper generation workflow."""
    
    @pytest.mark.asyncio
    async def test_table_and_figure_generation(self, sample_results):
        """Test generating tables and figures for paper."""
        from src.tools.writing import format_results_table
        from src.tools.visualization import plot_comparison_bar
        
        # Generate results table
        table_result = await format_results_table(results=sample_results)
        table_data = json.loads(table_result)
        
        assert "latex" in table_data
        assert "\\begin{table}" in table_data["latex"]
        
        # Generate comparison figure
        with patch("src.tools.visualization.plt") as mock_plt:
            mock_plt.savefig = MagicMock()
            mock_plt.figure = MagicMock()
            mock_plt.bar = MagicMock()
            mock_plt.close = MagicMock()
            
            with patch("src.tools.visualization.project_manager") as mock_pm:
                mock_pm.get_current_project = AsyncMock(return_value=None)
                
                with patch("src.tools.visualization.workflow_db") as mock_wf:
                    mock_wf.get_project_workflow = AsyncMock(return_value=None)
                    
                    fig_result = await plot_comparison_bar(
                        results={"method_a": 0.9, "method_b": 0.8},
                        metric="accuracy",
                    )
        
        # Should return path or success
        fig_data = json.loads(fig_result)
        assert "output_path" in fig_data or "success" in fig_data


class TestWorkflowEnforcement:
    """Tests for workflow enforcement across tools."""
    
    @pytest.mark.asyncio
    async def test_experiment_blocked_without_idea(self):
        """Experiments should be blocked without approved idea."""
        from src.tools.experiments import run_experiment
        from src.db.workflow import WorkflowState
        
        # Create workflow state without approved idea
        mock_workflow = WorkflowState(
            workflow_id="wf_123",
            project_id="test",
            stage="experimenting",
            approved_idea_id=None,  # No approved idea
        )
        
        mock_project = MagicMock()
        mock_project.project_id = "test"
        
        with patch("src.tools.experiments.project_manager") as mock_pm:
            mock_pm.get_current_project = AsyncMock(return_value=mock_project)
            
            with patch("src.tools.experiments.workflow_db") as mock_wf:
                mock_wf.get_project_workflow = AsyncMock(return_value=mock_workflow)
                mock_wf.validate_action = MagicMock(return_value=(False, "Approve idea first"))
                mock_wf.get_missing_prerequisites = MagicMock(return_value=[
                    {"tool": "approve_idea", "description": "Approve an idea first"}
                ])
                
                result = await run_experiment(script="train.py")
        
        data = json.loads(result)
        assert data["success"] is False
        assert "BLOCKED" in data.get("error", "")


class TestConferenceFormatting:
    """Tests for conference-specific formatting."""
    
    @pytest.mark.asyncio
    async def test_list_conferences(self):
        """Should list all supported conferences."""
        from src.tools.formatting import list_conferences
        
        result = await list_conferences()
        data = json.loads(result)
        
        assert "conferences" in data
        conference_names = [c["name"] for c in data["conferences"]]
        
        # Should include major conferences
        assert "neurips" in conference_names or "NeurIPS" in str(data)
        assert "icml" in conference_names or "ICML" in str(data)
    
    @pytest.mark.asyncio
    async def test_get_conference_requirements(self):
        """Should return conference requirements."""
        from src.tools.formatting import get_conference_requirements
        
        result = await get_conference_requirements(conference="neurips")
        data = json.loads(result)
        
        assert "page_limit" in data or "requirements" in data
