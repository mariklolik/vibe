"""Tests for workflow orchestration."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.db.workflow import WorkflowDB, WorkflowState


class TestWorkflowState:
    """Tests for workflow state management."""
    
    def test_initial_stage(self):
        """New workflow should start at context_gathering."""
        state = WorkflowState(
            workflow_id="wf_123",
            project_id="test_project",
        )
        assert state.stage == "context_gathering"
    
    def test_progress_summary(self):
        """Should provide progress summary."""
        state = WorkflowState(
            workflow_id="wf_123",
            project_id="test_project",
            stage="experimenting",
            approved_idea_id="idea_123",
        )
        
        summary = state.get_progress_summary()
        assert "stage" in summary
        assert "progress_percent" in summary


class TestWorkflowDB:
    """Tests for workflow database operations."""
    
    def test_validate_action_without_approval(self):
        """Should block experiments without approved idea."""
        db = WorkflowDB()
        
        state = WorkflowState(
            workflow_id="wf_123",
            project_id="test_project",
            stage="experimenting",
            approved_idea_id=None,  # No approved idea
        )
        
        is_valid, error = db.validate_action(state, "run_experiment")
        
        assert is_valid is False
        assert "idea" in error.lower() or "approval" in error.lower()
    
    def test_validate_action_with_approval(self):
        """Should allow experiments with approved idea."""
        db = WorkflowDB()
        
        state = WorkflowState(
            workflow_id="wf_123",
            project_id="test_project",
            stage="experimenting",
            approved_idea_id="idea_123",
            env_created=True,
            datasets_setup=True,
        )
        
        is_valid, error = db.validate_action(state, "run_experiment")
        
        assert is_valid is True
    
    def test_get_missing_prerequisites(self):
        """Should identify missing prerequisites."""
        db = WorkflowDB()
        
        state = WorkflowState(
            workflow_id="wf_123",
            project_id="test_project",
            stage="experimenting",
            approved_idea_id="idea_123",
            env_created=False,  # Missing
            datasets_setup=False,  # Missing
        )
        
        missing = db.get_missing_prerequisites(state, "run_experiment")
        
        assert len(missing) > 0
        prereq_tools = [p["tool"] for p in missing]
        assert "create_experiment_env" in prereq_tools or "env_created" in str(missing)


class TestWorkflowStages:
    """Tests for workflow stage transitions."""
    
    def test_stage_order(self):
        """Workflow stages should be in correct order."""
        db = WorkflowDB()
        
        stages = db.WORKFLOW_STAGES
        
        assert stages.index("context_gathering") < stages.index("idea_generation")
        assert stages.index("idea_generation") < stages.index("idea_approval")
        assert stages.index("idea_approval") < stages.index("experiment_setup")
        assert stages.index("experiment_setup") < stages.index("experimenting")
        assert stages.index("experimenting") < stages.index("analysis")
        assert stages.index("analysis") < stages.index("writing")
        assert stages.index("writing") < stages.index("formatting")
    
    def test_approval_required_stages(self):
        """Certain stages should require user approval."""
        db = WorkflowDB()
        
        assert "idea_approval" in db.APPROVAL_REQUIRED_STAGES


class TestActionPrerequisites:
    """Tests for action prerequisites."""
    
    def test_run_experiment_prerequisites(self):
        """run_experiment should require approved idea and env."""
        db = WorkflowDB()
        
        prereqs = db.ACTION_PREREQUISITES.get("run_experiment", {})
        
        assert "idea_approved" in prereqs.get("requires", [])
        assert "env_created" in prereqs.get("requires", [])
    
    def test_cast_to_format_prerequisites(self):
        """cast_to_format should require paper and figures."""
        db = WorkflowDB()
        
        prereqs = db.ACTION_PREREQUISITES.get("cast_to_format", {})
        
        requires = prereqs.get("requires", [])
        assert "paper_drafted" in requires or "figures_generated" in requires
