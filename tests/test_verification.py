"""Tests for statistical verification tools."""

import json
import pytest

from src.tools.verification import (
    check_significance,
    compare_to_baselines,
    compute_statistics,
    detect_anomalies,
)


class TestCheckSignificance:
    """Tests for statistical significance checking."""
    
    @pytest.mark.asyncio
    async def test_significant_difference(self, sample_results):
        """Should detect significant difference between methods."""
        results = {
            "our_method": [0.92, 0.91, 0.93, 0.90, 0.94],
            "baseline": [0.82, 0.81, 0.83, 0.80, 0.84],
        }
        
        result = await check_significance(
            method1="our_method",
            method2="baseline",
            results=results,
        )
        
        data = json.loads(result)
        assert "p_value" in data
        assert "significant" in data
    
    @pytest.mark.asyncio
    async def test_no_significant_difference(self):
        """Should detect no significant difference for similar results."""
        results = {
            "method_a": [0.90, 0.91, 0.89, 0.90, 0.91],
            "method_b": [0.90, 0.90, 0.91, 0.89, 0.90],
        }
        
        result = await check_significance(
            method1="method_a",
            method2="method_b",
            results=results,
        )
        
        data = json.loads(result)
        assert data["p_value"] > 0.05  # Not significant


class TestCompareToBaselines:
    """Tests for baseline comparison."""
    
    @pytest.mark.asyncio
    async def test_compare_multiple_baselines(self):
        """Should compare against multiple baselines."""
        results = {
            "our_method": {"accuracy": 0.95, "f1": 0.94},
            "baseline_1": {"accuracy": 0.85, "f1": 0.84},
            "baseline_2": {"accuracy": 0.88, "f1": 0.87},
        }
        
        result = await compare_to_baselines(
            method="our_method",
            baselines=["baseline_1", "baseline_2"],
            results=results,
        )
        
        data = json.loads(result)
        assert "comparisons" in data
        assert len(data["comparisons"]) == 2
    
    @pytest.mark.asyncio
    async def test_calculates_improvement(self):
        """Should calculate improvement over baselines."""
        results = {
            "ours": {"accuracy": 0.90},
            "baseline": {"accuracy": 0.80},
        }
        
        result = await compare_to_baselines(
            method="ours",
            baselines=["baseline"],
            results=results,
        )
        
        data = json.loads(result)
        comparison = data["comparisons"][0]
        # 0.90 vs 0.80 = 12.5% improvement
        assert comparison["improvement_percent"]["accuracy"] == pytest.approx(12.5, rel=0.1)


class TestComputeStatistics:
    """Tests for statistics computation."""
    
    @pytest.mark.asyncio
    async def test_compute_mean_std(self):
        """Should compute mean and standard deviation."""
        results = {
            "method": [0.90, 0.92, 0.88, 0.91, 0.89],
        }
        
        result = await compute_statistics(results=results)
        
        data = json.loads(result)
        stats = data["method"]
        assert stats["mean"] == pytest.approx(0.90, rel=0.01)
        assert "std" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats
    
    @pytest.mark.asyncio
    async def test_confidence_interval(self):
        """Should compute confidence interval."""
        results = {
            "method": [0.90, 0.90, 0.90, 0.90, 0.90],  # No variance
        }
        
        result = await compute_statistics(results=results, confidence_level=0.95)
        
        data = json.loads(result)
        stats = data["method"]
        # With no variance, CI should be tight around mean
        assert stats["ci_lower"] == pytest.approx(0.90, rel=0.01)
        assert stats["ci_upper"] == pytest.approx(0.90, rel=0.01)


class TestDetectAnomalies:
    """Tests for anomaly detection."""
    
    @pytest.mark.asyncio
    async def test_detect_outlier(self):
        """Should detect outliers in results."""
        results = {
            "runs": [0.90, 0.91, 0.89, 0.90, 0.50],  # 0.50 is outlier
        }
        
        result = await detect_anomalies(results=results, threshold=2.0)
        
        data = json.loads(result)
        assert data["anomalies_detected"] is True
        assert len(data["anomalies"]) > 0
    
    @pytest.mark.asyncio
    async def test_no_anomalies(self):
        """Should report no anomalies for consistent results."""
        results = {
            "runs": [0.90, 0.91, 0.89, 0.90, 0.91],
        }
        
        result = await detect_anomalies(results=results, threshold=2.0)
        
        data = json.loads(result)
        assert data["anomalies_detected"] is False
