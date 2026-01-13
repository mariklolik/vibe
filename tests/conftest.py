"""Pytest configuration and fixtures."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_paper():
    """Sample paper data for testing."""
    return {
        "id": "arxiv:2401.00001",
        "title": "Test Paper on Efficient Attention",
        "abstract": (
            "We propose a novel attention mechanism that reduces "
            "computational complexity from O(n²) to O(n log n). "
            "Our method achieves state-of-the-art performance on "
            "multiple benchmarks while maintaining efficiency."
        ),
        "authors": ["Alice Smith", "Bob Jones"],
        "categories": ["cs.LG", "cs.CL"],
    }


@pytest.fixture
def sample_idea():
    """Sample idea data for testing."""
    return {
        "title": "Unified Efficient Attention Framework",
        "description": (
            "We propose combining sparse attention with linear attention "
            "for a unified framework that adapts to different sequence lengths."
        ),
        "motivation": (
            "Current methods are either sparse or linear but not adaptive. "
            "We bridge this gap with a routing mechanism."
        ),
        "source_papers": ["arxiv:2401.00001"],
    }


@pytest.fixture
def sample_results():
    """Sample experiment results for testing."""
    return {
        "our_method": {
            "accuracy": 0.92,
            "f1": 0.91,
            "latency_ms": 15.2,
        },
        "baseline_1": {
            "accuracy": 0.85,
            "f1": 0.84,
            "latency_ms": 45.0,
        },
        "baseline_2": {
            "accuracy": 0.88,
            "f1": 0.87,
            "latency_ms": 32.0,
        },
    }
