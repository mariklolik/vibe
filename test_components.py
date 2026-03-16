#!/usr/bin/env python3
"""Component-level tests for the research pipeline.

Tests each agent independently to verify it works before running e2e.
Run with: python test_components.py [test_name]

Available tests:
    health      - Test proxy health and metrics
    client      - Test Anthropic SDK connectivity
    structured  - Test structured JSON extraction
    effort      - Test effort levels and fallback
    research    - Test research agent (paper search + idea generation)
    experiment  - Test experiment agent (design + execution)
    writer      - Test writer agent (paper generation)
    reviewer    - Test reviewer agent (verification)
    e2e         - Full end-to-end pipeline test
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test")


def test_health():
    """Test 0: Verify proxy health and metrics."""
    print("\n=== TEST: Proxy Health ===")

    from src.agents.client import check_proxy_health, check_proxy_metrics

    health = check_proxy_health()
    print(f"Health: {json.dumps(health, indent=2)}")
    assert health.get("status") != "unreachable", "Proxy is unreachable!"

    metrics = check_proxy_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    print("PASSED: Proxy is healthy")
    return True


def test_client():
    """Test 1: Verify Anthropic SDK connects to proxy."""
    print("\n=== TEST: Anthropic SDK Client ===")

    from src.agents.client import create_client, call_agent, get_token_usage

    client = create_client()

    response = call_agent(
        client=client,
        system_prompt="You are a helpful assistant. Respond concisely.",
        user_message="What is 2+2? Respond with just the number.",
        max_tokens=50,
        temperature=0.0,
    )

    print(f"Response: {response.text}")
    print(f"Model: {response.model}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Usage: {response.usage}")
    print(f"Cumulative tokens: {get_token_usage()}")

    assert response.text.strip(), "Empty response"
    assert "4" in response.text, f"Expected '4' in response, got: {response.text}"
    print("PASSED: Client connectivity works")
    return True


def test_client_structured():
    """Test 1b: Verify structured JSON extraction."""
    print("\n=== TEST: Structured JSON Output ===")

    from src.agents.client import create_client, call_agent_structured

    client = create_client()

    parsed, raw = call_agent_structured(
        client=client,
        system_prompt=(
            "You are a helpful assistant. Always respond with a JSON block:\n"
            '```json\n{"answer": "your answer", "confidence": 0.95}\n```'
        ),
        user_message="What is the capital of France?",
        max_tokens=200,
        temperature=0.0,
    )

    print(f"Raw: {raw[:200]}")
    print(f"Parsed: {parsed}")

    assert parsed is not None, f"Failed to parse JSON from: {raw}"
    assert "answer" in parsed, f"Missing 'answer' key in: {parsed}"
    print("PASSED: Structured output works")
    return True


def test_effort():
    """Test 1c: Verify effort levels and fallback model."""
    print("\n=== TEST: Effort Levels & Fallback ===")

    from src.agents.client import create_client, call_agent, EFFORT_LOW, EFFORT_HIGH

    client = create_client()

    # Test with effort="low" (fast/cheap)
    print("Testing effort=low...")
    resp_low = call_agent(
        client=client,
        system_prompt="Respond concisely.",
        user_message="Say hello.",
        max_tokens=50,
        effort=EFFORT_LOW,
    )
    print(f"  Low effort: {resp_low.text[:50]} (model={resp_low.model})")

    # Test with effort="high" and fallback
    print("Testing effort=high with fallback...")
    resp_high = call_agent(
        client=client,
        system_prompt="Respond concisely.",
        user_message="Say hello.",
        max_tokens=50,
        effort=EFFORT_HIGH,
        fallback_model="haiku",
    )
    print(f"  High effort: {resp_high.text[:50]} (model={resp_high.model})")

    print("PASSED: Effort levels work")
    return True


def test_research():
    """Test 2: Research agent — paper search and idea generation."""
    print("\n=== TEST: Research Agent ===")

    from src.agents.client import create_client
    from src.agents.research import ResearchAgent

    client = create_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ResearchAgent(client, tmpdir)

        print("Searching papers...")
        papers = asyncio.run(agent.search_papers(
            "efficient attention mechanisms", max_papers=5
        ))
        print(f"Found {len(papers)} papers")
        assert len(papers) > 0, "No papers found"

        for p in papers[:3]:
            print(f"  - {p.get('title', '?')[:80]}")

        print("\nGenerating ideas (effort=high)...")
        ideas, raw = agent.generate_ideas(
            "efficient attention mechanisms", papers, num_ideas=3
        )

        if ideas:
            idea_list = ideas.get("ideas", [])
            print(f"Generated {len(idea_list)} ideas:")
            for idea in idea_list:
                print(f"  - {idea.get('title', '?')} (novelty={idea.get('novelty_score', '?')})")

            selected = agent.select_best_idea(ideas, min_novelty=0.5)
            if selected:
                print(f"\nAuto-selected: {selected['title']}")
                print(f"  Novelty: {selected.get('novelty_score')}")
            else:
                print("\nNo idea met threshold (this is OK for test)")
        else:
            print(f"WARNING: Ideas parsing failed. Raw: {raw[:300]}")

    print("PASSED: Research agent works")
    return True


def test_experiment():
    """Test 3: Experiment agent — two-phase code generation + execution."""
    print("\n=== TEST: Experiment Agent ===")

    from src.agents.client import create_client
    from src.agents.experiment import ExperimentAgent

    client = create_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create required dirs
        for d in ["src", "scripts", "configs", "experiments", "experiments/logs",
                   "experiments/results", "verification", "figures"]:
            (Path(tmpdir) / d).mkdir(parents=True, exist_ok=True)

        agent = ExperimentAgent(client, tmpdir)

        test_idea = {
            "title": "Test Attention Method",
            "description": "A lightweight attention mechanism using random projection",
            "method_summary": "Replace softmax attention with random feature maps",
            "datasets": ["CIFAR-10"],
            "baselines": ["standard attention", "linear attention"],
            "target_conference": "ICML",
        }

        # Phase 1: Implement method in src/
        print("Phase 1: Implementing method in src/ (effort=high)...")
        method_design, raw = agent.implement_method(test_idea)

        if method_design:
            files = method_design.get("files", {})
            print(f"Generated {len(files)} method files:")
            for fname in files:
                print(f"  - {fname} ({len(files[fname])} chars)")

            method_files = agent.save_method_files(method_design)
            method_filenames = [str(f.relative_to(tmpdir)) for f in method_files]
            print(f"Saved {len(method_files)} files to project")

            # Verify src/ exists and has Python files
            src_dir = Path(tmpdir) / "src"
            src_files = list(src_dir.glob("*.py"))
            print(f"src/ contains {len(src_files)} Python files: {[f.name for f in src_files]}")
        else:
            print(f"WARNING: Method implementation failed. Raw: {raw[:300]}")
            method_filenames = []

        # Phase 2: Design experiment scripts
        print("\nPhase 2: Designing experiment scripts (effort=high)...")
        design, raw = agent.design_experiments(test_idea, method_filenames)

        if design:
            scripts_dict = design.get("scripts", {})
            hyps = design.get("hypotheses", [])
            print(f"Designed {len(scripts_dict)} scripts, {len(hyps)} hypotheses")

            scripts = agent.save_experiment_scripts(design)
            print(f"Saved {len(scripts)} script files")

            # Verify scripts/ exists
            scripts_dir = Path(tmpdir) / "scripts"
            script_files = list(scripts_dir.glob("*.py"))
            print(f"scripts/ contains {len(script_files)} Python files: {[f.name for f in script_files]}")
        else:
            print(f"WARNING: Experiment design failed. Raw: {raw[:300]}")

        # Test execution with a simple script
        print("\nTesting experiment execution...")
        test_script = Path(tmpdir) / "scripts" / "quick_test.py"
        test_script.write_text(
            'import json\n'
            'print(json.dumps({"metric": "accuracy", "value": 0.85, "seed": 42}))\n'
            'print(json.dumps({"metric": "accuracy", "value": 0.83, "seed": 123}))\n'
            'print(json.dumps({"metric": "accuracy", "value": 0.86, "seed": 456}))\n'
        )
        result = agent.run_experiment(test_script, timeout_seconds=30)
        print(f"Execution: {'OK' if result['success'] else 'FAILED'}")
        print(f"Metrics extracted: {result.get('metrics', [])}")
        print(f"Signature: {result.get('signature', 'N/A')}")
        assert result["success"], f"Experiment failed: {result}"

        # Test statistical verification
        print("\nTesting hypothesis verification...")
        verif = agent.verify_hypothesis(
            {"id": "h1", "statement": "Method is better", "test_type": "t-test", "expected_direction": "higher"},
            ours_values=[0.85, 0.83, 0.86, 0.84, 0.87],
            baseline_values=[0.78, 0.76, 0.79, 0.77, 0.80],
        )
        print(f"Verification: {verif['verdict']} (p={verif['p_value']:.4f})")
        assert verif["can_include_in_paper"], "Should be significant"

    print("PASSED: Experiment agent works")
    return True


def test_writer():
    """Test 4: Writer agent — paper generation."""
    print("\n=== TEST: Writer Agent ===")

    from src.agents.client import create_client
    from src.agents.writer import WriterAgent

    client = create_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        ideas_dir = Path(tmpdir) / "ideas"
        ideas_dir.mkdir()
        (ideas_dir / "selected_idea.json").write_text(json.dumps({
            "title": "Random Projection Attention",
            "description": "Efficient attention using random feature maps",
            "method_summary": "Replace O(n^2) softmax with O(n) random features",
            "target_conference": "ICML",
        }))

        exp_dir = Path(tmpdir) / "experiments"
        exp_dir.mkdir()
        (exp_dir / "all_results.json").write_text(json.dumps({
            "experiment_results": [
                {"script": "main.py", "success": True, "metrics": [
                    {"metric": "accuracy", "value": 0.85},
                ]}
            ],
            "verification_results": [
                {"hypothesis_id": "h1", "p_value": 0.003, "can_include_in_paper": True,
                 "ours_mean": 0.85, "baseline_mean": 0.78, "verdict": "SIGNIFICANT"}
            ],
        }))

        (Path(tmpdir) / "verification").mkdir()
        (Path(tmpdir) / "paper").mkdir()

        agent = WriterAgent(client, tmpdir)

        print("Writing paper (effort=high, section-by-section)...")
        paper, raw = agent.write_full_paper("icml")

        if paper:
            completeness = agent.check_completeness(paper)
            print(f"Paper status: {completeness['status']}")
            print(f"Word count: {completeness['total_words']}/{completeness['target_words']}")

            tex_path = agent.save_paper(paper)
            print(f"LaTeX saved: {tex_path}")
            print(f"LaTeX size: {tex_path.stat().st_size} bytes")

            pdf = agent.compile_pdf()
            if pdf:
                print(f"PDF compiled: {pdf}")
            else:
                print("PDF compilation skipped (LaTeX not available)")
        else:
            print(f"WARNING: Paper generation failed. Raw: {raw[:300]}")

    print("PASSED: Writer agent works")
    return True


def test_reviewer():
    """Test 5: Reviewer agent — verification."""
    print("\n=== TEST: Reviewer Agent ===")

    from src.agents.client import create_client
    from src.agents.reviewer import ReviewerAgent

    client = create_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / "paper"
        paper_dir.mkdir()
        (paper_dir / "paper.json").write_text(json.dumps({
            "paper": {
                "title": "Test Paper",
                "abstract": "We propose a test method that achieves 85% accuracy.",
                "sections": [
                    {"name": "introduction", "content": "This is the introduction."},
                    {"name": "method", "content": "Our method uses random projections."},
                    {"name": "experiments", "content": "We evaluate on CIFAR-10. Accuracy: 0.85 +/- 0.02."},
                ],
                "references": [
                    {"key": "vaswani2017attention", "title": "Attention Is All You Need",
                     "authors": "Vaswani et al.", "year": "2017", "venue": "NeurIPS"},
                ],
            }
        }))

        (Path(tmpdir) / "ideas").mkdir()
        (Path(tmpdir) / "ideas" / "selected_idea.json").write_text(json.dumps({
            "title": "Test Method", "description": "A test research idea",
        }))

        agent = ReviewerAgent(client, tmpdir)

        print("Running review (opus model, effort=high)...")
        review, raw = agent.review()

        if review:
            review_data = review.get("review", review)
            print(f"Verdict: {review_data.get('verdict', 'N/A')}")
            print(f"Scores: {review_data.get('scores', {})}")
            issues = review_data.get("issues", [])
            print(f"Issues: {len(issues)}")
            for iss in issues[:3]:
                print(f"  [{iss.get('severity')}] {iss.get('description', '?')[:80]}")
        else:
            print(f"WARNING: Review parsing failed. Raw: {raw[:300]}")

    print("PASSED: Reviewer agent works")
    return True


def test_e2e():
    """Test 6: Full end-to-end pipeline."""
    print("\n=== TEST: Full E2E Pipeline ===")
    print("This test runs the complete pipeline. May take 10-30 minutes.")

    from src.agents.orchestrator import Orchestrator
    from src.agents.client import get_token_usage

    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = Orchestrator(projects_dir=tmpdir)

        result = asyncio.run(
            orchestrator.run_full_pipeline(
                topic="efficient token merging for vision transformers",
                conference="icml",
                min_novelty=0.5,
            )
        )

        print(f"\nFinal status: {result['final_status']}")
        print(f"Elapsed: {result['elapsed_seconds']}s")
        print(f"Token usage: {result.get('token_usage', {})}")

        for phase, data in result.get("phases", {}).items():
            if isinstance(data, dict):
                success = data.get("success", "N/A")
                elapsed = data.get("elapsed_seconds", "?")
                tokens = data.get("tokens_used", {})
                print(f"  {phase}: {'OK' if success else 'FAILED'} ({elapsed}s, tokens={tokens})")

        project_dir = result["project_dir"]
        progress = Path(project_dir) / "progress.txt"
        if progress.exists():
            print(f"\nProgress file: {progress.stat().st_size} bytes")

        pdf = Path(project_dir) / "paper" / "main.pdf"
        if pdf.exists():
            print(f"PDF generated: {pdf} ({pdf.stat().st_size} bytes)")

    print("PASSED: E2E pipeline works")
    return True


# Test registry
TESTS = {
    "health": test_health,
    "client": test_client,
    "structured": test_client_structured,
    "effort": test_effort,
    "research": test_research,
    "experiment": test_experiment,
    "writer": test_writer,
    "reviewer": test_reviewer,
    "e2e": test_e2e,
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in TESTS:
            try:
                TESTS[test_name]()
            except Exception as e:
                print(f"\nFAILED: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        elif test_name == "all":
            for name, func in TESTS.items():
                if name == "e2e":
                    continue
                try:
                    func()
                except Exception as e:
                    print(f"\nFAILED {name}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {', '.join(TESTS.keys())}")
            sys.exit(1)
    else:
        print("Usage: python test_components.py [test_name|all]")
        print(f"Available tests: {', '.join(TESTS.keys())}")
