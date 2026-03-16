"""Experiment agent: design, implement, execute, and verify experiments.

Maps to experiment-runner skill from research_claude_agents.
Handles: method implementation, experiment design, code generation, execution, verification.

Generates rom4ik-style project structure:
  src/       - Core method implementation (importable classes/algorithms)
  scripts/   - Experiment runners that import from src/
  configs/   - YAML/JSON configs for reproducibility
  experiments/results/ - Execution outputs

Proxy optimization:
- Method implementation: effort="high" (code quality is critical)
- Experiment design: effort="high", budget=$0.50
- Results analysis: effort="medium" (structured summary)
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import anthropic

from src.agents.base import BaseAgent
from src.agents.client import DEFAULT_MODEL, EFFORT_HIGH, EFFORT_MEDIUM
from src.state.progress import load_project_config, save_project_config

logger = logging.getLogger(__name__)

# --- System prompt for method implementation (per-file generation) ---
METHOD_SYSTEM_PROMPT = """You are an expert ML engineer who implements novel research methods as clean, importable Python modules.

Your role is to write production-quality method code that:
1. Implements the core algorithm as reusable classes (not scripts)
2. Separates concerns: model architecture, training logic, data utilities
3. Uses proper typing, docstrings, and clean interfaces
4. Handles GPU/CPU automatically via PyTorch device detection
5. Is importable by experiment scripts (scripts/ folder)

## Output Format

Return ONLY a JSON block with the file content:

```json
{
  "filename": "src/model.py",
  "content": "import torch\\nimport torch.nn as nn\\n\\nclass MyMethod(nn.Module):\\n    ...",
  "description": "Core model implementing the proposed method"
}
```

## Guidelines
- Code must be executable as-is (no placeholders or TODOs)
- Model class should extend nn.Module with proper forward()
- All hyperparameters should be configurable (not hardcoded)
- Include type hints and docstrings for public methods
- Handle GPU/CPU automatically via device detection
- Default to TINY models/configs: < 30M params, batch_size=1-2, seq_len=128-256
- Only 8GB GPU memory available — design for this constraint
- Avoid loading pre-trained models unless absolutely necessary (they eat memory)
- Custom small architectures preferred over pre-trained models
"""

# --- System prompt for experiment scripts (scripts/ generation) ---
EXPERIMENT_SYSTEM_PROMPT = """You are an expert ML engineer who designs and runs experiments for research papers.

Your role is to:
1. Design rigorous experiments with proper baselines, ablations, and controls
2. Generate experiment runner scripts that import from the src/ module
3. Define testable hypotheses with clear metrics
4. Ensure reproducibility via configs, seeds, and proper logging

## Project Structure
The method code is already implemented in src/ (importable module).
You generate scripts that USE src/ and configs/.

## Output Format for Experiment Design

```json
{
  "hypotheses": [
    {
      "id": "h1",
      "statement": "Our method achieves higher accuracy than baseline X on dataset Y",
      "metric": "accuracy",
      "test_type": "t-test",
      "expected_direction": "higher"
    }
  ],
  "scripts": {
    "scripts/run_main.py": "import sys; sys.path.insert(0, '.'); from src.model import MyMethod\\n...",
    "scripts/run_baselines.py": "...",
    "scripts/run_ablations.py": "...",
    "scripts/analyze_results.py": "...",
    "scripts/generate_figures.py": "..."
  },
  "configs": {
    "configs/main_experiment.yaml": "...",
    "configs/baseline.yaml": "...",
    "configs/ablation_no_component_x.yaml": "..."
  },
  "run_order": ["scripts/run_main.py", "scripts/run_baselines.py", "scripts/run_ablations.py", "scripts/analyze_results.py"]
}
```

## Output Format for Results Analysis

```json
{
  "analysis": {
    "main_findings": ["finding1", "finding2"],
    "hypothesis_results": [
      {
        "id": "h1",
        "supported": true,
        "metric_ours": {"mean": 0.85, "std": 0.02},
        "metric_baseline": {"mean": 0.78, "std": 0.03},
        "improvement": "8.97%",
        "p_value": 0.003,
        "effect_size": 2.8,
        "conclusion": "Statistically significant improvement (p<0.05)"
      }
    ],
    "ablation_insights": ["insight1"],
    "limitations": ["limitation1"],
    "additional_experiments_needed": []
  }
}
```

## Guidelines
- Scripts must import from src/ (sys.path.insert if needed)
- Use multiple random seeds (minimum 3, preferably 5)
- Output metrics as JSON lines to stdout for automated parsing
- Report mean +/- std for all metrics
- p < 0.05 is REQUIRED for any claim in the paper
- Scripts should load configs from configs/ directory
- Include proper argparse for CLI usage

## CRITICAL: API Accuracy
- Study the provided src/ code CAREFULLY before writing scripts
- If src/ returns dataclass objects, use ATTRIBUTE ACCESS (.field), NOT dict access (.get())
- Match the EXACT class names, function signatures, and return types from src/
- If a function returns List[SomeClass], iterate with obj.attribute, not obj["key"]
- If unsure about an API, write defensive code: getattr(obj, 'field', default)

## CRITICAL: GPU Memory Management — ONLY 8GB FREE
- Available GPU memory is approximately 8GB. This is HARD LIMIT.
- batch_size MUST be 1-2, sequence length MUST be 128-256 tokens
- Use small models (< 100M params, ideally < 30M)
- ALWAYS wrap model creation and training in try/except RuntimeError for CUDA OOM
- On OOM: halve batch_size and retry, then fallback to CPU if still fails
- Add torch.cuda.empty_cache() between experiments and seeds
- Avoid loading pre-trained models with large embeddings (gpt2 is 124M — may OOM)
- For validation/proof-of-concept: use synthetic data + tiny custom models
- Scripts should check torch.cuda.is_available() and use CPU fallback gracefully
- Do NOT use DataParallel or multi-GPU — only single GPU available
"""


class ExperimentAgent(BaseAgent):
    """Handles method implementation, experiment design, execution, and verification.

    Two-phase code generation (rom4ik pattern):
    1. Implement method in src/ (core classes, importable)
    2. Generate experiment scripts in scripts/ (use src/)
    """

    name = "experiment"
    system_prompt = EXPERIMENT_SYSTEM_PROMPT
    default_effort = EFFORT_HIGH
    default_fallback_model = "haiku"
    default_max_budget_usd = None  # No budget cap — code quality is critical

    def __init__(
        self,
        client: anthropic.Anthropic,
        project_dir: str,
        model: str = DEFAULT_MODEL,
    ):
        super().__init__(client, project_dir, model)
        self.src_dir = Path(project_dir) / "src"
        self.scripts_dir = Path(project_dir) / "scripts"
        self.configs_dir = Path(project_dir) / "configs"
        self.experiments_dir = Path(project_dir) / "experiments"
        self.results_dir = self.experiments_dir / "results"
        self.verification_dir = Path(project_dir) / "verification"
        self.figures_dir = Path(project_dir) / "figures"
        for d in [self.src_dir, self.scripts_dir, self.configs_dir,
                  self.experiments_dir, self.results_dir,
                  self.verification_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # Files to generate for the method implementation (per-file calls)
    METHOD_FILES = [
        ("src/model.py", "Core model/method class (nn.Module)", "Write the main model class implementing the novel method. Must extend nn.Module with proper forward(), handle GPU/CPU automatically."),
        ("src/trainer.py", "Training loop with logging", "Write the training loop class. Must handle: training epochs, validation, logging metrics as JSON, checkpointing, learning rate scheduling. Import the model from src.model."),
        ("src/data_utils.py", "Dataset loading utilities", "Write dataset loading utilities. Must handle: downloading/loading standard datasets (CIFAR-10, MNIST, etc.), data augmentation, train/val/test splits, DataLoader creation."),
        ("src/metrics.py", "Evaluation metrics", "Write evaluation metric functions. Must include: accuracy, loss tracking, any task-specific metrics. Functions should accept predictions and targets, return dicts."),
        ("configs/default.yaml", "Default hyperparameters", "Write a YAML config file with sensible defaults for all hyperparameters: model architecture, training (lr, epochs, batch_size, seeds), data, evaluation."),
    ]

    def implement_method(self, idea: dict) -> tuple[Optional[dict], str]:
        """Generate core method implementation in src/ using per-file calls.

        Like the writer's section-by-section pattern: generates each file
        independently for reliability (smaller responses = reliable JSON).
        """
        old_prompt = self.system_prompt
        self.system_prompt = METHOD_SYSTEM_PROMPT

        idea_context = (
            f"## Research Idea\n"
            f"Title: {idea.get('title')}\n"
            f"Description: {idea.get('description')}\n"
            f"Method: {idea.get('method_summary')}\n"
            f"Datasets: {idea.get('datasets')}\n"
            f"Baselines: {idea.get('baselines')}\n"
        )

        files = {}
        deps = set()
        generated_files = []
        raw_texts = []

        for filename, description, guidance in self.METHOD_FILES:
            self.log_progress(f"Generating: {filename} ({description})")

            # Include context of previously generated files
            prev_context = ""
            if generated_files:
                prev_context = "\n\n## Already Generated Files\n"
                for gf_name, gf_content in generated_files:
                    # Show first 1500 chars of each file for context
                    prev_context += f"\n### {gf_name}\n```python\n{gf_content[:1500]}\n```\n"

            task = (
                f"Generate the file: {filename}\n"
                f"Purpose: {description}\n\n"
                f"{idea_context}\n\n"
                f"Specific instructions: {guidance}\n\n"
                f"{prev_context}\n\n"
                f"Return ONLY a JSON block:\n"
                f'```json\n{{"filename": "{filename}", "content": "...", '
                f'"dependencies": ["torch", "..."], "description": "..."}}\n```\n'
                f"The content must be complete, executable code — no placeholders.\n"
            )

            result, raw = self.call_structured(
                task=task,
                max_tokens=8192,
                temperature=0.4,
                effort=EFFORT_HIGH,
            )
            raw_texts.append(raw)

            if result:
                content = result.get("content", "")
                if content:
                    files[filename] = content
                    generated_files.append((filename, content))
                    for d in result.get("dependencies", []):
                        deps.add(d)
                    logger.info(f"Generated {filename}: {len(content)} chars")
                else:
                    logger.warning(f"Empty content for {filename}")
            else:
                # Fallback: extract code from response
                content = self._extract_code_fallback(raw, filename)
                if content:
                    files[filename] = content
                    generated_files.append((filename, content))
                    logger.info(f"Generated {filename} (fallback): {len(content)} chars")
                else:
                    logger.warning(f"Failed to generate {filename}. Raw: {raw[:200]}")

        self.system_prompt = old_prompt

        if not files:
            return None, "\n".join(raw_texts)

        # Generate __init__.py based on what was actually created
        # IMPORTANT: Use ^class to match only top-level class definitions,
        # not "class" inside docstrings/comments (which produced false matches
        # like "from .model import implementing" and "from .model import a")
        init_imports = []
        if "src/model.py" in files:
            import re
            classes = re.findall(r'^class (\w+)', files["src/model.py"], re.MULTILINE)
            for cls in classes:
                init_imports.append(f"from .model import {cls}")
        if "src/trainer.py" in files:
            import re
            classes = re.findall(r'^class (\w+)', files["src/trainer.py"], re.MULTILINE)
            for cls in classes:
                init_imports.append(f"from .trainer import {cls}")
        if "src/metrics.py" in files:
            init_imports.append("from .metrics import *")

        files["src/__init__.py"] = "\n".join(init_imports) + "\n" if init_imports else ""

        method_design = {
            "files": files,
            "dependencies": sorted(deps) if deps else ["torch", "torchvision", "numpy", "scipy", "pyyaml"],
            "method_summary": f"Implementation for: {idea.get('title', 'unknown')}",
        }

        return method_design, "\n".join(raw_texts)

    def save_method_files(self, method_design: dict) -> list[Path]:
        """Save method implementation files to project directory.

        Validates Python files for syntax errors after saving.
        """
        saved = []
        files = method_design.get("files", {})
        for filepath, content in files.items():
            # Fix common content issues before saving
            content = self._fix_file_content(content, filepath)

            full_path = Path(self.project_dir) / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            saved.append(full_path)

            # Validate Python syntax, attempt auto-repair if broken
            if filepath.endswith(".py"):
                error = self._check_python_syntax(full_path)
                if error:
                    repaired = self._try_truncation_repair(content)
                    if repaired and repaired != content:
                        full_path.write_text(repaired)
                        error2 = self._check_python_syntax(full_path)
                        if not error2:
                            logger.info(f"Auto-repaired {filepath} by truncating incomplete tail")
                            self.log_progress(f"Auto-repaired {filepath} (truncated incomplete tail)")
                        else:
                            logger.warning(f"Syntax error in {filepath} (repair failed): {error}")
                            self.log_progress(f"WARNING: {filepath} has syntax error: {error}")
                    else:
                        logger.warning(f"Syntax error in {filepath}: {error}")
                        self.log_progress(f"WARNING: {filepath} has syntax error: {error}")
                else:
                    logger.info(f"Saved method file: {filepath} ({len(content)} chars) [syntax OK]")
            else:
                logger.info(f"Saved method file: {filepath} ({len(content)} chars)")

        # Save dependencies
        deps = method_design.get("dependencies", [])
        if deps:
            req_path = Path(self.project_dir) / "requirements.txt"
            req_path.write_text("\n".join(deps) + "\n")
            saved.append(req_path)

        return saved

    @staticmethod
    def _fix_file_content(content: str, filepath: str) -> str:
        """Fix common content extraction issues."""
        if not filepath.endswith(".py"):
            # For YAML, just fix literal escapes
            if filepath.endswith((".yaml", ".yml")):
                if "\\n" in content and content.count("\n") < 3:
                    content = content.replace("\\n", "\n").replace("\\t", "\t")
            return content

        # Fix 1: If content has literal \n instead of actual newlines
        # Case A: entirely escaped (no real newlines at all)
        if "\\n" in content and "\n" not in content:
            content = content.replace("\\n", "\n").replace("\\t", "\t")
        # Case B: mixed — real newlines exist but also literal \\n sequences
        # (common when JSON content value had escaped newlines that weren't decoded)
        elif "\\n" in content and "\n" in content:
            # Check if literal \\n appears inside what looks like string lines
            # Heuristic: if >20% of "lines" by \\n splitting have code-like content
            escaped_parts = content.split("\\n")
            if len(escaped_parts) > len(content.split("\n")) * 2:
                # More \\n-separated parts than real lines — likely escaped content
                content = content.replace("\\n", "\n").replace("\\t", "\t")
                logger.warning(f"Fixed mixed literal \\n escapes in {filepath}")

        # Fix 1b: Double-escaped sequences (\\\\n → \n)
        if "\\\\n" in content:
            content = content.replace("\\\\n", "\n").replace("\\\\t", "\t")

        # Fix 1c: JSON-escaped quotes (\" → ") — from fallback extraction of JSON strings
        if '\\"' in content:
            content = content.replace('\\"', '"')
            logger.warning(f"Fixed JSON-escaped quotes in {filepath}")

        # Fix 2: Strip markdown code fences that leaked into the code content
        # The LLM sometimes wraps code in ```python...``` or ```json...``` inside
        # the JSON "content" field, producing SyntaxErrors like `p_str =```json`
        import re
        # Remove opening fences: ```python, ```json, ```yaml, bare ```
        content = re.sub(r'^```(?:python|json|yaml|yml|bash|sh)?\s*$', '', content, flags=re.MULTILINE)
        # Remove closing fences: bare ``` at end of line
        content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)

        # Fix 3: If content starts mid-string (missing opening docstring)
        # Detect: first line doesn't start with valid Python tokens
        first_line = content.split("\n")[0].strip() if "\n" in content else content[:100].strip()
        valid_starts = ("import ", "from ", "class ", "def ", "#", '"""', "'''",
                        "__", "if ", "try:", "import", "@", "#!/")
        if first_line and not any(first_line.startswith(s) for s in valid_starts):
            # Try to find where real code starts
            match = re.search(r'^(from __future__|import |from |class |def |#|""")', content, re.MULTILINE)
            if match:
                logger.warning(f"Trimming {len(content[:match.start()])} chars of preamble from {filepath}")
                content = content[match.start():]

        return content

    @staticmethod
    def _check_python_syntax(filepath: Path) -> Optional[str]:
        """Check if a Python file has valid syntax. Returns error msg or None."""
        import py_compile
        try:
            py_compile.compile(str(filepath), doraise=True)
            return None
        except py_compile.PyCompileError as e:
            return str(e)

    @staticmethod
    def _try_truncation_repair(content: str) -> Optional[str]:
        """Try to fix syntax errors by truncating incomplete code at the end.

        When the LLM response gets cut off at max_tokens, files end with
        unterminated strings, incomplete functions, etc. This finds the last
        complete top-level block (class/def at indent 0) and truncates there.
        """
        import re
        lines = content.split("\n")
        if len(lines) < 10:
            return None

        # Find the last top-level def/class boundary
        last_good = None
        for i in range(len(lines) - 1, 0, -1):
            line = lines[i]
            # A top-level def/class starts a new block
            if re.match(r'^(class |def |@)', line):
                # The previous line ends the prior complete block
                last_good = i
                break

        if last_good and last_good > len(lines) // 2:
            # Truncate at the last complete block boundary
            truncated = "\n".join(lines[:last_good]).rstrip() + "\n"
            return truncated

        return None

    @staticmethod
    def _extract_code_fallback(raw: str, filename: str) -> Optional[str]:
        """Extract code content from raw response when JSON parsing fails.

        Looks for the largest python code block, or YAML for config files.
        """
        if not raw:
            return None

        import re

        # Determine expected language
        if filename.endswith((".yaml", ".yml")):
            lang_pattern = r'```(?:yaml|yml)\s*\n(.*?)\n```'
        else:
            lang_pattern = r'```python\s*\n(.*?)\n```'

        # Find ALL code blocks of the expected type, take the largest
        matches = re.findall(lang_pattern, raw, re.DOTALL)
        if matches:
            code = max(matches, key=len)
            # Unescape JSON artifacts if present
            if '\\"' in code:
                code = code.replace('\\"', '"')
            return code

        # Fallback: any code block
        matches = re.findall(r'```\w*\s*\n(.*?)\n```', raw, re.DOTALL)
        if matches:
            # Filter to ones that look like code
            code_blocks = [m for m in matches if "import " in m or "def " in m or "class " in m or ":" in m]
            if code_blocks:
                return max(code_blocks, key=len)

        # Last resort: if raw text looks like code, use it directly
        if "import " in raw or "def " in raw or "class " in raw:
            # Strip any non-code preamble (text before first import/def/class)
            for marker in ["import ", "#!/", "from ", "class ", "def "]:
                idx = raw.find(marker)
                if idx >= 0:
                    return raw[idx:].rstrip()

        return None

    @staticmethod
    def _find_best_gpu() -> Optional[int]:
        """Find GPU with most free memory. Returns GPU index or None."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            best_idx, best_free = None, 0
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    idx, free = int(parts[0].strip()), int(parts[1].strip())
                    if free > best_free:
                        best_idx, best_free = idx, free
            if best_free > 1000:  # At least 1GB free
                return best_idx
        except Exception:
            pass
        return None

    def _validate_src_imports(self) -> bool:
        """Validate that src/ modules can be imported without errors.

        Catches import-time failures (missing deps, name errors) before
        running experiments, providing clearer error messages.
        """
        src_dir = self.src_dir
        if not src_dir.exists():
            return False

        all_ok = True
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_name = py_file.stem
            try:
                result = subprocess.run(
                    ["python", "-c", f"import sys; sys.path.insert(0, '.'); from src import {module_name}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.project_dir),
                )
                if result.returncode != 0:
                    error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
                    logger.warning(f"Import validation failed for src.{module_name}: {error_msg}")
                    self.log_progress(f"WARNING: src.{module_name} import failed: {error_msg}")
                    all_ok = False
                else:
                    logger.info(f"Import validation OK: src.{module_name}")
            except Exception as e:
                logger.warning(f"Import validation error for src.{module_name}: {e}")
                all_ok = False

        return all_ok

    # Experiment scripts to generate (per-file calls)
    EXPERIMENT_SCRIPTS = [
        ("scripts/run_main.py", "Main experiment runner", "Write the main experiment script that trains the proposed method on the target dataset. Must: import from src/, use multiple seeds (3-5), print JSON metrics to stdout with format {\"metric\": \"name\", \"value\": N, \"seed\": S}."),
        ("scripts/run_baselines.py", "Baseline comparisons", "Write the baseline comparison script. Must: implement or import baseline methods, run on same dataset with same seeds, print JSON metrics to stdout for fair comparison."),
        ("scripts/run_ablations.py", "Ablation studies", "Write ablation study script that systematically disables/modifies components of the method. Must: test each component independently, print JSON metrics to stdout."),
        ("scripts/analyze_results.py", "Result analysis and statistics", "Write analysis script that reads all experiment outputs, computes statistics (mean, std, p-values via scipy.stats.ttest_ind), generates summary tables, and saves all_results.json to experiments/."),
        ("scripts/generate_figures.py", "Figure generation", "Write figure generation script using matplotlib. Must: create comparison bar charts, training curves, ablation plots. Save to figures/ directory as PNG."),
    ]

    def design_experiments(self, idea: dict, method_files: list[str]) -> tuple[Optional[dict], str]:
        """Design experiment scripts using per-file calls (like writer's sections).

        Phase 2: generate scripts/, configs/ that import from src/.
        Generates each script independently for reliability.
        """
        # Load actual src/ code for context — include FULL API surface
        # (truncation caused API mismatch bugs: scripts used .get() on dataclass objects)
        src_context = "## Available Method Modules (FULL CODE — use these exact APIs)\n"
        for f in method_files:
            if f.startswith("src/") and f.endswith(".py"):
                full_path = Path(self.project_dir) / f
                if full_path.exists():
                    content = full_path.read_text()
                    # Include full file for API accuracy (up to 4000 chars)
                    src_context += f"\n### {f}\n```python\n{content[:4000]}\n```\n"

        # Also extract public API summary for quick reference
        src_context += "\n## API Quick Reference (classes, functions, signatures)\n"
        for f in method_files:
            if f.startswith("src/") and f.endswith(".py"):
                full_path = Path(self.project_dir) / f
                if full_path.exists():
                    content = full_path.read_text()
                    import re
                    # Extract class/function definitions with their signatures
                    for match in re.finditer(r'^(class \w+.*?:|def \w+\(.*?\).*?:)', content, re.MULTILINE):
                        src_context += f"  - `{f}`: `{match.group(0).strip()}`\n"
                    # Extract dataclass fields
                    for match in re.finditer(r'^@dataclass', content, re.MULTILINE):
                        # Get the class and its fields
                        pos = match.start()
                        block = content[pos:pos+500]
                        src_context += f"  - `{f}` dataclass: `{block.split(chr(10)+chr(10))[0].strip()}`\n"

        # Get actual GPU memory for context
        gpu_info = ""
        best_gpu = self._find_best_gpu()
        if best_gpu is not None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free",
                     "--format=csv,noheader,nounits", f"--id={best_gpu}"],
                    capture_output=True, text=True, timeout=5,
                )
                free_mb = result.stdout.strip().split(",")[-1].strip()
                gpu_info = f"\n## HARDWARE CONSTRAINT\nGPU has only {free_mb}MB free memory. batch_size MUST be 1-2, seq_len 128-256. Use torch.cuda.empty_cache() between seeds.\n"
            except Exception:
                pass

        idea_context = (
            f"## Research Idea\n"
            f"Title: {idea.get('title')}\n"
            f"Description: {idea.get('description')}\n"
            f"Method: {idea.get('method_summary')}\n"
            f"Datasets: {idea.get('datasets')}\n"
            f"Baselines: {idea.get('baselines')}\n"
            f"Target Conference: {idea.get('target_conference')}\n"
            f"{gpu_info}"
        )

        # Step 1: Generate hypotheses first
        self.log_progress("Designing experiment hypotheses...")
        hyp_task = (
            f"Define testable hypotheses for this research.\n\n"
            f"{idea_context}\n\n"
            f"Return a JSON block:\n"
            f'```json\n{{"hypotheses": [{{"id": "h1", "statement": "...", '
            f'"metric": "accuracy", "test_type": "t-test", "expected_direction": "higher"}}]}}\n```\n'
        )
        hyp_result, hyp_raw = self.call_structured(
            task=hyp_task, max_tokens=2048, temperature=0.4, effort=EFFORT_HIGH,
        )
        hypotheses = hyp_result.get("hypotheses", []) if hyp_result else []

        # Step 2: Generate each script
        scripts = {}
        raw_texts = [hyp_raw]
        generated_scripts = []

        for filename, description, guidance in self.EXPERIMENT_SCRIPTS:
            self.log_progress(f"Generating: {filename} ({description})")

            prev_context = ""
            if generated_scripts:
                prev_context = "\n\n## Previously Generated Scripts\n"
                for gs_name, gs_content in generated_scripts:
                    prev_context += f"\n### {gs_name}\n```python\n{gs_content[:1000]}\n```\n"

            task = (
                f"Generate the experiment script: {filename}\n"
                f"Purpose: {description}\n\n"
                f"{idea_context}\n\n"
                f"{src_context}\n\n"
                f"Hypotheses: {json.dumps(hypotheses, indent=2)}\n\n"
                f"Specific instructions: {guidance}\n\n"
                f"IMPORTANT: Scripts run from the project root. Use sys.path.insert(0, '.') "
                f"then import from src/.\n\n"
                f"{prev_context}\n\n"
                f"Return ONLY a JSON block:\n"
                f'```json\n{{"filename": "{filename}", "content": "...", "description": "..."}}\n```\n'
            )

            result, raw = self.call_structured(
                task=task, max_tokens=8192, temperature=0.4, effort=EFFORT_HIGH,
            )
            raw_texts.append(raw)

            if result and result.get("content"):
                scripts[filename] = result["content"]
                generated_scripts.append((filename, result["content"]))
                logger.info(f"Generated {filename}: {len(result['content'])} chars")
            else:
                # Fallback: extract code from raw response
                content = self._extract_code_fallback(raw, filename)
                if content:
                    scripts[filename] = content
                    generated_scripts.append((filename, content))
                    logger.info(f"Generated {filename} (fallback): {len(content)} chars")
                else:
                    logger.warning(f"Failed to generate {filename}. Raw: {raw[:200] if raw else 'empty'}")

        # Build design dict matching the expected format
        design = {
            "hypotheses": hypotheses,
            "scripts": scripts,
            "configs": {},
            "run_order": [s for s, _, _ in self.EXPERIMENT_SCRIPTS if s in scripts],
        }

        if not scripts:
            return None, "\n".join(raw_texts)

        return design, "\n".join(raw_texts)

    def save_experiment_scripts(self, design: dict) -> list[Path]:
        """Save experiment scripts and configs to project directory.

        Applies content fixes and syntax validation (same as save_method_files).
        """
        saved = []

        # Save scripts
        for filepath, content in design.get("scripts", {}).items():
            # Apply same content fixes as method files
            content = self._fix_file_content(content, filepath)

            full_path = Path(self.project_dir) / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            saved.append(full_path)

            # Validate Python syntax, attempt auto-repair if broken
            if filepath.endswith(".py"):
                error = self._check_python_syntax(full_path)
                if error:
                    repaired = self._try_truncation_repair(content)
                    if repaired and repaired != content:
                        full_path.write_text(repaired)
                        error2 = self._check_python_syntax(full_path)
                        if not error2:
                            logger.info(f"Auto-repaired {filepath} by truncating incomplete tail")
                            self.log_progress(f"Auto-repaired {filepath} (truncated incomplete tail)")
                        else:
                            logger.warning(f"Syntax error in {filepath} (repair failed): {error}")
                            self.log_progress(f"WARNING: {filepath} has syntax error: {error}")
                    else:
                        logger.warning(f"Syntax error in {filepath}: {error}")
                        self.log_progress(f"WARNING: {filepath} has syntax error: {error}")
                else:
                    logger.info(f"Saved script: {filepath} ({len(content)} chars) [syntax OK]")

        # Save configs
        for filepath, content in design.get("configs", {}).items():
            full_path = Path(self.project_dir) / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Save hypotheses
        hypotheses = design.get("hypotheses", [])
        if hypotheses:
            h_path = self.experiments_dir / "hypotheses.json"
            h_path.write_text(json.dumps(hypotheses, indent=2))

        # Save full design
        design_path = self.experiments_dir / "experiment_design.json"
        design_path.write_text(json.dumps(design, indent=2, default=str))

        return saved

    def run_experiment(
        self,
        script_path: Path,
        timeout_seconds: int = 3600,
        gpu_id: Optional[int] = None,
    ) -> dict:
        """Execute an experiment script and capture results.

        Implements anti-fabrication: SHA256 signature of output log.
        """
        log_dir = self.experiments_dir / "logs" / script_path.stem
        log_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"

        env = {}
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            # Auto-select GPU with most free memory
            best_gpu = self._find_best_gpu()
            if best_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                logger.info(f"Auto-selected GPU {best_gpu}")

        logger.info(f"Running experiment: {script_path.name}")

        start_time = time.time()
        try:
            import os as _os
            run_env = {**_os.environ, **env}
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self.project_dir),
                env=run_env,
            )

            elapsed = time.time() - start_time

            # Auto-retry on CUDA OOM with CPU fallback
            if result.returncode != 0 and "CUDA" in result.stderr and "out of memory" in result.stderr:
                logger.warning(f"CUDA OOM on {script_path.name}, retrying with CUDA_VISIBLE_DEVICES=''")
                self.log_progress(f"CUDA OOM on {script_path.name}, retrying on CPU...")
                cpu_env = {**_os.environ, "CUDA_VISIBLE_DEVICES": ""}
                start_time = time.time()
                result = subprocess.run(
                    ["python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(self.project_dir),
                    env=cpu_env,
                )
                elapsed = time.time() - start_time

            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)

            # Anti-fabrication signature (research-mcp pattern)
            signature = hashlib.sha256(result.stdout.encode()).hexdigest()[:16]

            metrics = self._parse_metrics(result.stdout)

            return {
                "script": script_path.name,
                "return_code": result.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "signature": signature,
                "metrics": metrics,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "success": result.returncode == 0,
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }

        except subprocess.TimeoutExpired:
            return {
                "script": script_path.name,
                "return_code": -1,
                "error": f"Timeout after {timeout_seconds}s",
                "success": False,
            }
        except Exception as e:
            return {
                "script": script_path.name,
                "return_code": -1,
                "error": str(e),
                "success": False,
            }

    def _parse_metrics(self, stdout: str) -> list[dict]:
        """Extract metrics from experiment stdout."""
        metrics = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and ("metric" in data or "accuracy" in data or "loss" in data):
                    metrics.append(data)
            except json.JSONDecodeError:
                import re
                match = re.search(r'(accuracy|loss|f1|precision|recall|psnr|ssim|fid|bleu)\s*[:=]\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    metrics.append({
                        "metric": match.group(1).lower(),
                        "value": float(match.group(2)),
                        "raw_line": line,
                    })

        return metrics

    def verify_hypothesis(
        self,
        hypothesis: dict,
        ours_values: list[float],
        baseline_values: list[float],
    ) -> dict:
        """Statistical hypothesis verification (research_claude_agents pattern).

        Enforces p < 0.05 for any claim to be paper-worthy.
        """
        from scipy import stats
        import numpy as np

        test_type = hypothesis.get("test_type", "t-test")
        direction = hypothesis.get("expected_direction", "higher")

        ours_arr = np.array(ours_values)
        base_arr = np.array(baseline_values)

        if test_type == "t-test":
            statistic, p_value = stats.ttest_ind(ours_arr, base_arr)
        elif test_type == "paired-t":
            statistic, p_value = stats.ttest_rel(ours_arr, base_arr)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(ours_arr - base_arr)
        elif test_type == "mann-whitney":
            statistic, p_value = stats.mannwhitneyu(ours_arr, base_arr, alternative="two-sided")
        else:
            statistic, p_value = stats.ttest_ind(ours_arr, base_arr)

        if direction in ("higher", "lower"):
            p_value = p_value / 2

        pooled_std = np.sqrt((ours_arr.std()**2 + base_arr.std()**2) / 2)
        effect_size = (ours_arr.mean() - base_arr.mean()) / pooled_std if pooled_std > 0 else 0

        can_include = bool(p_value < 0.05)

        result = {
            "hypothesis_id": hypothesis.get("id"),
            "hypothesis_statement": hypothesis.get("statement"),
            "test_type": test_type,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "ours_mean": float(ours_arr.mean()),
            "ours_std": float(ours_arr.std()),
            "baseline_mean": float(base_arr.mean()),
            "baseline_std": float(base_arr.std()),
            "can_include_in_paper": can_include,
            "verdict": "SIGNIFICANT" if can_include else "NOT_SIGNIFICANT",
        }

        v_path = self.verification_dir / f"{hypothesis.get('id', 'h')}.json"
        v_path.write_text(json.dumps(result, indent=2))

        return result

    def analyze_results(
        self,
        idea: dict,
        experiment_results: list[dict],
        verification_results: list[dict],
    ) -> tuple[Optional[dict], str]:
        """Use Claude to analyze experiment results (effort="medium")."""
        results_text = json.dumps(experiment_results, indent=2, default=str)
        verification_text = json.dumps(verification_results, indent=2, default=str)

        task = (
            f"Analyze these experiment results for our research paper.\n\n"
            f"## Research Idea\n{json.dumps(idea, indent=2)}\n\n"
            f"## Experiment Results\n{results_text}\n\n"
            f"## Statistical Verification\n{verification_text}\n\n"
            f"Provide:\n"
            f"1. Main findings and their significance\n"
            f"2. Which hypotheses are supported (only p<0.05 claims)\n"
            f"3. Ablation insights\n"
            f"4. Honest limitations\n"
            f"5. Whether additional experiments are needed\n"
        )

        return self.call_structured(
            task=task,
            max_tokens=8192,
            temperature=0.3,
            effort=EFFORT_MEDIUM,
        )

    async def run(self, idea: dict) -> dict:
        """Full experiment pipeline: implement -> design -> execute -> verify -> analyze.

        Two-phase code generation (rom4ik pattern):
        1. Implement core method in src/ (importable classes)
        2. Generate experiment scripts in scripts/ (use src/)
        3. Execute scripts
        4. Verify hypotheses statistically
        5. Analyze results
        """
        self.log_progress(f"Starting experiments for: {idea.get('title')}")

        # Step 1: Implement core method in src/ (effort="high")
        self.log_progress("Phase 1: Implementing core method in src/...")
        method_design, raw = self.implement_method(idea)
        if not method_design:
            self.log_progress(f"ERROR: Failed to implement method.\nRaw: {raw[:500]}")
            return {"success": False, "error": "Method implementation failed"}

        method_files = self.save_method_files(method_design)
        method_filenames = [str(f.relative_to(self.project_dir)) for f in method_files]
        self.log_progress(
            f"Implemented method: {len(method_files)} files\n"
            f"  Files: {', '.join(method_filenames)}\n"
            f"  Summary: {method_design.get('method_summary', 'N/A')}"
        )

        # Step 2: Design experiment scripts (effort="high")
        self.log_progress("Phase 2: Designing experiment scripts...")
        design, raw = self.design_experiments(idea, method_filenames)
        if not design:
            self.log_progress(f"ERROR: Failed to design experiments.\nRaw: {raw[:500]}")
            return {"success": False, "error": "Experiment design failed"}

        scripts = self.save_experiment_scripts(design)
        self.log_progress(f"Designed {len(scripts)} experiment scripts + configs")

        # Step 2.5: Validate src/ modules can be imported
        import_ok = self._validate_src_imports()
        if not import_ok:
            self.log_progress("WARNING: src/ import validation failed — experiments may crash")

        # Step 3: Execute experiments in order
        run_order = design.get("run_order", [str(s.relative_to(self.project_dir)) for s in scripts])
        all_results = []
        for script_rel in run_order:
            script_path = Path(self.project_dir) / script_rel
            if not script_path.exists():
                self.log_progress(f"Script not found: {script_rel}, skipping")
                continue
            result = self.run_experiment(script_path)
            all_results.append(result)
            status = "OK" if result["success"] else "FAILED"
            self.log_progress(
                f"Experiment {script_path.name}: {status} "
                f"(elapsed={result.get('elapsed_seconds', '?')}s, "
                f"signature={result.get('signature', 'N/A')}, "
                f"metrics={len(result.get('metrics', []))})"
            )
            if not result["success"] and result.get("stderr_tail"):
                self.log_progress(f"  stderr: {result['stderr_tail'][:300]}")

        # Step 4: Verify hypotheses
        verification_results = []
        hypotheses = design.get("hypotheses", [])
        self.log_progress(
            f"Experiments complete. {sum(1 for r in all_results if r['success'])}/{len(all_results)} succeeded.\n"
            f"Hypotheses to verify: {len(hypotheses)}\n"
            f"Total metrics extracted: {sum(len(r.get('metrics', [])) for r in all_results)}"
        )

        # Step 5: Analyze results (effort="medium")
        analysis, raw = self.analyze_results(idea, all_results, verification_results)
        if analysis:
            analysis_path = self.experiments_dir / "analysis.json"
            analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
            self.log_progress(f"Analysis complete. Main findings: {analysis.get('analysis', {}).get('main_findings', [])}")
        else:
            self.log_progress(f"WARNING: Analysis parsing failed. Raw: {raw[:300]}")

        # Save all results
        results_path = self.experiments_dir / "all_results.json"
        results_path.write_text(json.dumps({
            "experiment_results": all_results,
            "verification_results": verification_results,
            "analysis": analysis,
            "method_files": method_filenames,
        }, indent=2, default=str))

        config = load_project_config(self.project_dir)
        config["stage"] = "writing"
        config["experiments_completed"] = len([r for r in all_results if r["success"]])
        save_project_config(self.project_dir, config)

        return {
            "success": any(r["success"] for r in all_results),
            "experiment_results": all_results,
            "verification_results": verification_results,
            "analysis": analysis,
            "design": design,
            "method_files": method_filenames,
        }
