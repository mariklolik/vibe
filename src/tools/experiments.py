"""Experiment execution tools - run experiments with logging and checkpoints."""

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.db.experiments_db import experiments_db, Experiment
from src.db.workflow import workflow_db
from src.project.manager import project_manager


EXPERIMENTS_DIR = Path("./experiments")
LOGS_DIR = Path("./logs")
CHECKPOINTS_DIR = Path("./checkpoints")


def _ensure_dirs():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


async def _check_workflow_prerequisites(action: str) -> tuple[bool, str]:
    """Check if workflow prerequisites are met for an action."""
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return True, ""  # No project, skip validation
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return True, ""  # No workflow, skip validation
    
    is_valid, error_msg = workflow_db.validate_action(workflow, action)
    if not is_valid:
        return False, error_msg
    
    return True, ""


async def run_experiment(
    script: str,
    config: Optional[str] = None,
    env_name: Optional[str] = None,
    gpu_ids: str = "0",
    name: Optional[str] = None,
) -> str:
    """Run an experiment script with logging.
    
    Prerequisites: Must have approved idea and created experiment environment.
    """
    # Check workflow prerequisites
    is_valid, error_msg = await _check_workflow_prerequisites("run_experiment")
    if not is_valid:
        current_project_obj = await project_manager.get_current_project()
        workflow = await workflow_db.get_project_workflow(
            current_project_obj.project_id if current_project_obj else None
        )
        missing = workflow_db.get_missing_prerequisites(workflow, "run_experiment") if workflow else []
        return json.dumps({
            "success": False,
            "error": "WORKFLOW_BLOCKED",
            "message": error_msg,
            "missing_prerequisites": missing,
            "action_required": "Call get_next_action() to see required steps",
        }, indent=2)
    
    _ensure_dirs()
    
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    exp_name = name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    script_path = Path(script)
    if not script_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Script not found: {script}",
        })
    
    log_dir = LOGS_DIR / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env_vars["EXPERIMENT_NAME"] = exp_name
    env_vars["LOG_DIR"] = str(log_dir)
    
    cmd_parts = []
    
    if env_name:
        conda_path = shutil.which("conda")
        if conda_path:
            cmd_parts.append(f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} &&")
        else:
            venv_path = Path(f"./.venvs/{env_name}")
            if venv_path.exists():
                cmd_parts.append(f"source {venv_path}/bin/activate &&")
    
    cmd_parts.append(f"python {script}")
    
    if config:
        config_path = Path(config)
        if config_path.exists():
            cmd_parts.append(f"--config {config}")
    
    cmd = " ".join(cmd_parts)
    
    exp = Experiment(
        experiment_id=exp_id,
        name=exp_name,
        status="running",
        config={"script": script, "config": config, "gpu_ids": gpu_ids},
        metrics={},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        logs_dir=str(log_dir),
        checkpoint_path=None,
        extra_data={},
    )
    await experiments_db.save_experiment(exp)
    
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    
    try:
        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            process = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=stdout_f,
                stderr=stderr_f,
                env=env_vars,
                cwd=str(script_path.parent) if script_path.parent != Path(".") else None,
            )
        
        return json.dumps({
            "success": True,
            "experiment_id": exp_id,
            "name": exp_name,
            "pid": process.pid,
            "log_dir": str(log_dir),
            "status": "running",
            "monitor_cmd": f"tail -f {stdout_log}",
        })
    except Exception as e:
        exp.status = "failed"
        exp.extra_data["error"] = str(e)
        await experiments_db.save_experiment(exp)
        
        return json.dumps({
            "success": False,
            "experiment_id": exp_id,
            "error": str(e),
        })


async def run_baseline(
    baseline_dir: str,
    config: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Run a baseline method for comparison.
    
    Prerequisites: Must have approved idea and created experiment environment.
    """
    # Check workflow prerequisites
    is_valid, error_msg = await _check_workflow_prerequisites("run_baseline")
    if not is_valid:
        return json.dumps({
            "success": False,
            "error": "WORKFLOW_BLOCKED",
            "message": error_msg,
            "action_required": "Call get_next_action() to see required steps",
        }, indent=2)
    
    baseline_path = Path(baseline_dir)
    
    if not baseline_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Baseline directory not found: {baseline_dir}",
        })
    
    train_scripts = ["train.py", "main.py", "run.py", "run_experiment.py"]
    script = None
    
    for s in train_scripts:
        if (baseline_path / s).exists():
            script = str(baseline_path / s)
            break
    
    if not script:
        return json.dumps({
            "success": False,
            "error": f"No training script found in {baseline_dir}",
            "checked": train_scripts,
        })
    
    exp_name = name or f"baseline_{baseline_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return await run_experiment(script, config, name=exp_name)


async def run_ablation(
    script: str,
    base_config: Optional[str] = None,
    ablation_params: Optional[dict] = None,
) -> str:
    if not ablation_params:
        return json.dumps({
            "success": False,
            "error": "ablation_params required",
        })
    
    _ensure_dirs()
    
    ablation_id = f"ablation_{uuid.uuid4().hex[:8]}"
    ablation_dir = EXPERIMENTS_DIR / ablation_id
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    configurations = [{}]
    for param, values in ablation_params.items():
        new_configs = []
        for config in configurations:
            for value in values:
                new_config = config.copy()
                new_config[param] = value
                new_configs.append(new_config)
        configurations = new_configs
    
    experiments = []
    
    for i, config in enumerate(configurations):
        config_name = "_".join(f"{k}={v}" for k, v in config.items())
        exp_name = f"{ablation_id}_{config_name}"
        
        config_file = ablation_dir / f"config_{i}.json"
        config_file.write_text(json.dumps(config, indent=2))
        
        experiments.append({
            "name": exp_name,
            "config": config,
            "config_file": str(config_file),
            "status": "pending",
        })
    
    return json.dumps({
        "success": True,
        "ablation_id": ablation_id,
        "total_experiments": len(experiments),
        "ablation_params": ablation_params,
        "experiments": experiments,
        "note": "Use run_experiment for each configuration to execute",
    }, indent=2)


async def monitor_training(
    experiment_name: str,
    metrics: Optional[list[str]] = None,
) -> str:
    exp = await experiments_db.get_experiment(experiment_name)
    
    if not exp:
        return json.dumps({
            "success": False,
            "error": f"Experiment not found: {experiment_name}",
        })
    
    result = {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "status": exp.status,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
        "current_metrics": exp.metrics,
    }
    
    if exp.logs_dir:
        log_dir = Path(exp.logs_dir)
        stdout_log = log_dir / "stdout.log"
        
        if stdout_log.exists():
            with open(stdout_log, "r") as f:
                lines = f.readlines()
                result["recent_output"] = "".join(lines[-20:])
    
    if metrics:
        history = await experiments_db.get_metric_history(exp.experiment_id)
        filtered_history = [h for h in history if h["metric_name"] in metrics]
        result["metric_history"] = filtered_history
    
    return json.dumps(result, indent=2)


async def save_checkpoint(
    experiment_name: str,
    checkpoint_name: Optional[str] = None,
) -> str:
    exp = await experiments_db.get_experiment(experiment_name)
    
    if not exp:
        return json.dumps({
            "success": False,
            "error": f"Experiment not found: {experiment_name}",
        })
    
    _ensure_dirs()
    
    checkpoint_name = checkpoint_name or f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = CHECKPOINTS_DIR / exp.name / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "checkpoint_name": checkpoint_name,
        "created_at": datetime.now().isoformat(),
        "metrics": exp.metrics,
        "config": exp.config,
    }
    
    meta_file = checkpoint_dir / "checkpoint_meta.json"
    meta_file.write_text(json.dumps(meta, indent=2))
    
    exp.checkpoint_path = str(checkpoint_dir)
    exp.updated_at = datetime.now().isoformat()
    await experiments_db.save_experiment(exp)
    
    return json.dumps({
        "success": True,
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_name": checkpoint_name,
        "experiment": exp.name,
    })


async def resume_experiment(
    experiment_name: str,
    checkpoint: str,
) -> str:
    exp = await experiments_db.get_experiment(experiment_name)
    
    if not exp:
        return json.dumps({
            "success": False,
            "error": f"Experiment not found: {experiment_name}",
        })
    
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINTS_DIR / exp.name / checkpoint
    
    if not checkpoint_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Checkpoint not found: {checkpoint}",
        })
    
    meta_file = checkpoint_path / "checkpoint_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        meta = {}
    
    new_name = f"{exp.name}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = exp.config.copy()
    config["resume_from"] = str(checkpoint_path)
    
    script = config.get("script")
    if script:
        return await run_experiment(
            script=script,
            config=config.get("config"),
            name=new_name,
        )
    
    return json.dumps({
        "success": True,
        "original_experiment": exp.name,
        "checkpoint": str(checkpoint_path),
        "new_experiment_name": new_name,
        "resume_config": config,
        "note": "Manually run the experiment with resume_from in config",
    })
