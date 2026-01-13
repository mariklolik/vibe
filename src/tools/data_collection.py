"""Data collection tools - collect metrics from experiment logs."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.db.experiments_db import experiments_db


LOGS_DIR = Path("./logs")


async def collect_metrics(
    experiments: list[str],
    metrics: Optional[list[str]] = None,
) -> str:
    results = {}
    
    for exp_name in experiments:
        exp = await experiments_db.get_experiment(exp_name)
        
        if exp:
            results[exp_name] = {
                "status": exp.status,
                "metrics": exp.metrics,
            }
            
            if metrics:
                history = await experiments_db.get_metric_history(exp.experiment_id)
                filtered = [h for h in history if h["metric_name"] in metrics]
                results[exp_name]["history"] = filtered
        else:
            log_dir = LOGS_DIR / exp_name
            if log_dir.exists():
                stdout_log = log_dir / "stdout.log"
                if stdout_log.exists():
                    parsed = _parse_log_file(stdout_log, metrics)
                    results[exp_name] = {
                        "status": "completed",
                        "metrics": parsed,
                    }
    
    return json.dumps({
        "experiments": list(experiments),
        "results": results,
    }, indent=2)


def _parse_log_file(log_path: Path, metrics: Optional[list[str]] = None) -> dict:
    content = log_path.read_text()
    
    patterns = {
        "loss": r"(?:loss|Loss)[:=]\s*([\d.]+)",
        "accuracy": r"(?:accuracy|Accuracy|acc|Acc)[:=]\s*([\d.]+)",
        "f1": r"(?:f1|F1)[:=]\s*([\d.]+)",
        "precision": r"(?:precision|Precision)[:=]\s*([\d.]+)",
        "recall": r"(?:recall|Recall)[:=]\s*([\d.]+)",
        "epoch": r"(?:epoch|Epoch)[:=]\s*(\d+)",
        "lr": r"(?:lr|learning_rate)[:=]\s*([\d.e-]+)",
        "val_loss": r"(?:val_loss|val loss)[:=]\s*([\d.]+)",
        "val_accuracy": r"(?:val_accuracy|val_acc|val accuracy)[:=]\s*([\d.]+)",
    }
    
    result = {}
    
    for name, pattern in patterns.items():
        if metrics is None or name in metrics:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                values = [float(m) for m in matches]
                result[name] = {
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
    
    return result


async def parse_tensorboard(
    log_dir: str,
    tags: Optional[list[str]] = None,
) -> str:
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Log directory not found: {log_dir}",
        })
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(str(log_path))
        ea.Reload()
        
        available_tags = ea.Tags()
        scalar_tags = available_tags.get("scalars", [])
        
        if tags:
            scalar_tags = [t for t in scalar_tags if t in tags]
        
        data = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            data[tag] = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events
            ]
        
        return json.dumps({
            "success": True,
            "log_dir": log_dir,
            "tags": scalar_tags,
            "data": data,
        }, indent=2)
    
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "tensorboard not installed. Run: pip install tensorboard",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def parse_wandb(
    project: str,
    run_ids: Optional[list[str]] = None,
) -> str:
    try:
        import wandb
        
        api = wandb.Api()
        
        if run_ids:
            runs = [api.run(f"{project}/{run_id}") for run_id in run_ids]
        else:
            runs = api.runs(project, per_page=10)
        
        results = []
        for run in runs:
            history = run.history()
            
            results.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
                "summary": dict(run.summary),
                "history_length": len(history),
            })
        
        return json.dumps({
            "success": True,
            "project": project,
            "runs": results,
        }, indent=2)
    
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "wandb not installed. Run: pip install wandb",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def aggregate_results(
    experiments: list[str],
    group_by: Optional[str] = None,
) -> str:
    all_metrics = {}
    
    for exp_name in experiments:
        exp = await experiments_db.get_experiment(exp_name)
        if exp and exp.metrics:
            for metric, value in exp.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                if isinstance(value, (int, float)):
                    all_metrics[metric].append(value)
                elif isinstance(value, dict) and "final" in value:
                    all_metrics[metric].append(value["final"])
    
    aggregated = {}
    for metric, values in all_metrics.items():
        if values:
            arr = np.array(values)
            aggregated[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }
    
    return json.dumps({
        "experiments": experiments,
        "aggregated": aggregated,
    }, indent=2)


async def export_to_csv(
    results: dict,
    output_path: str,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if isinstance(results, dict):
            if "experiments" in results and "results" in results:
                rows = []
                for exp_name, exp_data in results["results"].items():
                    row = {"experiment": exp_name}
                    if isinstance(exp_data, dict):
                        metrics = exp_data.get("metrics", {})
                        for metric, value in metrics.items():
                            if isinstance(value, dict):
                                row[metric] = value.get("final", value.get("mean", ""))
                            else:
                                row[metric] = value
                    rows.append(row)
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame([results])
        else:
            df = pd.DataFrame(results)
        
        df.to_csv(output, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": str(output),
            "rows": len(df),
            "columns": list(df.columns),
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


async def compute_statistics(
    results: dict,
    confidence_level: float = 0.95,
) -> str:
    statistics = {}
    
    for key, values in results.items():
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            arr = np.array(values)
            n = len(arr)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if n > 1 else 0
            se = std / np.sqrt(n) if n > 0 else 0
            
            if n > 1:
                ci = stats.t.interval(confidence_level, n - 1, loc=mean, scale=se)
            else:
                ci = (mean, mean)
            
            statistics[key] = {
                "n": n,
                "mean": float(mean),
                "std": float(std),
                "se": float(se),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "confidence_level": confidence_level,
            }
    
    return json.dumps({
        "statistics": statistics,
    }, indent=2)
