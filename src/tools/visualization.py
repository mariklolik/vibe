"""Visualization tools - generate plots and figures for papers."""

import json
from io import BytesIO
from pathlib import Path
from typing import Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.db.conferences import get_conference
from src.db.workflow import workflow_db
from src.project.manager import project_manager


OUTPUTS_DIR = Path("./figures")


async def _track_figure_generated(figure_path: str):
    """Track that a figure was generated in the workflow."""
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if workflow and figure_path not in workflow.figures_generated:
        workflow.figures_generated.append(figure_path)
        await workflow_db.save_workflow(workflow)

CONFERENCE_STYLES = {
    "neurips": {
        "figure.figsize": (6.75, 4.5),
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    },
    "icml": {
        "figure.figsize": (6.75, 4.5),
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    },
    "cvpr": {
        "figure.figsize": (6.875, 4.5),
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    },
    "acl": {
        "figure.figsize": (6.5, 4.0),
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    },
}

COLOR_PALETTES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "colorblind": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"],
    "grayscale": ["#000000", "#555555", "#888888", "#aaaaaa", "#cccccc"],
}


def _ensure_output_dir():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_conference_style(conference: Optional[str] = None):
    style = CONFERENCE_STYLES.get(conference, CONFERENCE_STYLES["neurips"])
    plt.rcParams.update(style)
    sns.set_palette(COLOR_PALETTES["colorblind"])


async def plot_training_curves(
    experiments: list[str],
    metrics: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    conference: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    _apply_conference_style(conference)
    
    if not metrics:
        metrics = ["loss", "accuracy"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    if len(metrics) == 1:
        axes = [axes]
    
    from src.db.experiments_db import experiments_db
    
    for exp_name in experiments:
        exp = await experiments_db.get_experiment(exp_name)
        if exp:
            history = await experiments_db.get_metric_history(exp.experiment_id)
            
            for ax, metric in zip(axes, metrics):
                metric_data = [h for h in history if h["metric_name"] == metric]
                if metric_data:
                    steps = [h.get("step", i) for i, h in enumerate(metric_data)]
                    values = [h["metric_value"] for h in metric_data]
                    ax.plot(steps, values, label=exp_name)
    
    for ax, metric in zip(axes, metrics):
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "training_curves.pdf"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Track figure in workflow
    await _track_figure_generated(str(output_path))
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "experiments": experiments,
        "metrics": metrics,
        "conference_style": conference,
        "workflow_updated": True,
    })


async def plot_comparison_bar(
    results: dict,
    metric: str,
    output_path: Optional[str] = None,
    conference: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    _apply_conference_style(conference)
    
    methods = list(results.keys())
    values = []
    errors = []
    
    for method in methods:
        data = results[method]
        if isinstance(data, dict):
            values.append(data.get(metric, data.get("mean", 0)))
            errors.append(data.get("std", 0))
        else:
            values.append(data)
            errors.append(0)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    colors = COLOR_PALETTES["colorblind"][:len(methods)]
    x = np.arange(len(methods))
    
    bars = ax.bar(x, values, yerr=errors, capsize=4, color=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric.capitalize())
    ax.grid(True, axis="y", alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / f"comparison_{metric}.pdf"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Track figure in workflow
    await _track_figure_generated(str(output_path))
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "methods": methods,
        "metric": metric,
        "workflow_updated": True,
    })


async def plot_ablation_table(
    results: dict,
    output_path: Optional[str] = None,
    conference: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "ablation_table.tex"
    else:
        output_path = Path(output_path)
    
    headers = ["Configuration"]
    metrics = set()
    for config_data in results.values():
        if isinstance(config_data, dict):
            metrics.update(config_data.keys())
    metrics = sorted(metrics)
    headers.extend(metrics)
    
    rows = []
    best_values = {m: float("-inf") for m in metrics}
    
    for config, data in results.items():
        row = [config.replace("_", "\\_")]
        for metric in metrics:
            val = data.get(metric, "-")
            if isinstance(val, float):
                row.append(f"{val:.2f}")
                if val > best_values[metric]:
                    best_values[metric] = val
            else:
                row.append(str(val))
        rows.append(row)
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Ablation study results.}\n"
    latex += "\\label{tab:ablation}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    latex += "\\toprule\n"
    latex += " & ".join(headers) + " \\\\\n"
    latex += "\\midrule\n"
    
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "configurations": list(results.keys()),
        "metrics": metrics,
        "latex_preview": latex[:500],
    })


async def plot_scatter(
    x_data: list,
    y_data: list,
    labels: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    regression: bool = False,
) -> str:
    _ensure_output_dir()
    _apply_conference_style(None)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    colors = COLOR_PALETTES["colorblind"]
    
    ax.scatter(x_data, y_data, c=colors[0], alpha=0.7, s=50)
    
    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), fontsize=7, alpha=0.8)
    
    if regression and len(x_data) > 2:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_data), max(x_data), 100)
        ax.plot(x_line, p(x_line), "--", color=colors[1], alpha=0.8, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "scatter.pdf"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "points": len(x_data),
        "regression": regression,
    })


async def plot_heatmap(
    data: list,
    x_labels: Optional[list[str]] = None,
    y_labels: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    _apply_conference_style(None)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    arr = np.array(data)
    
    sns.heatmap(arr, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=x_labels, yticklabels=y_labels, ax=ax)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "heatmap.pdf"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "shape": arr.shape,
    })


async def plot_qualitative(
    images: list[str],
    labels: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    
    from PIL import Image
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    
    for i, (ax, img_path) in enumerate(zip(axes, images)):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, f"Image not found:\n{img_path}", 
                   ha="center", va="center", transform=ax.transAxes)
        
        ax.axis("off")
        if labels and i < len(labels):
            ax.set_title(labels[i])
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "qualitative.pdf"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "images": len(images),
    })


async def generate_architecture_diagram(
    components: list[dict],
    connections: Optional[list[dict]] = None,
    output_path: Optional[str] = None,
) -> str:
    _ensure_output_dir()
    
    mermaid_code = "graph LR\n"
    
    for comp in components:
        comp_id = comp.get("id", comp.get("name", "node"))
        comp_label = comp.get("label", comp.get("name", "Node"))
        shape = comp.get("shape", "rect")
        
        if shape == "circle":
            mermaid_code += f"    {comp_id}(({comp_label}))\n"
        elif shape == "diamond":
            mermaid_code += f"    {comp_id}{{{comp_label}}}\n"
        else:
            mermaid_code += f"    {comp_id}[{comp_label}]\n"
    
    if connections:
        for conn in connections:
            src = conn.get("from", conn.get("source"))
            dst = conn.get("to", conn.get("target"))
            label = conn.get("label", "")
            
            if label:
                mermaid_code += f"    {src} -->|{label}| {dst}\n"
            else:
                mermaid_code += f"    {src} --> {dst}\n"
    
    if output_path is None:
        output_path = OUTPUTS_DIR / "architecture.md"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = f"```mermaid\n{mermaid_code}```\n"
    output_path.write_text(content)
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "mermaid_code": mermaid_code,
        "components": len(components),
        "connections": len(connections) if connections else 0,
    })


async def style_for_conference(
    figure_path: str,
    conference: str,
    output_path: Optional[str] = None,
) -> str:
    conf = get_conference(conference)
    
    if not conf:
        return json.dumps({
            "success": False,
            "error": f"Conference not found: {conference}",
        })
    
    fig_path = Path(figure_path)
    if not fig_path.exists():
        return json.dumps({
            "success": False,
            "error": f"Figure not found: {figure_path}",
        })
    
    guidelines = conf.figure_guidelines
    
    if output_path is None:
        output_path = fig_path.parent / f"{fig_path.stem}_{conference}{fig_path.suffix}"
    else:
        output_path = Path(output_path)
    
    import shutil
    shutil.copy(fig_path, output_path)
    
    return json.dumps({
        "success": True,
        "output_path": str(output_path),
        "conference": conference,
        "guidelines": guidelines,
        "note": "Figure copied. Apply manual adjustments per guidelines.",
    })
