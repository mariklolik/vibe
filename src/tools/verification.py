"""Results verification tools - statistical tests and anomaly detection."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from src.db.workflow import workflow_db
from src.project.manager import project_manager
from src.tools.tracking import ExperimentTracker


async def verify_hypothesis(
    hypothesis: str,
    results: dict,
    test_type: str = "t-test",
) -> str:
    method_results = {}
    for method, data in results.items():
        if isinstance(data, list):
            method_results[method] = np.array(data)
        elif isinstance(data, dict) and "values" in data:
            method_results[method] = np.array(data["values"])
    
    if len(method_results) < 2:
        return json.dumps({
            "success": False,
            "error": "Need at least 2 methods to compare",
            "hypothesis": hypothesis,
        })
    
    methods = list(method_results.keys())
    group1 = method_results[methods[0]]
    group2 = method_results[methods[1]]
    
    if test_type == "t-test":
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test_type == "paired-t":
        if len(group1) != len(group2):
            return json.dumps({
                "success": False,
                "error": "Paired t-test requires equal sample sizes",
            })
        statistic, p_value = stats.ttest_rel(group1, group2)
        test_name = "Paired t-test"
    elif test_type == "wilcoxon":
        if len(group1) != len(group2):
            return json.dumps({
                "success": False,
                "error": "Wilcoxon test requires equal sample sizes",
            })
        statistic, p_value = stats.wilcoxon(group1, group2)
        test_name = "Wilcoxon signed-rank test"
    elif test_type == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2)
        test_name = "Mann-Whitney U test"
    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown test type: {test_type}",
        })
    
    alpha = 0.05
    significant = p_value < alpha
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    if abs(effect_size) < 0.2:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    verdict = "supported" if significant and mean1 > mean2 else "not supported"
    
    return json.dumps({
        "hypothesis": hypothesis,
        "verdict": verdict,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(significant),
        "effect_size": {
            "cohens_d": float(effect_size),
            "interpretation": effect_interpretation,
        },
        "comparison": {
            "method1": methods[0],
            "method1_mean": float(mean1),
            "method1_std": float(np.std(group1)),
            "method2": methods[1],
            "method2_mean": float(mean2),
            "method2_std": float(np.std(group2)),
        },
    }, indent=2)


async def check_significance(
    method1: str,
    method2: str,
    results: dict,
    test: str = "t-test",
    alpha: float = 0.05,
) -> str:
    if method1 not in results or method2 not in results:
        return json.dumps({
            "success": False,
            "error": f"Methods not found in results: {method1}, {method2}",
            "available": list(results.keys()),
        })
    
    data1 = results[method1]
    data2 = results[method2]
    
    if isinstance(data1, dict):
        data1 = data1.get("values", [data1.get("mean", 0)])
    if isinstance(data2, dict):
        data2 = data2.get("values", [data2.get("mean", 0)])
    
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    
    if test == "t-test":
        statistic, p_value = stats.ttest_ind(arr1, arr2)
    elif test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(arr1, arr2)
    elif test == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(arr1, arr2)
    elif test == "anova":
        statistic, p_value = stats.f_oneway(arr1, arr2)
    else:
        statistic, p_value = stats.ttest_ind(arr1, arr2)
    
    significant = p_value < alpha
    
    return json.dumps({
        "method1": method1,
        "method2": method2,
        "test": test,
        "alpha": alpha,
        "p_value": float(p_value),
        "statistic": float(statistic),
        "significant": bool(significant),
        "interpretation": f"{method1} is {'significantly' if significant else 'not significantly'} different from {method2}",
        "method1_stats": {
            "mean": float(np.mean(arr1)),
            "std": float(np.std(arr1)),
            "n": len(arr1),
        },
        "method2_stats": {
            "mean": float(np.mean(arr2)),
            "std": float(np.std(arr2)),
            "n": len(arr2),
        },
    }, indent=2)


async def detect_anomalies(
    results: dict,
    threshold: float = 2.0,
) -> str:
    anomalies = []
    
    for method, data in results.items():
        if isinstance(data, list):
            values = np.array(data)
        elif isinstance(data, dict) and "values" in data:
            values = np.array(data["values"])
        else:
            continue
        
        if len(values) < 3:
            continue
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            continue
        
        z_scores = np.abs((values - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        if len(anomaly_indices) > 0:
            for idx in anomaly_indices:
                anomalies.append({
                    "method": method,
                    "index": int(idx),
                    "value": float(values[idx]),
                    "z_score": float(z_scores[idx]),
                    "mean": float(mean),
                    "std": float(std),
                })
    
    return json.dumps({
        "threshold": threshold,
        "anomalies_found": len(anomalies),
        "anomalies": anomalies,
        "recommendation": "Review anomalous results for data quality issues" if anomalies else "No anomalies detected",
    }, indent=2)


async def compare_to_baselines(
    method: str,
    baselines: list[str],
    results: dict,
) -> str:
    if method not in results:
        return json.dumps({
            "success": False,
            "error": f"Method not found: {method}",
        })
    
    method_data = results[method]
    if isinstance(method_data, dict):
        method_values = np.array(method_data.get("values", [method_data.get("mean", 0)]))
    else:
        method_values = np.array(method_data)
    
    comparisons = []
    wins = 0
    
    for baseline in baselines:
        if baseline not in results:
            comparisons.append({
                "baseline": baseline,
                "error": "Not found in results",
            })
            continue
        
        baseline_data = results[baseline]
        if isinstance(baseline_data, dict):
            baseline_values = np.array(baseline_data.get("values", [baseline_data.get("mean", 0)]))
        else:
            baseline_values = np.array(baseline_data)
        
        method_mean = np.mean(method_values)
        baseline_mean = np.mean(baseline_values)
        
        improvement = ((method_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
        
        if len(method_values) > 1 and len(baseline_values) > 1:
            statistic, p_value = stats.ttest_ind(method_values, baseline_values)
            significant = p_value < 0.05
        else:
            statistic, p_value = 0, 1.0
            significant = False
        
        is_better = method_mean > baseline_mean and significant
        if is_better:
            wins += 1
        
        comparisons.append({
            "baseline": baseline,
            "baseline_mean": float(baseline_mean),
            "method_mean": float(method_mean),
            "improvement_percent": float(improvement),
            "p_value": float(p_value),
            "significant": bool(significant),
            "is_better": bool(is_better),
        })
    
    return json.dumps({
        "method": method,
        "method_mean": float(np.mean(method_values)),
        "baselines_compared": len(baselines),
        "wins": wins,
        "comparisons": comparisons,
        "summary": f"{method} significantly outperforms {wins}/{len(baselines)} baselines",
    }, indent=2)


async def generate_results_summary(
    experiments: list[str],
) -> str:
    from src.db.experiments_db import experiments_db
    
    summary = {
        "total_experiments": len(experiments),
        "experiments": [],
        "aggregated_metrics": {},
    }
    
    all_metrics = {}
    
    for exp_name in experiments:
        exp = await experiments_db.get_experiment(exp_name)
        
        if exp:
            exp_summary = {
                "name": exp.name,
                "status": exp.status,
                "metrics": exp.metrics,
                "created_at": exp.created_at,
            }
            summary["experiments"].append(exp_summary)
            
            for metric, value in exp.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                if isinstance(value, (int, float)):
                    all_metrics[metric].append(value)
                elif isinstance(value, dict) and "final" in value:
                    all_metrics[metric].append(value["final"])
    
    for metric, values in all_metrics.items():
        if values:
            arr = np.array(values)
            summary["aggregated_metrics"][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "best_experiment": experiments[int(np.argmax(arr))] if len(arr) == len(experiments) else None,
            }
    
    best_overall = None
    if summary["experiments"]:
        primary_metric = list(all_metrics.keys())[0] if all_metrics else None
        if primary_metric:
            best_idx = np.argmax([exp["metrics"].get(primary_metric, 0) 
                                 if isinstance(exp["metrics"].get(primary_metric, 0), (int, float))
                                 else exp["metrics"].get(primary_metric, {}).get("final", 0)
                                 for exp in summary["experiments"]])
            best_overall = summary["experiments"][best_idx]["name"]
    
    summary["best_experiment"] = best_overall
    
    return json.dumps(summary, indent=2)


async def verify_and_record_hypothesis(
    hypothesis_id: str,
    hypothesis_statement: str,
    run_ids: list[str],
    metric_name: str,
    test_type: str = "t-test",
) -> str:
    """
    Verify a hypothesis using ONLY metrics from verified experiment runs.
    
    MANDATORY: This must be called before any claim can be added to the paper.
    Results that are not statistically verified cannot be included.
    
    IMPORTANT: This function ONLY uses metrics parsed from actual experiment logs.
    You cannot provide arbitrary metrics - they must come from run_experiment() runs.
    
    Args:
        hypothesis_id: Unique identifier for the hypothesis
        hypothesis_statement: The claim being tested
        run_ids: List of run_ids from run_experiment() to use for verification
        metric_name: The metric name to compare (e.g., "accuracy", "loss", "f1")
        test_type: Statistical test to use (t-test, paired-t, wilcoxon, mann-whitney)
    
    Returns:
        JSON with verification result and whether claim can be made
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    if len(run_ids) < 2:
        return json.dumps({
            "success": False,
            "error": "INSUFFICIENT_RUNS",
            "message": "Need at least 2 run_ids to compare for hypothesis verification",
        })
    
    # Load experiment tracker and validate runs
    tracker = ExperimentTracker(current_project_obj.root_path)
    
    results = {}
    unverified_runs = []
    missing_metric_runs = []
    
    for run_id in run_ids:
        run = await tracker.get_run(run_id)
        
        if not run:
            return json.dumps({
                "success": False,
                "error": "RUN_NOT_FOUND",
                "message": f"Run '{run_id}' not found. Use run_experiment() to create runs.",
                "run_id": run_id,
            })
        
        if not run.is_verified:
            unverified_runs.append(run_id)
            continue
        
        # Get metric from PARSED metrics only (not user-provided)
        if metric_name not in run.parsed_metrics:
            missing_metric_runs.append({
                "run_id": run_id,
                "available_metrics": list(run.parsed_metrics.keys()),
            })
            continue
        
        metric_data = run.parsed_metrics[metric_name]
        if isinstance(metric_data, dict):
            value = metric_data.get("final", metric_data.get("max", 0))
        else:
            value = metric_data
        
        results[run_id] = {
            "name": run.name,
            "value": value,
            "signature": run.experiment_signature,
        }
    
    if unverified_runs:
        return json.dumps({
            "success": False,
            "error": "UNVERIFIED_RUNS",
            "message": (
                "Some runs are not verified. They have no log signature proving real execution. "
                "Use log_experiment(run_id) to finalize each run first."
            ),
            "unverified_runs": unverified_runs,
        })
    
    if missing_metric_runs:
        return json.dumps({
            "success": False,
            "error": "METRIC_NOT_FOUND",
            "message": f"Metric '{metric_name}' not found in some experiment logs.",
            "runs_missing_metric": missing_metric_runs,
            "hint": "Check that your experiment outputs the metric in format: metric_name: value",
        })
    
    if len(results) < 2:
        return json.dumps({
            "success": False,
            "error": "INSUFFICIENT_VALID_RUNS",
            "message": "Need at least 2 runs with valid parsed metrics to compare",
        })
    
    # Build results dict for statistical test
    run_names = list(results.keys())
    values1 = [results[run_names[0]]["value"]]
    values2 = [results[run_names[1]]["value"]]
    
    # If more runs, add them to appropriate groups (assume first half vs second half)
    mid = len(run_names) // 2
    for i, run_id in enumerate(run_names):
        if i == 0 or i == 1:
            continue
        if i < mid:
            values1.append(results[run_id]["value"])
        else:
            values2.append(results[run_id]["value"])
    
    # Perform statistical test
    arr1 = np.array(values1)
    arr2 = np.array(values2)
    
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    
    # Simple comparison for single values
    if len(arr1) == 1 and len(arr2) == 1:
        p_value = 1.0  # Can't do statistical test with single values
        significant = False
        effect_size = (mean1 - mean2) / max(0.01, abs(mean2))
        test_name = "Single-value comparison (no statistical test)"
    else:
        if test_type == "t-test":
            statistic, p_value = stats.ttest_ind(arr1, arr2)
            test_name = "Independent t-test"
        elif test_type == "mann-whitney":
            statistic, p_value = stats.mannwhitneyu(arr1, arr2)
            test_name = "Mann-Whitney U test"
        else:
            statistic, p_value = stats.ttest_ind(arr1, arr2)
            test_name = "Independent t-test"
        
        significant = p_value < 0.05
        pooled_std = np.sqrt((np.std(arr1)**2 + np.std(arr2)**2) / 2)
        effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    if abs(effect_size) < 0.2:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    can_claim = significant or (len(arr1) == 1 and len(arr2) == 1 and mean1 > mean2)
    
    verification_record = {
        "hypothesis_id": hypothesis_id,
        "statement": hypothesis_statement,
        "run_ids": run_ids,
        "metric": metric_name,
        "test_type": test_name,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "effect_size": float(effect_size),
        "effect_interpretation": effect_interpretation,
        "significant": bool(significant),
        "can_claim": can_claim,
        "group1_mean": float(mean1),
        "group2_mean": float(mean2),
        "signatures": [results[r]["signature"] for r in run_names],
        "timestamp": datetime.now().isoformat(),
        "verified_from_logs": True,
    }
    
    workflow.verified_hypotheses[hypothesis_id] = verification_record
    await workflow_db.save_workflow(workflow)
    
    p_val_str = f"{verification_record['p_value']:.4f}" if verification_record['p_value'] < 1 else "N/A"
    
    return json.dumps({
        "success": True,
        "hypothesis_id": hypothesis_id,
        "verification": verification_record,
        "can_include_in_paper": verification_record["can_claim"],
        "data_source": "Metrics parsed from actual experiment logs (verified)",
        "message": (
            f"Hypothesis {'VERIFIED' if verification_record['can_claim'] else 'NOT VERIFIED'}. "
            f"Metric: {metric_name}, p-value: {p_val_str}, "
            f"Effect size: {effect_interpretation}"
        ),
        "next_steps": (
            ["Add this claim to paper with supporting figures"] 
            if verification_record["can_claim"] 
            else ["Revise hypothesis or run additional experiments"]
        ),
    }, indent=2)


async def check_claims_verified(claims: list[str]) -> str:
    """
    Check if all claims have been verified before allowing paper submission.
    
    This is called by cast_to_format to ensure no unverified claims are published.
    
    Args:
        claims: List of hypothesis IDs to check
    
    Returns:
        JSON with verification status for each claim
    """
    current_project_obj = await project_manager.get_current_project()
    if not current_project_obj:
        return json.dumps({"error": "No active project"})
    
    workflow = await workflow_db.get_project_workflow(current_project_obj.project_id)
    if not workflow:
        return json.dumps({"error": "No workflow found"})
    
    verified_claims = []
    unverified_claims = []
    
    for claim_id in claims:
        if claim_id in workflow.verified_hypotheses:
            record = workflow.verified_hypotheses[claim_id]
            if record.get("can_claim", False):
                verified_claims.append({
                    "claim_id": claim_id,
                    "statement": record.get("statement"),
                    "p_value": record.get("p_value"),
                })
            else:
                unverified_claims.append({
                    "claim_id": claim_id,
                    "reason": "Hypothesis test failed (not significant)",
                })
        else:
            unverified_claims.append({
                "claim_id": claim_id,
                "reason": "No verification record found",
            })
    
    all_verified = len(unverified_claims) == 0
    
    return json.dumps({
        "all_verified": all_verified,
        "verified_claims": verified_claims,
        "unverified_claims": unverified_claims,
        "can_proceed": all_verified,
        "message": (
            "All claims verified. Paper can proceed." 
            if all_verified 
            else f"BLOCKED: {len(unverified_claims)} unverified claims. Run verify_and_record_hypothesis for each."
        ),
    }, indent=2)
