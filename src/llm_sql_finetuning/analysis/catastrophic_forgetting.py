"""
Analyze catastrophic forgetting metrics from benchmark results across training steps.

This module provides functions to:
1. Load benchmark results from multiple steps
2. Calculate catastrophic forgetting metrics
3. Analyze performance degradation on different tasks
4. Generate visualizations and reports
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class CatastrophicForgettingMetrics:
    """Metrics for catastrophic forgetting analysis."""
    step: int
    task_name: str
    initial_performance: float
    current_performance: float
    absolute_degradation: float
    relative_degradation: float  # percentage change
    is_degraded: bool  # whether performance decreased
    retention_rate: Optional[float] = None  # acc_t / acc_0 * 100%
    forgetting_measure: Optional[float] = None  # max(acc_k) - acc_t for k < t
    backward_transfer: Optional[float] = None  # acc_t - acc_0


@dataclass
class StepSummary:
    """Summary statistics for a training step."""
    step: int
    avg_performance: float  # ACC: Average Accuracy across all tasks
    avg_degradation: float
    num_degraded_tasks: int
    total_tasks: int
    degradation_rate: float  # percentage of tasks with degradation
    avg_retention_rate: Optional[float] = None  # average retention rate
    avg_backward_transfer: Optional[float] = None  # BWT: Backward Transfer
    avg_forgetting_measure: Optional[float] = None  # FORG: Forgetting Measure
    catastrophic_forgetting_test: Optional[float] = None  # CBT: Catastrophic Forgetting Test


def load_benchmark_results(results_dir: Path) -> Dict[int, Dict]:
    """
    Load all benchmark result files from a directory.
    
    Args:
        results_dir: Path to directory containing step_*_results.json files
        
    Returns:
        Dictionary mapping step number to results
    """
    results = {}
    json_files = sorted(results_dir.glob("step_*_results.json"))
    
    for json_file in json_files:
        # Extract step number from filename
        step_str = json_file.stem.split("_")[1]
        try:
            step = int(step_str)
            with open(json_file, "r") as f:
                results[step] = json.load(f)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def extract_metric_by_task(
    results: Dict[int, Dict], 
    metric_key: str = "acc,none"
) -> Dict[int, Dict[str, float]]:
    """
    Extract specific metric values for each task across all steps.
    
    Args:
        results: Dictionary mapping step to results
        metric_key: The metric key to extract (default: "acc,none")
        
    Returns:
        Dictionary mapping step to {task_name: metric_value}
    """
    metric_data = {}
    
    for step, step_results in results.items():
        metric_data[step] = {}
        for task_name, task_results in step_results.get("results", {}).items():
            if metric_key in task_results:
                metric_data[step][task_name] = task_results[metric_key]
    
    return metric_data


def calculate_forgetting_metrics(
    metric_data: Dict[int, Dict[str, float]],
    baseline_step: int = 0
) -> List[CatastrophicForgettingMetrics]:
    """
    Calculate catastrophic forgetting metrics.
    
    Args:
        metric_data: Dictionary mapping step to {task_name: metric_value}
        baseline_step: Step to use as baseline (default: step 0)
        
    Returns:
        List of CatastrophicForgettingMetrics objects
    """
    metrics = []
    
    if baseline_step not in metric_data:
        raise ValueError(f"Baseline step {baseline_step} not found in data")
    
    baseline_performance = metric_data[baseline_step]
    sorted_steps = sorted(metric_data.keys())
    
    # Track best performance for each task (for Forgetting Measure)
    task_best_performance = {task: perf for task, perf in baseline_performance.items()}
    
    for step in sorted_steps:
        current_performance = metric_data[step]
        
        for task_name, current_perf in current_performance.items():
            baseline_perf = baseline_performance.get(task_name, current_perf)
            
            # Update best performance seen so far
            if task_name in task_best_performance:
                task_best_performance[task_name] = max(
                    task_best_performance[task_name], 
                    current_perf
                )
            else:
                task_best_performance[task_name] = current_perf
            
            # Calculate basic metrics
            abs_degradation = baseline_perf - current_perf
            rel_degradation = ((baseline_perf - current_perf) / baseline_perf * 100) if baseline_perf != 0 else 0
            is_degraded = current_perf < baseline_perf
            
            # Calculate enhanced metrics
            retention_rate = (current_perf / baseline_perf * 100) if baseline_perf != 0 else None
            backward_transfer = current_perf - baseline_perf
            
            # Forgetting Measure: max performance so far - current performance
            best_perf_so_far = task_best_performance.get(task_name, baseline_perf)
            forgetting_measure = best_perf_so_far - current_perf
            
            metrics.append(
                CatastrophicForgettingMetrics(
                    step=step,
                    task_name=task_name,
                    initial_performance=baseline_perf,
                    current_performance=current_perf,
                    absolute_degradation=abs_degradation,
                    relative_degradation=rel_degradation,
                    is_degraded=is_degraded,
                    retention_rate=retention_rate,
                    forgetting_measure=forgetting_measure,
                    backward_transfer=backward_transfer
                )
            )
    
    return metrics


def calculate_step_summaries(
    metrics: List[CatastrophicForgettingMetrics]
) -> Dict[int, StepSummary]:
    """
    Calculate summary statistics for each training step.
    
    Args:
        metrics: List of CatastrophicForgettingMetrics
        
    Returns:
        Dictionary mapping step to StepSummary
    """
    step_metrics = {}
    
    for metric in metrics:
        if metric.step not in step_metrics:
            step_metrics[metric.step] = []
        step_metrics[metric.step].append(metric)
    
    summaries = {}
    for step, step_metric_list in step_metrics.items():
        degradations = [m.absolute_degradation for m in step_metric_list]
        degraded_count = sum(1 for m in step_metric_list if m.is_degraded)
        
        # Calculate enhanced metrics
        retention_rates = [m.retention_rate for m in step_metric_list if m.retention_rate is not None]
        backward_transfers = [m.backward_transfer for m in step_metric_list if m.backward_transfer is not None]
        forgetting_measures = [m.forgetting_measure for m in step_metric_list if m.forgetting_measure is not None]
        
        # CBT: Catastrophic Forgetting Test = average forgetting measure
        # This measures the average difference between best performance and current performance
        cbt = float(np.mean(forgetting_measures)) if forgetting_measures else None
        
        summaries[step] = StepSummary(
            step=step,
            avg_performance=np.mean([m.current_performance for m in step_metric_list]),  # ACC
            avg_degradation=np.mean(degradations),
            num_degraded_tasks=degraded_count,
            total_tasks=len(step_metric_list),
            degradation_rate=(degraded_count / len(step_metric_list) * 100) if step_metric_list else 0,
            avg_retention_rate=float(np.mean(retention_rates)) if retention_rates else None,
            avg_backward_transfer=float(np.mean(backward_transfers)) if backward_transfers else None,  # BWT
            avg_forgetting_measure=float(np.mean(forgetting_measures)) if forgetting_measures else None,  # FORG
            catastrophic_forgetting_test=cbt  # CBT
        )
    
    return summaries


def get_most_affected_tasks(
    metrics: List[CatastrophicForgettingMetrics],
    top_n: int = 10,
    step_filter: Optional[int] = None
) -> List[Tuple[str, float, float]]:
    """
    Get the most affected tasks by catastrophic forgetting.
    
    Args:
        metrics: List of CatastrophicForgettingMetrics
        top_n: Number of top tasks to return
        step_filter: Filter to specific step (if None, uses final step)
        
    Returns:
        List of tuples (task_name, absolute_degradation, relative_degradation)
    """
    if step_filter is not None:
        filtered_metrics = [m for m in metrics if m.step == step_filter]
    else:
        # Use the last step
        max_step = max(m.step for m in metrics)
        filtered_metrics = [m for m in metrics if m.step == max_step]
    
    # Sort by absolute degradation
    sorted_metrics = sorted(
        filtered_metrics,
        key=lambda m: m.absolute_degradation,
        reverse=True
    )
    
    return [
        (m.task_name, m.absolute_degradation, m.relative_degradation)
        for m in sorted_metrics[:top_n]
    ]


def generate_report(
    results_dir: Path,
    output_file: Optional[Path] = None,
    metric_key: str = "acc,none"
) -> Dict:
    """
    Generate a comprehensive catastrophic forgetting analysis report.
    
    Args:
        results_dir: Path to directory containing benchmark results
        output_file: Optional path to save report as JSON
        metric_key: Metric key to analyze
        
    Returns:
        Dictionary containing analysis report
    """
    # Load and process data
    results = load_benchmark_results(results_dir)
    metric_data = extract_metric_by_task(results, metric_key)
    metrics = calculate_forgetting_metrics(metric_data)
    step_summaries = calculate_step_summaries(metrics)
    
    # Get top affected tasks
    most_affected = get_most_affected_tasks(metrics, top_n=15)
    
    # Prepare report
    report = {
        "metadata": {
            "total_steps": len(results),
            "metric_analyzed": metric_key,
            "steps": sorted(results.keys())
        },
        "step_summaries": {
            str(step): {
                "step": summary.step,
                "avg_performance": float(summary.avg_performance),
                "avg_degradation": float(summary.avg_degradation),
                "num_degraded_tasks": summary.num_degraded_tasks,
                "total_tasks": summary.total_tasks,
                "degradation_rate": float(summary.degradation_rate),
                "avg_retention_rate": float(summary.avg_retention_rate) if summary.avg_retention_rate is not None else None,
                "avg_backward_transfer": float(summary.avg_backward_transfer) if summary.avg_backward_transfer is not None else None,
                "avg_forgetting_measure": float(summary.avg_forgetting_measure) if summary.avg_forgetting_measure is not None else None,
                "catastrophic_forgetting_test": float(summary.catastrophic_forgetting_test) if summary.catastrophic_forgetting_test is not None else None,
            }
            for step, summary in step_summaries.items()
        },
        "most_affected_tasks": [
            {
                "task_name": name,
                "absolute_degradation": float(abs_deg),
                "relative_degradation_percent": float(rel_deg)
            }
            for name, abs_deg, rel_deg in most_affected
        ]
    }
    
    # Calculate overall statistics
    all_degradations = [m.absolute_degradation for m in metrics if m.step == max(step_summaries.keys())]
    report["overall_statistics"] = {
        "mean_degradation": float(np.mean(all_degradations)),
        "std_degradation": float(np.std(all_degradations)),
        "max_degradation": float(np.max(all_degradations)),
        "min_degradation": float(np.min(all_degradations)),
        "median_degradation": float(np.median(all_degradations))
    }
    
    # Save report if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    
    return report


def print_summary_table(step_summaries: Dict[int, StepSummary]) -> None:
    """
    Print a formatted summary table of forgetting metrics by step.
    
    Args:
        step_summaries: Dictionary mapping step to StepSummary
    """
    print("\n" + "="*120)
    print("CATASTROPHIC FORGETTING ANALYSIS - SUMMARY BY STEP")
    print("="*120)
    header = f"{'Step':<10} {'ACC':<10} {'BWT':<10} {'FORG':<10} {'CBT':<10} {'Degraded':<12} {'Rate%':<10}"
    print(header)
    print("-"*120)
    
    for step in sorted(step_summaries.keys()):
        summary = step_summaries[step]
        acc = summary.avg_performance
        bwt = summary.avg_backward_transfer if summary.avg_backward_transfer is not None else 0.0
        forg = summary.avg_forgetting_measure if summary.avg_forgetting_measure is not None else 0.0
        cbt = summary.catastrophic_forgetting_test if summary.catastrophic_forgetting_test is not None else 0.0
        
        row = f"{step:<10} {acc:<10.4f} {bwt:<10.4f} {forg:<10.4f} {cbt:<10.4f} " \
              f"{summary.num_degraded_tasks}/{summary.total_tasks:<8} {summary.degradation_rate:<9.1f}%"
        print(row)
    
    print("="*120 + "\n")


def print_most_affected_table(
    most_affected: List[Tuple[str, float, float]]
) -> None:
    """
    Print a formatted table of most affected tasks.
    
    Args:
        most_affected: List of tuples (task_name, abs_degradation, rel_degradation)
    """
    print("\n" + "="*100)
    print("TOP AFFECTED TASKS (by absolute degradation)")
    print("="*100)
    print(f"{'Rank':<6} {'Task Name':<50} {'Abs. Degradation':<20} {'Rel. Degradation (%)':<15}")
    print("-"*100)
    
    for rank, (task_name, abs_deg, rel_deg) in enumerate(most_affected, 1):
        print(f"{rank:<6} {task_name:<50} {abs_deg:<20.6f} {rel_deg:<14.2f}%")
    
    print("="*100 + "\n")
