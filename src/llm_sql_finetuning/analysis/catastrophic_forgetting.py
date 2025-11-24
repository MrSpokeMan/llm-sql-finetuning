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
from dataclasses import dataclass
from collections import defaultdict
from typing import Any


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


def load_benchmark_results(results_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
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


def identify_benchmark(task_name: str) -> Optional[str]:
    """
    Identify which benchmark a task belongs to.

    Args:
        task_name: Name of the task

    Returns:
        Benchmark name: 'MMLU_en', 'MMLU_pl', 'MMLU_ru', 'Hellaswag', or None
    """
    task_lower = task_name.lower()
    if task_lower.startswith("global_mmlu_full_en"):
        return "MMLU_en"
    elif task_lower.startswith("global_mmlu_full_pl"):
        return "MMLU_pl"
    elif task_lower.startswith("global_mmlu_full_ru"):
        return "MMLU_ru"
    elif task_lower.startswith("hellaswag"):
        return "Hellaswag"
    return None


def extract_mmlu_subcategory(task_name: str) -> Optional[str]:
    """
    Extract MMLU subcategory from task name.

    Args:
        task_name: Name of the task (e.g., 'global_mmlu_full_en_machine_learning')

    Returns:
        Subcategory name (e.g., 'machine_learning') or None
    """
    parts = task_name.split("_")
    if len(parts) >= 5 and parts[0] == "global" and parts[1] == "mmlu" and parts[2] == "full":
        # Extract subcategory (everything after language code)
        # Format: global_mmlu_full_{lang}_{subcategory}
        if len(parts) > 4:
            return "_".join(parts[4:])
    return None


def filter_tasks_by_benchmark(metric_data: Dict[int, Dict[str, float]], benchmark: str) -> Dict[int, Dict[str, float]]:
    """
    Filter metric data to include only tasks from a specific benchmark.

    Args:
        metric_data: Dictionary mapping step to {task_name: metric_value}
        benchmark: Benchmark name ('MMLU_en', 'MMLU_pl', 'MMLU_ru', 'Hellaswag')

    Returns:
        Filtered metric data
    """
    filtered_data = {}
    for step, tasks in metric_data.items():
        filtered_data[step] = {task_name: value for task_name, value in tasks.items() if identify_benchmark(task_name) == benchmark}
    return filtered_data


def apply_noise_reduction(values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply noise reduction by averaging the last N values.

    Args:
        values: List of values to smooth
        window_size: Number of last values to average (default: 5)

    Returns:
        List of smoothed values
    """
    if len(values) < window_size:
        return values

    smoothed = values.copy()
    for i in range(len(values) - window_size, len(values)):
        # Average the last window_size values up to and including current index
        start_idx = max(0, i - window_size + 1)
        smoothed[i] = np.mean(values[start_idx : i + 1])

    return smoothed


def extract_metric_by_task(
    results: Dict[int, Dict[str, Dict[str, Any]]], metric_key: str = "acc,none", benchmark_filter: Optional[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Extract specific metric values for each task across all steps.

    Args:
        results: Dictionary mapping step to results
        metric_key: The metric key to extract (default: "acc,none")
        benchmark_filter: Optional benchmark to filter by ('MMLU_en', 'MMLU_pl', 'MMLU_ru', 'Hellaswag')

    Returns:
        Dictionary mapping step to {task_name: metric_value}
    """
    metric_data: Dict[int, Dict[str, float]] = {}

    for step, step_results in results.items():
        metric_data[step] = {}
        for task_name, task_results in step_results.get("results", {}).items():
            if metric_key in task_results:
                # Apply benchmark filter if specified
                if benchmark_filter is None or identify_benchmark(task_name) == benchmark_filter:
                    metric_value = task_results[metric_key]
                    # Ensure the value is a float
                    metric_data[step][task_name] = float(metric_value) if not isinstance(metric_value, float) else metric_value

    return metric_data


def calculate_forgetting_metrics(
    metric_data: Dict[int, Dict[str, float]], baseline_step: int = 0
) -> List[CatastrophicForgettingMetrics]:
    """
    Calculate catastrophic forgetting metrics.

    Args:
        metric_data: Dictionary mapping step to {task_name: metric_value}
        baseline_step: Step to use as baseline (default: step 0, or lowest step if 0 not found)

    Returns:
        List of CatastrophicForgettingMetrics objects
    """
    metrics = []

    # If baseline step not found, use the lowest step number as baseline
    if baseline_step not in metric_data:
        if not metric_data:
            raise ValueError("No metric data available")
        actual_baseline_step = min(metric_data.keys())
        print(f"⚠️  Warning: Baseline step {baseline_step} not found. Using step {actual_baseline_step} as baseline instead.")
        baseline_step = actual_baseline_step

    baseline_performance: Dict[str, float] = metric_data[baseline_step]
    sorted_steps = sorted(metric_data.keys())

    # Track best performance for each task (for Forgetting Measure)
    task_best_performance: Dict[str, float] = {task: float(perf) for task, perf in baseline_performance.items()}

    for step in sorted_steps:
        current_performance: Dict[str, float] = metric_data[step]

        for task_name, current_perf in current_performance.items():
            # Ensure we have float values - current_perf is already float from Dict[str, float]
            current_perf_float: float = float(current_perf) if not isinstance(current_perf, float) else current_perf
            # Get baseline performance, defaulting to current if not found
            baseline_perf_value = baseline_performance.get(task_name)
            if baseline_perf_value is None:
                baseline_perf: float = current_perf_float
            else:
                baseline_perf = float(baseline_perf_value) if not isinstance(baseline_perf_value, float) else baseline_perf_value

            # Update best performance seen so far
            if task_name in task_best_performance:
                task_best_performance[task_name] = max(task_best_performance[task_name], current_perf_float)
            else:
                task_best_performance[task_name] = current_perf_float

            # Calculate basic metrics
            abs_degradation = baseline_perf - current_perf_float
            rel_degradation = ((baseline_perf - current_perf_float) / baseline_perf * 100) if baseline_perf != 0 else 0.0
            is_degraded = current_perf_float < baseline_perf

            # Calculate enhanced metrics
            retention_rate: Optional[float] = (current_perf_float / baseline_perf * 100) if baseline_perf != 0 else None
            backward_transfer = current_perf_float - baseline_perf

            # Forgetting Measure: max performance so far - current performance
            best_perf_value = task_best_performance.get(task_name)
            if best_perf_value is None:
                best_perf_so_far: float = baseline_perf
            else:
                best_perf_so_far = float(best_perf_value) if not isinstance(best_perf_value, float) else best_perf_value
            forgetting_measure = best_perf_so_far - current_perf_float

            metrics.append(
                CatastrophicForgettingMetrics(
                    step=step,
                    task_name=task_name,
                    initial_performance=baseline_perf,
                    current_performance=current_perf_float,
                    absolute_degradation=abs_degradation,
                    relative_degradation=rel_degradation,
                    is_degraded=is_degraded,
                    retention_rate=retention_rate,
                    forgetting_measure=forgetting_measure,
                    backward_transfer=backward_transfer,
                )
            )

    return metrics


def calculate_step_summaries(
    metrics: List[CatastrophicForgettingMetrics],
    use_percentage_focus: bool = True,
    apply_smoothing: bool = True,
    smoothing_window: int = 5,
) -> Dict[int, StepSummary]:
    """
    Calculate summary statistics for each training step.

    Args:
        metrics: List of CatastrophicForgettingMetrics
        use_percentage_focus: If True, focus on percentage-based metrics
        apply_smoothing: If True, apply noise reduction to degradation metrics
        smoothing_window: Window size for smoothing (default: 5)

    Returns:
        Dictionary mapping step to StepSummary
    """
    step_metrics: Dict[int, List[CatastrophicForgettingMetrics]] = {}

    for metric in metrics:
        if metric.step not in step_metrics:
            step_metrics[metric.step] = []
        step_metrics[metric.step].append(metric)

    summaries = {}
    sorted_steps = sorted(step_metrics.keys())

    # Collect all degradation values for smoothing
    all_avg_degradations = []
    all_relative_degradations = []

    for step in sorted_steps:
        step_metric_list = step_metrics[step]
        degradations = [m.absolute_degradation for m in step_metric_list]
        relative_degradations = [m.relative_degradation for m in step_metric_list]
        degraded_count = sum(1 for m in step_metric_list if m.is_degraded)

        # Calculate enhanced metrics
        retention_rates = [m.retention_rate for m in step_metric_list if m.retention_rate is not None]
        backward_transfers = [m.backward_transfer for m in step_metric_list if m.backward_transfer is not None]
        forgetting_measures = [m.forgetting_measure for m in step_metric_list if m.forgetting_measure is not None]

        # CBT: Catastrophic Forgetting Test = average forgetting measure
        # This measures the average difference between best performance and current performance
        cbt = float(np.mean(forgetting_measures)) if forgetting_measures else None

        avg_degradation = float(np.mean(degradations))
        avg_relative_degradation = float(np.mean(relative_degradations)) if relative_degradations else 0.0

        all_avg_degradations.append(avg_degradation)
        all_relative_degradations.append(avg_relative_degradation)

        # Use relative degradation as primary metric if percentage focus is enabled
        if use_percentage_focus:
            primary_degradation = avg_relative_degradation
        else:
            primary_degradation = avg_degradation

        summaries[step] = StepSummary(
            step=step,
            avg_performance=float(np.mean([m.current_performance for m in step_metric_list])),  # ACC
            avg_degradation=float(primary_degradation),  # Use percentage if focus enabled
            num_degraded_tasks=degraded_count,
            total_tasks=len(step_metric_list),
            degradation_rate=(degraded_count / len(step_metric_list) * 100) if step_metric_list else 0,
            avg_retention_rate=float(np.mean(retention_rates)) if retention_rates else None,
            avg_backward_transfer=float(np.mean(backward_transfers)) if backward_transfers else None,  # BWT
            avg_forgetting_measure=float(np.mean(forgetting_measures)) if forgetting_measures else None,  # FORG
            catastrophic_forgetting_test=cbt,  # CBT
        )

    # Apply smoothing if requested
    if apply_smoothing and len(sorted_steps) >= smoothing_window:
        smoothed_degradations = apply_noise_reduction(all_avg_degradations, smoothing_window)
        smoothed_relative = apply_noise_reduction(all_relative_degradations, smoothing_window)

        for i, step in enumerate(sorted_steps):
            if use_percentage_focus:
                summaries[step].avg_degradation = float(smoothed_relative[i])
            else:
                summaries[step].avg_degradation = float(smoothed_degradations[i])

    return summaries


def get_most_affected_tasks(
    metrics: List[CatastrophicForgettingMetrics], top_n: int = 10, step_filter: Optional[int] = None, use_percentage_focus: bool = True
) -> List[Tuple[str, float, float]]:
    """
    Get the most affected tasks by catastrophic forgetting.

    Args:
        metrics: List of CatastrophicForgettingMetrics
        top_n: Number of top tasks to return
        step_filter: Filter to specific step (if None, uses final step)
        use_percentage_focus: If True, sort by relative degradation instead of absolute

    Returns:
        List of tuples (task_name, absolute_degradation, relative_degradation)
    """
    if step_filter is not None:
        filtered_metrics = [m for m in metrics if m.step == step_filter]
    else:
        # Use the last step
        max_step = max(m.step for m in metrics)
        filtered_metrics = [m for m in metrics if m.step == max_step]

    # Sort by relative degradation if percentage focus is enabled, otherwise by absolute
    if use_percentage_focus:
        sorted_metrics = sorted(filtered_metrics, key=lambda m: m.relative_degradation, reverse=True)
    else:
        sorted_metrics = sorted(filtered_metrics, key=lambda m: m.absolute_degradation, reverse=True)

    return [(m.task_name, m.absolute_degradation, m.relative_degradation) for m in sorted_metrics[:top_n]]


def divide_by_category(metrics: List[CatastrophicForgettingMetrics]) -> Dict[str, List[CatastrophicForgettingMetrics]]:
    """
    Divide metrics into categories: Science, Humanities, Social Sciences, Medical.

    Uses keyword matching to categorize MMLU tasks. Falls back to API if needed.

    Args:
        metrics: List of CatastrophicForgettingMetrics

    Returns:
        Dictionary mapping category name to list of metrics
    """
    # Define category keywords
    category_keywords = {
        "Science": [
            "physics",
            "chemistry",
            "biology",
            "mathematics",
            "computer_science",
            "astronomy",
            "anatomy",
            "electrical_engineering",
            "machine_learning",
            "statistics",
            "abstract_algebra",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "conceptual_physics",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
        ],
        "Humanities": [
            "history",
            "philosophy",
            "religion",
            "world_religions",
            "prehistory",
            "formal_logic",
            "logical_fallacies",
            "moral_disputes",
            "moral_scenarios",
            "high_school_european_history",
            "high_school_us_history",
            "high_school_world_history",
            "jurisprudence",
            "professional_law",
            "international_law",
        ],
        "Social_Sciences": [
            "psychology",
            "sociology",
            "economics",
            "geography",
            "government",
            "politics",
            "macroeconomics",
            "microeconomics",
            "econometrics",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_microeconomics",
            "high_school_psychology",
            "professional_psychology",
            "security_studies",
            "us_foreign_policy",
            "public_relations",
            "human_sexuality",
        ],
        "Medical": [
            "medicine",
            "clinical",
            "medical_genetics",
            "virology",
            "nutrition",
            "human_aging",
            "college_medicine",
            "professional_medicine",
            "clinical_knowledge",
        ],
    }

    # Initialize result dictionary
    categorized: Dict[str, List[CatastrophicForgettingMetrics]] = {
        "Science": [],
        "Humanities": [],
        "Social_Sciences": [],
        "Medical": [],
        "Other": [],
    }

    # Categorize each metric
    for metric in metrics:
        task_name_lower = metric.task_name.lower()
        categorized_flag = False

        for category, keywords in category_keywords.items():
            if any(keyword in task_name_lower for keyword in keywords):
                categorized[category].append(metric)
                categorized_flag = True
                break

        if not categorized_flag:
            categorized["Other"].append(metric)

    # Try API-based categorization as fallback for uncategorized items
    if categorized["Other"]:
        try:
            from google import genai
            import os

            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
                other_task_names = [m.task_name for m in categorized["Other"] if m.task_name.startswith("global_mmlu_full_")]
                if other_task_names:
                    prompt = (
                        f"Divide the following MMLU tasks into 4 categories: "
                        f"{other_task_names}. "
                        f"Categories: 1. Science, 2. Humanities, 3. Social Sciences, 4. Medical. "
                        f"Return ONLY valid JSON with format: "
                        f'{{"Science": ["task1", "task2"], "Humanities": [...], "Social_Sciences": [...], "Medical": [...]}}'
                    )
                    response = client.models.generate_content(model="gemini-3-pro-preview", contents=prompt)
                    response_text = response.text if response.text is not None else "{}"
                    api_categories = json.loads(response_text)

                    # Re-categorize based on API response
                    remaining_other = []
                    for metric in categorized["Other"]:
                        found = False
                        for cat, tasks in api_categories.items():
                            if metric.task_name in tasks:
                                categorized[cat].append(metric)
                                found = True
                                break
                        if not found:
                            remaining_other.append(metric)
                    categorized["Other"] = remaining_other
        except Exception as e:
            print(f"Warning: API categorization failed: {e}. Using keyword-based categorization only.")

    return categorized


def categorize_metrics_and_save(
    metrics: List[CatastrophicForgettingMetrics], output_file: Path, step_filter: Optional[int] = None, use_percentage: bool = True
) -> Any:
    """
    Categorize metrics by category and save to JSON with statistics.

    Args:
        metrics: List of CatastrophicForgettingMetrics
        output_file: Path to save JSON file
        step_filter: Filter to specific step (if None, uses final step)
        use_percentage: If True, use relative degradation, else absolute

    Returns:
        Dictionary with categorized metrics and statistics
    """
    if step_filter is not None:
        filtered_metrics = [m for m in metrics if m.step == step_filter]
    else:
        max_step = max(m.step for m in metrics)
        filtered_metrics = [m for m in metrics if m.step == max_step]

    # Divide by category
    categorized = divide_by_category(filtered_metrics)

    # Build result structure with statistics
    result: Dict[str, Any] = {
        "metadata": {
            "step": step_filter if step_filter is not None else max_step,
            "use_percentage": use_percentage,
            "total_tasks": len(filtered_metrics),
        },
        "categories": {},
    }

    for category, category_metrics in categorized.items():
        if not category_metrics:
            continue

        degradations = [m.relative_degradation if use_percentage else m.absolute_degradation for m in category_metrics]

        performances = [m.current_performance for m in category_metrics]
        baseline_performances = [m.initial_performance for m in category_metrics]

        result["categories"][category] = {
            "task_count": len(category_metrics),
            "tasks": [
                {
                    "task_name": m.task_name,
                    "initial_performance": float(m.initial_performance),
                    "current_performance": float(m.current_performance),
                    "absolute_degradation": float(m.absolute_degradation),
                    "relative_degradation_percent": float(m.relative_degradation),
                    "retention_rate_percent": float(m.retention_rate) if m.retention_rate else None,
                }
                for m in category_metrics
            ],
            "statistics": {
                "mean_degradation": float(np.mean(degradations)),
                "std_degradation": float(np.std(degradations)),
                "median_degradation": float(np.median(degradations)),
                "max_degradation": float(np.max(degradations)),
                "min_degradation": float(np.min(degradations)),
                "mean_performance": float(np.mean(performances)),
                "mean_baseline_performance": float(np.mean(baseline_performances)),
                "mean_retention_rate": float(np.mean([m.retention_rate for m in category_metrics if m.retention_rate]))
                if any(m.retention_rate for m in category_metrics)
                else None,
            },
        }

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Category analysis saved to {output_file}")

    return result


def group_mmlu_subcategories_by_degradation(
    metrics: List[CatastrophicForgettingMetrics], step_filter: Optional[int] = None, use_percentage: bool = True
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Group MMLU subcategories by degradation level across different languages.

    Args:
        metrics: List of CatastrophicForgettingMetrics
        step_filter: Filter to specific step (if None, uses final step)
        use_percentage: If True, use relative degradation, else absolute

    Returns:
        Dictionary mapping subcategory to list of (task_name, degradation) tuples
        grouped by language
    """
    if step_filter is not None:
        filtered_metrics = [m for m in metrics if m.step == step_filter]
    else:
        max_step = max(m.step for m in metrics)
        filtered_metrics = [m for m in metrics if m.step == max_step]

    # Group by subcategory
    subcategory_groups: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for metric in filtered_metrics:
        benchmark = identify_benchmark(metric.task_name)
        if benchmark and benchmark.startswith("MMLU"):
            subcategory = extract_mmlu_subcategory(metric.task_name)
            if subcategory:
                degradation = metric.relative_degradation if use_percentage else metric.absolute_degradation
                subcategory_groups[subcategory][benchmark].append((metric.task_name, degradation))

    # Convert to final format: subcategory -> list of (task_name, degradation) for each language
    result: Dict[str, List[Tuple[str, float]]] = {}
    for subcategory, lang_dict in subcategory_groups.items():
        all_tasks = []
        for lang, tasks in lang_dict.items():
            all_tasks.extend(tasks)
        # Sort by degradation level (highest first)
        all_tasks.sort(key=lambda x: x[1], reverse=True)
        result[subcategory] = all_tasks

    return result


def generate_report(
    results_dir: Path,
    output_file: Optional[Path] = None,
    metric_key: str = "acc,none",
    benchmark_filter: Optional[str] = None,
    use_percentage_focus: bool = True,
    apply_smoothing: bool = True,
) -> Any:
    """
    Generate a comprehensive catastrophic forgetting analysis report.

    Args:
        results_dir: Path to directory containing benchmark results
        output_file: Optional path to save report as JSON
        metric_key: Metric key to analyze
        benchmark_filter: Optional benchmark to filter by ('MMLU_en', 'MMLU_pl', 'MMLU_ru', 'Hellaswag')
        use_percentage_focus: If True, focus on percentage-based metrics
        apply_smoothing: If True, apply noise reduction

    Returns:
        Dictionary containing analysis report
    """
    # Load and process data
    results = load_benchmark_results(results_dir)
    metric_data = extract_metric_by_task(results, metric_key, benchmark_filter=benchmark_filter)
    metrics = calculate_forgetting_metrics(metric_data)
    step_summaries = calculate_step_summaries(metrics, use_percentage_focus=use_percentage_focus, apply_smoothing=apply_smoothing)

    # Get top affected tasks
    most_affected = get_most_affected_tasks(metrics, top_n=15, use_percentage_focus=use_percentage_focus)

    # Prepare report
    report = {
        "metadata": {
            "total_steps": len(results),
            "metric_analyzed": metric_key,
            "benchmark_filter": benchmark_filter,
            "use_percentage_focus": use_percentage_focus,
            "apply_smoothing": apply_smoothing,
            "steps": sorted(results.keys()),
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
                "avg_forgetting_measure": float(summary.avg_forgetting_measure)
                if summary.avg_forgetting_measure is not None
                else None,
                "catastrophic_forgetting_test": float(summary.catastrophic_forgetting_test)
                if summary.catastrophic_forgetting_test is not None
                else None,
            }
            for step, summary in step_summaries.items()
        },
        "most_affected_tasks": [
            {"task_name": name, "absolute_degradation": float(abs_deg), "relative_degradation_percent": float(rel_deg)}
            for name, abs_deg, rel_deg in most_affected
        ],
    }

    # Calculate overall statistics (use relative degradation if percentage focus)
    final_step = max(step_summaries.keys())
    if use_percentage_focus:
        all_degradations = [m.relative_degradation for m in metrics if m.step == final_step]
    else:
        all_degradations = [m.absolute_degradation for m in metrics if m.step == final_step]

    report["overall_statistics"] = {
        "mean_degradation": float(np.mean(all_degradations)),
        "std_degradation": float(np.std(all_degradations)),
        "max_degradation": float(np.max(all_degradations)),
        "min_degradation": float(np.min(all_degradations)),
        "median_degradation": float(np.median(all_degradations)),
    }

    # Add MMLU subcategory grouping if analyzing MMLU benchmarks
    if benchmark_filter and benchmark_filter.startswith("MMLU"):
        mmlu_groups = group_mmlu_subcategories_by_degradation(metrics, use_percentage=use_percentage_focus)
        report["mmlu_subcategory_groups"] = {
            subcat: [{"task_name": task_name, "degradation": float(deg)} for task_name, deg in tasks]
            for subcat, tasks in mmlu_groups.items()
        }

        # Add category-based analysis
        if output_file:
            category_output = output_file.parent / f"categories_{benchmark_filter.lower()}.json"
            categorize_metrics_and_save(metrics, category_output, use_percentage=use_percentage_focus)
            report["category_analysis_file"] = str(category_output)

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
    print("\n" + "=" * 120)
    print("CATASTROPHIC FORGETTING ANALYSIS - SUMMARY BY STEP")
    print("=" * 120)
    header = f"{'Step':<10} {'ACC':<10} {'BWT':<10} {'FORG':<10} {'CBT':<10} {'Degraded':<12} {'Rate%':<10}"
    print(header)
    print("-" * 120)

    for step in sorted(step_summaries.keys()):
        summary = step_summaries[step]
        acc = summary.avg_performance
        bwt = summary.avg_backward_transfer if summary.avg_backward_transfer is not None else 0.0
        forg = summary.avg_forgetting_measure if summary.avg_forgetting_measure is not None else 0.0
        cbt = summary.catastrophic_forgetting_test if summary.catastrophic_forgetting_test is not None else 0.0

        row = (
            f"{step:<10} {acc:<10.4f} {bwt:<10.4f} {forg:<10.4f} {cbt:<10.4f} "
            f"{summary.num_degraded_tasks}/{summary.total_tasks:<8} {summary.degradation_rate:<9.1f}%"
        )
        print(row)

    print("=" * 120 + "\n")


def print_most_affected_table(most_affected: List[Tuple[str, float, float]]) -> None:
    """
    Print a formatted table of most affected tasks.

    Args:
        most_affected: List of tuples (task_name, abs_degradation, rel_degradation)
    """
    print("\n" + "=" * 100)
    print("TOP AFFECTED TASKS (by absolute degradation)")
    print("=" * 100)
    print(f"{'Rank':<6} {'Task Name':<50} {'Abs. Degradation':<20} {'Rel. Degradation (%)':<15}")
    print("-" * 100)

    for rank, (task_name, abs_deg, rel_deg) in enumerate(most_affected, 1):
        print(f"{rank:<6} {task_name:<50} {abs_deg:<20.6f} {rel_deg:<14.2f}%")

    print("=" * 100 + "\n")
