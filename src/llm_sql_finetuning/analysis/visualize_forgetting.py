"""
Visualize catastrophic forgetting metrics from the analysis report.

This module creates comprehensive plots showing:
1. Average performance and degradation over training steps
2. Degradation rate over time
3. Most affected tasks
4. Distribution of degradation across tasks
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any
import seaborn as sns
from llm_sql_finetuning.analysis import generate_report

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_report(report_path: Path) -> Any:
    """Load the catastrophic forgetting report."""
    with open(report_path, "r") as f:
        return json.load(f)


def plot_performance_over_time(report: Dict[str, Any], output_dir: Path) -> None:
    """Plot average performance and degradation over training steps."""
    step_summaries = report["step_summaries"]
    steps = sorted([int(k) for k in step_summaries.keys()])

    avg_performance = [step_summaries[str(step)]["avg_performance"] for step in steps]  # ACC
    avg_degradation = [step_summaries[str(step)]["avg_degradation"] for step in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Average Performance (ACC) - show as percentage
    avg_performance_percent = [p * 100 for p in avg_performance]
    baseline_percent = avg_performance_percent[0]
    ax1.plot(steps, avg_performance_percent, marker="o", linewidth=2, markersize=6, color="#2E86AB", label="Average Accuracy (ACC)")
    ax1.axhline(y=baseline_percent, color="r", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_percent:.1f}%)")
    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Average Performance Across Training Steps", fontsize=14, fontweight="bold", pad=20)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Set ylim with some padding
    y_min, y_max = min(avg_performance_percent), max(avg_performance_percent)
    y_range = y_max - y_min
    ax1.set_ylim([y_min - y_range * 0.05, y_max + y_range * 0.05])

    # Plot 2: Average Degradation (Percentage)
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    avg_degradation_percent = [d * 100 for d in avg_degradation]
    colors = ["red" if d > 0 else "green" for d in avg_degradation_percent]
    ax2.bar(steps, avg_degradation_percent, color=colors, alpha=0.6, width=100)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Degradation (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Average Degradation from Baseline - Percentage (Positive = Forgetting)", fontsize=14, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "performance_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_degradation_rate(report: Dict[str, Any], output_dir: Path) -> None:
    """Plot the percentage of tasks showing degradation over time."""
    step_summaries = report["step_summaries"]
    steps = sorted([int(k) for k in step_summaries.keys()])

    degradation_rates = [step_summaries[str(step)]["degradation_rate"] for step in steps]
    num_degraded = [step_summaries[str(step)]["num_degraded_tasks"] for step in steps]
    total_tasks = step_summaries[str(steps[0])]["total_tasks"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create dual y-axis
    ax2 = ax.twinx()

    # Plot degradation rate as line
    ax.plot(steps, degradation_rates, marker="o", linewidth=2, markersize=6, color="#E63946", label="Degradation Rate (%)")
    ax.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Degradation Rate (%)", fontsize=12, fontweight="bold", color="#E63946")
    ax.tick_params(axis="y", labelcolor="#E63946")
    ax.set_ylim([0, max(degradation_rates) * 1.1 if degradation_rates else 100])

    # Plot number of degraded tasks as bars
    # Calculate bar width based on step range, not max step value
    # Set x-axis limits to prevent oversized figures
    if steps:
        step_range = max(steps) - min(steps) if len(steps) > 1 else max(steps)
        # Use a reasonable bar width: either 2% of range or a fixed value, whichever is smaller
        bar_width = min(step_range * 0.02 if step_range > 0 else 100, 500)
        # Set x-axis limits with padding to prevent oversized figures
        x_padding = step_range * 0.05 if step_range > 0 else 100
        x_min = min(steps) - x_padding
        x_max = max(steps) + x_padding
    else:
        bar_width = 100
        x_min = 0
        x_max = 1000
    ax2.bar(steps, num_degraded, alpha=0.5, color="#F77F00", label="Number of Degraded Tasks", width=bar_width)
    ax2.set_ylabel("Number of Degraded Tasks", fontsize=12, fontweight="bold", color="#F77F00")
    ax2.tick_params(axis="y", labelcolor="#F77F00")
    ax2.set_ylim([0, total_tasks * 1.1 if total_tasks else 1])
    # Set x-axis limits on both axes
    ax.set_xlim([x_min, x_max])
    ax2.set_xlim([x_min, x_max])

    # Add value labels (only if not too many steps to avoid clutter)
    if len(steps) <= 20:
        for step, rate in zip(steps, degradation_rates):
            ax.text(
                step,
                rate + max(degradation_rates) * 0.02 if degradation_rates else 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Task Degradation Rate Over Training Steps", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "degradation_rate.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_most_affected_tasks(report: Dict[str, Any], output_dir: Path, top_n: int = 15) -> None:
    """Plot the most affected tasks by catastrophic forgetting."""
    most_affected_list = report.get("most_affected_tasks", [])
    if not isinstance(most_affected_list, list):
        raise TypeError("most_affected_tasks must be a list")
    most_affected = most_affected_list[:top_n]
    use_percentage = report.get("metadata", {}).get("use_percentage_focus", True)

    task_names = [task["task_name"].replace("global_mmlu_full_", "").replace("hellaswag", "Hellaswag") for task in most_affected]
    abs_degradations = [task["absolute_degradation"] for task in most_affected]
    rel_degradations = [task["relative_degradation_percent"] for task in most_affected]

    if use_percentage:
        # If percentage focus, show relative degradation prominently
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Relative Degradation (Primary)
        y_pos = np.arange(len(task_names))
        bars1 = ax1.barh(y_pos, rel_degradations, color="#F77F00", alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(task_names, fontsize=9)
        ax1.set_xlabel("Relative Degradation (%)", fontsize=12, fontweight="bold")
        ax1.set_title(f"Top {top_n} Most Affected Tasks (Relative Degradation %)", fontsize=14, fontweight="bold", pad=20)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, rel_degradations)):
            ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")

        # Plot 2: Absolute Degradation (Secondary)
        bars2 = ax2.barh(y_pos, abs_degradations, color="#D62828", alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(task_names, fontsize=9)
        ax2.set_xlabel("Absolute Degradation", fontsize=12, fontweight="bold")
        ax2.set_title(f"Top {top_n} Most Affected Tasks (Absolute Degradation)", fontsize=14, fontweight="bold", pad=20)
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, abs_degradations)):
            ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, fontweight="bold")
    else:
        # Original layout if not using percentage focus
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Absolute Degradation
        y_pos = np.arange(len(task_names))
        bars1 = ax1.barh(y_pos, abs_degradations, color="#D62828", alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(task_names, fontsize=9)
        ax1.set_xlabel("Absolute Degradation", fontsize=12, fontweight="bold")
        ax1.set_title(f"Top {top_n} Most Affected Tasks (Absolute Degradation)", fontsize=14, fontweight="bold", pad=20)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, abs_degradations)):
            ax1.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, fontweight="bold")

        # Plot 2: Relative Degradation
        bars2 = ax2.barh(y_pos, rel_degradations, color="#F77F00", alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(task_names, fontsize=9)
        ax2.set_xlabel("Relative Degradation (%)", fontsize=12, fontweight="bold")
        ax2.set_title(f"Top {top_n} Most Affected Tasks (Relative Degradation)", fontsize=14, fontweight="bold", pad=20)
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, rel_degradations)):
            ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "most_affected_tasks.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_standard_metrics(report: Dict[str, Any], output_dir: Path) -> None:
    """Plot standard metrics: FORG, BWT, CBT, ACC over training steps (with percentages)."""
    step_summaries = report["step_summaries"]
    steps = sorted([int(k) for k in step_summaries.keys()])

    # Extract metrics
    acc = [step_summaries[str(step)]["avg_performance"] for step in steps]
    bwt = [step_summaries[str(step)].get("avg_backward_transfer", 0) for step in steps]
    forg = [step_summaries[str(step)].get("avg_forgetting_measure", 0) for step in steps]
    cbt = [step_summaries[str(step)].get("catastrophic_forgetting_test", 0) for step in steps]

    # Convert to percentages for BWT, FORG, CBT if needed
    # For BWT: convert to percentage change
    baseline_acc = acc[0] if acc else 1.0
    bwt_percent = [(b / baseline_acc * 100) if baseline_acc != 0 else 0 for b in bwt]
    forg_percent = [(f / baseline_acc * 100) if baseline_acc != 0 else 0 for f in forg]
    cbt_percent = [(c / baseline_acc * 100) if baseline_acc != 0 else 0 for c in cbt]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Average Accuracy (ACC) - show as percentage
    acc_percent = [a * 100 for a in acc]
    baseline_percent = acc_percent[0]
    ax1.plot(steps, acc_percent, marker="o", linewidth=2.5, markersize=7, color="#2E86AB", label="ACC")
    ax1.axhline(y=baseline_percent, color="r", linestyle="--", alpha=0.6, label=f"Baseline ({baseline_percent:.1f}%)")
    ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Accuracy (ACC) - Percentage", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Backward Transfer (BWT) - as percentage
    # bwt_percent is already in percentage form (calculated on line 233)
    colors_bwt = ["red" if b < 0 else "green" for b in bwt_percent]
    ax2.bar(steps, bwt_percent, color=colors_bwt, alpha=0.6, width=100)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Backward Transfer (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Backward Transfer (BWT) - Percentage (Negative = Forgetting)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    # Skip value labels on bars to avoid clutter - values are visible from bar heights

    # Plot 3: Forgetting Measure (FORG) - as percentage
    # forg_percent is already in percentage form (calculated on line 234)
    ax3.plot(steps, forg_percent, marker="s", linewidth=2.5, markersize=7, color="#E63946", label="FORG")
    ax3.fill_between(steps, forg_percent, 0, alpha=0.2, color="#E63946")
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Forgetting Measure (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Forgetting Measure (FORG) - Percentage", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Catastrophic Forgetting Test (CBT) - as percentage
    # cbt_percent is already in percentage form (calculated on line 235)
    ax4.plot(steps, cbt_percent, marker="^", linewidth=2.5, markersize=7, color="#F77F00", label="CBT")
    ax4.fill_between(steps, cbt_percent, 0, alpha=0.2, color="#F77F00")
    ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Catastrophic Forgetting Test (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Catastrophic Forgetting Test (CBT) - Percentage", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Standard Catastrophic Forgetting Metrics (All in Percentages)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path = output_dir / "standard_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_degradation_distribution(report: Dict[str, Any], output_dir: Path) -> None:
    """Plot the distribution of degradation across all tasks at final step (percentages)."""
    step_summaries = report["step_summaries"]

    overall_stats = report["overall_statistics"]

    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(16, 12))

    steps = sorted([int(k) for k in step_summaries.keys()])

    # Plot 2: Degradation over steps (percentage)
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    avg_degradations = [step_summaries[str(step)]["avg_degradation"] for step in steps]
    ax2.plot(steps, avg_degradations, marker="o", linewidth=2, markersize=6, color="#2E86AB")
    ax2.fill_between(steps, avg_degradations, 0, alpha=0.3, where=[d > 0 for d in avg_degradations], color="red")
    ax2.fill_between(steps, avg_degradations, 0, alpha=0.3, where=[d <= 0 for d in avg_degradations], color="green")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Degradation (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Average Degradation Trend - Percentage", fontsize=14, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Statistics summary - in percentage
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    stats_names = ["Mean", "Std", "Max", "Min", "Median"]
    stats_values = [
        overall_stats["mean_degradation"],
        overall_stats["std_degradation"],
        overall_stats["max_degradation"],
        overall_stats["min_degradation"],
        overall_stats["median_degradation"],
    ]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    bars = ax3.bar(stats_names, stats_values, color=colors, alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax3.set_ylabel("Degradation (%)", fontsize=12, fontweight="bold")
    ax3.set_title("Degradation Statistics Summary - Percentage", fontsize=14, fontweight="bold", pad=20)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, stats_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.5 if height > 0 else -1),
            f"{val:.2f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    # Plot 4: Degradation rate over time
    degradation_rates = [step_summaries[str(step)]["degradation_rate"] for step in steps]
    ax4.plot(steps, degradation_rates, marker="s", linewidth=2.5, markersize=7, color="#E63946", label="% Tasks Degraded")
    ax4.fill_between(steps, degradation_rates, alpha=0.2, color="#E63946")
    ax4.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Degradation Rate (%)", fontsize=12, fontweight="bold")
    ax4.set_title("Percentage of Tasks Showing Degradation", fontsize=14, fontweight="bold", pad=20)
    ax4.set_ylim([0, max(degradation_rates) * 1.1])
    ax4.grid(True, alpha=0.3)

    # Add value labels (only if not too many steps to avoid clutter and oversized figures)
    if len(steps) <= 20:
        for step, rate in zip(steps, degradation_rates):
            ax4.text(
                step,
                rate + max(degradation_rates) * 0.02 if degradation_rates else 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    output_path = output_dir / "degradation_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_comprehensive_dashboard(report: Dict[str, Any], output_dir: Path) -> None:
    """Create a comprehensive dashboard with all key metrics."""
    step_summaries = report.get("step_summaries", {})
    if not isinstance(step_summaries, dict):
        raise TypeError("step_summaries must be a dict")
    steps = sorted([int(k) for k in step_summaries.keys()])

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Average Performance (top left, spans 2 columns) - as percentage
    ax1 = fig.add_subplot(gs[0, :2])
    avg_performance = [step_summaries[str(step)]["avg_performance"] for step in steps]
    avg_performance_percent = [p * 100 for p in avg_performance]
    baseline_percent = avg_performance_percent[0]
    ax1.plot(steps, avg_performance_percent, marker="o", linewidth=2.5, markersize=7, color="#2E86AB", label="Average Performance")
    ax1.axhline(y=baseline_percent, color="r", linestyle="--", alpha=0.6, label=f"Baseline ({baseline_percent:.1f}%)")
    ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Performance Over Time - Percentage", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Degradation Rate (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    degradation_rates = [step_summaries[str(step)]["degradation_rate"] for step in steps]
    ax2.plot(steps, degradation_rates, marker="s", linewidth=2, markersize=6, color="#E63946")
    ax2.fill_between(steps, degradation_rates, alpha=0.2, color="#E63946")
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Degradation Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Task Degradation Rate", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Average Degradation (middle left) - percentage
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    ax3 = fig.add_subplot(gs[1, 0])
    avg_degradation = [step_summaries[str(step)]["avg_degradation"] for step in steps]
    colors = ["#D62828" if d > 0 else "#06A77D" for d in avg_degradation]
    ax3.bar(steps, avg_degradation, color=colors, alpha=0.6, width=100)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax3.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Avg Degradation (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Average Degradation - Percentage", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Number of Degraded Tasks (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    num_degraded = [step_summaries[str(step)]["num_degraded_tasks"] for step in steps]
    total_tasks = step_summaries[str(steps[0])]["total_tasks"]
    ax4.plot(steps, num_degraded, marker="o", linewidth=2, markersize=6, color="#F77F00", label="Degraded Tasks")
    ax4.axhline(y=total_tasks / 2, color="gray", linestyle="--", alpha=0.5, label="50% Threshold")
    ax4.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Number of Tasks", fontsize=11, fontweight="bold")
    ax4.set_title(f"Degraded Tasks Count (Total: {total_tasks})", fontsize=12, fontweight="bold")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Top 10 Most Affected Tasks (bottom, spans 2 columns) - percentage
    ax5 = fig.add_subplot(gs[2, :])
    most_affected_list = report.get("most_affected_tasks", [])
    if not isinstance(most_affected_list, list):
        raise TypeError("most_affected_tasks must be a list")
    most_affected = most_affected_list[:10]
    task_names = [task["task_name"].replace("global_mmlu_full_", "").replace("hellaswag", "Hellaswag") for task in most_affected]
    rel_degradations = [task["relative_degradation_percent"] for task in most_affected]

    y_pos = np.arange(len(task_names))
    bars = ax5.barh(y_pos, rel_degradations, color="#D62828", alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(task_names, fontsize=9)
    ax5.set_xlabel("Relative Degradation (%)", fontsize=11, fontweight="bold")
    ax5.set_title("Top 10 Most Affected Tasks - Percentage", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="x")
    ax5.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, rel_degradations):
        ax5.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")

    plt.suptitle("Catastrophic Forgetting Analysis Dashboard (All Metrics in Percentages)", fontsize=16, fontweight="bold", y=0.995)

    output_path = output_dir / "comprehensive_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_benchmark_comparison(reports: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """Plot comparison across different benchmarks."""
    benchmarks = ["MMLU_en", "MMLU_pl", "MMLU_ru", "Hellaswag"]
    available_benchmarks = [b for b in benchmarks if b in reports]

    if len(available_benchmarks) < 2:
        print("Skipping benchmark comparison: need at least 2 benchmarks")
        return

    # Create figure with better spacing
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, left=0.1, right=0.96, top=0.94, bottom=0.1)

    # Plot 1: Average Performance Comparison - as percentage
    ax1 = fig.add_subplot(gs[0, 0])
    for benchmark in available_benchmarks:
        report = reports[benchmark]
        step_summaries = report["step_summaries"]
        steps = sorted([int(k) for k in step_summaries.keys()])
        avg_performance = [step_summaries[str(step)]["avg_performance"] for step in steps]
        avg_performance_percent = [p * 100 for p in avg_performance]
        ax1.plot(steps, avg_performance_percent, marker="o", linewidth=2, markersize=5, label=benchmark)
    ax1.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Performance Comparison - Percentage", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Degradation Rate Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for benchmark in available_benchmarks:
        report = reports[benchmark]
        step_summaries = report["step_summaries"]
        steps = sorted([int(k) for k in step_summaries.keys()])
        degradation_rates = [step_summaries[str(step)]["degradation_rate"] for step in steps]
        ax2.plot(steps, degradation_rates, marker="s", linewidth=2, markersize=5, label=benchmark)
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Degradation Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Degradation Rate Comparison", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])  # Ensure full range is visible

    # Plot 3: Average Degradation Comparison (Percentage)
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    ax3 = fig.add_subplot(gs[1, 0])
    for benchmark in available_benchmarks:
        report = reports[benchmark]
        step_summaries = report["step_summaries"]
        steps = sorted([int(k) for k in step_summaries.keys()])
        avg_degradation = [step_summaries[str(step)]["avg_degradation"] for step in steps]
        ax3.plot(steps, avg_degradation, marker="^", linewidth=2, markersize=5, label=benchmark)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax3.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Average Degradation (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Average Degradation Comparison (Percentage)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Step Statistics Comparison
    # Values in JSON are in decimal form, need to multiply by 100 to get percentage
    ax4 = fig.add_subplot(gs[1, 1])
    final_stats = {}
    for benchmark in available_benchmarks:
        report = reports[benchmark]
        final_stats[benchmark] = report["overall_statistics"]["mean_degradation"]

    colors = ["#2E86AB", "#E63946", "#F77F00", "#06A77D"]
    bars = ax4.bar(final_stats.keys(), final_stats.values(), color=colors[: len(final_stats)], alpha=0.7)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax4.set_ylabel("Mean Degradation (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Final Step: Mean Degradation by Benchmark", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=10)

    # Add value labels only if they won't clutter (few benchmarks)
    if len(final_stats) <= 4:
        y_min, y_max = min(final_stats.values()), max(final_stats.values())
        y_range = y_max - y_min if y_max != y_min else 1
        label_offset = max(abs(y_range) * 0.1, 1.0) if y_range != 0 else 1.0
        for bar, val in zip(bars, final_stats.values()):
            height = bar.get_height()
            offset = label_offset if height > 0 else -label_offset
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset,
                f"{val:.2f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle("Benchmark Comparison", fontsize=14, fontweight="bold", y=0.98)
    output_path = output_dir / "benchmark_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_category_analysis(category_file: Path, output_dir: Path) -> None:
    """Plot category-based analysis from JSON file."""
    if not category_file.exists():
        print(f"Skipping category plots: file not found {category_file}")
        return

    with open(category_file, "r") as f:
        category_data = json.load(f)

    categories = category_data.get("categories", {})
    if not categories:
        print("No category data found in file")
        return

    # Filter out 'Other' category if it exists and is empty or has very few tasks
    category_names = [cat for cat in categories.keys() if cat != "Other" or categories[cat]["task_count"] > 0]
    if not category_names:
        category_names = list(categories.keys())

    # Create comprehensive category analysis plot with optimized spacing
    # Use better layout: 2 rows, 2 columns for cleaner display
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35, left=0.12, right=0.96, top=0.94, bottom=0.12)

    colors = ["#2E86AB", "#E63946", "#F77F00", "#06A77D", "#6A994E"]

    # Plot 1: Mean Degradation by Category (top left)
    # mean_degradation in category JSON is already in percentage form (from relative_degradation_percent)
    ax1 = fig.add_subplot(gs[0, 0])
    mean_degradations = [categories[cat]["statistics"]["mean_degradation"] for cat in category_names]
    # Values are already in percentage form, use directly
    bars1 = ax1.bar(category_names, mean_degradations, color=colors[: len(category_names)], alpha=0.7)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax1.set_ylabel("Mean Degradation (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Degradation by Category", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # Add value labels only if they won't clutter
    if len(category_names) <= 5:
        constant_offset = 0.25
        y_min, y_max = min(mean_degradations), max(mean_degradations)
        y_range = y_max - y_min if y_max != y_min else 1
        label_offset = max(abs(y_range) * 0.12, 1.5) if y_range != 0 else 1.5
        for bar, val in zip(bars1, mean_degradations):
            height = bar.get_height()
            offset = constant_offset if height >= 0 else -constant_offset
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset,
                f"{val:.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    # Plot 2: Mean Performance by Category (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    mean_performances = [categories[cat]["statistics"]["mean_performance"] * 100 for cat in category_names]
    bars2 = ax2.bar(category_names, mean_performances, color=colors[: len(category_names)], alpha=0.7)
    ax2.set_ylabel("Mean Performance (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Performance by Category", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # Add value labels only if few categories
    if len(category_names) <= 5:
        perf_min, perf_max = min(mean_performances), max(mean_performances)
        perf_range = perf_max - perf_min if perf_max != perf_min else 1
        label_offset = max(perf_range * 0.08, 1.5) if perf_range > 0 else 1.5
        for bar, val in zip(bars2, mean_performances):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + label_offset,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Plot 3: Task Count by Category (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    task_counts = [categories[cat]["task_count"] for cat in category_names]
    bars3 = ax3.bar(category_names, task_counts, color=colors[: len(category_names)], alpha=0.7)
    ax3.set_ylabel("Number of Tasks", fontsize=11, fontweight="bold")
    ax3.set_title("Task Count by Category", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # Add value labels only if few categories
    if len(category_names) <= 5:
        max_count = max(task_counts) if task_counts else 1
        label_offset = max(max_count * 0.06, 2.5)
        for bar, val in zip(bars3, task_counts):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + label_offset,
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Plot 4: Degradation Distribution by Category (Box plot style, bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    category_degradations = []
    category_labels = []
    for cat in category_names:
        tasks = categories[cat]["tasks"]
        degradations = [t["relative_degradation_percent"] for t in tasks]
        if degradations:  # Only add if there are degradations
            category_degradations.append(degradations)
            category_labels.append(f"{cat}\n(n={len(degradations)})")

    if category_degradations:
        bp = ax4.boxplot(category_degradations, labels=category_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors[: len(category_degradations)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax4.set_ylabel("Degradation (%)", fontsize=11, fontweight="bold")
        ax4.set_title("Degradation Distribution by Category", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")
        plt.setp(ax4.xaxis.get_majorticklabels(), fontsize=9)

    plt.suptitle("Category-Based Analysis", fontsize=14, fontweight="bold", y=0.99)
    output_path = output_dir / "category_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Create detailed category comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for grouped comparison
    x = np.arange(len(category_names))
    width = 0.35

    # Mean degradation and mean performance
    mean_degs = mean_degradations
    mean_perfs = mean_performances

    bars1 = ax.bar(x - width / 2, mean_degs, width, label="Mean Degradation (%)", color="#E63946", alpha=0.7)
    bars2 = ax.bar(x + width / 2, mean_perfs, width, label="Mean Performance (%)", color="#2E86AB", alpha=0.7)

    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Category Comparison: Degradation vs Performance", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add value labels only if few categories to avoid clutter
    if len(category_names) <= 5:
        y_min = min(min(mean_degs), min(mean_perfs))
        y_max = max(max(mean_degs), max(mean_perfs))
        y_range = y_max - y_min if y_max != y_min else 1
        label_offset = max(abs(y_range) * 0.08, 2.0) if y_range != 0 else 2.0

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if abs(height) > max(abs(y_range) * 0.05, 0.5):  # Only label significant values
                    offset = label_offset if height > 0 else -label_offset
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + offset,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                        fontsize=8,
                        fontweight="bold",
                    )

    plt.tight_layout()
    output_path = output_dir / "category_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_all_plots_for_benchmark(report: Dict[str, Any], benchmark_name: str, output_dir: Path) -> None:
    """Generate all plots for a specific benchmark."""
    benchmark_output_dir = output_dir / benchmark_name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Generating plots for {benchmark_name}...")
    print("-" * 60)

    plot_performance_over_time(report, benchmark_output_dir)
    plot_degradation_rate(report, benchmark_output_dir)
    plot_most_affected_tasks(report, benchmark_output_dir)
    plot_standard_metrics(report, benchmark_output_dir)
    plot_degradation_distribution(report, benchmark_output_dir)
    plot_comprehensive_dashboard(report, benchmark_output_dir)

    # Plot category analysis for MMLU benchmarks if available
    if benchmark_name.startswith("MMLU"):
        category_file = report.get("category_analysis_file")
        if category_file and isinstance(category_file, (str, Path)):
            plot_category_analysis(Path(str(category_file)), benchmark_output_dir)

    print(f"✅ All plots for {benchmark_name} saved to: {benchmark_output_dir}")


def main() -> None:
    """Generate all visualizations from the catastrophic forgetting report."""

    benchmarks_base_dir = Path("data/benchmarks")
    reports_base_dir = Path("data/reports")

    # Get all subdirectories in benchmarks (e.g., smolLM, smolLM_DoRA, smolLM_LoRA, lfm2)
    benchmark_dirs = [d for d in benchmarks_base_dir.iterdir() if d.is_dir() and any(d.glob("step_*_results.json"))]

    if not benchmark_dirs:
        print(f"❌ No benchmark directories found in {benchmarks_base_dir}")
        return

    benchmarks = ["MMLU_en", "MMLU_pl", "MMLU_ru", "Hellaswag"]

    print("=" * 80)
    print("GENERATING ANALYSIS FOR EACH BENCHMARK DIRECTORY")
    print("=" * 80)
    print(f"Found {len(benchmark_dirs)} benchmark directories:")
    for bd in benchmark_dirs:
        print(f"  - {bd.name}")
    print("=" * 80)

    # Process each benchmark directory
    for benchmark_dir in sorted(benchmark_dirs):
        print(f"\n{'=' * 80}")
        print(f"📁 Processing directory: {benchmark_dir.name}")
        print(f"{'=' * 80}")

        # Create output directory for this benchmark directory
        model_output_dir = reports_base_dir / benchmark_dir.name
        plots_output_dir = model_output_dir / "plots"
        plots_output_dir.mkdir(parents=True, exist_ok=True)

        all_reports = {}

        # Generate report and plots for each benchmark type
        for benchmark in benchmarks:
            print(f"\n  Processing benchmark: {benchmark}")

            # Generate report for this benchmark with percentage focus
            report_output = model_output_dir / f"analysis_{benchmark.lower()}.json"
            report = generate_report(
                results_dir=benchmark_dir,
                output_file=report_output,
                metric_key="acc,none",
                benchmark_filter=benchmark,
                use_percentage_focus=True,  # Always use percentages
                apply_smoothing=True,  # Apply noise reduction
            )

            all_reports[benchmark] = report

            # Generate all plots for this benchmark
            generate_all_plots_for_benchmark(report, benchmark, plots_output_dir)

            # If category file exists, also plot it directly
            if benchmark.startswith("MMLU"):
                category_file = model_output_dir / f"categories_{benchmark.lower()}.json"
                if category_file.exists():
                    plot_category_analysis(category_file, plots_output_dir / benchmark)

        # Generate comparison plots if we have multiple benchmarks
        if len(all_reports) >= 2:
            print("\n  Generating benchmark comparison plots...")
            plot_benchmark_comparison(all_reports, plots_output_dir)

        print(f"\n  ✅ Completed analysis for {benchmark_dir.name}")
        print(f"     Reports: {model_output_dir}")
        print(f"     Plots: {plots_output_dir}")

    print("\n" + "=" * 80)
    print("✅ ALL ANALYSES COMPLETE")
    print("=" * 80)
    print(f"\nProcessed {len(benchmark_dirs)} benchmark directories:")
    for bd in sorted(benchmark_dirs):
        model_output_dir = reports_base_dir / bd.name
        print(f"  - {bd.name}: {model_output_dir}")


if __name__ == "__main__":
    main()
