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
from typing import Dict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_report(report_path: Path) -> Dict:
    """Load the catastrophic forgetting report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def plot_performance_over_time(report: Dict, output_dir: Path):
    """Plot average performance and degradation over training steps."""
    step_summaries = report['step_summaries']
    steps = sorted([int(k) for k in step_summaries.keys()])
    
    avg_performance = [step_summaries[str(step)]['avg_performance'] for step in steps]  # ACC
    avg_degradation = [step_summaries[str(step)]['avg_degradation'] for step in steps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Average Performance (ACC)
    ax1.plot(steps, avg_performance, marker='o', linewidth=2, markersize=6, 
             color='#2E86AB', label='Average Accuracy (ACC)')
    ax1.axhline(y=avg_performance[0], color='r', linestyle='--', alpha=0.5, 
                label=f'Baseline ({avg_performance[0]:.4f})')
    ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Accuracy (ACC)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Performance Across Training Steps', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(avg_performance) * 0.999, max(avg_performance) * 1.001])
    
    # Plot 2: Average Degradation
    colors = ['red' if d > 0 else 'green' for d in avg_degradation]
    ax2.bar(steps, avg_degradation, color=colors, alpha=0.6, width=100)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Degradation', fontsize=12, fontweight='bold')
    ax2.set_title('Average Degradation from Baseline (Positive = Forgetting)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (step, deg) in enumerate(zip(steps, avg_degradation)):
        if abs(deg) > 0.0001:  # Only label significant values
            ax2.text(step, deg + (0.0002 if deg > 0 else -0.0002), 
                    f'{deg:.4f}', ha='center', va='bottom' if deg > 0 else 'top',
                    fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'performance_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_degradation_rate(report: Dict, output_dir: Path):
    """Plot the percentage of tasks showing degradation over time."""
    step_summaries = report['step_summaries']
    steps = sorted([int(k) for k in step_summaries.keys()])
    
    degradation_rates = [step_summaries[str(step)]['degradation_rate'] for step in steps]
    num_degraded = [step_summaries[str(step)]['num_degraded_tasks'] for step in steps]
    total_tasks = step_summaries[str(steps[0])]['total_tasks']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create dual y-axis
    ax2 = ax.twinx()
    
    # Plot degradation rate as line
    line = ax.plot(steps, degradation_rates, marker='o', linewidth=2.5, 
                   markersize=8, color='#E63946', label='Degradation Rate (%)')
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degradation Rate (%)', fontsize=12, fontweight='bold', 
                  color='#E63946')
    ax.tick_params(axis='y', labelcolor='#E63946')
    ax.set_ylim([0, max(degradation_rates) * 1.1])
    
    # Plot number of degraded tasks as bars
    bars = ax2.bar(steps, num_degraded, alpha=0.3, color='#F77F00', 
                   width=100, label=f'# Degraded Tasks (out of {total_tasks})')
    ax2.set_ylabel('Number of Degraded Tasks', fontsize=12, fontweight='bold', 
                   color='#F77F00')
    ax2.tick_params(axis='y', labelcolor='#F77F00')
    ax2.set_ylim([0, total_tasks * 1.1])
    
    # Add value labels
    for step, rate in zip(steps, degradation_rates):
        ax.text(step, rate + 1, f'{rate:.1f}%', ha='center', va='bottom',
               fontsize=9, fontweight='bold')
    
    ax.set_title('Task Degradation Rate Over Training Steps', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'degradation_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_most_affected_tasks(report: Dict, output_dir: Path, top_n: int = 15):
    """Plot the most affected tasks by catastrophic forgetting."""
    most_affected = report['most_affected_tasks'][:top_n]
    
    task_names = [task['task_name'].replace('global_mmlu_full_', '') 
                  for task in most_affected]
    abs_degradations = [task['absolute_degradation'] for task in most_affected]
    rel_degradations = [task['relative_degradation_percent'] for task in most_affected]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Absolute Degradation
    y_pos = np.arange(len(task_names))
    bars1 = ax1.barh(y_pos, abs_degradations, color='#D62828', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(task_names, fontsize=9)
    ax1.set_xlabel('Absolute Degradation', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Most Affected Tasks (Absolute Degradation)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, abs_degradations)):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Relative Degradation
    bars2 = ax2.barh(y_pos, rel_degradations, color='#F77F00', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(task_names, fontsize=9)
    ax2.set_xlabel('Relative Degradation (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Most Affected Tasks (Relative Degradation)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, rel_degradations)):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'most_affected_tasks.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_standard_metrics(report: Dict, output_dir: Path):
    """Plot standard metrics: FORG, BWT, CBT, ACC over training steps."""
    step_summaries = report['step_summaries']
    steps = sorted([int(k) for k in step_summaries.keys()])
    
    # Extract metrics
    acc = [step_summaries[str(step)]['avg_performance'] for step in steps]
    bwt = [step_summaries[str(step)].get('avg_backward_transfer', 0) for step in steps]
    forg = [step_summaries[str(step)].get('avg_forgetting_measure', 0) for step in steps]
    cbt = [step_summaries[str(step)].get('catastrophic_forgetting_test', 0) for step in steps]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average Accuracy (ACC)
    ax1.plot(steps, acc, marker='o', linewidth=2.5, markersize=7, 
            color='#2E86AB', label='ACC')
    ax1.axhline(y=acc[0], color='r', linestyle='--', alpha=0.6, label='Baseline')
    ax1.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Average Accuracy (ACC)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Backward Transfer (BWT)
    colors_bwt = ['red' if b < 0 else 'green' for b in bwt]
    ax2.bar(steps, bwt, color=colors_bwt, alpha=0.6, width=100)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Backward Transfer', fontsize=11, fontweight='bold')
    ax2.set_title('Backward Transfer (BWT) - Negative = Forgetting', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Forgetting Measure (FORG)
    ax3.plot(steps, forg, marker='s', linewidth=2.5, markersize=7, 
            color='#E63946', label='FORG')
    ax3.fill_between(steps, forg, alpha=0.2, color='#E63946')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Forgetting Measure', fontsize=11, fontweight='bold')
    ax3.set_title('Forgetting Measure (FORG)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Catastrophic Forgetting Test (CBT)
    ax4.plot(steps, cbt, marker='^', linewidth=2.5, markersize=7, 
            color='#F77F00', label='CBT')
    ax4.fill_between(steps, cbt, alpha=0.2, color='#F77F00')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Catastrophic Forgetting Test', fontsize=11, fontweight='bold')
    ax4.set_title('Catastrophic Forgetting Test (CBT)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Standard Catastrophic Forgetting Metrics', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'standard_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_degradation_distribution(report: Dict, output_dir: Path):
    """Plot the distribution of degradation across all tasks at final step."""
    step_summaries = report['step_summaries']
    final_step = max([int(k) for k in step_summaries.keys()])
    
    # We need to calculate individual task degradations
    # For this, we'll need to load the actual benchmark data
    # But we can use the overall statistics from the report
    overall_stats = report['overall_statistics']
    
    # Since we don't have individual task data in the report,
    # we'll create a visualization based on the statistics we have
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get degradation data from step summaries
    steps = sorted([int(k) for k in step_summaries.keys()])
    degradations_by_step = {}
    
    # We'll need to estimate the distribution from the statistics
    # For now, let's show the statistics we have
    
    # Plot 1: Box plot representation (using statistics)
    stats_data = {
        'Min': overall_stats['min_degradation'],
        'Q1': overall_stats['median_degradation'] - overall_stats['std_degradation'],
        'Median': overall_stats['median_degradation'],
        'Q3': overall_stats['median_degradation'] + overall_stats['std_degradation'],
        'Max': overall_stats['max_degradation'],
        'Mean': overall_stats['mean_degradation']
    }
    
    # Create a violin-like plot using the statistics
    ax1.barh(['Overall'], [overall_stats['max_degradation'] - overall_stats['min_degradation']], 
            left=overall_stats['min_degradation'], color='lightblue', alpha=0.5)
    ax1.axvline(overall_stats['mean_degradation'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {overall_stats['mean_degradation']:.4f}")
    ax1.axvline(overall_stats['median_degradation'], color='green', linestyle='--', 
               linewidth=2, label=f"Median: {overall_stats['median_degradation']:.4f}")
    ax1.set_xlabel('Degradation', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Degradation Statistics (Final Step)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Degradation over steps (heatmap-like)
    avg_degradations = [step_summaries[str(step)]['avg_degradation'] for step in steps]
    ax2.plot(steps, avg_degradations, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax2.fill_between(steps, avg_degradations, 0, alpha=0.3, 
                    where=[d > 0 for d in avg_degradations], color='red')
    ax2.fill_between(steps, avg_degradations, 0, alpha=0.3, 
                    where=[d <= 0 for d in avg_degradations], color='green')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Degradation', fontsize=12, fontweight='bold')
    ax2.set_title('Average Degradation Trend', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics summary
    stats_names = ['Mean', 'Std', 'Max', 'Min', 'Median']
    stats_values = [
        overall_stats['mean_degradation'],
        overall_stats['std_degradation'],
        overall_stats['max_degradation'],
        overall_stats['min_degradation'],
        overall_stats['median_degradation']
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax3.bar(stats_names, stats_values, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Degradation Value', fontsize=12, fontweight='bold')
    ax3.set_title('Degradation Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, stats_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.002),
                f'{val:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # Plot 4: Degradation rate over time
    degradation_rates = [step_summaries[str(step)]['degradation_rate'] for step in steps]
    ax4.plot(steps, degradation_rates, marker='s', linewidth=2.5, markersize=7, 
            color='#E63946', label='% Tasks Degraded')
    ax4.fill_between(steps, degradation_rates, alpha=0.2, color='#E63946')
    ax4.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Degradation Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Percentage of Tasks Showing Degradation', 
                 fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylim([0, max(degradation_rates) * 1.1])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for step, rate in zip(steps, degradation_rates):
        ax4.text(step, rate + 1, f'{rate:.1f}%', ha='center', va='bottom',
               fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'degradation_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comprehensive_dashboard(report: Dict, output_dir: Path):
    """Create a comprehensive dashboard with all key metrics."""
    step_summaries = report['step_summaries']
    steps = sorted([int(k) for k in step_summaries.keys()])
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Average Performance (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    avg_performance = [step_summaries[str(step)]['avg_performance'] for step in steps]
    ax1.plot(steps, avg_performance, marker='o', linewidth=2.5, markersize=7, 
            color='#2E86AB', label='Average Performance')
    ax1.axhline(y=avg_performance[0], color='r', linestyle='--', alpha=0.6, 
               label=f'Baseline ({avg_performance[0]:.4f})')
    ax1.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Average Performance Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Degradation Rate (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    degradation_rates = [step_summaries[str(step)]['degradation_rate'] for step in steps]
    ax2.plot(steps, degradation_rates, marker='s', linewidth=2, markersize=6, 
            color='#E63946')
    ax2.fill_between(steps, degradation_rates, alpha=0.2, color='#E63946')
    ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Degradation Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Task Degradation Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average Degradation (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    avg_degradation = [step_summaries[str(step)]['avg_degradation'] for step in steps]
    colors = ['#D62828' if d > 0 else '#06A77D' for d in avg_degradation]
    ax3.bar(steps, avg_degradation, color=colors, alpha=0.6, width=100)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Avg Degradation', fontsize=11, fontweight='bold')
    ax3.set_title('Average Degradation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Number of Degraded Tasks (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    num_degraded = [step_summaries[str(step)]['num_degraded_tasks'] for step in steps]
    total_tasks = step_summaries[str(steps[0])]['total_tasks']
    ax4.plot(steps, num_degraded, marker='o', linewidth=2, markersize=6, 
            color='#F77F00', label='Degraded Tasks')
    ax4.axhline(y=total_tasks/2, color='gray', linestyle='--', alpha=0.5, 
               label='50% Threshold')
    ax4.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax4.set_title(f'Degraded Tasks Count (Total: {total_tasks})', 
                 fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistics Summary (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    overall_stats = report['overall_statistics']
    stats_names = ['Mean', 'Std', 'Max', 'Min', 'Median']
    stats_values = [
        overall_stats['mean_degradation'],
        overall_stats['std_degradation'],
        overall_stats['max_degradation'],
        overall_stats['min_degradation'],
        overall_stats['median_degradation']
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax5.bar(stats_names, stats_values, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Degradation', fontsize=11, fontweight='bold')
    ax5.set_title('Final Step Statistics', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Top 5 Most Affected Tasks (bottom, spans 3 columns)
    ax6 = fig.add_subplot(gs[2, :])
    most_affected = report['most_affected_tasks'][:10]
    task_names = [task['task_name'].replace('global_mmlu_full_', '') 
                  for task in most_affected]
    abs_degradations = [task['absolute_degradation'] for task in most_affected]
    
    y_pos = np.arange(len(task_names))
    bars = ax6.barh(y_pos, abs_degradations, color='#D62828', alpha=0.7)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(task_names, fontsize=9)
    ax6.set_xlabel('Absolute Degradation', fontsize=11, fontweight='bold')
    ax6.set_title('Top 10 Most Affected Tasks', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, abs_degradations):
        ax6.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    
    plt.suptitle('Catastrophic Forgetting Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'comprehensive_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations from the catastrophic forgetting report."""
    report_path = Path('data/reports/analysis.json')
    output_dir = Path('data/reports/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading catastrophic forgetting report...")
    report = load_report(report_path)
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    plot_performance_over_time(report, output_dir)
    plot_degradation_rate(report, output_dir)
    plot_most_affected_tasks(report, output_dir)
    plot_standard_metrics(report, output_dir)  # NEW: FORG, BWT, CBT, ACC
    plot_degradation_distribution(report, output_dir)
    plot_comprehensive_dashboard(report, output_dir)
    
    print("-" * 60)
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   Generated 6 plot files")


if __name__ == '__main__':
    main()

