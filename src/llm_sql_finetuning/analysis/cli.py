#!/usr/bin/env python3
"""
Command-line script to run catastrophic forgetting analysis on benchmark results.

Usage:
    python -m llm_sql_finetuning.utils.analysis.cli <benchmarks_dir> [--output <report_file>] [--metric <metric_key>]
"""

import argparse
import sys
from pathlib import Path

try:
    from llm_sql_finetuning.analysis import (
        load_benchmark_results,
        extract_metric_by_task,
        calculate_forgetting_metrics,
        calculate_step_summaries,
        get_most_affected_tasks,
        print_summary_table,
        print_most_affected_table,
        generate_report,
    )
except ImportError as e:
    print(f"Error: Could not import analysis module. {e}")
    sys.exit(1)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze catastrophic forgetting from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze benchmarks with default settings
  python -m llm_sql_finetuning.utils.analysis.cli data/benchmarks
  
  # Save report to specific location
  python -m llm_sql_finetuning.utils.analysis.cli data/benchmarks --output reports/analysis.json
  
  # Analyze different metric
  python -m llm_sql_finetuning.utils.analysis.cli data/benchmarks --metric acc_norm,none
        """,
    )

    parser.add_argument(
        "benchmarks_dir",
        type=Path,
        help="Path to directory containing step_*_results.json files, or parent directory with subdirectories",
    )

    parser.add_argument(
        "--all-subdirs",
        "-a",
        action="store_true",
        help="Process all subdirectories in benchmarks_dir (each subdirectory will have separate reports)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for JSON report (default: data/reports/catastrophic_forgetting_report.json or data/reports/{subdir}/catastrophic_forgetting_report.json)",
    )

    parser.add_argument("--metric", "-m", type=str, default="acc,none", help="Metric key to analyze (default: acc,none)")

    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        choices=["MMLU_en", "MMLU_pl", "MMLU_ru", "Hellaswag"],
        default=None,
        help="Filter to specific benchmark (MMLU_en, MMLU_pl, MMLU_ru, Hellaswag)",
    )

    parser.add_argument("--no-smoothing", action="store_true", help="Disable noise reduction (averaging last 5 values)")

    parser.add_argument(
        "--absolute-values", action="store_true", help="Use absolute values instead of percentages (default: percentages)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    # Validate input directory
    if not args.benchmarks_dir.exists():
        print(f"Error: Benchmarks directory not found: {args.benchmarks_dir}")
        sys.exit(1)

    # If --all-subdirs is set, process each subdirectory separately
    if args.all_subdirs:
        # Get all subdirectories that contain step_*_results.json files
        subdirs = [d for d in args.benchmarks_dir.iterdir() if d.is_dir() and any(d.glob("step_*_results.json"))]

        if not subdirs:
            print(f"❌ No subdirectories with benchmark results found in {args.benchmarks_dir}")
            sys.exit(1)

        print(f"📊 Catastrophic Forgetting Analysis (Processing {len(subdirs)} directories)")
        print(f"{'=' * 80}")
        print(f"Parent directory: {args.benchmarks_dir.resolve()}")
        print(f"Found {len(subdirs)} subdirectories:")
        for sd in sorted(subdirs):
            print(f"  - {sd.name}")
        print(f"{'=' * 80}\n")

        # Process each subdirectory
        for subdir in sorted(subdirs):
            print(f"\n{'=' * 80}")
            print(f"📁 Processing: {subdir.name}")
            print(f"{'=' * 80}")

            # Set output for this subdirectory
            if args.output is None:
                output_file = subdir.parent.parent / "reports" / subdir.name / "catastrophic_forgetting_report.json"
            else:
                # If output is specified, use it as base and append subdir name
                output_file = args.output.parent / subdir.name / args.output.name

            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Process this subdirectory (recursive call to main logic)
            result = process_single_directory(
                subdir, output_file, args.metric, args.benchmark, not args.absolute_values, not args.no_smoothing, args.verbose
            )

            if result != 0:
                print(f"⚠️  Warning: Analysis for {subdir.name} completed with errors")

        print(f"\n{'=' * 80}")
        print("✅ ALL ANALYSES COMPLETE")
        print(f"{'=' * 80}")
        return 0
    else:
        # Single directory processing (original behavior)
        if args.output is None:
            args.output = args.benchmarks_dir.parent / "reports" / "catastrophic_forgetting_report.json"

        return process_single_directory(
            args.benchmarks_dir,
            args.output,
            args.metric,
            args.benchmark,
            not args.absolute_values,
            not args.no_smoothing,
            args.verbose,
        )


def process_single_directory(
    benchmarks_dir: Path,
    output_file: Path,
    metric: str,
    benchmark_filter: str,
    use_percentage_focus: bool,
    apply_smoothing: bool,
    verbose: bool,
) -> int:
    """Process a single benchmark directory."""
    print("📊 Catastrophic Forgetting Analysis")
    print(f"{'=' * 80}")
    print(f"Benchmarks dir: {benchmarks_dir.resolve()}")
    print(f"Output file: {output_file.resolve()}")
    print(f"Metric: {metric}")
    print(f"{'=' * 80}\n")

    try:
        # Load data
        print("1. Loading benchmark results...")
        results = load_benchmark_results(benchmarks_dir)
        print(f"   ✓ Loaded {len(results)} files")
        print(f"   ✓ Steps: {sorted(results.keys())}")

        # Extract metrics
        print("\n2. Extracting performance metrics...")
        metric_data = extract_metric_by_task(results, metric)
        total_tasks = len(set(t for step_data in metric_data.values() for t in step_data.keys()))
        print(f"- Extracted metrics for {total_tasks} unique tasks")

        # Calculate forgetting metrics
        print("\n3. Calculating forgetting metrics...")
        forgetting_metrics = calculate_forgetting_metrics(metric_data)
        print(f"- Calculated {len(forgetting_metrics)} metric entries")

        # Calculate step summaries
        print("\n4. Computing step summaries...")
        step_summaries = calculate_step_summaries(forgetting_metrics)
        print(f"- Computed summaries for {len(step_summaries)} steps")

        if verbose:
            print("\n" + "=" * 80)
            print_summary_table(step_summaries)

        # Get most affected tasks
        print("\n5. Identifying most affected tasks...")
        most_affected = get_most_affected_tasks(forgetting_metrics, top_n=15)
        print(f"- Found {len(most_affected)} top affected tasks")

        if verbose:
            print("\n" + "=" * 80)
            print_most_affected_table(most_affected)

        # Generate report
        print("\n6️⃣  Generating comprehensive report...")
        report = generate_report(
            benchmarks_dir,
            output_file,
            metric,
            benchmark_filter=benchmark_filter,
            use_percentage_focus=use_percentage_focus,
            apply_smoothing=apply_smoothing,
        )
        print(f"- Report saved to {output_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("📈 ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nMetric Analyzed: {report['metadata']['metric_analyzed']}")
        print(f"Total Steps: {report['metadata']['total_steps']}")
        print("\nOverall Statistics (across all steps and tasks):")
        for key, value in report["overall_statistics"].items():
            print(f"  {key:.<40} {value:.6f}")

        print("\n- Analysis completed successfully!")
        return 0

    except Exception as e:
        print(f"\n!!! Error during analysis: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
