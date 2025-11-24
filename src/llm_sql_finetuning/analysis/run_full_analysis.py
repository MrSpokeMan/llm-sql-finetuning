"""
Run full catastrophic forgetting analysis for all benchmark directories.

This script:
1. Analyzes all subdirectories in data/benchmarks/
2. Generates reports for each benchmark type (MMLU_en, MMLU_pl, MMLU_ru, Hellaswag)
3. Creates visualizations for each model directory

Usage:
    python -m llm_sql_finetuning.analysis.run_full_analysis
    python -m llm_sql_finetuning.analysis.run_full_analysis --benchmarks-dir data/benchmarks
    python -m llm_sql_finetuning.analysis.run_full_analysis --no-smoothing --absolute-values
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from llm_sql_finetuning.analysis import generate_report
    from llm_sql_finetuning.analysis.visualize_forgetting import (
        generate_all_plots_for_benchmark,
        plot_category_analysis,
        plot_benchmark_comparison,
    )
except ImportError as e:
    print(f"Error: Could not import analysis module. {e}")
    sys.exit(1)


def run_full_analysis(
    benchmarks_base_dir: Optional[Path] = None,
    reports_base_dir: Optional[Path] = None,
    use_percentage_focus: bool = True,
    apply_smoothing: bool = True,
    metric_key: str = "acc,none",
) -> int:
    """
    Run full analysis for all benchmark directories.

    Args:
        benchmarks_base_dir: Base directory containing benchmark subdirectories
        reports_base_dir: Base directory for output reports
        use_percentage_focus: Whether to use percentage-based metrics
        apply_smoothing: Whether to apply noise reduction
        metric_key: Metric key to analyze
    """
    if benchmarks_base_dir is None:
        benchmarks_base_dir = Path("data/benchmarks")
    if reports_base_dir is None:
        reports_base_dir = Path("data/reports")

    # Get all subdirectories in benchmarks (e.g., smolLM, smolLM_DoRA, smolLM_LoRA, lfm2)
    benchmark_dirs = [d for d in benchmarks_base_dir.iterdir() if d.is_dir() and any(d.glob("step_*_results.json"))]

    if not benchmark_dirs:
        print(f"❌ No benchmark directories found in {benchmarks_base_dir}")
        print("   Looking for directories containing step_*_results.json files")
        return 1

    benchmarks = ["MMLU_en", "MMLU_pl", "MMLU_ru", "Hellaswag"]

    print("=" * 80)
    print("FULL CATASTROPHIC FORGETTING ANALYSIS")
    print("=" * 80)
    print(f"Benchmarks base directory: {benchmarks_base_dir.resolve()}")
    print(f"Reports base directory: {reports_base_dir.resolve()}")
    print(f"Found {len(benchmark_dirs)} benchmark directories:")
    for bd in sorted(benchmark_dirs):
        print(f"  - {bd.name}")
    print(f"Metric: {metric_key}")
    print(f"Percentage focus: {use_percentage_focus}")
    print(f"Smoothing: {apply_smoothing}")
    print("=" * 80)

    # Process each benchmark directory
    processed_count = 0
    for benchmark_dir in sorted(benchmark_dirs):
        print(f"\n{'=' * 80}")
        print(f"📁 Processing directory: {benchmark_dir.name}")
        print(f"{'=' * 80}")

        try:
            # Create output directory for this benchmark directory
            model_output_dir = reports_base_dir / benchmark_dir.name
            plots_output_dir = model_output_dir / "plots"
            plots_output_dir.mkdir(parents=True, exist_ok=True)

            all_reports = {}

            # Generate report and plots for each benchmark type
            for benchmark in benchmarks:
                print(f"\n  📊 Processing benchmark: {benchmark}")

                # Generate report for this benchmark
                report_output = model_output_dir / f"analysis_{benchmark.lower()}.json"
                print(f"     Generating report: {report_output.name}")

                report = generate_report(
                    results_dir=benchmark_dir,
                    output_file=report_output,
                    metric_key=metric_key,
                    benchmark_filter=benchmark,
                    use_percentage_focus=use_percentage_focus,
                    apply_smoothing=apply_smoothing,
                )

                all_reports[benchmark] = report
                print("     ✓ Report saved")

                # Generate all plots for this benchmark
                print("     Generating plots...")
                generate_all_plots_for_benchmark(report, benchmark, plots_output_dir)

                # If category file exists, also plot it directly
                if benchmark.startswith("MMLU"):
                    category_file = model_output_dir / f"categories_{benchmark.lower()}.json"
                    if category_file.exists():
                        print("     Generating category analysis plots...")
                        plot_category_analysis(category_file, plots_output_dir / benchmark)

            # Generate comparison plots if we have multiple benchmarks
            if len(all_reports) >= 2:
                print("\n  📈 Generating benchmark comparison plots...")
                plot_benchmark_comparison(all_reports, plots_output_dir)

            print(f"\n  ✅ Completed analysis for {benchmark_dir.name}")
            print(f"     Reports: {model_output_dir}")
            print(f"     Plots: {plots_output_dir}")
            processed_count += 1

        except Exception as e:
            print(f"\n  ❌ Error processing {benchmark_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✅ ALL ANALYSES COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed {processed_count}/{len(benchmark_dirs)} directories:")
    for bd in sorted(benchmark_dirs):
        model_output_dir = reports_base_dir / bd.name
        if model_output_dir.exists():
            print(f"  ✓ {bd.name}: {model_output_dir}")

    return 0 if processed_count == len(benchmark_dirs) else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full catastrophic forgetting analysis for all benchmark directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with default settings
  python -m llm_sql_finetuning.analysis.run_full_analysis
  
  # Specify custom directories
  python -m llm_sql_finetuning.analysis.run_full_analysis --benchmarks-dir data/benchmarks --reports-dir data/reports
  
  # Disable smoothing and use absolute values
  python -m llm_sql_finetuning.analysis.run_full_analysis --no-smoothing --absolute-values
        """,
    )

    parser.add_argument(
        "--benchmarks-dir",
        "-b",
        type=Path,
        default=None,
        help="Base directory containing benchmark subdirectories (default: data/benchmarks)",
    )

    parser.add_argument(
        "--reports-dir", "-r", type=Path, default=None, help="Base directory for output reports (default: data/reports)"
    )

    parser.add_argument("--metric", "-m", type=str, default="acc,none", help="Metric key to analyze (default: acc,none)")

    parser.add_argument("--no-smoothing", action="store_true", help="Disable noise reduction (averaging last 5 values)")

    parser.add_argument(
        "--absolute-values", action="store_true", help="Use absolute values instead of percentages (default: percentages)"
    )

    args = parser.parse_args()

    return run_full_analysis(
        benchmarks_base_dir=args.benchmarks_dir,
        reports_base_dir=args.reports_dir,
        use_percentage_focus=not args.absolute_values,
        apply_smoothing=not args.no_smoothing,
        metric_key=args.metric,
    )


if __name__ == "__main__":
    sys.exit(main())
