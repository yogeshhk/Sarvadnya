#!/usr/bin/env python3
"""
Yogasutra GraphRAG Benchmark Test Runner

This script provides a command-line interface for running comprehensive benchmarks
on the GraphRAG and LinearRAG systems with various configurations.

Usage:
    python run_benchmark.py --config baseline_fast
    python run_benchmark.py --all
    python run_benchmark.py --config custom_config --benchmark-file custom_dataset.json

Author: Suraj Kulkarni
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any

from test_framework import (
    TestConfig, GraphRAGTester, load_test_configs,
    print_results_summary, save_results_to_file, generate_comparison_report
)

def validate_environment():
    """Validate that required environment variables and files are present."""
    errors = []

    # Check GROQ API key
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY environment variable not set")

    # Check benchmark dataset
    if not os.path.exists("benchmark_dataset.json"):
        errors.append("benchmark_dataset.json not found")

    # Check test configurations
    if not os.path.exists("test_configs.json"):
        errors.append("test_configs.json not found")

    # Check data files
    if not os.path.exists("data/graph_small.json"):
        errors.append("data/graph_small.json not found")

    if errors:
        print("Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues before running benchmarks.")
        return False

    return True

def run_single_configuration(config_name: str, benchmark_file: str, output_dir: str, verbose: bool = False) -> str:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*80}")
    print(f"RUNNING CONFIGURATION: {config_name}")
    print(f"{'='*80}")

    try:
        # Load configurations
        configs = load_test_configs("test_configs.json")
        if config_name not in configs:
            raise Exception(f"Configuration '{config_name}' not found in test_configs.json")

        config = configs[config_name]

        # Initialize tester
        tester = GraphRAGTester(config)

        # Run benchmark
        start_time = time.time()
        summary = tester.run_benchmark(benchmark_file)
        end_time = time.time()

        # Print results
        print_results_summary(summary)
        print(".2f")

        # Save results
        result_file = save_results_to_file(summary, output_dir)

        if verbose:
            print("\nDetailed Results:")
            print("-" * 40)
            for result in summary.detailed_results:
                status = "✓" if result.passed else "✗"
                print(f"{status} {result.test_id}: {result.overall_score:.3f} "
                      f"(K:{result.keyword_score:.3f}, S:{result.sutra_reference_score:.3f}, Sem:{result.semantic_score:.3f})")

        return result_file

    except Exception as e:
        print(f"Error running configuration {config_name}: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def run_all_configurations(benchmark_file: str, output_dir: str, verbose: bool = False, parallel: bool = False) -> List[str]:
    """Run benchmark for all configurations."""
    print(f"\n{'='*80}")
    print("RUNNING ALL CONFIGURATIONS")
    print(f"{'='*80}")

    try:
        # Load all configurations
        configs = load_test_configs("test_configs.json")

        if not configs:
            raise Exception("No configurations found in test_configs.json")

        print(f"Found {len(configs)} configurations to test:")
        for name, config in configs.items():
            print(f"  - {name}: {config.description}")
        print()

        result_files = []

        if parallel:
            # TODO: Implement parallel execution
            print("Parallel execution not yet implemented. Running sequentially...")
            parallel = False

        # Run each configuration
        for i, (config_name, config) in enumerate(configs.items(), 1):
            print(f"\n--- Configuration {i}/{len(configs)} ---")
            result_file = run_single_configuration(config_name, benchmark_file, output_dir, verbose)
            if result_file:
                result_files.append(result_file)

        # Generate comparison report
        if result_files:
            print(f"\n{'='*80}")
            print("GENERATING COMPARISON REPORT")
            print(f"{'='*80}")

            try:
                comparison_file = generate_comparison_report(output_dir)
                print(f"Comparison report generated: {comparison_file}")

                # Print top 3 configurations
                print_top_configurations(comparison_file)

            except Exception as e:
                print(f"Error generating comparison report: {str(e)}")

        return result_files

    except Exception as e:
        print(f"Error running all configurations: {str(e)}")
        return []

def print_top_configurations(comparison_file: str):
    """Print top performing configurations from comparison report."""
    try:
        with open(comparison_file, 'r') as f:
            data = json.load(f)

        print("\nTOP PERFORMING CONFIGURATIONS:")
        print("-" * 40)

        configs = data.get("configurations", [])
        if configs:
            for i, config in enumerate(configs[:3], 1):
                print(f"{i}. {config['config_name']}")
                print(".1f")
                print(".3f")
                print(".3f")
                print(".3f")
                print(".3f")
                print()

    except Exception as e:
        print(f"Error reading comparison file: {str(e)}")

def list_configurations():
    """List all available test configurations."""
    try:
        configs = load_test_configs("test_configs.json")

        print("Available Test Configurations:")
        print("=" * 50)

        for name, config in configs.items():
            print(f"\n{name}:")
            print(f"  Description: {config.description}")
            print(f"  Backend: {config.backend_type}")
            print(f"  Embedding Model: {config.embedding_model}")
            print(f"  LLM Model: {config.groq_model}")
            print(f"  Temperature: {config.temperature}")
            print(f"  Max Tokens: {config.max_tokens}")
            print(f"  Conversation Mode: {config.conversation_mode}")
            print(f"  Response Mode: {config.response_mode}")
            print(f"  Graph File: {config.graph_file}")

    except Exception as e:
        print(f"Error loading configurations: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Yogasutra GraphRAG Benchmark Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --config baseline_fast
  python run_benchmark.py --all --verbose
  python run_benchmark.py --config baseline_quality --output-dir my_results
  python run_benchmark.py --list-configs
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Run a specific configuration (see --list-configs for options)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available configurations"
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available test configurations"
    )

    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="benchmark_dataset.json",
        help="Path to benchmark dataset JSON file (default: benchmark_dataset.json)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save test results (default: test_results)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed results"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation (not recommended)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run configurations in parallel (not yet implemented)"
    )

    args = parser.parse_args()

    # Handle list configurations
    if args.list_configs:
        list_configurations()
        return

    # Validate environment unless skipped
    if not args.skip_validation:
        if not validate_environment():
            sys.exit(1)

    # Validate arguments
    if not args.config and not args.all:
        print("Error: Must specify either --config <name> or --all")
        parser.print_help()
        sys.exit(1)

    if args.config and args.all:
        print("Error: Cannot specify both --config and --all")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmarks
    try:
        if args.config:
            result_file = run_single_configuration(
                args.config,
                args.benchmark_file,
                args.output_dir,
                args.verbose
            )
            if result_file:
                print(f"\nBenchmark completed successfully. Results saved to: {result_file}")
            else:
                print("\nBenchmark failed.")
                sys.exit(1)

        elif args.all:
            result_files = run_all_configurations(
                args.benchmark_file,
                args.output_dir,
                args.verbose,
                args.parallel
            )
            if result_files:
                print(f"\nAll benchmarks completed successfully. {len(result_files)} result files generated.")
            else:
                print("\nAll benchmarks failed.")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
