#!/usr/bin/env python3
"""
Plot comparison between polynomial degrees (D3, D4, D5).
Shows eigenvalues of covariance matrix and explained variance ratios.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from function_encoder.utils.experiment_plotter import ExperimentPlotter

def find_degree_experiments(results_dir, method='progressive'):
    """Find polynomial experiments for different degrees."""
    results_path = Path(results_dir)

    experiments = {}
    for degree in [3, 4, 5]:
        pattern = f"polynomial_{method}_d{degree}*"
        degree_exps = list(results_path.glob(pattern))

        if degree_exps:
            # Get most recent for this degree
            experiments[f'd{degree}'] = sorted(degree_exps, key=lambda p: p.stat().st_mtime)[-1]

    return experiments

def main():
    parser = argparse.ArgumentParser(description='Plot comparison between polynomial degrees 3, 4, and 5')
    parser.add_argument('--d3-path', type=str, help='Path to degree 3 experiment')
    parser.add_argument('--d4-path', type=str, help='Path to degree 4 experiment')
    parser.add_argument('--d5-path', type=str, help='Path to degree 5 experiment')
    parser.add_argument('--method', choices=['progressive', 'prune'], default='progressive',
                       help='Method to compare (progressive or prune)')
    parser.add_argument('--save-dir', default='plot_outputs', help='Directory to save plots')
    parser.add_argument('--save-as', type=str, help='Custom filename for the plot (without extension)')
    parser.add_argument('--list', action='store_true', help='List available polynomial degree experiments')

    args = parser.parse_args()

    plotter = ExperimentPlotter()
    results_dir = Path("results")

    if args.list:
        print("Available polynomial degree experiments:")
        for method in ['progressive', 'prune']:
            experiments = find_degree_experiments(results_dir, method)
            if experiments:
                print(f"\n{method.upper()}:")
                for degree, path in experiments.items():
                    print(f"  {degree.upper()}: {path.name}")
        return

    # Determine experiment paths
    if args.d3_path and args.d4_path and args.d5_path:
        degree3_path = Path(args.d3_path)
        degree4_path = Path(args.d4_path)
        degree5_path = Path(args.d5_path)
    else:
        # Auto-find experiments
        experiments = find_degree_experiments(results_dir, args.method)

        missing_degrees = []
        for degree in ['d3', 'd4', 'd5']:
            if degree not in experiments:
                missing_degrees.append(degree)

        if missing_degrees:
            print(f"❌ Missing {args.method} experiments for degrees: {', '.join(missing_degrees)}")
            print(f"Please run polynomial experiments for these degrees first.")
            return

        degree3_path = experiments['d3']
        degree4_path = experiments['d4']
        degree5_path = experiments['d5']

    # Validate paths
    for degree, path in [('D3', degree3_path), ('D4', degree4_path), ('D5', degree5_path)]:
        if not path.exists():
            print(f"❌ {degree} experiment path not found: {path}")
            return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print(f"📊 Plotting polynomial degree comparison ({args.method}):")
    print(f"  D3: {degree3_path.name}")
    print(f"  D4: {degree4_path.name}")
    print(f"  D5: {degree5_path.name}")
    print(f"💾 Saving to: {save_dir}")

    try:
        filename = args.save_as if args.save_as else f"polynomial_degree_comparison_{args.method}"

        plotter.plot_polynomial_degree_comparison(
            degree3_path=degree3_path,
            degree4_path=degree4_path,
            degree5_path=degree5_path,
            save_dir=save_dir
        )

        print(f"✅ Polynomial degree comparison plot saved: {save_dir}/polynomial_degree_comparison.png")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()