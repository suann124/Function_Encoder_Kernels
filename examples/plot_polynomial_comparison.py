#!/usr/bin/env python3

import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.utils.experiment_plotter import ExperimentPlotter
from pathlib import Path

def main():
    """Plot comparison between polynomial degrees 3, 4, and 5."""

    parser = argparse.ArgumentParser(description='Plot comparison between polynomial degrees 3, 4, and 5')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    plotter = ExperimentPlotter()

    # Find the most recent experiments for each degree
    results_dir = Path("results")

    # Look for degree 3 experiments (multiple naming patterns)
    d3_paths = []
    d3_paths.extend(list(results_dir.glob("polynomial_progressive_poly_degree3_*")))  # poly_degree3 pattern
    d3_paths.extend(list(results_dir.glob("polynomial_progressive_polynomial_d3_*")))  # polynomial_d3 pattern
    # Also check for any polynomial experiments without specific degree in name
    general_poly = list(results_dir.glob("polynomial_progressive_polynomial_*"))
    d3_paths.extend([p for p in general_poly if "polynomial_d" not in p.name])

    # Look for degree 4
    d4_paths = list(results_dir.glob("polynomial_progressive_polynomial_d4_*"))

    # Look for degree 5
    d5_paths = list(results_dir.glob("polynomial_progressive_polynomial_d5_*"))

    if not d3_paths:
        print("❌ No degree 3 experiments found. Please run polynomial_pca.py first.")
        return

    if not d4_paths:
        print("❌ No degree 4 experiments found. Please run polynomial_pca_degree4.py first.")
        return

    if not d5_paths:
        print("❌ No degree 5 experiments found. Please run polynomial_pca_degree5.py first.")
        return

    # Get the most recent experiment for each degree
    degree3_path = sorted(d3_paths)[-1]  # Most recent
    degree4_path = sorted(d4_paths)[-1]  # Most recent
    degree5_path = sorted(d5_paths)[-1]  # Most recent

    print(f"📊 Plotting comparison:")
    print(f"  D3: {degree3_path.name}")
    print(f"  D4: {degree4_path.name}")
    print(f"  D5: {degree5_path.name}")

    # Create comparison plot
    try:
        plotter.plot_polynomial_degree_comparison(
            degree3_path=degree3_path,
            degree4_path=degree4_path,
            degree5_path=degree5_path,
            save_dir=args.save_dir
        )
        print(f"✅ Comparison plot created successfully! Saved to: {args.save_dir}/polynomial_degree_comparison.png")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()