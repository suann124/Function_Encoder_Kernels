#!/usr/bin/env python3

import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.utils.experiment_plotter import ExperimentPlotter
from pathlib import Path

def main():
    """Plot comparison between progressive and train-then-prune methods."""

    parser = argparse.ArgumentParser(description='Plot comparison between progressive and train-then-prune methods')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    plotter = ExperimentPlotter()

    # Find the most recent experiments for each method
    results_dir = Path("results")

    # Look for progressive experiments
    progressive_paths = []
    progressive_paths.extend(list(results_dir.glob("polynomial_progressive_*")))

    # Look for train-then-prune experiments
    prune_paths = list(results_dir.glob("polynomial_train_then_prune_*"))

    if not progressive_paths:
        print("❌ No progressive experiments found. Please run a progressive experiment first.")
        return

    if not prune_paths:
        print("❌ No train-then-prune experiments found. Please run a train-then-prune experiment first.")
        return

    # Get the most recent experiment for each method
    progressive_path = sorted(progressive_paths)[-1]  # Most recent
    prune_path = sorted(prune_paths)[-1]  # Most recent

    print(f"📊 Plotting method comparison:")
    print(f"  Progressive: {progressive_path.name}")
    print(f"  Train-then-prune: {prune_path.name}")

    # Create comparison plot
    try:
        plotter.plot_method_comparison(
            progressive_path=progressive_path,
            prune_path=prune_path,
            save_dir=args.save_dir
        )
        print(f"✅ Method comparison plot created successfully! Saved to: {args.save_dir}/method_comparison.png")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()