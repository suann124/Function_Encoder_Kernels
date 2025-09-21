#!/usr/bin/env python3

import sys, os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.utils.experiment_plotter import ExperimentPlotter
from pathlib import Path

def main():
    """Plot comparison between progressive and train-then-prune methods."""

    parser = argparse.ArgumentParser(description='Plot comparison between progressive and train-then-prune methods')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--dataset', nargs=2, metavar=('PROGRESSIVE_DATASET', 'PRUNE_DATASET'),
                       help='Two dataset names: first for progressive, second for train-then-prune (e.g., --dataset dt01 64)')
    parser.add_argument('--save-as', type=str, help='Custom filename for the plot (without extension)')
    args = parser.parse_args()

    plotter = ExperimentPlotter()

    # Find the most recent experiments for each method
    results_dir = Path("results")

    # Look for progressive experiments
    progressive_paths = []
    if args.dataset:
        progressive_dataset = args.dataset[0]
        progressive_paths.extend(list(results_dir.glob(f"kepler_progressive_*{progressive_dataset}*")))
    else:
        # Use most recent progressive experiment from any dataset
        all_progressive = list(results_dir.glob("kepler_progressive_*"))
        if all_progressive:
            latest_progressive = sorted(all_progressive, key=lambda p: p.stat().st_mtime)[-1]
            progressive_paths = [latest_progressive]

    # Look for train-then-prune experiments
    if args.dataset:
        prune_dataset = args.dataset[1]
        prune_paths = list(results_dir.glob(f"kepler_prune_*{prune_dataset}*"))
    else:
        # Use most recent prune experiment from any dataset
        all_prune = list(results_dir.glob("kepler_prune_*"))
        if all_prune:
            latest_prune = sorted(all_prune, key=lambda p: p.stat().st_mtime)[-1]
            prune_paths = [latest_prune]

    if not progressive_paths:
        if args.dataset:
            print(f"❌ No progressive experiments found for dataset '{args.dataset[0]}'. Please run a progressive experiment first.")
        else:
            print("❌ No progressive experiments found. Please run a progressive experiment first.")
        return

    if not prune_paths:
        if args.dataset:
            print(f"❌ No train-then-prune experiments found for dataset '{args.dataset[1]}'. Please run a train-then-prune experiment first.")
        else:
            print("❌ No train-then-prune experiments found. Please run a train-then-prune experiment first.")
        return

    # Get the experiments (already selected as most recent if no dataset specified)
    if args.dataset:
        # If datasets specified, get most recent from each filtered set
        progressive_path = sorted(progressive_paths, key=lambda p: p.stat().st_mtime)[-1]
        prune_path = sorted(prune_paths, key=lambda p: p.stat().st_mtime)[-1]
    else:
        # Already selected the most recent ones above
        progressive_path = progressive_paths[0]
        prune_path = prune_paths[0]

    print(f"📊 Plotting method comparison:")
    if args.dataset:
        print(f"  Progressive dataset: {args.dataset[0]}")
        print(f"  Train-then-prune dataset: {args.dataset[1]}")
    else:
        print(f"  Using most recent experiments from any dataset")
    print(f"  Progressive: {progressive_path.name}")
    print(f"  Train-then-prune: {prune_path.name}")

    # Create comparison plot
    try:
        # Determine filename
        if args.save_as:
            filename = f"{args.save_as}.png"
        else:
            filename = "Eig_spectrum_comparison.png"

        plotter.plot_method_comparison(
            progressive_path=progressive_path,
            prune_path=prune_path,
            save_dir=args.save_dir,
            custom_filename=filename
        )
        print(f"✅ Method comparison plot created successfully! Saved to: {args.save_dir}/{filename}")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()