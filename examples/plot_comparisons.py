#!/usr/bin/env python3
"""
Plot comparisons between progressive and train-then-prune methods.
Supports method comparison, basis comparison, and variance comparison.
Works with polynomial, kepler, and vdp experiments.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from function_encoder.utils.experiment_plotter import ExperimentPlotter

def find_experiments(results_dir, problem_type=None, method=None, dataset=None):
    """Find experiments matching criteria."""
    results_path = Path(results_dir)

    # Build pattern based on what's specified
    if problem_type and method:
        pattern = f"{problem_type}_{method}_*"
    elif problem_type:
        pattern = f"{problem_type}_*"
    elif method:
        pattern = f"*{method}*"
    else:
        pattern = "*"

    experiments = list(results_path.glob(pattern))

    # Filter by dataset keyword if specified
    if dataset:
        experiments = [exp for exp in experiments if dataset in exp.name]

    return experiments

def get_most_recent(paths):
    """Get most recent experiment from a list of paths."""
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]

def main():
    parser = argparse.ArgumentParser(description='Plot comparisons between progressive and train-then-prune methods')
    parser.add_argument('--progressive-path', type=str, help='Path to progressive experiment')
    parser.add_argument('--prune-path', type=str, help='Path to train-then-prune experiment')
    parser.add_argument('--problem-type', choices=['polynomial', 'kepler', 'vdp'],
                       help='Problem type (auto-detected if paths provided)')
    parser.add_argument('--dataset', nargs=2, metavar=('PROGRESSIVE_DATASET', 'PRUNE_DATASET'),
                       help='Dataset names for progressive and prune experiments')
    parser.add_argument('--save-dir', default='plot_outputs', help='Directory to save plots')
    parser.add_argument('--comparison-type', choices=['method', 'basis', 'all'], default='all',
                       help='Type of comparison to plot')
    parser.add_argument('--save-as', type=str, help='Custom filename prefix for plots')
    parser.add_argument('--list', action='store_true', help='List available experiment pairs')

    args = parser.parse_args()

    plotter = ExperimentPlotter()
    results_dir = Path("results")

    if args.list:
        print("Available experiment pairs:")
        for problem_type in ['polynomial', 'kepler', 'vdp']:
            progressive_exps = find_experiments(results_dir, problem_type, 'progressive')
            prune_exps = find_experiments(results_dir, problem_type, 'prune')

            if progressive_exps and prune_exps:
                print(f"\n{problem_type.upper()}:")
                print(f"  Progressive: {[p.name for p in progressive_exps]}")
                print(f"  Prune: {[p.name for p in prune_exps]}")
        return

    # Determine experiment paths
    if args.progressive_path and args.prune_path:
        progressive_path = Path(args.progressive_path)
        prune_path = Path(args.prune_path)
    else:
        # Auto-find experiments
        if args.problem_type:
            progressive_exps = find_experiments(results_dir, args.problem_type, 'progressive')
            prune_exps = find_experiments(results_dir, args.problem_type, 'prune')
        else:
            # Find any experiments
            progressive_exps = find_experiments(results_dir, method='progressive')
            prune_exps = find_experiments(results_dir, method='prune')

        if args.dataset:
            # Filter by dataset keywords
            prog_dataset, prune_dataset = args.dataset
            progressive_exps = [p for p in progressive_exps if prog_dataset in p.name]
            prune_exps = [p for p in prune_exps if prune_dataset in p.name]

        progressive_path = get_most_recent(progressive_exps)
        prune_path = get_most_recent(prune_exps)

        if not progressive_path:
            print("❌ No progressive experiments found. Please run a progressive experiment first.")
            return
        if not prune_path:
            print("❌ No train-then-prune experiments found. Please run a train-then-prune experiment first.")
            return

    # Validate paths
    if not progressive_path.exists():
        print(f"❌ Progressive experiment path not found: {progressive_path}")
        return
    if not prune_path.exists():
        print(f"❌ Train-then-prune experiment path not found: {prune_path}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print(f"📊 Plotting comparisons:")
    print(f"  Progressive: {progressive_path.name}")
    print(f"  Train-then-prune: {prune_path.name}")
    print(f"💾 Saving to: {save_dir}")

    try:
        # Method comparison (eigenvalue spectra)
        if args.comparison_type in ['method', 'all']:
            method_filename = f"{args.save_as}_method.png" if args.save_as else "method_comparison.png"
            plotter.plot_method_comparison(
                progressive_path=progressive_path,
                prune_path=prune_path,
                save_dir=save_dir,
                custom_filename=method_filename
            )
            print(f"✅ Method comparison plot saved: {save_dir}/{method_filename}")

            # Cumulative variance comparison
            cumvar_filename = f"{args.save_as}_cumvar.png" if args.save_as else "cumulative_variance_comparison.png"
            plotter.plot_cumulative_variance_comparison(
                progressive_path=progressive_path,
                prune_path=prune_path,
                save_dir=save_dir,
                custom_filename=cumvar_filename
            )
            print(f"✅ Cumulative variance comparison plot saved: {save_dir}/{cumvar_filename}")

        # Basis comparison (2x4 basis functions)
        if args.comparison_type in ['basis', 'all']:
            basis_filename = f"{args.save_as}_basis.png" if args.save_as else "basis_comparison.png"
            plotter.plot_basis_selection_comparison(
                progressive_path=progressive_path,
                prune_path=prune_path,
                save_dir=save_dir,
                custom_filename=basis_filename
            )
            print(f"✅ Basis comparison plot saved: {save_dir}/{basis_filename}")

        print("🎉 All comparison plots completed successfully!")

    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()