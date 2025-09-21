#!/usr/bin/env python3
"""
Command line script to plot saved experiments.
Usage: python plot_experiments.py [experiment_path] [--save-dir output_dir]
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from function_encoder.utils.experiment_plotter import ExperimentPlotter

def main():
    parser = argparse.ArgumentParser(description='Plot saved experiments')
    parser.add_argument('experiment_path', nargs='?', help='Path to experiment directory')
    parser.add_argument('--save-dir', default='plot_outputs', help='Directory to save plots')
    parser.add_argument('--list', action='store_true', help='List all available experiments')

    args = parser.parse_args()

    plotter = ExperimentPlotter()

    if args.list:
        # List all experiments
        experiments = plotter.list_all_experiments()
        print(f"Found {len(experiments)} experiments:")
        for i, exp in enumerate(experiments):
            print(f"{i+1:2d}. {exp}")

        if experiments:
            print(f"\nTo plot an experiment, run:")
            print(f"python plot_experiments.py \"{experiments[0]}\" --save-dir plot_outputs")
        return

    if not args.experiment_path:
        print("Please provide an experiment path or use --list to see available experiments")
        print("Usage: python plot_experiments.py [experiment_path] [--save-dir output_dir]")
        return

    exp_path = Path(args.experiment_path)
    if not exp_path.exists():
        print(f"Error: Experiment path does not exist: {exp_path}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print(f"Plotting experiment: {exp_path}")
    print(f"Saving plots to: {save_dir}")

    try:
        plotter.plot_experiment(exp_path, save_dir=save_dir)
        print("✅ Plotting completed successfully!")
        print(f"📁 Check {save_dir}/ for saved plot files")
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()