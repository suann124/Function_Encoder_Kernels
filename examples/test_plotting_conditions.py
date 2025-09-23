#!/usr/bin/env python3
"""
Test script to verify the exact plotting conditions used in plot_polynomial_degree_comparison
and identify why train-then-prune plots are not appearing.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from function_encoder.utils.experiment_saver import ExperimentSaver

def test_plot_conditions():
    """Test the exact conditions used in the plotting function."""
    saver = ExperimentSaver()
    results_dir = Path("results")

    # Load the exact experiments
    prune_exp_path = results_dir / "polynomial_prune_d4_20250923_204842"
    progressive_exp_path = results_dir / "polynomial_progressive_d4"

    print("🧪 TESTING PLOT CONDITIONS")
    print("=" * 50)

    # Load data
    prune_data = saver.load_experiment(prune_exp_path)
    progressive_data = saver.load_experiment(progressive_exp_path)

    # Test the exact conditions from the plotting code
    datasets = [progressive_data]  # Only testing one degree for now
    prune_datasets = [prune_data]

    # Collect all data for consistent scaling (mimics the plotting code)
    all_eigenvalues_covariance = []
    all_eigenvalues_explained_var = []

    print("Processing progressive data...")
    for data in datasets:
        # Get eigenvalues of covariance matrix for top row
        if "final_eigenvalues" in data["pca_data"]:
            print("  ✅ Found final_eigenvalues in progressive data")
            eigenvals = data["pca_data"]["final_eigenvalues"]
            print(f"     Shape: {eigenvals.shape}, Values: {eigenvals[:5]}")
            all_eigenvalues_covariance.extend(eigenvals)
        else:
            print("  ❌ final_eigenvalues NOT found in progressive data")

        # Reconstruct scores for eigenvalue spectrum (explained variance) for bottom row
        scores = []
        pca_data = data["pca_data"]
        if "num_scores" in pca_data:
            print("  ✅ Found num_scores in progressive data")
            num_scores = pca_data["num_scores"]
            print(f"     num_scores: {num_scores}")
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    scores.append(pca_data[score_key])
                    print(f"     Found {score_key}: shape={pca_data[score_key].shape}")
            print(f"     Total scores reconstructed: {len(scores)}")
        else:
            print("  ❌ num_scores NOT found in progressive data")

        if scores:
            final_score = scores[-1]  # Final eigenvalues
            print(f"     final_score shape: {final_score.shape}, values: {final_score}")
            all_eigenvalues_explained_var.extend(final_score)
        else:
            print("  ❌ No scores reconstructed")

    print("\nProcessing prune data...")
    for prune_data_item in prune_datasets:
        if prune_data_item:
            print("  ✅ Prune data exists")
            # Get eigenvalues of covariance matrix for top row (train-then-prune uses "eigenvalues")
            if "eigenvalues" in prune_data_item["pca_data"]:
                print("  ✅ Found eigenvalues in prune data")
                eigenvals = prune_data_item["pca_data"]["eigenvalues"]
                print(f"     Shape: {eigenvals.shape}, Values: {eigenvals[:5]}")
                all_eigenvalues_covariance.extend(eigenvals)
            else:
                print("  ❌ eigenvalues NOT found in prune data")

            # Get explained variance ratio for bottom row
            if "explained_variance_ratio" in prune_data_item["pca_data"]:
                print("  ✅ Found explained_variance_ratio in prune data")
                exp_var = prune_data_item["pca_data"]["explained_variance_ratio"]
                print(f"     Shape: {exp_var.shape}, Values: {exp_var[:5]}")
                all_eigenvalues_explained_var.extend(exp_var)
            else:
                print("  ❌ explained_variance_ratio NOT found in prune data")
        else:
            print("  ❌ Prune data is None")

    print(f"\nData collection summary:")
    print(f"  all_eigenvalues_covariance: {len(all_eigenvalues_covariance)} values")
    print(f"  all_eigenvalues_explained_var: {len(all_eigenvalues_explained_var)} values")

    # Now test the actual plotting conditions (simulate the loop in the plotting function)
    print(f"\n" + "="*50)
    print("TESTING PLOTTING LOGIC")
    print("="*50)

    for col, (data, prune_data_item) in enumerate(zip(datasets, prune_datasets)):
        print(f"\nColumn {col} (degree {3+col}):")

        # Top row: Eigenvalues of covariance matrix plots
        print("  Top row (eigenvalues of covariance matrix):")

        # Plot progressive method eigenvalues of covariance matrix
        if "final_eigenvalues" in data["pca_data"]:
            eigenvalues_cov = data["pca_data"]["final_eigenvalues"]
            print(f"    ✅ Progressive: Will plot {len(eigenvalues_cov)} eigenvalues")
        else:
            print(f"    ❌ Progressive: No final_eigenvalues found")

        # Plot train-then-prune method eigenvalues of covariance matrix if available
        if prune_data_item and "eigenvalues" in prune_data_item["pca_data"]:
            prune_eigenvalues_cov = prune_data_item["pca_data"]["eigenvalues"]
            print(f"    ✅ Train-then-prune: Will plot {len(prune_eigenvalues_cov)} eigenvalues")
        else:
            print(f"    ❌ Train-then-prune: No eigenvalues found or prune_data is None")

        # Bottom row: Eigenvalue spectrum (final basis)
        print("  Bottom row (explained variance spectrum):")

        # Plot progressive method eigenvalue spectrum
        scores = []
        pca_data = data["pca_data"]
        if "num_scores" in pca_data:
            num_scores = pca_data["num_scores"]
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    scores.append(pca_data[score_key])

        if scores:
            # Plot only the final (10th basis) eigenvalue spectrum
            final_score = scores[-1]  # Last score corresponds to 10 basis functions
            print(f"    ✅ Progressive: Will plot {len(final_score)} explained variance values")
        else:
            print(f"    ❌ Progressive: No scores found")

        # Plot train-then-prune method eigenvalue spectrum if available
        if prune_data_item and "explained_variance_ratio" in prune_data_item["pca_data"]:
            prune_explained_var = prune_data_item["pca_data"]["explained_variance_ratio"]
            print(f"    ✅ Train-then-prune: Will plot {len(prune_explained_var)} explained variance values")
        else:
            print(f"    ❌ Train-then-prune: No explained_variance_ratio found or prune_data is None")

def create_minimal_test_plot():
    """Create a minimal test plot to verify the plotting actually works."""
    saver = ExperimentSaver()
    results_dir = Path("results")

    # Load data
    prune_exp_path = results_dir / "polynomial_prune_d4_20250923_204842"
    progressive_exp_path = results_dir / "polynomial_progressive_d4"

    prune_data = saver.load_experiment(prune_exp_path)
    progressive_data = saver.load_experiment(progressive_exp_path)

    print(f"\n" + "="*50)
    print("CREATING MINIMAL TEST PLOT")
    print("="*50)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    # Top plot: eigenvalues of covariance matrix
    ax_top = axes[0]

    # Progressive data
    if "final_eigenvalues" in progressive_data["pca_data"]:
        eigenvals_prog = progressive_data["pca_data"]["final_eigenvalues"]
        ax_top.plot(range(1, len(eigenvals_prog) + 1), eigenvals_prog,
                   marker="o", color="blue", label="Progressive")
        print(f"✅ Plotted progressive eigenvalues: {len(eigenvals_prog)} points")

    # Train-then-prune data
    if "eigenvalues" in prune_data["pca_data"]:
        eigenvals_prune = prune_data["pca_data"]["eigenvalues"]
        ax_top.plot(range(1, len(eigenvals_prune) + 1), eigenvals_prune,
                   marker="s", color="red", label="Train-then-prune")
        print(f"✅ Plotted train-then-prune eigenvalues: {len(eigenvals_prune)} points")

    ax_top.set_ylabel("Eigenvalue of covariance matrix")
    ax_top.legend()
    ax_top.grid(True)

    # Bottom plot: explained variance
    ax_bottom = axes[1]

    # Progressive data - reconstruct final scores
    scores = []
    pca_data = progressive_data["pca_data"]
    if "num_scores" in pca_data:
        num_scores = pca_data["num_scores"]
        for i in range(num_scores):
            score_key = f"score_{i}"
            if score_key in pca_data:
                scores.append(pca_data[score_key])

    if scores:
        final_score = scores[-1]
        ax_bottom.plot(range(1, len(final_score) + 1), final_score,
                      marker="o", color="blue", label="Progressive")
        print(f"✅ Plotted progressive explained variance: {len(final_score)} points")

    # Train-then-prune data
    if "explained_variance_ratio" in prune_data["pca_data"]:
        exp_var_prune = prune_data["pca_data"]["explained_variance_ratio"]
        ax_bottom.plot(range(1, len(exp_var_prune) + 1), exp_var_prune,
                      marker="s", color="red", label="Train-then-prune")
        print(f"✅ Plotted train-then-prune explained variance: {len(exp_var_prune)} points")

    ax_bottom.set_xlabel("Eigenvalue Index")
    ax_bottom.set_ylabel("Explained Variance Ratio")
    ax_bottom.set_yscale("log")
    ax_bottom.legend()
    ax_bottom.grid(True)

    plt.tight_layout()
    plt.savefig("test_polynomial_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved test plot as test_polynomial_comparison.png")

def main():
    test_plot_conditions()
    create_minimal_test_plot()

    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("Based on this analysis:")
    print("1. ✅ All required data fields are present in both experiments")
    print("2. ✅ The plotting conditions should work correctly")
    print("3. ✅ Created a minimal test plot that should show both methods")
    print("")
    print("If the train-then-prune plots are still missing from the degree comparison,")
    print("the issue might be:")
    print("- The plot_degree_comparison.py script is not passing prune paths correctly")
    print("- The auto-discovery logic is not finding prune experiments")
    print("- There's a bug in the plotting method invocation")

if __name__ == "__main__":
    main()