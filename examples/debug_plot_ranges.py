#!/usr/bin/env python3
"""
Debug script to check the data ranges for progressive vs train-then-prune experiments
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from function_encoder.utils.experiment_saver import ExperimentSaver

def debug_data_ranges():
    """Debug the data ranges to understand why train-then-prune plots disappeared."""

    saver = ExperimentSaver("results")

    # Progressive experiment paths
    progressive_d4 = "results/polynomial_progressive_d4"

    # Train-then-prune experiment paths
    prune_d4 = "results/polynomial_prune_d4_20250923_204842"

    print("=== PROGRESSIVE EXPERIMENT DEBUG ===")
    prog_data = saver.load_experiment(progressive_d4)
    pca_data = prog_data["pca_data"]

    print("Available keys in pca_data:")
    for key in pca_data.keys():
        print(f"  - {key}")

    if "final_eigenvalues" in pca_data:
        final_eigenvals = pca_data["final_eigenvalues"]
        print(f"final_eigenvalues length: {len(final_eigenvals)}")
        print(f"final_eigenvalues range: x-axis will be 1 to {len(final_eigenvals)}")
        print(f"final_eigenvalues values: {final_eigenvals[:10]}...")  # First 10 values

    # Check progressive scores
    scores = []
    if "num_scores" in pca_data:
        num_scores = pca_data["num_scores"]
        print(f"num_scores: {num_scores}")
        for i in range(num_scores):
            score_key = f"score_{i}"
            if score_key in pca_data:
                scores.append(pca_data[score_key])
        if scores:
            final_score = scores[-1]
            print(f"final score (score_{num_scores-1}) length: {len(final_score)}")
            print(f"final score range: x-axis will be 1 to {len(final_score)}")

    print("\n=== TRAIN-THEN-PRUNE EXPERIMENT DEBUG ===")
    prune_data = saver.load_experiment(prune_d4)
    prune_pca_data = prune_data["pca_data"]

    print("Available keys in pca_data:")
    for key in prune_pca_data.keys():
        print(f"  - {key}")

    if "eigenvalues" in prune_pca_data:
        eigenvals = prune_pca_data["eigenvalues"]
        print(f"eigenvalues length: {len(eigenvals)}")
        print(f"eigenvalues range: x-axis will be 1 to {len(eigenvals)}")
        print(f"eigenvalues values: {eigenvals[:10]}...")  # First 10 values

    if "explained_variance_ratio" in prune_pca_data:
        explained_var = prune_pca_data["explained_variance_ratio"]
        print(f"explained_variance_ratio length: {len(explained_var)}")
        print(f"explained_variance_ratio range: x-axis will be 1 to {len(explained_var)}")
        print(f"explained_variance_ratio values: {explained_var[:10]}...")  # First 10 values

    if "keep_indices" in prune_pca_data:
        keep_indices = prune_pca_data["keep_indices"]
        print(f"keep_indices: {keep_indices}")
        print(f"Number of kept components: {len(keep_indices)}")

    print("\n=== ANALYSIS ===")
    print("If xlim(1,10) was set:")
    print("- Progressive data would show points from 1 to", len(final_eigenvals) if "final_eigenvalues" in pca_data else "unknown")
    print("- Train-then-prune eigenvalues would show points from 1 to", len(eigenvals) if "eigenvalues" in prune_pca_data else "unknown")
    print("- Train-then-prune explained variance would show points from 1 to", len(explained_var) if "explained_variance_ratio" in prune_pca_data else "unknown")

    # Check if train-then-prune data would be visible in xlim(1,10)
    if "eigenvalues" in prune_pca_data:
        if len(prune_pca_data["eigenvalues"]) < 10:
            print(f"⚠️  ISSUE: Train-then-prune eigenvalues only go to {len(prune_pca_data['eigenvalues'])}, so xlim(1,10) would show them!")
        else:
            print("✅ Train-then-prune eigenvalues should be visible in xlim(1,10)")

    if "explained_variance_ratio" in prune_pca_data:
        if len(prune_pca_data["explained_variance_ratio"]) < 10:
            print(f"⚠️  ISSUE: Train-then-prune explained variance only goes to {len(prune_pca_data['explained_variance_ratio'])}, so xlim(1,10) would show them!")
        else:
            print("✅ Train-then-prune explained variance should be visible in xlim(1,10)")

if __name__ == "__main__":
    debug_data_ranges()