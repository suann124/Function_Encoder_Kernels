#!/usr/bin/env python3
"""
Diagnostic script to debug why train-then-prune plots are not appearing
in the polynomial degree comparison plot.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pathlib import Path
from function_encoder.utils.experiment_saver import ExperimentSaver

def examine_experiment_data(experiment_path):
    """Examine the data structure of an experiment."""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {experiment_path}")
    print(f"{'='*60}")

    saver = ExperimentSaver()
    try:
        data = saver.load_experiment(experiment_path)
    except Exception as e:
        print(f"❌ Failed to load experiment: {e}")
        return None

    # Basic metadata
    print(f"Metadata:")
    if "metadata" in data:
        for key, value in data["metadata"].items():
            print(f"  {key}: {value}")

    # PCA data structure - this is the key for our investigation
    print(f"\nPCA Data Structure:")
    if "pca_data" in data:
        pca_data = data["pca_data"]
        print(f"  Available keys: {list(pca_data.keys())}")

        # Check for expected keys
        expected_keys = ["eigenvalues", "explained_variance_ratio", "final_eigenvalues", "num_scores"]
        for key in expected_keys:
            if key in pca_data:
                value = pca_data[key]
                if hasattr(value, 'shape'):
                    print(f"  ✅ {key}: shape={value.shape}, type={type(value)}")
                else:
                    print(f"  ✅ {key}: value={value}, type={type(value)}")
            else:
                print(f"  ❌ {key}: NOT FOUND")

        # Print shapes and sample values for key arrays
        print(f"\nDetailed PCA Data Analysis:")

        if "eigenvalues" in pca_data:
            eigenvals = pca_data["eigenvalues"]
            print(f"  eigenvalues: {eigenvals[:10] if len(eigenvals) > 10 else eigenvals}")

        if "explained_variance_ratio" in pca_data:
            exp_var = pca_data["explained_variance_ratio"]
            print(f"  explained_variance_ratio: {exp_var[:10] if len(exp_var) > 10 else exp_var}")

        if "final_eigenvalues" in pca_data:
            final_eigenvals = pca_data["final_eigenvalues"]
            print(f"  final_eigenvalues: {final_eigenvals[:10] if len(final_eigenvals) > 10 else final_eigenvals}")

        # Check for score data (progressive method)
        if "num_scores" in pca_data:
            num_scores = pca_data["num_scores"]
            print(f"  num_scores: {num_scores}")
            for i in range(min(3, num_scores)):  # Show first 3 scores
                score_key = f"score_{i}"
                if score_key in pca_data:
                    score = pca_data[score_key]
                    print(f"  {score_key}: shape={score.shape if hasattr(score, 'shape') else 'N/A'}")
    else:
        print("  ❌ NO PCA DATA FOUND")

    # Other data sections
    print(f"\nOther Data Sections:")
    for section in ["training_data", "visualization_data", "comparison_data", "performance_summary"]:
        if section in data:
            section_data = data[section]
            if isinstance(section_data, dict):
                print(f"  ✅ {section}: {list(section_data.keys())}")
            else:
                print(f"  ✅ {section}: {type(section_data)}")
        else:
            print(f"  ❌ {section}: NOT FOUND")

    return data

def test_plotting_conditions(progressive_data, prune_data):
    """Test the specific conditions used in plot_polynomial_degree_comparison."""
    print(f"\n{'='*60}")
    print(f"TESTING PLOTTING CONDITIONS")
    print(f"{'='*60}")

    print("Progressive data conditions:")
    if progressive_data and "pca_data" in progressive_data:
        pca_data = progressive_data["pca_data"]

        # Test condition for top row (eigenvalues of covariance matrix)
        final_eigenvals_exists = "final_eigenvalues" in pca_data
        print(f"  final_eigenvalues exists: {final_eigenvals_exists}")

        # Test condition for bottom row (explained variance)
        num_scores_exists = "num_scores" in pca_data
        print(f"  num_scores exists: {num_scores_exists}")

        if num_scores_exists:
            num_scores = pca_data["num_scores"]
            print(f"  num_scores value: {num_scores}")

            scores = []
            for i in range(num_scores):
                score_key = f"score_{i}"
                if score_key in pca_data:
                    scores.append(pca_data[score_key])
            print(f"  scores reconstructed: {len(scores)} scores found")

            if scores:
                final_score = scores[-1]
                print(f"  final_score shape: {final_score.shape if hasattr(final_score, 'shape') else 'N/A'}")

    print("\nTrain-then-prune data conditions:")
    if prune_data and "pca_data" in prune_data:
        pca_data = prune_data["pca_data"]

        # Test condition for top row (eigenvalues of covariance matrix)
        eigenvals_exists = "eigenvalues" in pca_data
        print(f"  eigenvalues exists: {eigenvals_exists}")

        # Test condition for bottom row (explained variance)
        exp_var_exists = "explained_variance_ratio" in pca_data
        print(f"  explained_variance_ratio exists: {exp_var_exists}")

        if eigenvals_exists:
            eigenvals = pca_data["eigenvalues"]
            print(f"  eigenvalues shape: {eigenvals.shape if hasattr(eigenvals, 'shape') else 'N/A'}")

        if exp_var_exists:
            exp_var = pca_data["explained_variance_ratio"]
            print(f"  explained_variance_ratio shape: {exp_var.shape if hasattr(exp_var, 'shape') else 'N/A'}")

def main():
    results_dir = Path("results")

    # Find a train-then-prune experiment as requested
    prune_pattern = "polynomial_prune_d4_20250923_204842"
    prune_exp_path = results_dir / prune_pattern

    # Find a progressive experiment for comparison
    progressive_pattern = "polynomial_progressive_d4"
    progressive_exp_path = results_dir / progressive_pattern

    print("🔍 DEBUGGING POLYNOMIAL DEGREE COMPARISON PLOT")
    print("=" * 80)

    # Examine train-then-prune experiment
    prune_data = examine_experiment_data(prune_exp_path)

    # Examine progressive experiment for comparison
    progressive_data = examine_experiment_data(progressive_exp_path)

    # Test the plotting conditions
    test_plotting_conditions(progressive_data, prune_data)

    # Summary and recommendations
    print(f"\n{'='*60}")
    print("SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*60}")

    if prune_data:
        pca_data = prune_data.get("pca_data", {})

        has_eigenvalues = "eigenvalues" in pca_data
        has_exp_var_ratio = "explained_variance_ratio" in pca_data

        print(f"Train-then-prune experiment has:")
        print(f"  eigenvalues: {'✅' if has_eigenvalues else '❌'}")
        print(f"  explained_variance_ratio: {'✅' if has_exp_var_ratio else '❌'}")

        if not has_eigenvalues or not has_exp_var_ratio:
            print(f"\n🚨 ISSUE FOUND:")
            print(f"The train-then-prune experiment is missing the required fields:")
            if not has_eigenvalues:
                print(f"  - 'eigenvalues' in pca_data")
            if not has_exp_var_ratio:
                print(f"  - 'explained_variance_ratio' in pca_data")
            print(f"\nThis explains why the train-then-prune plots are not appearing!")
            print(f"The plotting code expects these exact field names.")
        else:
            print(f"\n✅ All required fields are present!")
            print(f"The issue might be elsewhere in the plotting logic.")

    if progressive_data:
        pca_data = progressive_data.get("pca_data", {})

        has_final_eigenvals = "final_eigenvalues" in pca_data
        has_num_scores = "num_scores" in pca_data

        print(f"\nProgressive experiment has:")
        print(f"  final_eigenvalues: {'✅' if has_final_eigenvals else '❌'}")
        print(f"  num_scores: {'✅' if has_num_scores else '❌'}")

if __name__ == "__main__":
    main()