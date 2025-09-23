#!/usr/bin/env python3
"""
Test script to verify that the streamplot fix works with actual loaded models.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the function encoder to path
sys.path.append(str(Path(__file__).parent))

from function_encoder.utils.experiment_plotter import ExperimentPlotter

def test_model_loading():
    """Test the model loading functionality."""
    plotter = ExperimentPlotter()

    # Test loading Kepler models
    examples_dir = Path(__file__).parent / "examples"
    progressive_model_path = examples_dir / "kepler_pca_model.pth"
    prune_model_path = examples_dir / "kepler_train_prune_model.pth"

    print("Testing model loading...")
    print(f"Progressive model path exists: {progressive_model_path.exists()}")
    print(f"Prune model path exists: {prune_model_path.exists()}")

    if progressive_model_path.exists():
        print("\nLoading progressive model...")
        progressive_model = plotter._load_kepler_model(progressive_model_path, num_basis=10)
        if progressive_model is not None:
            print("✓ Progressive model loaded successfully!")

            # Test vector field generation
            print("Testing vector field generation...")
            x_range = [-2, 2]
            y_range = [-2, 2]
            x_grid = np.linspace(x_range[0], x_range[1], 10)
            y_grid = np.linspace(y_range[0], y_range[1], 10)
            X, Y = np.meshgrid(x_grid, y_grid)

            U, V = plotter._generate_real_vector_field(progressive_model, 0, X, Y)
            if U is not None and V is not None:
                print("✓ Vector field generation successful!")
                print(f"Vector field shape: U={U.shape}, V={V.shape}")
            else:
                print("✗ Vector field generation failed")
        else:
            print("✗ Progressive model loading failed")

    if prune_model_path.exists():
        print("\nLoading prune model...")
        prune_model = plotter._load_kepler_model(prune_model_path, num_basis=10)
        if prune_model is not None:
            print("✓ Prune model loaded successfully!")
        else:
            print("✗ Prune model loading failed")

def test_streamplot_visualization():
    """Test the actual streamplot visualization with real models."""
    print("\nTesting streamplot visualization...")

    plotter = ExperimentPlotter()

    # Create a mock comparison setup
    examples_dir = Path(__file__).parent / "examples"
    progressive_model_path = examples_dir / "kepler_pca_model.pth"
    prune_model_path = examples_dir / "kepler_train_prune_model.pth"

    if not (progressive_model_path.exists() and prune_model_path.exists()):
        print("Model files not found, skipping visualization test")
        return

    # Create a simple 2x2 subplot to test the functionality
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Create mock data structures
    class MockData:
        def __init__(self):
            self.pca_data = {"keep_indices": [0, 1, 2, 3]}

    data_progressive = MockData()
    data_prune = MockData()

    # Create grid for testing
    x_range = [-3, 3]
    y_range = [-3, 3]
    x_grid = np.linspace(x_range[0], x_range[1], 15)
    y_grid = np.linspace(y_range[0], y_range[1], 15)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Load models
    progressive_model = plotter._load_kepler_model(progressive_model_path, num_basis=10)
    prune_model = plotter._load_kepler_model(prune_model_path, num_basis=10)

    # Test plotting the first 4 basis functions
    for j in range(2):
        for k in range(2):
            idx = j * 2 + k
            ax = axes[j, k]

            # Try progressive model first
            model = progressive_model if idx < 2 else prune_model
            model_name = "Progressive" if idx < 2 else "Prune"
            basis_idx = idx % 2

            if model is not None:
                U, V = plotter._generate_real_vector_field(model, basis_idx, X, Y)
                if U is not None and V is not None:
                    ax.streamplot(X, Y, U, V, density=1.0, color='darkblue', arrowsize=1.0)
                    ax.set_title(f"{model_name} - ψ{basis_idx+1} (Learned)")
                else:
                    ax.text(0.5, 0.5, 'Vector field\ngeneration failed',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{model_name} - ψ{basis_idx+1} (Failed)")
            else:
                ax.text(0.5, 0.5, 'Model loading\nfailed',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{model_name} - ψ{basis_idx+1} (No Model)")

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_streamplot_output.png", dpi=150, bbox_inches='tight')
    print("✓ Test visualization saved as 'test_streamplot_output.png'")

if __name__ == "__main__":
    print("Testing streamplot fix with real models...")
    print("=" * 50)

    try:
        test_model_loading()
        test_streamplot_visualization()
        print("\n" + "=" * 50)
        print("All tests completed!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()