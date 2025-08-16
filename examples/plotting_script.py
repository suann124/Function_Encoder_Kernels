import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_npz(path):
    return np.load(path)

def plot_eigenvalue_spectrum(eigenvalues, keep_indices, outdir, tag=""):
    plt.figure()
    plt.semilogy(eigenvalues, 'b.-', label='Eigenvalues')
    if keep_indices is not None:
        plt.axvline(x=len(keep_indices)-1, color='r', linestyle='--', label=f'Cutoff (n={len(keep_indices)})')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title(f'PCA Eigenvalue Spectrum {tag}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'eigenvalue_spectrum{tag}.png'), dpi=300)
    plt.close()

def plot_cumulative_variance(explained_var, keep_indices, outdir, tag=""):
    plt.figure()
    cumsum_var = np.cumsum(explained_var)
    plt.plot(cumsum_var, 'g.-')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
    if keep_indices is not None:
        plt.axvline(x=len(keep_indices)-1, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Variance Explained {tag}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'cumulative_variance{tag}.png'), dpi=300)
    plt.close()

def plot_basis_importance(n_basis, keep_indices, outdir, tag=""):
    plt.figure()
    basis_indices = np.arange(n_basis)
    colors = ['red' if (keep_indices is not None and i in keep_indices) else 'blue' for i in basis_indices]
    plt.bar(basis_indices, np.ones(n_basis), color=colors)
    plt.xlabel('Basis Function Index')
    plt.ylabel('Selected')
    plt.title(f'Selected Basis Functions (Red = Kept) {tag}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'basis_importance{tag}.png'), dpi=300)
    plt.close()

def plot_function_approximation(path, outdir, tag=""):
    data = np.load(path)
    plt.figure()
    plt.plot(data['X_sorted'], data['y_sorted'], 'k-', label='True', linewidth=1)
    plt.plot(data['X_sorted'], data['y_pred_orig'], 'b--', label='Original', alpha=0.8)
    plt.plot(data['X_sorted'], data['y_pred_pruned'], 'g--', label='Pruned', alpha=0.8)
    plt.plot(data['X_sorted'], data['y_pred_pruned_refined'], 'r:', label='Pruned & Refined', linewidth=2)
    plt.scatter(data['example_X'], data['example_y'], c='red', s=20, zorder=5, alpha=0.5, label='Example Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Function Approximation Comparison {tag}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'function_approximation{tag}.png'), dpi=300)
    plt.close()

def plot_coeff_comparison(path, outdir, tag=""):
    data = np.load(path)
    plt.figure()
    x_pos = np.arange(len(data['coeffs_orig']))
    plt.bar(x_pos - 0.2, data['coeffs_orig'], 0.4, label='Original', alpha=0.7)
    x_pos_pruned = np.arange(len(data['coeffs_pruned']))
    plt.bar(x_pos_pruned + 0.2, data['coeffs_pruned'], 0.4, label='Pruned', alpha=0.7)
    x_pos_pruned_refined = np.arange(len(data['coeffs_pruned_refined']))
    plt.bar(x_pos_pruned_refined + 0.4, data['coeffs_pruned_refined'], 0.4, label='Pruned & Refined', alpha=0.7)
    plt.xlabel('Basis Index')
    plt.ylabel('Coefficient Value')
    plt.title(f'Coefficient Comparison {tag}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'coeff_comparison{tag}.png'), dpi=300)
    plt.close()

def print_performance_summary(path):
    import json
    with open(path, 'r') as f:
        summary = json.load(f)
    print("Performance Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="General-purpose results plotting tool for function encoder experiments.")
    parser.add_argument('--pca', type=str, required=True, help='Path to PCA results .npz file')
    parser.add_argument('--comp', type=str, required=True, help='Path to comparison results .npz file')
    parser.add_argument('--func', type=str, required=False, help='Path to function approximation .npz file')
    parser.add_argument('--coeff', type=str, required=False, help='Path to coefficient comparison .npz file')
    parser.add_argument('--summary', type=str, required=False, help='Path to performance summary .json file')
    parser.add_argument('--outdir', type=str, default='results/plots', help='Output directory for plots')
    parser.add_argument('--tag', type=str, default='', help='Tag to append to plot filenames (e.g. dataset name)')
    parser.add_argument('--all', action='store_true', help='Plot all figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pca = load_npz(args.pca)
    comp = load_npz(args.comp)

    if args.all:
        # Existing plots
        plot_eigenvalue_spectrum(pca['eigenvalues'], pca.get('keep_indices', None), args.outdir, args.tag)
        plot_cumulative_variance(pca['explained_var'], pca.get('keep_indices', None), args.outdir, args.tag)
        n_basis = None
        if 'coeffs_original' in comp:
            n_basis = comp['coeffs_original'].shape[1] if len(comp['coeffs_original'].shape) > 1 else len(comp['coeffs_original'])
        elif 'n_basis' in pca:
            n_basis = int(pca['n_basis'])
        elif 'eigenvectors' in pca:
            n_basis = pca['eigenvectors'].shape[0]
        if n_basis is not None:
            plot_basis_importance(n_basis, pca.get('keep_indices', None), args.outdir, args.tag)
        # New plots
        if args.func:
            plot_function_approximation(args.func, args.outdir, args.tag)
        if args.coeff:
            plot_coeff_comparison(args.coeff, args.outdir, args.tag)
        if args.summary:
            print_performance_summary(args.summary)

if __name__ == '__main__':
    main()
