import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_results():
    pca = np.load('results/pca_results.npz')
    comp = np.load('results/comparison_results.npz')
    return pca, comp

def plot_eigenvalue_spectrum(eigenvalues, keep_indices, outdir):
    plt.figure()
    plt.semilogy(eigenvalues, 'b.-', label='Eigenvalues')
    plt.axvline(x=len(keep_indices)-1, color='r', linestyle='--', label=f'Cutoff (n={len(keep_indices)})')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('PCA Eigenvalue Spectrum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'eigenvalue_spectrum.png'), dpi=300)
    plt.close()

def plot_cumulative_variance(explained_var, keep_indices, outdir):
    plt.figure()
    cumsum_var = np.cumsum(explained_var)
    plt.plot(cumsum_var, 'g.-')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
    plt.axvline(x=len(keep_indices)-1, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cumulative_variance.png'), dpi=300)
    plt.close()

def plot_basis_importance(n_basis, keep_indices, outdir):
    plt.figure()
    basis_indices = np.arange(n_basis)
    colors = ['red' if i in keep_indices else 'blue' for i in basis_indices]
    plt.bar(basis_indices, np.ones(n_basis), color=colors)
    plt.xlabel('Basis Function Index')
    plt.ylabel('Selected')
    plt.title('Selected Basis Functions (Red = Kept)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'basis_importance.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot polynomial pruning results for publication.")
    parser.add_argument('--outdir', type=str, default='results/plots', help='Output directory for plots')
    parser.add_argument('--all', action='store_true', help='Plot all figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pca, comp = load_results()

    if args.all:
        plot_eigenvalue_spectrum(pca['eigenvalues'], pca['keep_indices'], args.outdir)
        plot_cumulative_variance(pca['explained_var'], pca['keep_indices'], args.outdir)
        n_basis = comp['coeffs_original'].shape[1] if len(comp['coeffs_original'].shape) > 1 else len(comp['coeffs_original'])
        plot_basis_importance(n_basis, pca['keep_indices'], args.outdir)

if __name__ == '__main__':
    main()
