import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


class ExperimentSaver:
    """
    Centralized experiment data saver for function encoder experiments.

    Handles saving of training data, PCA analysis, model comparisons, and
    visualization data for easy plotting and analysis.
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _ensure_dir(self, path: Path) -> Path:
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _tensor_to_numpy(self, data: Any) -> Any:
        """Convert tensors to numpy arrays recursively."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, dict):
            return {k: self._tensor_to_numpy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._tensor_to_numpy(item) for item in data]
        else:
            return data

    def save_experiment(self,
                       problem_type: str,  # "polynomial", "van_der_pol", "kepler"
                       method: str,        # "progressive", "train_then_prune"
                       experiment_data: Dict[str, Any],
                       dataset_name: Optional[str] = None,
                       timestamp: Optional[str] = None) -> Path:
        """
        Save complete experiment data.

        Args:
            problem_type: Type of problem (polynomial, van_der_pol, kepler)
            method: Training method (progressive, train_then_prune)
            experiment_data: Dictionary containing all experiment data
            timestamp: Optional timestamp string for unique naming

        Returns:
            Path to saved experiment directory
        """

        # Create experiment directory with new naming convention
        if dataset_name is None:
            dataset_name = experiment_data.get("dataset_params", {}).get("name", "default")

        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use format: problemtype_method_dataset_timestamp
        exp_name = f"{problem_type}_{method}_{dataset_name}_{timestamp}"
        exp_dir = self._ensure_dir(self.base_dir / exp_name)

        # Convert tensors to numpy
        data = self._tensor_to_numpy(experiment_data)

        # Save metadata
        metadata = {
            "problem_type": problem_type,
            "method": method,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_basis": data.get("num_basis"),
            "dataset_params": data.get("dataset_params", {}),
            "training_params": data.get("training_params", {})
        }

        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training data
        if "training_data" in data:
            np.savez(exp_dir / "training_data.npz", **data["training_data"])

        # Save PCA analysis
        if "pca_data" in data:
            # Handle scores separately due to varying sizes
            pca_data_to_save = dict(data["pca_data"])
            if "scores" in data:
                # Save each score individually
                for i, score in enumerate(data["scores"]):
                    score_array = score.detach().cpu().numpy() if hasattr(score, 'detach') else np.array(score)
                    pca_data_to_save[f"score_{i}"] = score_array
                pca_data_to_save["num_scores"] = len(data["scores"])
            np.savez(exp_dir / "pca_data.npz", **pca_data_to_save)

        # Save model comparison
        if "comparison_data" in data:
            np.savez(exp_dir / "comparison_data.npz", **data["comparison_data"])

        # Save visualization data
        if "visualization_data" in data:
            np.savez(exp_dir / "visualization_data.npz", **data["visualization_data"])

        # Save performance summary
        if "performance_summary" in data:
            with open(exp_dir / "performance_summary.json", "w") as f:
                json.dump(data["performance_summary"], f, indent=2)

        print(f"Experiment saved to: {exp_dir}")
        return exp_dir

    def prepare_progressive_data(self,
                               problem_type: str,
                               num_basis: int,
                               losses: List[float],
                               scores: List[torch.Tensor],
                               eigenvalues: torch.Tensor,
                               gram_eigenvalues: torch.Tensor,
                               visualization_data: Dict[str, Any],
                               dataset_params: Optional[Dict] = None,
                               training_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepare data structure for progressive training experiments.

        Args:
            problem_type: Type of problem
            num_basis: Number of basis functions
            losses: Training losses
            scores: Explained variance scores for each progressive step
            eigenvalues: Final eigenvalues from coefficient covariance
            gram_eigenvalues: Final eigenvalues from Gram matrix
            visualization_data: Data for plotting (function evals, trajectories, etc.)
            dataset_params: Dataset parameters
            training_params: Training parameters

        Returns:
            Structured experiment data dictionary
        """

        return {
            "num_basis": num_basis,
            "dataset_params": dataset_params or {},
            "training_params": training_params or {},
            "training_data": {
                "losses": np.array(losses),
                "method": "progressive"
            },
            "pca_data": {
                "final_eigenvalues": eigenvalues.detach().cpu().numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues,
                "final_gram_eigenvalues": gram_eigenvalues.detach().cpu().numpy() if isinstance(gram_eigenvalues, torch.Tensor) else gram_eigenvalues,
            },
            "scores": scores,  # Add scores at the top level for saving
            "visualization_data": visualization_data,
            "performance_summary": {
                "problem_type": problem_type,
                "method": "progressive",
                "num_basis_final": num_basis,
                "final_loss": float(losses[-1]) if losses else None
            }
        }

    def prepare_prune_data(self,
                          problem_type: str,
                          num_basis_original: int,
                          num_basis_pruned: int,
                          train_losses: List[float],
                          finetune_losses: List[float],
                          eigenvalues: np.ndarray,
                          eigenvectors: np.ndarray,
                          explained_variance_ratio: np.ndarray,
                          keep_indices: List[int],
                          comparison_results: Dict[str, Any],
                          visualization_data: Dict[str, Any],
                          dataset_params: Optional[Dict] = None,
                          training_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Prepare data structure for train-then-prune experiments.

        Args:
            problem_type: Type of problem
            num_basis_original: Original number of basis functions
            num_basis_pruned: Number of basis functions after pruning
            train_losses: Initial training losses
            finetune_losses: Fine-tuning losses after pruning
            eigenvalues: PCA eigenvalues
            eigenvectors: PCA eigenvectors
            explained_variance_ratio: Explained variance ratios
            keep_indices: Indices of kept basis functions
            comparison_results: Model comparison results
            visualization_data: Data for plotting
            dataset_params: Dataset parameters
            training_params: Training parameters

        Returns:
            Structured experiment data dictionary
        """

        return {
            "num_basis": num_basis_original,
            "dataset_params": dataset_params or {},
            "training_params": training_params or {},
            "training_data": {
                "train_losses": np.array(train_losses),
                "finetune_losses": np.array(finetune_losses),
                "method": "train_then_prune"
            },
            "pca_data": {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "explained_variance_ratio": explained_variance_ratio,
                "keep_indices": np.array(keep_indices),
                "num_components_kept": len(keep_indices)
            },
            "comparison_data": {
                "mse_original": comparison_results.get("mse_original"),
                "mse_pruned": comparison_results.get("mse_pruned"),
                "mse_pruned_refined": comparison_results.get("mse_pruned_refined"),
                "coeffs_original": comparison_results.get("coeffs_original"),
                "coeffs_pruned": comparison_results.get("coeffs_pruned"),
                "coeffs_pruned_refined": comparison_results.get("coeffs_pruned_refined"),
                "y_pred_original": comparison_results.get("y_pred_original"),
                "y_pred_pruned": comparison_results.get("y_pred_pruned"),
                "y_pred_pruned_refined": comparison_results.get("y_pred_pruned_refined")
            },
            "visualization_data": visualization_data,
            "performance_summary": {
                "problem_type": problem_type,
                "method": "train_then_prune",
                "num_basis_original": num_basis_original,
                "num_basis_pruned": num_basis_pruned,
                "compression_ratio": num_basis_pruned / num_basis_original,
                "performance_ratio": comparison_results.get("mse_pruned_refined", 0) / comparison_results.get("mse_original", 1),
                "keep_indices": keep_indices
            }
        }

    def load_experiment(self, experiment_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load experiment data from saved directory.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            Dictionary containing all experiment data
        """
        exp_path = Path(experiment_path)

        # Load metadata
        with open(exp_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        data = {"metadata": metadata}

        # Load each data file if it exists
        for data_file in ["training_data.npz", "pca_data.npz", "comparison_data.npz", "visualization_data.npz"]:
            file_path = exp_path / data_file
            if file_path.exists():
                loaded = np.load(file_path, allow_pickle=True)
                data[data_file.replace(".npz", "")] = dict(loaded)

        # Load performance summary
        perf_file = exp_path / "performance_summary.json"
        if perf_file.exists():
            with open(perf_file, "r") as f:
                data["performance_summary"] = json.load(f)

        return data

    def list_experiments(self, problem_type: Optional[str] = None, method: Optional[str] = None) -> List[Path]:
        """
        List available experiments.

        Args:
            problem_type: Filter by problem type
            method: Filter by method

        Returns:
            List of experiment directory paths
        """
        experiments = []

        search_pattern = self.base_dir
        if problem_type:
            search_pattern = search_pattern / problem_type
        if method:
            search_pattern = search_pattern / method

        if search_pattern.exists():
            for item in search_pattern.rglob("experiment_*"):
                if item.is_dir() and (item / "metadata.json").exists():
                    experiments.append(item)

        return sorted(experiments)


def create_visualization_data_polynomial(X_sorted: np.ndarray,
                                        y_sorted: np.ndarray,
                                        y_pred: np.ndarray,
                                        example_X: np.ndarray,
                                        example_y: np.ndarray,
                                        basis_outputs: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """Helper function to create visualization data for polynomial problems."""
    viz_data = {
        "X_sorted": X_sorted,
        "y_sorted": y_sorted,
        "y_pred": y_pred,
        "example_X": example_X,
        "example_y": example_y
    }

    if basis_outputs is not None:
        viz_data["basis_outputs"] = basis_outputs

    return viz_data


def create_visualization_data_dynamics(trajectories_true: List[np.ndarray],
                                     trajectories_pred: List[np.ndarray],
                                     initial_conditions: List[np.ndarray],
                                     system_params: List[float]) -> Dict[str, Any]:
    """Helper function to create visualization data for dynamical systems."""
    return {
        "trajectories_true": trajectories_true,
        "trajectories_pred": trajectories_pred,
        "initial_conditions": initial_conditions,
        "system_params": system_params
    }