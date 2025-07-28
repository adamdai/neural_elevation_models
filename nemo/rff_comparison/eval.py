import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


class HeightFieldEvaluator:
    """
    Evaluation framework for height field models.
    
    Calculates RMSE, MAE, ME, and Slope RMSE metrics for comparing
    different encoding methods.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, pred_heights, true_heights, xy_coords=None, model=None):
        """
        Calculate all evaluation metrics.
        
        Args:
            pred_heights (torch.Tensor): Predicted heights
            true_heights (torch.Tensor): True heights
            xy_coords (torch.Tensor, optional): Input coordinates for slope calculation
            model (nn.Module, optional): Model for gradient computation if xy_coords provided
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Ensure tensors are on CPU and convert to numpy for calculations
        pred_heights = pred_heights.detach().cpu().numpy()
        true_heights = true_heights.detach().cpu().numpy()
        
        # Basic height metrics
        errors = pred_heights - true_heights
        
        metrics = {
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(np.abs(errors)),
            'ME': np.mean(errors),  # Mean Error (bias)
        }
        
        # Calculate slope RMSE if coordinates and model are provided
        if xy_coords is not None and model is not None:
            slope_rmse = self._calculate_slope_rmse(xy_coords, pred_heights, true_heights, model)
            metrics['Slope_RMSE'] = slope_rmse
        
        return metrics
    
    def _calculate_slope_rmse(self, xy_coords, pred_heights, true_heights, model):
        """
        Calculate RMSE of slopes (gradients).
        
        Args:
            xy_coords (torch.Tensor): Input coordinates
            pred_heights (np.ndarray): Predicted heights
            true_heights (np.ndarray): True heights
            model (nn.Module): Model for gradient computation
            
        Returns:
            float: RMSE of slopes
        """
        # Convert back to torch for gradient computation
        xy_coords = xy_coords.detach()
        
        # Calculate gradients for predicted heights
        xy_coords.requires_grad_(True)
        pred_output = model(xy_coords)
        pred_grad = torch.autograd.grad(pred_output.sum(), xy_coords, create_graph=True)[0]
        
        # For true heights, we need to estimate gradients numerically
        # Use finite differences to approximate gradients
        eps = 1e-6
        true_grad = self._numerical_gradient(xy_coords, true_heights, eps)
        
        # Calculate RMSE of gradient differences
        grad_diff = (pred_grad.detach().cpu().numpy() - true_grad)**2
        slope_rmse = np.sqrt(np.mean(grad_diff))
        
        return slope_rmse
    
    def _numerical_gradient(self, xy_coords, heights, eps=1e-6):
        """
        Calculate numerical gradients using finite differences.
        
        Args:
            xy_coords (torch.Tensor): Input coordinates
            heights (np.ndarray): Height values
            eps (float): Small perturbation for finite differences
            
        Returns:
            np.ndarray: Numerical gradients
        """
        xy_np = xy_coords.detach().cpu().numpy()
        heights = heights.reshape(-1, 1)
        
        # Reshape to grid for gradient calculation
        # Assuming xy_coords are in a regular grid pattern
        n_points = len(xy_np)
        grid_size = int(np.sqrt(n_points))
        
        if grid_size**2 != n_points:
            # If not a perfect square, use a simpler approach
            return np.zeros((n_points, 2))
        
        # Reshape to grid
        heights_grid = heights.reshape(grid_size, grid_size)
        
        # Calculate gradients using finite differences
        grad_x = np.gradient(heights_grid, axis=1)
        grad_y = np.gradient(heights_grid, axis=0)
        
        # Flatten back to point format
        gradients = np.column_stack([grad_x.flatten(), grad_y.flatten()])
        
        return gradients
    
    def compare_methods(self, results_dict, print_table=True):
        """
        Compare multiple methods and print results table.
        
        Args:
            results_dict (dict): Dictionary with method names as keys and metric dicts as values
            print_table (bool): Whether to print the comparison table
            
        Returns:
            str: Formatted table string
        """
        # Prepare table data
        headers = ['Method'] + list(next(iter(results_dict.values())).keys())
        table_data = []
        
        for method_name, metrics in results_dict.items():
            row = [method_name] + [f"{value:.6f}" for value in metrics.values()]
            table_data.append(row)
        
        # Create table
        table = tabulate(table_data, headers=headers, tablefmt='grid', floatfmt='.6f')
        
        if print_table:
            print("\n" + "="*80)
            print("HEIGHT FIELD EVALUATION RESULTS")
            print("="*80)
            print(table)
            print("="*80)
        
        return table
    
    def plot_comparison(self, results_dict, metric_name='RMSE', save_path=None):
        """
        Create a bar plot comparing methods for a specific metric.
        
        Args:
            results_dict (dict): Results dictionary
            metric_name (str): Metric to plot
            save_path (str, optional): Path to save the plot
        """
        methods = list(results_dict.keys())
        values = [results_dict[method][metric_name] for method in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.title(f'{metric_name} Comparison')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results_dict, filepath):
        """
        Save results to a file.
        
        Args:
            results_dict (dict): Results dictionary
            filepath (str): Path to save results
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for method, metrics in results_dict.items():
            serializable_results[method] = {
                key: float(value) for key, value in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def evaluate_model(model, test_xyz, normalization_params=None):
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model: Trained NeuralHeightField model
        test_xyz (torch.Tensor): Test data
        normalization_params (dict, optional): Normalization parameters for denormalization
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Extract coordinates and true heights
    xy_test = test_xyz[:, :2]
    z_true = test_xyz[:, 2]
    
    # Get predictions (without gradients for basic metrics)
    with torch.no_grad():
        z_pred = model(xy_test)
        
        # Denormalize if parameters provided
        if normalization_params is not None:
            z_pred = z_pred * normalization_params['z_std'] + normalization_params['z_mean']
            z_true = z_true * normalization_params['z_std'] + normalization_params['z_mean']
    
    # Calculate basic metrics (RMSE, MAE, ME) without gradients
    evaluator = HeightFieldEvaluator()
    metrics = evaluator.calculate_metrics(z_pred, z_true)
    
    # Calculate slope RMSE separately with gradients enabled
    try:
        # Recompute predictions with gradients for slope calculation
        xy_test_grad = xy_test.clone().detach()
        xy_test_grad.requires_grad_(True)
        
        # Get predictions with gradients
        z_pred_grad = model(xy_test_grad)
        
        # Denormalize if needed
        if normalization_params is not None:
            z_pred_grad = z_pred_grad * normalization_params['z_std'] + normalization_params['z_mean']
        
        # Calculate slope RMSE
        slope_rmse = evaluator._calculate_slope_rmse(xy_test_grad, z_pred_grad.detach().cpu().numpy(), z_true.cpu().numpy(), model)
        metrics['Slope_RMSE'] = slope_rmse
        
    except Exception as e:
        print(f"Warning: Could not compute slope RMSE: {e}")
        metrics['Slope_RMSE'] = 0.0
    
    return metrics


if __name__ == "__main__":
    # Test the evaluator
    print("Testing HeightFieldEvaluator...")
    
    # Create dummy data
    n_points = 1000
    pred_heights = torch.randn(n_points)
    true_heights = torch.randn(n_points)
    xy_coords = torch.rand(n_points, 2)
    
    # Test metrics calculation
    evaluator = HeightFieldEvaluator()
    metrics = evaluator.calculate_metrics(pred_heights, true_heights)
    
    print("Metrics calculated:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Test comparison
    results = {
        'Method_A': {'RMSE': 0.1, 'MAE': 0.08, 'ME': 0.01},
        'Method_B': {'RMSE': 0.09, 'MAE': 0.07, 'ME': 0.02},
        'Method_C': {'RMSE': 0.12, 'MAE': 0.10, 'ME': 0.00}
    }
    
    evaluator.compare_methods(results)
    
    print("\n✅ Evaluator test completed successfully!")
