import torch
import torch.nn as nn
import time
import json
from pathlib import Path

from nemo.field import NeuralHeightField
from nemo.rff_comparison.data_utils import load_moon_data_for_comparison
from nemo.rff_comparison.eval import HeightFieldEvaluator, evaluate_model


def train_model(model, train_xyz, max_iters=5000, lr=0.001, print_freq=500):
    """
    Train a neural height field model.
    
    Args:
        model: NeuralHeightField model to train
        train_xyz (torch.Tensor): Training data
        max_iters (int): Maximum training iterations
        lr (float): Learning rate
        print_freq (int): How often to print progress
        
    Returns:
        dict: Training history and final loss
    """
    print(f"Training {model.__class__.__name__}...")
    
    # Move model to device
    device = next(model.parameters()).device
    train_xyz = train_xyz.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training history
    history = {'losses': [], 'iterations': []}
    
    start_time = time.time()
    
    for i in range(max_iters):
        # Forward pass
        xy_train = train_xyz[:, :2]
        z_train = train_xyz[:, 2]
        
        z_pred = model(xy_train)
        loss = criterion(z_pred, z_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record history
        if i % print_freq == 0:
            history['losses'].append(loss.item())
            history['iterations'].append(i)
            print(f"Iteration {i:5d} | Loss: {loss.item():.6f}")
    
    training_time = time.time() - start_time
    final_loss = loss.item()
    
    print(f"Training completed in {training_time:.2f}s | Final loss: {final_loss:.6f}")
    
    return {
        'final_loss': final_loss,
        'training_time': training_time,
        'history': history
    }


def train_and_evaluate_method(method_name, method_config, train_xyz, test_xyz, 
                            normalization_params, max_iters=5000):
    """
    Train and evaluate a single method.
    
    Args:
        method_name (str): Name of the method
        method_config (dict): Configuration for the method
        train_xyz (torch.Tensor): Training data
        test_xyz (torch.Tensor): Test data
        normalization_params (dict): Normalization parameters
        max_iters (int): Maximum training iterations
        
    Returns:
        dict: Results including metrics and training info
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {method_name}")
    print(f"{'='*60}")
    
    # Create model
    model = NeuralHeightField(**method_config)
    
    # Train model
    training_info = train_model(model, train_xyz, max_iters=max_iters)
    
    # Evaluate model
    print(f"Evaluating {method_name}...")
    metrics = evaluate_model(model, test_xyz, normalization_params)
    
    # Combine results
    results = {
        'method_name': method_name,
        'metrics': metrics,
        'training_info': training_info,
        'model_config': method_config
    }
    
    return results


def run_fair_comparison(data_path="data/Moon_Map_01_0_rep0.dat", 
                       test_size=0.2, max_iters=5000, save_results=True):
    """
    Run a FAIR comparison between RFF and Fourier MLP methods.
    
    FAIR COMPARISON (Option 1): Match input dimensions to MLP
    - Both methods get the same input dimension to the MLP
    - This ensures equal representational capacity
    
    Args:
        data_path (str): Path to moon DEM data
        test_size (float): Fraction of data for testing
        max_iters (int): Maximum training iterations
        save_results (bool): Whether to save results to file
        
    Returns:
        dict: Complete results dictionary
    """
    print("Starting FAIR RFF vs Fourier MLP Comparison")
    print("="*60)
    print("FAIR COMPARISON: Matching input dimensions to MLP")
    print("="*60)
    
    # Load data
    print("Loading moon data...")
    train_xyz, test_xyz, normalization_params = load_moon_data_for_comparison(
        data_path=data_path, test_size=test_size
    )
    
    # FAIR comparison - same input dimensions to MLP
    methods = {
        'RFF_128freq': {
            'encoding_type': 'rff',
            'num_frequencies': 128,  # → 256 input dim (2 * 128)
            'scale': 10.0,
            'hidden_dim': 64,
            'n_layers': 3,
            'activation': 'relu'
        },
        'Fourier_MLP_128freq': {
            'encoding_type': 'fourier_mlp',
            'encoding_kwargs': {'num_frequencies': 128, 'scale': 10.0},  # → 256 input dim (2 * 128)
            'hidden_dim': 64,
            'n_layers': 3,
            'activation': 'relu'
        },
        'RFF_256freq': {
            'encoding_type': 'rff',
            'num_frequencies': 256,  # → 512 input dim (2 * 256)
            'scale': 10.0,
            'hidden_dim': 64,
            'n_layers': 3,
            'activation': 'relu'
        },
        'Fourier_MLP_256freq': {
            'encoding_type': 'fourier_mlp',
            'encoding_kwargs': {'num_frequencies': 256, 'scale': 10.0},  # → 512 input dim (2 * 256)
            'hidden_dim': 64,
            'n_layers': 3,
            'activation': 'relu'
        }
    }
    
    # Print comparison setup
    print("\nComparison Setup:")
    print("-" * 40)
    for method_name, config in methods.items():
        if config['encoding_type'] == 'rff':
            input_dim = 2 * config['num_frequencies']
        else:  # fourier_mlp
            input_dim = 2 * config['encoding_kwargs']['num_frequencies']
        print(f"{method_name}: {input_dim} input dimensions to MLP")
    
    # Train and evaluate each method
    all_results = {}
    evaluator = HeightFieldEvaluator()
    
    for method_name, method_config in methods.items():
        try:
            results = train_and_evaluate_method(
                method_name, method_config, train_xyz, test_xyz, 
                normalization_params, max_iters=max_iters
            )
            all_results[method_name] = results
            
        except Exception as e:
            print(f"Error training {method_name}: {e}")
            continue
    
    # Prepare comparison table
    metrics_dict = {}
    for method_name, results in all_results.items():
        metrics_dict[method_name] = results['metrics']
    
    # Print comparison table
    evaluator.compare_methods(metrics_dict)
    
    # Save results if requested
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"fair_comparison_results_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {}
        for method_name, results in all_results.items():
            serializable_results[method_name] = {
                'metrics': {k: float(v) for k, v in results['metrics'].items()},
                'training_info': {
                    'final_loss': float(results['training_info']['final_loss']),
                    'training_time': float(results['training_info']['training_time'])
                },
                'model_config': results['model_config']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FAIR COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    best_rmse_method = min(metrics_dict.items(), key=lambda x: x[1]['RMSE'])
    best_mae_method = min(metrics_dict.items(), key=lambda x: x[1]['MAE'])
    
    print(f"Best RMSE: {best_rmse_method[0]} ({best_rmse_method[1]['RMSE']:.6f})")
    print(f"Best MAE:  {best_mae_method[0]} ({best_mae_method[1]['MAE']:.6f})")
    
    # Key differences analysis
    print(f"\n{'='*60}")
    print("KEY DIFFERENCES ANALYSIS")
    print(f"{'='*60}")
    print("RFF Encoding:")
    print("  - Trainable frequencies (B matrix is learned)")
    print("  - Can adapt frequencies during training")
    print("  - More parameters (encoding + MLP)")
    print()
    print("Fourier MLP Encoding:")
    print("  - Fixed frequencies (B matrix is fixed)")
    print("  - Frequencies set at initialization")
    print("  - Fewer parameters (only MLP)")
    
    return all_results


def quick_test():
    """
    Run a quick test with fewer iterations to verify everything works.
    """
    print("Running quick test with reduced iterations...")
    
    results = run_fair_comparison(
        data_path="data/Moon_Map_01_0_rep0.dat",
        test_size=0.2,
        max_iters=100,  # Reduced for quick test
        save_results=False
    )
    
    print("\n✅ Quick test completed successfully!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FAIR RFF vs Fourier MLP Comparison')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run a quick test with fewer iterations')
    parser.add_argument('--data-path', type=str, 
                       default="data/Moon_Map_01_0_rep0.dat",
                       help='Path to moon DEM data')
    parser.add_argument('--max-iters', type=int, default=5000,
                       help='Maximum training iterations')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for testing')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        run_fair_comparison(
            data_path=args.data_path,
            test_size=args.test_size,
            max_iters=args.max_iters
        )
