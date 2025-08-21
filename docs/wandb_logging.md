# Weights & Biases Logging for NEMoV2

This document describes how to use the new Weights & Biases (wandb) logging system with the NEMoV2 neural elevation model.

## Overview

The logging system has been designed with separation of concerns in mind:
- **NEMoV2 class**: Handles the neural network model and training logic
- **NEMoLogger class**: Handles all wandb logging operations

This architecture makes the code more maintainable, testable, and allows for easy switching between different logging backends in the future.

## Features

The logging system provides comprehensive tracking of:

### Training Metrics
- Total loss, height loss, regularization losses
- Learning rate changes
- Gradient norms (optional)
- Memory usage (GPU/CPU)
- Training timing
- Early stopping information

### Model Information
- Model architecture and parameters
- Parameter counts (total and trainable)
- Model saving/loading events

### Evaluation Metrics
- MSE, MAE, RMSE
- Relative errors
- Dataset sizes

### System Information
- GPU memory usage
- Training duration
- Epoch timing

## Quick Start

### 1. Basic Usage

```python
from nemo.nemov2 import NEMoV2
from nemo.logger import NEMoLogger

# Initialize the model
model = NEMoV2()

# Set up logging
logger = NEMoLogger(
    project_name="my-elevation-project",
    run_name="experiment-1",
    config={"learning_rate": 1e-3, "max_epochs": 1000}
)

# Train with logging
losses = model.fit(xy_train, z_train, logger=logger)

# Evaluate with logging
metrics = model.evaluate(xy_test, z_test, logger=logger)

# Save model with logging
model.save_model("model.pth", logger=logger)

# Finish logging
logger.finish()
```

### 2. Context Manager Usage

```python
with NEMoLogger(project_name="my-project") as logger:
    model.fit(xy_train, z_train, logger=logger)
    model.evaluate(xy_test, z_test, logger=logger)
    model.save_model("model.pth", logger=logger)
# Logger automatically finishes when exiting context
```

## Configuration Options

### Logger Initialization

```python
logger = NEMoLogger(
    project_name="neural-elevation-models",  # wandb project name
    run_name="experiment-1",                 # specific run name (auto-generated if None)
    config={                                 # configuration dictionary
        "learning_rate": 1e-3,
        "max_epochs": 1000,
        "batch_size": 5000,
    },
    log_gradients=True,                      # log gradient norms
    log_memory=True,                         # log memory usage
    log_timing=True,                         # log timing information
)
```

### Training Configuration

The logger automatically logs all training parameters when you call `start_training()`:

```python
# These are automatically logged:
config = {
    "model": "NEMoV2",
    "encoding": "HashGrid",
    "network": "FullyFusedMLP",
    "learning_rate": 1e-3,
    "max_epochs": 500,
    "batch_size": 5000,
    "grad_clip": 1.0,
    "grad_weight": 0.01,
    "laplacian_weight": 0.001,
    "patience": 50,
    "early_stopping": True,
    "spatial_method": "finite_diff",
    "enable_spatial": True,
}
```

## Logged Metrics

### Training Metrics (per epoch)
- `training/total_loss`: Combined loss (height + regularization)
- `training/height_loss`: MSE loss on height predictions
- `training/spatial_smoothness_loss`: Spatial regularization loss
- `training/height_variance_loss`: Height variance regularization loss
- `training/learning_rate`: Current learning rate
- `training/gradient_norm`: Total gradient norm (if enabled)
- `training/memory_usage_mb`: GPU memory usage (if available)
- `training/epoch_time`: Time per epoch (if enabled)

### Best Performance Tracking
- `training/best_loss`: Best loss achieved
- `training/best_epoch`: Epoch with best loss

### Evaluation Metrics
- `evaluation/{split}/mse`: Mean squared error
- `evaluation/{split}/mae`: Mean absolute error
- `evaluation/{split}/rmse`: Root mean squared error
- `evaluation/{split}/relative_mse`: Relative MSE
- `evaluation/{split}/relative_mae`: Relative MAE
- `evaluation/{split}/dataset_size`: Number of evaluation points

### Model Information
- `model/total_parameters`: Total parameter count
- `model/trainable_parameters`: Trainable parameter count
- `model/saved`: Model save events
- `model/loaded`: Model load events

### System Information
- `system/gpu_memory_allocated_mb`: GPU memory allocated
- `system/gpu_memory_reserved_mb`: GPU memory reserved

## Advanced Usage

### Custom Metrics

```python
# Log custom metrics
logger.log_custom_metric("custom/validation_score", 0.95)
logger.log_custom_metric("custom/feature_importance", feature_importance, step=epoch)
```

### Gradient Logging

```python
# Enable detailed gradient logging
logger = NEMoLogger(log_gradients=True)

# This will log:
# - training/total_gradient_norm
# - gradients/{parameter_name} for each parameter
```

### Memory Monitoring

```python
# Enable memory usage logging
logger = NEMoLogger(log_memory=True)

# This will log GPU memory usage every epoch
```

## Example Workflow

See `examples/training_with_wandb.py` for a complete example that demonstrates:

1. Setting up wandb logging
2. Training a model with comprehensive logging
3. Evaluating the model and logging metrics
4. Saving the model with logging
5. Error handling and proper cleanup

## Best Practices

### 1. Always Finish the Logger
```python
try:
    # Your training code
    pass
finally:
    logger.finish()  # Ensure wandb run is properly closed
```

### 2. Use Context Manager
```python
with NEMoLogger(project_name="my-project") as logger:
    # Your code here
    pass
# Logger automatically finishes
```

### 3. Organize Your Config
```python
config = {
    "model": {...},
    "training": {...},
    "data": {...},
    "system": {...},
}
```

### 4. Monitor Key Metrics
- Watch for gradient explosion (`training/gradient_norm`)
- Monitor memory usage (`system/gpu_memory_*`)
- Track learning rate changes (`training/learning_rate`)

## Troubleshooting

### Common Issues

1. **wandb not initialized**: Make sure you have a valid wandb API key
2. **Memory logging errors**: Check if CUDA is available
3. **Gradient logging errors**: Ensure `log_gradients=True` and gradients exist

### Debug Mode

```python
import wandb
wandb.init(mode="disabled")  # For testing without actual logging
```

## Migration from Print Statements

The new logging system replaces the verbose print statements in the original code:

| Old (Print) | New (Wandb) |
|-------------|-------------|
| `print(f"Loss: {loss:.6f}")` | `logger.log_training_metrics(metrics)` |
| `print(f"Best loss: {best_loss}")` | Automatically logged via `log_training_metrics` |
| `print(f"Model saved to {path}")` | `logger.log_model_save(path)` |

The verbose mode is still available for console output, but all structured data is now logged to wandb for better tracking and visualization.
