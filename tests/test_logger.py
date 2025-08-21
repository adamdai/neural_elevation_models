#!/usr/bin/env python3
"""
Test script for the NEMoLogger class.

This script tests the logging functionality without requiring actual wandb authentication.
"""

import sys
import os
import tempfile
import torch
import numpy as np

# Add the parent directory to the path to import nemo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo.logger import NEMoLogger, TrainingMetrics, EvaluationMetrics


def test_logger_initialization():
    """Test logger initialization with disabled wandb."""
    import wandb

    # Disable wandb for testing
    wandb.init(mode="disabled")

    logger = NEMoLogger(
        project_name="test-project",
        run_name="test-run",
        config={"test_param": 42},
        log_gradients=False,
        log_memory=False,
        log_timing=False,
    )

    assert logger.project_name == "test-project"
    assert logger.run_name == "test-run"
    assert logger.config["test_param"] == 42
    assert not logger.log_gradients
    assert not logger.log_memory
    assert not logger.log_timing

    logger.finish()
    print("✓ Logger initialization test passed")


def test_training_metrics():
    """Test TrainingMetrics dataclass."""
    metrics = TrainingMetrics(
        epoch=10,
        total_loss=0.5,
        height_loss=0.4,
        spatial_smoothness_loss=0.05,
        height_variance_loss=0.05,
        learning_rate=1e-3,
        gradient_norm=0.1,
        memory_usage=512.0,
        training_time=1.5,
    )

    assert metrics.epoch == 10
    assert metrics.total_loss == 0.5
    assert metrics.height_loss == 0.4
    assert metrics.spatial_smoothness_loss == 0.05
    assert metrics.height_variance_loss == 0.05
    assert metrics.learning_rate == 1e-3
    assert metrics.gradient_norm == 0.1
    assert metrics.memory_usage == 512.0
    assert metrics.training_time == 1.5

    print("✓ TrainingMetrics dataclass test passed")


def test_evaluation_metrics():
    """Test EvaluationMetrics dataclass."""
    metrics = EvaluationMetrics(
        mse=0.1,
        mae=0.2,
        rmse=0.3,
        rel_mse=0.05,
        rel_mae=0.1,
        dataset_size=1000,
    )

    assert metrics.mse == 0.1
    assert metrics.mae == 0.2
    assert metrics.rmse == 0.3
    assert metrics.rel_mse == 0.05
    assert metrics.rel_mae == 0.1
    assert metrics.dataset_size == 1000

    print("✓ EvaluationMetrics dataclass test passed")


def test_logger_methods():
    """Test logger methods with disabled wandb."""
    import wandb

    wandb.init(mode="disabled")

    logger = NEMoLogger(
        project_name="test-project",
        config={"test": True},
        log_gradients=True,
        log_memory=True,
        log_timing=True,
    )

    # Test training start
    logger.start_training(max_epochs=100, data_size=1000, batch_size=100)
    assert logger.training_start_time is not None

    # Test epoch start
    logger.start_epoch(epoch=5)
    assert logger.epoch_start_time is not None

    # Test training metrics logging
    metrics = TrainingMetrics(
        epoch=5,
        total_loss=0.5,
        height_loss=0.4,
        spatial_smoothness_loss=0.05,
        height_variance_loss=0.05,
        learning_rate=1e-3,
    )
    logger.log_training_metrics(metrics)

    # Test evaluation metrics logging
    eval_metrics = EvaluationMetrics(
        mse=0.1,
        mae=0.2,
        rmse=0.3,
        rel_mse=0.05,
        rel_mae=0.1,
        dataset_size=1000,
    )
    logger.log_evaluation_metrics(eval_metrics, split="test")

    # Test custom metric logging
    logger.log_custom_metric("test/custom", 42.0)

    # Test model save/load logging
    logger.log_model_save("/tmp/test_model.pth", model_size_mb=10.5)
    logger.log_model_load("/tmp/test_model.pth")

    # Test early stopping logging
    logger.log_early_stopping(epoch=50, patience=10)

    # Test training completion logging
    logger.log_training_completion(final_loss=0.1, total_epochs=50)

    logger.finish()
    print("✓ Logger methods test passed")


def test_context_manager():
    """Test logger context manager functionality."""
    import wandb

    wandb.init(mode="disabled")

    with NEMoLogger(project_name="test-project") as logger:
        assert logger.run is not None
        logger.log_custom_metric("test", 1.0)

    # Logger should be finished after exiting context
    print("✓ Context manager test passed")


def test_memory_logging():
    """Test memory logging functionality."""
    import wandb

    wandb.init(mode="disabled")

    logger = NEMoLogger(log_memory=True)

    # Test memory logging (should not crash even without CUDA)
    logger.log_memory_usage(step=1)

    logger.finish()
    print("✓ Memory logging test passed")


def test_gradient_logging():
    """Test gradient logging functionality."""
    import wandb

    wandb.init(mode="disabled")

    logger = NEMoLogger(log_gradients=True)

    # Create a simple model for testing
    model = torch.nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    # Forward pass
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)

    # Backward pass
    loss.backward()

    # Test gradient logging
    logger.log_gradient_norms(model, step=1)

    logger.finish()
    print("✓ Gradient logging test passed")


def main():
    """Run all tests."""
    print("Running NEMoLogger tests...")
    print("=" * 50)

    try:
        test_logger_initialization()
        test_training_metrics()
        test_evaluation_metrics()
        test_logger_methods()
        test_context_manager()
        test_memory_logging()
        test_gradient_logging()

        print("=" * 50)
        print("✓ All tests passed successfully!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
