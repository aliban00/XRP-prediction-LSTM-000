import torch
from rich.console import Console

console = Console()

def predict_with_uncertainty(model, x, n_samples=50):
    """
    Performs prediction with uncertainty estimation using Monte Carlo Dropout.

    Args:
        model: The PyTorch model with dropout layers.
        x: The input tensor.
        n_samples (int): The number of samples for Monte Carlo Dropout.

    Returns:
        The mean and standard deviation of the predictions.
    """
    console.log(f"Performing prediction with uncertainty (MC Dropout, {n_samples} samples)...")

    # Enable dropout
    model.train()

    predictions = [model(x) for _ in range(n_samples)]
    predictions = torch.stack(predictions)

    mean = predictions.mean(0)
    std = predictions.std(0)

    console.log("Uncertainty quantification complete.")
    return mean, std
