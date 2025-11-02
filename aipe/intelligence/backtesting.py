import torch
import numpy as np
from rich.console import Console

console = Console()

def backtest(model, test_sequences, test_labels, scaler, close_idx):
    """
    Backtests the model on historical data.

    Args:
        model: The trained model.
        test_sequences: A list of test sequences.
        test_labels: A list of test labels.
        scaler: The scaler used to normalize the data.
        close_idx: The index of the 'close' column.

    Returns:
        The average Mean Absolute Percentage Error (MAPE).
    """
    console.log("Starting backtesting...")

    total_mape = 0
    with torch.no_grad():
        for seq, labels in zip(test_sequences, test_labels):
            y_pred = model(seq.unsqueeze(0))

            # Inverse transform the prediction and the label
            dummy_pred = np.zeros((1, scaler.n_features_in_))
            dummy_pred[0, close_idx] = y_pred.detach().numpy()[0][0]
            y_pred_unscaled = scaler.inverse_transform(dummy_pred)

            dummy_label = np.zeros((1, scaler.n_features_in_))
            dummy_label[0, close_idx] = labels
            label_unscaled = scaler.inverse_transform(dummy_label)

            total_mape += abs((y_pred_unscaled[0, close_idx] - label_unscaled[0, close_idx]) / label_unscaled[0, close_idx]) * 100

    avg_mape = total_mape / len(test_sequences)
    console.log(f"Backtesting complete. Average MAPE: {avg_mape:.2f}%")
    return avg_mape
