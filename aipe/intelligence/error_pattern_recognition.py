import numpy as np
from sklearn.cluster import DBSCAN
from rich.console import Console

console = Console()

def recognize_error_patterns_dbscan(errors):
    """
    Recognizes patterns in prediction errors using DBSCAN.

    Args:
        errors (np.array): An array of prediction errors.

    Returns:
        The cluster labels for each error.
    """
    console.log("Recognizing error patterns with DBSCAN...")

    # Reshape for DBSCAN
    errors_reshaped = errors.reshape(-1, 1)

    # DBSCAN clustering
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(errors_reshaped)

    console.log("Error pattern recognition complete.")
    return clustering.labels_
