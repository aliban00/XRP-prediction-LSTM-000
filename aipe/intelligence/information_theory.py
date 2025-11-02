import numpy as np
from sklearn.feature_selection import mutual_info_regression
from rich.console import Console

console = Console()

def mutual_information(X, Y):
    """
    Calculates the mutual information between two continuous variables
    using the KSG estimator.

    Args:
        X (np.array): The first variable.
        Y (np.array): The second variable.

    Returns:
        The mutual information between X and Y.
    """
    mi = mutual_info_regression(X.reshape(-1, 1), Y)
    return mi[0]

def transfer_entropy(X, Y):
    """
    Calculates the transfer entropy from X to Y.
    """
    console.log("Calculating transfer entropy...")
    # To be implemented
    pass

def entropy_rate(X):
    """
    Calculates the entropy rate of a time series.
    """
    console.log("Calculating entropy rate...")
    # To be implemented
    pass
