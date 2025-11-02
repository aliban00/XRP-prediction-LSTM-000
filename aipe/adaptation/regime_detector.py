import numpy as np
import ruptures as rpt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from rich.console import Console

console = Console()

def detect_regimes_hmm(data, n_regimes=3):
    """
    Detects regimes in the data using a Hidden Markov Model.

    Args:
        data (np.array): The input data.
        n_regimes (int): The number of regimes to detect.

    Returns:
        The detected regimes (states) for each data point.
    """
    console.log(f"Detecting {n_regimes} regimes with HMM...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=2000, tol=1e-4, random_state=42)
    model.fit(scaled_data)
    regimes = model.predict(scaled_data)
    console.log("Regime detection complete.")
    return regimes

def detect_regimes_bcp(data, n_regimes=3):
    """
    Detects regimes in the data using Bayesian Change Point Detection.

    Args:
        data (np.array): The input data.
        n_regimes (int): The number of regimes to detect.

    Returns:
        The detected regimes (states) for each data point.
    """
    console.log(f"Detecting {n_regimes} regimes with Bayesian Change Point Detection...")
    algo = rpt.Binseg(model="rbf").fit(data)
    result = algo.predict(n_bkps=n_regimes-1)

    regimes = np.zeros(len(data), dtype=int)
    for i, bkpt in enumerate(result[:-1]):
        regimes[bkpt:] = i + 1

    console.log("Regime detection complete.")
    return regimes
