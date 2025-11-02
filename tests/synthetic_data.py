import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=5000, n_regimes=3):
    """
    Generates synthetic data with a known causal structure and distinct regimes.
    """
    data = []
    samples_per_regime = n_samples // n_regimes

    for i in range(n_regimes):
        mean = i * 20
        noise_scale = 2
        # Use non-Gaussian noise (Laplace distribution) for LiNGAM
        X = np.random.laplace(loc=mean, scale=noise_scale, size=samples_per_regime)
        Y = 2 * X + np.random.laplace(loc=mean, scale=noise_scale, size=samples_per_regime)
        Z = 3 * Y + np.random.laplace(loc=mean, scale=noise_scale, size=samples_per_regime)
        regime_data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'regime': i}, columns=['X', 'Y', 'Z', 'regime'])
        data.append(regime_data)

    return pd.concat(data, ignore_index=True)
