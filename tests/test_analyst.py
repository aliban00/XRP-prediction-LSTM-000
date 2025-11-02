import unittest
import pandas as pd
import numpy as np
from aipe.core.analyst import Analyst
from tests.synthetic_data import generate_synthetic_data

class TestAnalyst(unittest.TestCase):
    def test_causal_discovery(self):
        """
        Tests the causal discovery method with synthetic data.
        """
        analyst = Analyst()
        data = generate_synthetic_data()
        # Only use X, Y, Z for causal discovery, not the regime column
        model = analyst.discover_causal_structure(data[['X', 'Y', 'Z']])

        # In the true model, the causal order is X -> Y -> Z.
        # LiNGAM should identify this order.
        self.assertEqual(model.causal_order_, [0, 1, 2])

    def test_information_theory(self):
        """
        Tests the information theory analysis.
        """
        analyst = Analyst()
        data = generate_synthetic_data()
        mi_matrix = analyst.analyze_information_theory(data)

        # X and Y have a direct relationship, so their MI should be high.
        self.assertTrue(mi_matrix.loc['X', 'Y'] > 0.5)
        # Y and Z also have a direct relationship.
        self.assertTrue(mi_matrix.loc['Y', 'Z'] > 0.5)
        # X and Z are related through Y, so their MI should be positive,
        # but likely less than the direct relationships.
        self.assertTrue(mi_matrix.loc['X', 'Z'] > 0)

    def test_regime_detection(self):
        """
        Tests the regime detection method.
        """
        analyst = Analyst()
        n_samples = 3000
        n_regimes = 3
        data = generate_synthetic_data(n_samples=n_samples, n_regimes=n_regimes)

        # Bayesian Change Point detection requires sorted data
        data = data.sort_values(by='X').reset_index(drop=True)

        regimes = analyst.detect_regimes(data[['X']])

        # Check if the number of detected regimes is correct
        self.assertEqual(len(set(regimes)), n_regimes)

        # The detected changepoints should be close to the true changepoints.
        true_changepoints = [n_samples // n_regimes * i for i in range(1, n_regimes)]

        # Extract changepoints from the detected regimes
        detected_changepoints = sorted([i for i, (r1, r2) in enumerate(zip(regimes[:-1], regimes[1:])) if r1 != r2])

        self.assertEqual(len(detected_changepoints), len(true_changepoints))
        for tcp, dcp in zip(true_changepoints, detected_changepoints):
            self.assertTrue(abs(tcp - dcp) < 50)

    def test_complexity_analysis(self):
        """
        Tests the complexity analysis method.
        """
        analyst = Analyst()
        data = generate_synthetic_data()
        results = analyst.analyze_complexity(data[['X', 'Y', 'Z']])

        for col in ['X', 'Y', 'Z']:
            self.assertTrue(0 < results[col]['hurst'] < 1)
            self.assertTrue(isinstance(results[col]['lyapunov'], float))


if __name__ == '__main__':
    unittest.main()
