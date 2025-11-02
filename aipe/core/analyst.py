import pandas as pd
import yaml
import numpy as np
import lingam
import ta
from rich.console import Console

console = Console()

class Analyst:
    def __init__(self, config_path="aipe/config.yaml"):
        """
        Initializes The Analyst with a configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            console.log("Analyst configuration loaded successfully.")
        except FileNotFoundError:
            console.log(f"[bold red]Error: Configuration file not found at {config_path}[/bold red]")
            self.config = {}

    def discover_causal_structure(self, data: pd.DataFrame):
        """
        Discovers the causal structure of the data using the LiNGAM algorithm.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            A causal graph object from LiNGAM.
        """
        if not self.config.get('intelligence', {}).get('causal_discovery'):
            console.log("Causal discovery is disabled in the configuration.")
            return None

        causal_method = self.config.get('analyst', {}).get('causal_method', 'lingam')

        if causal_method == 'lingam':
            console.log("Starting causal discovery with VARMA-LiNGAM algorithm...")
            model = lingam.VARMALiNGAM(max_iter=50) # Reduce max_iter for performance
            try:
                model.fit(data)
            except Exception as e:
                console.log(f"[bold red]Error in VARMA-LiNGAM: {e}[/bold red]")
                return None
            console.log("Causal discovery complete.")
            return model
        else:
            console.log(f"[bold red]Unsupported causal discovery method: {causal_method}[/bold red]")
            return None

    def analyze_information_theory(self, data: pd.DataFrame):
        """
        Performs information-theoretic analysis on the data.
        """
        console.log("Performing information-theoretic analysis...")
        from aipe.intelligence.information_theory import mutual_information

        # Calculate mutual information between all pairs of variables
        mi_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
        for col1 in data.columns:
            for col2 in data.columns:
                if col1 == col2:
                    mi_matrix.loc[col1, col2] = np.nan
                else:
                    mi = mutual_information(data[col1].values, data[col2].values)
                    mi_matrix.loc[col1, col2] = mi

        console.log("Mutual Information Matrix:")
        console.print(mi_matrix)

        return mi_matrix

    def detect_regimes(self, data: pd.DataFrame):
        """
        Detects regimes in the data.
        """
        from aipe.adaptation.regime_detector import detect_regimes_bcp, detect_regimes_hmm

        regime_detection_method = self.config.get('analyst', {}).get('regime_detection', 'bcp')

        if regime_detection_method == 'bcp':
            regimes = detect_regimes_bcp(data.values)
            return regimes
        elif regime_detection_method == 'hmm':
            # Reshape data for HMM
            data_for_hmm = data.values.reshape(-1, 1) if isinstance(data, pd.Series) else data.values
            regimes = detect_regimes_hmm(data_for_hmm)
            return regimes
        else:
            console.log(f"[bold red]Unsupported regime detection method: {regime_detection_method}[/bold red]")
            return None

    def analyze_complexity(self, data: pd.DataFrame):
        """
        Analyzes the complexity of the data.
        """
        console.log("Analyzing complexity...")
        from aipe.intelligence.complexity import hurst_exponent, lyapunov_exponent

        results = {}
        for col in data.columns:
            results[col] = {
                'hurst': hurst_exponent(data[col].values),
                'lyapunov': lyapunov_exponent(data[col].values)
            }

        return results

    def engineer_features(self, data: pd.DataFrame):
        """
        Engineers features for the model.
        """
        console.log("Engineering features...")
        # Add all ta features
        data = ta.add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="volume"
        )
        console.log("Feature engineering complete.")
        return data
