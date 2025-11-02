from aipe.core.analyst import Analyst
from aipe.core.predictor import Predictor
from aipe.core.philosopher import Philosopher
from aipe.interface.terminal_ui import TerminalUI

class Orchestrator:
    def __init__(self):
        self.analyst = Analyst()
        self.predictor = Predictor()
        self.philosopher = Philosopher()
        self.ui = TerminalUI()

    def run(self, data):
        """
        Runs the main intelligence loop.
        """
        # 1. The Analyst understands the data
        causal_model = self.analyst.discover_causal_structure(data)
        mi_matrix = self.analyst.analyze_information_theory(data)
        regimes = self.analyst.detect_regimes(data)
        complexity = self.analyst.analyze_complexity(data)

        analyst_results = {
            'causal_model': causal_model,
            'mi_matrix': mi_matrix,
            'regimes': regimes,
            'complexity': complexity
        }

        # 2. The Predictor makes a prediction
        architecture = self.predictor.search_architecture(data)
        predictor_results = {'architecture': architecture}

        # 3. The Philosopher learns from experience
        prediction = 1.0
        actual = 1.1
        context = {}
        outcomes = self.philosopher.reason_counterfactually(prediction, actual, context)
        philosopher_results = {'outcomes': outcomes}

        # 4. Display insights
        self.ui.display_insights(analyst_results, predictor_results, philosopher_results)
