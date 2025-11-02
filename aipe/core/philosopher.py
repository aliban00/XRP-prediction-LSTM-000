import yaml
from rich.console import Console

console = Console()

class Philosopher:
    def __init__(self, config_path="aipe/config.yaml"):
        """
        Initializes The Philosopher with a configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            console.log("Philosopher configuration loaded successfully.")
        except FileNotFoundError:
            console.log(f"[bold red]Error: Configuration file not found at {config_path}[/bold red]")
            self.config = {}

    def reason_counterfactually(self, model, prediction, actual, context):
        """
        Performs counterfactual reasoning.
        """
        from aipe.intelligence.counterfactual_reasoning import counterfactual_analysis

        counterfactual_depth = self.config.get('philosopher', {}).get('counterfactual_depth', 3)

        console.log(f"Reasoning counterfactually (depth: {counterfactual_depth})...")
        outcomes = counterfactual_analysis(model, prediction, actual, context)

        return outcomes

    def recognize_error_patterns(self, errors):
        """
        Recognizes patterns in the model's errors.
        """
        from aipe.intelligence.error_pattern_recognition import recognize_error_patterns_dbscan

        error_clustering_method = self.config.get('philosopher', {}).get('error_clustering', 'dbscan')

        if error_clustering_method == 'dbscan':
            labels = recognize_error_patterns_dbscan(errors)
            return labels
        else:
            console.log(f"[bold red]Unsupported error clustering method: {error_clustering_method}[/bold red]")
            return None

    def evolve_strategies(self, strategies):
        """
        Evolves the prediction strategies.
        """
        from aipe.adaptation.strategy_evolver import evolve_strategies as evolve

        population_size = self.config.get('philosopher', {}).get('strategy_evolution', {}).get('population_size', 10)
        mutation_rate = self.config.get('philosopher', {}).get('strategy_evolution', {}).get('mutation_rate', 0.1)
        selection_method = self.config.get('philosopher', {}).get('strategy_evolution', {}).get('selection', 'tournament')

        new_generation = evolve(strategies, population_size=population_size, mutation_rate=mutation_rate, selection_method=selection_method)
        return new_generation

    def distill_knowledge(self, model):
        """
        Distills knowledge from the model.
        """
        from aipe.intelligence.knowledge_distillation import distill_knowledge as distill

        compression_ratio = self.config.get('philosopher', {}).get('knowledge_distillation', {}).get('compression_ratio', 0.1)
        rule_extraction = self.config.get('philosopher', {}).get('knowledge_distillation', {}).get('rule_extraction', True)

        distilled_knowledge = distill(model, compression_ratio=compression_ratio, rule_extraction=rule_extraction)
        return distilled_knowledge

    def backtest(self, model, test_sequences, test_labels, scaler, close_idx):
        """
        Backtests the model on historical data.
        """
        from aipe.intelligence.backtesting import backtest as bt

        avg_mape = bt(model, test_sequences, test_labels, scaler, close_idx)
        return avg_mape
