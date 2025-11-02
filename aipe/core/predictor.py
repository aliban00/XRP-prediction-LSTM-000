import yaml
import torch
from rich.console import Console

console = Console()

class Predictor:
    def __init__(self, config_path="aipe/config.yaml"):
        """
        Initializes The Predictor with a configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            console.log("Predictor configuration loaded successfully.")
        except FileNotFoundError:
            console.log(f"[bold red]Error: Configuration file not found at {config_path}[/bold red]")
            self.config = {}

    def search_architecture(self, train_sequences, train_labels):
        """
        Performs neural architecture search.
        """
        import os
        from aipe.intelligence.architecture_search import find_best_architecture, NASModel

        model = find_best_architecture(train_sequences, train_labels)
        return model

    def augment_with_memory(self, input_size):
        """
        Augments the model with a memory network.
        """
        from aipe.intelligence.memory_network import MemoryAugmentedNetwork

        memory_enabled = self.config.get('predictor', {}).get('memory', {}).get('enabled', False)
        if not memory_enabled:
            console.log("Memory augmentation is disabled in the configuration.")
            return None

        capacity = self.config.get('predictor', {}).get('memory', {}).get('capacity', 1000)
        retrieval_method = self.config.get('predictor', {}).get('memory', {}).get('retrieval_method', 'attention')

        console.log(f"Augmenting with memory (capacity: {capacity}, retrieval: {retrieval_method})...")
        model = MemoryAugmentedNetwork(input_size, memory_capacity=capacity, retrieval_method=retrieval_method)

        console.log("Memory augmentation complete.")
        return model

    def quantify_uncertainty(self, model, data):
        """
        Quantifies the uncertainty of the model's predictions.
        """
        from aipe.intelligence.uncertainty import predict_with_uncertainty

        uncertainty_method = self.config.get('predictor', {}).get('uncertainty', {}).get('method', 'mc_dropout')

        if uncertainty_method == 'mc_dropout':
            samples = self.config.get('predictor', {}).get('uncertainty', {}).get('samples', 50)
            mean, std = predict_with_uncertainty(model, data, n_samples=samples)
            return mean, std
        else:
            console.log(f"[bold red]Unsupported uncertainty quantification method: {uncertainty_method}[/bold red]")
            return None, None

    def meta_learn(self, model, task_data):
        """
        Performs meta-learning to adapt to new data.
        """
        from aipe.intelligence.meta_learner import MAML

        meta_learning_algorithm = self.config.get('predictor', {}).get('meta_learning', {}).get('algorithm', 'maml')

        if meta_learning_algorithm == 'maml':
            inner_lr = self.config.get('predictor', {}).get('meta_learning', {}).get('inner_lr', 0.01)
            adaptation_steps = self.config.get('predictor', {}).get('meta_learning', {}).get('adaptation_steps', 5)

            maml = MAML(model, inner_lr=inner_lr, adaptation_steps=adaptation_steps)
            adapted_model = maml.adapt(task_data)
            return adapted_model
        else:
            console.log(f"[bold red]Unsupported meta-learning algorithm: {meta_learning_algorithm}[/bold red]")
            return None
