import torch
import numpy as np
from rich.console import Console

console = Console()

def counterfactual_analysis(model, prediction, actual, context):
    """
    Performs counterfactual analysis by perturbing input features.

    Args:
        model: The trained model.
        prediction: The model's original prediction.
        actual: The actual outcome.
        context: The context in which the prediction was made, including the test_inputs.

    Returns:
        A list of "what if" scenarios and their potential outcomes.
    """
    console.log("Performing counterfactual analysis...")

    test_inputs = context.get('test_inputs')
    if test_inputs is None:
        console.log("[bold red]Error: 'test_inputs' not found in context for counterfactual analysis.[/bold red]")
        return []

    outcomes = []

    # Perturb each feature by a small amount and see how the prediction changes.
    for i in range(test_inputs.shape[1]):
        for perturbation in [-0.1, 0.1]:
            perturbed_inputs = test_inputs.clone()
            perturbed_inputs[-1, i] = perturbed_inputs[-1, i] * (1 + perturbation)

            with torch.no_grad():
                new_prediction = model(perturbed_inputs.unsqueeze(0)).item()

            if abs(new_prediction - prediction) > 0.01: # Only report significant changes
                feature_names = context.get('feature_names')
                if feature_names is not None and i < len(feature_names):
                    feature_name = feature_names[i]
                else:
                    feature_name = f"Feature {i}"
                outcome = f"If {feature_name} had been {'10% higher' if perturbation > 0 else '10% lower'}, the prediction would have been {new_prediction:.2f}."
                outcomes.append(outcome)

    console.log("Counterfactual analysis complete.")
    return outcomes
