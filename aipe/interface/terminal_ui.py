from rich.console import Console
from rich.table import Table

console = Console()

class TerminalUI:
    def display_insights(self, analyst_results, predictor_results, philosopher_results):
        """
        Displays the insights from the intelligence loop in a formatted way.
        """
        console.rule("[bold magenta]Adaptive Intelligence Prediction Engine (AIPE)[/bold magenta]")

        # Display Analyst insights
        console.print("\n[bold green]The Analyst: Deep Understanding Phase[/bold green]")
        causal_model = analyst_results.get('causal_model')
        if causal_model:
            console.print(f"✓ Causal structure discovered: {causal_model.causal_order_}")

        regimes = analyst_results.get('regimes')
        if regimes is not None:
            console.print(f"✓ Identified {len(set(regimes))} distinct regimes in historical data")

        complexity = analyst_results.get('complexity')
        if complexity:
            table = Table(title="Complexity Analysis")
            table.add_column("Variable", style="cyan")
            table.add_column("Hurst Exponent", style="magenta")
            table.add_column("Lyapunov Exponent", style="yellow")

            for var, values in complexity.items():
                table.add_row(var, f"{values['hurst']:.4f}", f"{values['lyapunov']:.4f}")
            console.print(table)

        # Display Predictor insights
        console.print("\n[bold blue]The Predictor: Architecture Evolution[/bold blue]")
        prediction = predictor_results.get('prediction')
        if prediction:
            console.print(f"Prediction: {prediction:.2f}")

        # Display Philosopher insights (placeholders)
        console.print("\n[bold yellow]The Philosopher: Strategic Learning[/bold yellow]")
        outcomes = philosopher_results.get('outcomes')
        if outcomes:
            console.print("Counterfactual Outcomes:")
            for outcome in outcomes:
                console.print(f"  - {outcome}")
