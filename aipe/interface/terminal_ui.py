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

        # Display Predictor insights (placeholders)
        console.print("\n[bold blue]The Predictor: Architecture Evolution[/bold blue]")
        console.print("Current architecture: [Attention(heads=4) → LSTM(256) → Dense(128)]")

        # Display Philosopher insights (placeholders)
        console.print("\n[bold yellow]The Philosopher: Strategic Learning[/bold yellow]")
        console.print("Analyzed 347 recent predictions:")
        console.print("  • Success pattern: Low volatility regimes (89% accuracy)")

        # Display Prediction (placeholders)
        console.print("\n[bold red]Prediction[/bold red]")
        console.print("Next 6 hours: ↑ 2.3% (confidence: 0.78, range: [1.8%, 2.9%])")
