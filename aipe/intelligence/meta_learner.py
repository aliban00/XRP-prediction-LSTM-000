import torch
import torch.optim as optim
from copy import deepcopy
from rich.console import Console

console = Console()

class MAML:
    def __init__(self, model, inner_lr=0.01, adaptation_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.adaptation_steps = adaptation_steps

    def adapt(self, task_data):
        """
        Adapts the model to a new task using MAML.
        """
        console.log(f"Adapting to new task with MAML (inner_lr: {self.inner_lr}, steps: {self.adaptation_steps})...")

        # Create a fast model for inner loop updates
        fast_model = deepcopy(self.model)
        optimizer = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        loss_fn = torch.nn.MSELoss()

        # Inner loop updates
        for _ in range(self.adaptation_steps):
            x, y = task_data
            y_pred = fast_model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        console.log("MAML adaptation complete.")
        return fast_model
