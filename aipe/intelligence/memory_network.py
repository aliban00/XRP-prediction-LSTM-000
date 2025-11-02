import torch
import torch.nn as nn
from rich.console import Console

console = Console()

class MemoryAugmentedNetwork(nn.Module):
    def __init__(self, input_size, memory_capacity=1000, retrieval_method='attention'):
        super(MemoryAugmentedNetwork, self).__init__()
        self.memory_capacity = memory_capacity
        self.retrieval_method = retrieval_method
        self.memory = nn.Parameter(torch.randn(memory_capacity, input_size), requires_grad=False)

        if self.retrieval_method == 'attention':
            self.attention = nn.MultiheadAttention(input_size, 1, batch_first=True)

    def forward(self, x):
        """
        Retrieves relevant memories and combines them with the input.
        """
        if self.retrieval_method == 'attention':
            # Use attention to retrieve relevant memories
            # This is a simplified approach for demonstration purposes.
            if x.dim() == 1:
                x_reshaped = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                x_reshaped = x.unsqueeze(1)
            else:
                x_reshaped = x

            memory_expanded = self.memory.unsqueeze(0).expand(x_reshaped.size(0), -1, -1)

            attn_output, _ = self.attention(x_reshaped, memory_expanded, memory_expanded)

            # Combine the input with the retrieved memory
            output = x + attn_output.squeeze(1)
            return output
        else:
            console.log(f"[bold red]Unsupported retrieval method: {self.retrieval_method}[/bold red]")
            return x
