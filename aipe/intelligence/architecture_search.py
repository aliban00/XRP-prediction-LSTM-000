import torch
import torch.nn as nn
from rich.console import Console

console = Console()

class SearchableArchitecture(nn.Module):
    def __init__(self, search_space, num_layers=8):
        super(SearchableArchitecture, self).__init__()
        self.search_space = search_space
        self.num_layers = num_layers
        self.architecture_weights = nn.Parameter(torch.randn(num_layers, len(search_space)))

        # Define the operations in the search space
        self.operations = nn.ModuleDict({
            'lstm': nn.LSTM(1, 1, batch_first=True),
            'attention': nn.MultiheadAttention(1, 1, batch_first=True),
            'transformer_block': nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first=True)
        })

    def forward(self, x):
        """
        Mixes operations based on learned architecture weights.
        """
        for i in range(self.num_layers):
            # Apply softmax to the architecture weights to get a probability distribution
            weights = torch.softmax(self.architecture_weights[i], dim=0)

            # Apply each operation to the input and weight the output
            # This is a simplified approach for demonstration purposes.
            # A true DARTS implementation would be more complex.
            output = 0
            for j, op_name in enumerate(self.search_space):
                op = self.operations[op_name]

                # Reshape input for different layer types
                if op_name == 'lstm':
                    # LSTM expects (batch, seq_len, features)
                    if x.dim() == 2:
                        x_reshaped = x.unsqueeze(1)
                    else:
                        x_reshaped = x
                    op_output, _ = op(x_reshaped)
                elif op_name == 'attention':
                    # Attention expects (batch, seq_len, features)
                    if x.dim() == 2:
                        x_reshaped = x.unsqueeze(1)
                    else:
                        x_reshaped = x
                    op_output, _ = op(x_reshaped, x_reshaped, x_reshaped)
                elif op_name == 'transformer_block':
                     # Transformer expects (batch, seq_len, features)
                    if x.dim() == 2:
                        x_reshaped = x.unsqueeze(1)
                    else:
                        x_reshaped = x
                    op_output = op(x_reshaped)

                output += weights[j] * op_output.squeeze(1)
            x = output

        return x
