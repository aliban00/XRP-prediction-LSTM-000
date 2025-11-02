import torch
import torch.nn as nn
import optuna
from rich.console import Console

console = Console()

class NASModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(NASModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

def objective(trial, train_sequences, train_labels):
    """
    Defines the objective function for Optuna.
    """
    # Define the search space
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 50, 200)

    # Create the model
    model = NASModel(input_size=train_sequences[0].shape[1], hidden_size=hidden_size, n_layers=n_layers)

    # Train the model (simplified for demonstration)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2): # Simplified training loop
        for seq, labels in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            y_pred = model(seq.unsqueeze(0))
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    # Evaluate the model
    val_loss = 0
    with torch.no_grad():
        for seq, labels in zip(train_sequences, train_labels): # Using train data for validation for simplicity
            y_pred = model(seq.unsqueeze(0))
            val_loss += criterion(y_pred, labels).item()

    return val_loss / len(train_sequences)

def find_best_architecture(train_sequences, train_labels):
    """
    Finds the best architecture using Optuna.
    """
    console.log("Starting neural architecture search with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_sequences, train_labels), n_trials=30)

    console.log(f"Best trial: {study.best_trial.value}")
    console.log(f"Best params: {study.best_params}")

    # Save the best model and parameters
    best_params = study.best_params
    model = NASModel(input_size=train_sequences[0].shape[1], hidden_size=best_params['hidden_size'], n_layers=best_params['n_layers'])
    torch.save({
        'params': best_params,
        'state_dict': model.state_dict()
    }, 'best_model.pt')

    return model
