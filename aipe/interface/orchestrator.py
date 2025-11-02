import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console
from aipe.core.analyst import Analyst
from aipe.core.predictor import Predictor
from aipe.core.philosopher import Philosopher
from aipe.interface.terminal_ui import TerminalUI

# Configure logging
logging.basicConfig(filename='aipe_performance.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Orchestrator:
    def __init__(self, data):
        self.analyst = Analyst()
        self.predictor = Predictor()
        self.philosopher = Philosopher()
        self.ui = TerminalUI()
        self.model = None
        self.loss_function = nn.MSELoss()
        self.optimizer = None
        self.console = Console()

    def run(self, data):
        """
        Runs the main intelligence loop.
        """
        # 1. The Analyst understands the data
        start_time = time.time()
        data = self.analyst.engineer_features(data)
        data.fillna(data.mean(), inplace=True) # Fill NaN values with mean
        if data.empty:
            self.console.log("[bold red]Error: No data left after feature engineering.[/bold red]")
            return

        causal_model = self.analyst.discover_causal_structure(data)
        mi_matrix = self.analyst.analyze_information_theory(data)
        regimes = self.analyst.detect_regimes(data)
        complexity = self.analyst.analyze_complexity(data)
        analyst_latency = time.time() - start_time
        logging.info(f"Analyst Latency: {analyst_latency:.2f}s")

        analyst_results = {
            'causal_model': causal_model,
            'mi_matrix': mi_matrix,
            'regimes': regimes,
            'complexity': complexity
        }

        # 2. Preprocess data
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[0:train_size], data[train_size:len(data)]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_data.values)
        train_data_normalized = torch.FloatTensor(train_data_normalized)

        # Create sequences
        tw = 12 # The sequence length
        close_idx = data.columns.get_loc('close')
        train_sequences, train_labels = self._create_sequences(train_data_normalized, tw, close_idx)

        # 3. The Predictor makes a prediction
        start_time = time.time()

        # Create a model for each regime
        self.models = {}

        # Align regimes with sequences. The regime of a sequence is the regime of its last data point.
        sequence_regimes = [regimes[i+tw-1] for i in range(len(train_sequences))]

        for regime in set(sequence_regimes):
            # Find the best architecture for this regime
            regime_train_sequences = [seq for i, seq in enumerate(train_sequences) if sequence_regimes[i] == regime]
            regime_train_labels = [label for i, label in enumerate(train_labels) if sequence_regimes[i] == regime]

            # Split data into training and validation sets
            val_size = int(len(regime_train_sequences) * 0.2)
            train_seqs = regime_train_sequences[:-val_size]
            train_labs = regime_train_labels[:-val_size]
            val_seqs = regime_train_sequences[-val_size:]
            val_labs = regime_train_labels[-val_size:]

            model = self.predictor.search_architecture(train_seqs, train_labs)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model with early stopping
            epochs = 30
            patience = 10
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(epochs):
                # Training
                for seq, labels in zip(train_seqs, train_labs):
                    optimizer.zero_grad()
                    y_pred = model(seq.unsqueeze(0))
                    single_loss = self.loss_function(y_pred, labels)
                    single_loss.backward()
                    optimizer.step()

                # Validation
                val_loss = 0
                with torch.no_grad():
                    for seq, labels in zip(val_seqs, val_labs):
                        y_pred = model(seq.unsqueeze(0))
                        val_loss += self.loss_function(y_pred, labels).item()
                val_loss /= len(val_seqs)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        self.console.log(f"Early stopping at epoch {epoch+1}")
                        break

            model.load_state_dict(best_model_state)
            self.models[regime] = model

        # Make a prediction using the model for the current regime
        current_regime = regimes[-1]
        self.model = self.models[current_regime]

        test_data_normalized = scaler.transform(test_data.values)
        test_data_normalized = torch.FloatTensor(test_data_normalized)
        test_sequences, test_labels = self._create_sequences(test_data_normalized, tw, close_idx)

        # Make a prediction on the last sequence of the training data
        test_inputs = train_data_normalized[-tw:]
        prediction = self.model(test_inputs.unsqueeze(0))

        # We need to inverse transform the prediction, but the scaler expects all features.
        # We'll create a dummy array with all features and replace the 'close' price with our prediction.
        dummy_prediction = np.zeros((1, len(data.columns)))
        dummy_prediction[0, close_idx] = prediction.detach().numpy()[0]
        prediction_unscaled = scaler.inverse_transform(dummy_prediction)
        predictor_latency = time.time() - start_time
        logging.info(f"Predictor Latency: {predictor_latency:.2f}s")

        predictor_results = {'prediction': prediction_unscaled[0, 3]}

        # 4. The Philosopher learns from experience
        start_time = time.time()
        actual = test_data['close'].values[0]
        context = {
            'test_inputs': test_inputs,
            'feature_names': data.columns
        }
        outcomes = self.philosopher.reason_counterfactually(self.model, predictor_results['prediction'], actual, context)
        philosopher_latency = time.time() - start_time
        logging.info(f"Philosopher Latency: {philosopher_latency:.2f}s")

        philosopher_results = {'outcomes': outcomes}

        # 5. Backtest the model
        avg_mape = self.philosopher.backtest(self.model, test_sequences, test_labels, scaler, close_idx)
        logging.info(f"Backtest MAPE: {avg_mape:.2f}%")

        # 6. Display insights and log performance
        self.ui.display_insights(analyst_results, predictor_results, philosopher_results)

        # Log performance
        mape = abs((predictor_results['prediction'] - actual) / actual) * 100
        logging.info(f"Prediction Accuracy (MAPE): {float(mape):.2f}%")

    def _create_sequences(self, input_data, tw, close_idx):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1, close_idx:close_idx+1] # Predict the 'close' price
            inout_seq.append((train_seq ,train_label))
        return [i[0] for i in inout_seq], [i[1] for i in inout_seq]
