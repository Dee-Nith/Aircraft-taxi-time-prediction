"""
Backpropagation Neural Network implementation for aircraft taxi time prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from ..utils.data_preprocessing import load_and_preprocess_data
    from ..utils.evaluation_metrics import calculate_metrics, print_metrics
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.data_preprocessing import load_and_preprocess_data
    from utils.evaluation_metrics import calculate_metrics, print_metrics


def random_number(a, b):
    """Generate random number between a and b."""
    return np.random.uniform(a, b)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def derived_leaky_relu(x, alpha=0.01):
    """Derivative of Leaky ReLU activation function."""
    return np.where(x > 0, 1, alpha)


class BPNN:
    """
    Backpropagation Neural Network with three hidden layers.
    """
    
    def __init__(self, num_in, num_hidden1, num_hidden2, num_hidden3, num_out):
        """
        Initialize the neural network.
        
        Args:
            num_in (int): Number of input features
            num_hidden1 (int): Number of neurons in first hidden layer
            num_hidden2 (int): Number of neurons in second hidden layer
            num_hidden3 (int): Number of neurons in third hidden layer
            num_out (int): Number of output neurons
        """
        # Nodes of input, hidden, and output layers (with bias nodes)
        self.num_in = num_in + 1
        self.num_hidden1 = num_hidden1 + 1
        self.num_hidden2 = num_hidden2 + 1
        self.num_hidden3 = num_hidden3 + 1
        self.num_out = num_out

        # Activation nodes
        self.active_in = np.array([1.0] * self.num_in)
        self.active_hidden1 = np.array([1.0] * self.num_hidden1)
        self.active_hidden2 = np.array([1.0] * self.num_hidden2)
        self.active_hidden3 = np.array([1.0] * self.num_hidden3)
        self.active_out = np.array([1.0] * self.num_out)

        # Initialize weight matrices with small random values
        self.weight_in = random_number(-0.1, 0.1) * np.random.randn(self.num_in, self.num_hidden1 - 1)
        self.weight_hidden1 = random_number(-0.1, 0.1) * np.random.randn(self.num_hidden1, self.num_hidden2 - 1)
        self.weight_hidden2 = random_number(-0.1, 0.1) * np.random.randn(self.num_hidden2, self.num_hidden3 - 1)
        self.weight_out = random_number(-0.1, 0.1) * np.random.randn(self.num_hidden3, self.num_out)

    def feedforward(self, inputs):
        """
        Forward propagation through the network.
        
        Args:
            inputs (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Network output
        """
        # Input to first hidden layer
        self.active_in[1:self.num_in] = inputs
        self.sum_hidden1 = np.dot(self.weight_in.T, np.array([self.active_in]).T)
        self.active_hidden1 = np.vstack((1, leaky_relu(self.sum_hidden1)))

        # First hidden layer to second hidden layer
        self.sum_hidden2 = np.dot(self.weight_hidden1.T, self.active_hidden1)
        self.active_hidden2 = np.vstack((1, leaky_relu(self.sum_hidden2)))

        # Second hidden layer to third hidden layer
        self.sum_hidden3 = np.dot(self.weight_hidden2.T, self.active_hidden2)
        self.active_hidden3 = np.vstack((1, leaky_relu(self.sum_hidden3)))

        # Third hidden layer to output layer
        self.sum_out = np.dot(self.weight_out.T, self.active_hidden3)
        self.active_out = self.sum_out
        
        return self.active_out

    def backpropagate(self, targets, lr):
        """
        Backpropagation algorithm for weight updates.
        
        Args:
            targets (numpy.ndarray): Target values
            lr (float): Learning rate
            
        Returns:
            float: Error value
        """
        # Calculate error
        self.error = (1 / 2) * np.dot((self.active_out - targets.T).T, (self.active_out - targets.T))
        
        # Output layer gradients
        self.gradient_out = self.active_out - targets.T
        self.gradient_w_out = np.dot(self.gradient_out, self.active_hidden3.T).T

        # Third hidden layer gradients
        self.gradient_hidden3 = np.dot(self.weight_out[1:], self.gradient_out) * derived_leaky_relu(self.sum_hidden3)
        self.gradient_w_hidden2 = np.dot(self.gradient_hidden3, self.active_hidden2.T).T

        # Second hidden layer gradients
        self.gradient_hidden2 = np.dot(self.weight_hidden2[1:], self.gradient_hidden3) * derived_leaky_relu(self.sum_hidden2)
        self.gradient_w_hidden1 = np.dot(self.gradient_hidden2, self.active_hidden1.T).T

        # First hidden layer gradients
        self.gradient_hidden1 = np.dot(self.weight_hidden1[1:], self.gradient_hidden2) * derived_leaky_relu(self.sum_hidden1)
        self.gradient_w_in = np.dot(self.gradient_hidden1, np.array([self.active_in])).T

        # Update weights
        self.weight_out -= lr * self.gradient_w_out
        self.weight_hidden2 -= lr * self.gradient_w_hidden2
        self.weight_hidden1 -= lr * self.gradient_w_hidden1
        self.weight_in -= lr * self.gradient_w_in
        
        return self.error

    def train(self, pattern, itera=2000, lr=0.01, verbose=True):
        """
        Train the neural network.
        
        Args:
            pattern (numpy.ndarray): Training data (features + targets)
            itera (int): Number of training iterations
            lr (float): Learning rate
            verbose (bool): Whether to print training progress
        """
        for i in range(itera):
            error = 0
            for j in pattern:
                inputs = np.array([j[0:self.num_in - 1]])
                targets = np.array([j[self.num_in - 1:]])
                self.feedforward(inputs)
                self.backpropagate(targets, lr)
                error += self.error
            
            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Error: {error[0][0]:.5f}")

    def predict(self, patterns):
        """
        Make predictions on new data.
        
        Args:
            patterns (numpy.ndarray): Input data
            
        Returns:
            list: Predictions
        """
        predictions = []
        for i in patterns:
            inputs = np.array([i[0:self.num_in - 1]])
            predictions.append(self.feedforward(inputs))
        return predictions


def train_and_evaluate_bpnn(data_path, input_columns=None, output_column='taxi_time'):
    """
    Train and evaluate the BPNN model.
    
    Args:
        data_path (str): Path to the dataset
        input_columns (list): List of input feature columns
        output_column (str): Name of the target column
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    # Load and preprocess data
    X_scaled, y_scaled, scaler_X, scaler_y, X_train, X_test, y_train, y_test = load_and_preprocess_data(
        data_path, input_columns, output_column
    )
    
    # Prepare training patterns
    pattern = np.hstack((X_train, y_train.reshape(-1, 1)))
    
    # Initialize and train the network
    nn = BPNN(X_train.shape[1], 128, 64, 32, 1)
    nn.train(pattern, itera=5000, lr=0.005)
    
    # Make predictions
    test_patterns = np.hstack((X_test, y_test.reshape(-1, 1)))
    predictions = nn.predict(test_patterns)
    
    # Denormalize predictions and actual values
    y_pred = np.array([pred[0][0] for pred in predictions])
    y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    MSE, RMSE, MBE, MAE, MAPE, TIC, R, R2 = calculate_metrics(y_test_denorm, y_pred_denorm)
    
    metrics = {
        "MSE": MSE,
        "RMSE": RMSE,
        "MBE": MBE,
        "MAE": MAE,
        "MAPE": MAPE,
        "TIC": TIC,
        "R": R,
        "R2": R2
    }
    
    # Print results
    print_metrics(metrics, "BPNN Test Data")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test_denorm)), y_test_denorm, 'b.', label='Actual', alpha=0.7)
    plt.plot(range(len(y_pred_denorm)), y_pred_denorm, 'g-', label='Predicted', alpha=0.7)
    plt.legend()
    plt.title("BPNN: Actual vs Predicted Values (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Taxi Time (minutes)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return nn, metrics, (y_test_denorm, y_pred_denorm)


if __name__ == "__main__":
    # Example usage
    data_path = "../../data/processed_dataset.csv"
    input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
    
    model, metrics, predictions = train_and_evaluate_bpnn(data_path, input_columns)
