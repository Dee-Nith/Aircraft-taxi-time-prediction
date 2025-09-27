"""
ANFIS (Adaptive Neuro-Fuzzy Inference System) implementation for aircraft taxi time prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skfuzzy.cluster import cmeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
try:
    from ..utils.evaluation_metrics import calculate_metrics, print_metrics
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.evaluation_metrics import calculate_metrics, print_metrics


class ANFISModel:
    """
    Simplified ANFIS model using Fuzzy C-Means clustering and Linear Regression.
    """
    
    def __init__(self, n_clusters=3, m=2, error=0.005, maxiter=1000):
        """
        Initialize ANFIS model.
        
        Args:
            n_clusters (int): Number of fuzzy clusters
            m (float): Fuzziness parameter
            error (float): Convergence error threshold
            maxiter (int): Maximum iterations
        """
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.cluster_centers = None
        self.membership_matrix = None
        self.regression_models = []
        self.scaler_X = None
        self.scaler_y = None
        
    def fit(self, X, y):
        """
        Train the ANFIS model.
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Target values
        """
        # Normalize the data
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Apply Fuzzy C-Means Clustering
        self.cluster_centers, self.membership_matrix, _, _, _, _, _ = cmeans(
            X_scaled.T, c=self.n_clusters, m=self.m, 
            error=self.error, maxiter=self.maxiter, init=None
        )
        
        # Train linear regression models for each cluster
        self.regression_models = []
        for i in range(self.n_clusters):
            # Get membership weights for this cluster
            weights = self.membership_matrix[i, :]
            
            # Train weighted linear regression
            model = LinearRegression()
            model.fit(X_scaled, y_scaled, sample_weight=weights)
            self.regression_models.append(model)
    
    def predict(self, X):
        """
        Make predictions using the trained ANFIS model.
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predictions
        """
        X_scaled = self.scaler_X.transform(X)
        
        # Calculate membership degrees for new data
        predictions = np.zeros(X_scaled.shape[0])
        
        for i, x in enumerate(X_scaled):
            # Calculate distance to cluster centers
            distances = np.linalg.norm(self.cluster_centers - x.reshape(1, -1), axis=1)
            
            # Calculate membership degrees (inverse distance weighting)
            if np.min(distances) == 0:
                # If exactly on a cluster center
                membership = np.zeros(self.n_clusters)
                membership[np.argmin(distances)] = 1.0
            else:
                # Inverse distance weighting
                membership = 1.0 / (distances ** 2)
                membership = membership / np.sum(membership)
            
            # Weighted prediction from all clusters
            cluster_predictions = []
            for j, model in enumerate(self.regression_models):
                pred = model.predict(x.reshape(1, -1))[0]
                cluster_predictions.append(pred * membership[j])
            
            predictions[i] = np.sum(cluster_predictions)
        
        # Denormalize predictions
        predictions_denorm = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions_denorm


def train_and_evaluate_anfis(data_path, input_columns=None, output_column='taxi_time'):
    """
    Train and evaluate the ANFIS model.
    
    Args:
        data_path (str): Path to the dataset
        input_columns (list): List of input feature columns
        output_column (str): Name of the target column
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Default input columns if not specified
    if input_columns is None:
        input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
    
    # Extract features and target
    X = data[input_columns].values
    y = data[output_column].values
    
    # Remove outliers using IQR method
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    non_outlier_indices = [i for i in range(len(y)) if lower_bound <= y[i] <= upper_bound]
    X = X[non_outlier_indices]
    y = y[non_outlier_indices]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ANFIS model
    anfis = ANFISModel(n_clusters=3)
    anfis.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = anfis.predict(X_train)
    y_test_pred = anfis.predict(X_test)
    
    # Calculate metrics for test data
    MSE, RMSE, MBE, MAE, MAPE, TIC, R, R2 = calculate_metrics(y_test, y_test_pred)
    
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
    print_metrics(metrics, "ANFIS Test Data")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Training data plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, label='Train Data')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 
             color='red', linestyle='--', label='Ideal Fit')
    plt.title("ANFIS: Actual vs Predicted (Training)")
    plt.xlabel("Actual Taxi Time")
    plt.ylabel("Predicted Taxi Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test data plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test Data')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
             color='red', linestyle='--', label='Ideal Fit')
    plt.title("ANFIS: Actual vs Predicted (Test)")
    plt.xlabel("Actual Taxi Time")
    plt.ylabel("Predicted Taxi Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return anfis, metrics, (y_test, y_test_pred)


if __name__ == "__main__":
    # Example usage
    data_path = "../../data/processed_dataset.csv"
    input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
    
    model, metrics, predictions = train_and_evaluate_anfis(data_path, input_columns)