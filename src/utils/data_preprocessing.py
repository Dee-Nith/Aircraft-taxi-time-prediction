"""
Data preprocessing utilities for aircraft taxi time prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path, input_columns=None, output_column='taxi_time'):
    """
    Load and preprocess the taxi time dataset.
    
    Args:
        file_path (str): Path to the CSV file
        input_columns (list): List of input feature columns
        output_column (str): Name of the target column
        
    Returns:
        tuple: (X_scaled, y_scaled, scaler_X, scaler_y, X_train, X_test, y_train, y_test)
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Default input columns if not specified
    if input_columns is None:
        input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
    
    # Extract features and target
    X = data[input_columns].values
    y = data[output_column].values
    
    # Remove outliers using IQR method
    X, y = remove_outliers_iqr(X, y)
    
    # Normalize the data
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    return X_scaled, y_scaled, scaler_X, scaler_y, X_train, X_test, y_train, y_test


def remove_outliers_iqr(X, y, factor=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
        factor (float): IQR factor for outlier detection
        
    Returns:
        tuple: (X_filtered, y_filtered)
    """
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Find non-outlier indices
    non_outlier_indices = np.where((y >= lower_bound) & (y <= upper_bound))[0]
    
    return X[non_outlier_indices], y[non_outlier_indices]


def normalize_data(X, y, feature_range=(-1, 1)):
    """
    Normalize input and output data.
    
    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
        feature_range (tuple): Range for normalization
        
    Returns:
        tuple: (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(random_state)
    
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(round(train_ratio * num_samples))
    validation_end = train_end + int(round(val_ratio * num_samples))
    
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:validation_end]
    test_idx = indices[validation_end:]
    
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_val = X[validation_idx, :]
    y_val = y[validation_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
