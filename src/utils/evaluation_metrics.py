"""
Evaluation metrics for machine learning models.
"""

import numpy as np


def calculate_metrics(true_out, predict_out):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        true_out (array-like): True values
        predict_out (array-like): Predicted values
        
    Returns:
        tuple: (MSE, RMSE, MBE, MAE, MAPE, TIC, R, R2)
    """
    true_out = np.array(true_out).ravel()
    predict_out = np.array(predict_out).ravel()
    N = len(predict_out)
    
    # Mean Squared Error
    MSE = np.mean((predict_out - true_out) ** 2)
    
    # Root Mean Squared Error
    RMSE = np.sqrt(MSE)
    
    # Mean Absolute Error
    MAE = np.mean(np.abs(predict_out - true_out))
    
    # Mean Bias Error
    MBE = np.mean(predict_out - true_out)
    
    # Mean Absolute Percentage Error
    MAPE = np.mean(np.abs((true_out - predict_out) / true_out)) * 100
    
    # Theil's Inequality Coefficient
    numerator = np.sqrt(np.mean((true_out - predict_out) ** 2))
    denominator = np.sqrt(np.mean((true_out) ** 2)) + np.sqrt(np.mean((predict_out) ** 2))
    TIC = numerator / denominator if denominator != 0 else 0.0
    
    # Correlation Coefficient
    correlation_matrix = np.corrcoef(true_out, predict_out)
    R = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    # R-squared
    SS_res = np.sum((true_out - predict_out) ** 2)
    SS_tot = np.sum((true_out - np.mean(true_out)) ** 2)
    R2 = 1 - (SS_res / SS_tot) if SS_tot != 0 else 0.0
    
    return MSE, RMSE, MBE, MAE, MAPE, TIC, R, R2


def compute_metrics(y_actual, y_pred):
    """
    Compute basic regression metrics.
    
    Args:
        y_actual (array-like): Actual values
        y_pred (array-like): Predicted values
        
    Returns:
        tuple: (rmse, mae, mre)
    """
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    mae = np.mean(np.abs(y_actual - y_pred))
    mre = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    return rmse, mae, mre


def print_metrics(metrics_dict, dataset_name="Dataset"):
    """
    Print formatted metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metric values
        dataset_name (str): Name of the dataset
    """
    print(f"\n{dataset_name} Performance Metrics:")
    print("-" * 40)
    
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


def create_metrics_summary(train_metrics, test_metrics, model_name="Model"):
    """
    Create a summary of metrics for comparison.
    
    Args:
        train_metrics (dict): Training metrics
        test_metrics (dict): Test metrics
        model_name (str): Name of the model
        
    Returns:
        dict: Summary metrics
    """
    summary = {
        "Model": model_name,
        "Train_RMSE": train_metrics.get("RMSE", 0),
        "Test_RMSE": test_metrics.get("RMSE", 0),
        "Train_MAE": train_metrics.get("MAE", 0),
        "Test_MAE": test_metrics.get("MAE", 0),
        "Train_R2": train_metrics.get("R2", 0),
        "Test_R2": test_metrics.get("R2", 0),
        "Overfitting": abs(train_metrics.get("RMSE", 0) - test_metrics.get("RMSE", 0))
    }
    
    return summary
