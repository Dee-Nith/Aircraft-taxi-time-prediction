"""
Main script to run aircraft taxi time prediction models.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.neural_networks.bpnn_model import train_and_evaluate_bpnn
from src.anfis.anfis_model import train_and_evaluate_anfis
from src.utils.evaluation_metrics import create_metrics_summary


def main():
    """
    Main function to run both models and compare results.
    """
    print("=" * 60)
    print("AIRCRAFT TAXI TIME PREDICTION")
    print("=" * 60)
    
    # Data configuration
    data_path = "data/processed_dataset.csv"
    input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
    output_column = 'taxi_time'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the correct location.")
        return
    
    print(f"Using dataset: {data_path}")
    print(f"Input features: {input_columns}")
    print(f"Target variable: {output_column}")
    print()
    
    # Load and display dataset info
    try:
        data = pd.read_csv(data_path)
        print(f"Dataset shape: {data.shape}")
        print(f"Features: {list(data.columns)}")
        print(f"Target variable range: {data[output_column].min():.2f} - {data[output_column].max():.2f}")
        print()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    results = {}
    
    # Run BPNN Model
    print("=" * 40)
    print("TRAINING BACKPROPAGATION NEURAL NETWORK")
    print("=" * 40)
    try:
        bpnn_model, bpnn_metrics, bpnn_predictions = train_and_evaluate_bpnn(
            data_path, input_columns, output_column
        )
        results['BPNN'] = bpnn_metrics
        print("BPNN training completed successfully!")
    except Exception as e:
        print(f"Error training BPNN: {e}")
        results['BPNN'] = None
    
    print()
    
    # Run ANFIS Model
    print("=" * 40)
    print("TRAINING ANFIS MODEL")
    print("=" * 40)
    try:
        anfis_model, anfis_metrics, anfis_predictions = train_and_evaluate_anfis(
            data_path, input_columns, output_column
        )
        results['ANFIS'] = anfis_metrics
        print("ANFIS training completed successfully!")
    except Exception as e:
        print(f"Error training ANFIS: {e}")
        results['ANFIS'] = None
    
    print()
    
    # Compare Results
    print("=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)
    
    if results['BPNN'] and results['ANFIS']:
        comparison_df = pd.DataFrame({
            'BPNN': results['BPNN'],
            'ANFIS': results['ANFIS']
        })
        
        print("\nPerformance Comparison:")
        print(comparison_df.round(4))
        
        # Determine best model
        bpnn_rmse = results['BPNN']['RMSE']
        anfis_rmse = results['ANFIS']['RMSE']
        
        if bpnn_rmse < anfis_rmse:
            print(f"\nBest Model: BPNN (RMSE: {bpnn_rmse:.4f})")
        else:
            print(f"\nBest Model: ANFIS (RMSE: {anfis_rmse:.4f})")
    
    elif results['BPNN']:
        print("BPNN Results Available:")
        for metric, value in results['BPNN'].items():
            print(f"  {metric}: {value:.4f}")
    
    elif results['ANFIS']:
        print("ANFIS Results Available:")
        for metric, value in results['ANFIS'].items():
            print(f"  {metric}: {value:.4f}")
    
    else:
        print("No models completed successfully.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
