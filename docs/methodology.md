# Methodology

**Author:** Deepak Annanth  
**GitHub:** [@Dee-Nith](https://github.com/Dee-Nith)  
**Email:** deepak.annanth@gmail.com

## Problem Formulation

The aircraft taxi time prediction problem is formulated as a regression task where we aim to predict the time an aircraft takes to taxi from gate to runway (or vice versa) based on various operational and environmental factors.

## Dataset Description

### Features
- **Temporal Features**: Gate hour, runway hour, day of week
- **Spatial Features**: Distance, angle, shortest path, gate distance
- **Operational Features**: Operation mode, other moving aircraft count, queue information

### Target Variable
- **taxi_time**: Time taken for aircraft to taxi (in minutes)

### Data Preprocessing
1. **Outlier Removal**: Using Interquartile Range (IQR) method
2. **Normalization**: MinMax scaling to range [-1, 1]
3. **Train-Test Split**: 80-20 split with random state for reproducibility

## Model Architectures

### 1. ANFIS (Adaptive Neuro-Fuzzy Inference System)

ANFIS combines the learning capabilities of neural networks with the interpretability of fuzzy logic systems.

#### Architecture:
- **Fuzzy Layer**: Fuzzy C-Means clustering for rule extraction
- **Rule Layer**: Membership functions for each cluster
- **Consequent Layer**: Linear regression models for each rule
- **Output Layer**: Weighted combination of rule outputs

#### Training Process:
1. Apply Fuzzy C-Means clustering to input data
2. Train separate linear regression models for each cluster
3. Use membership degrees for weighted predictions

### 2. Backpropagation Neural Network (BPNN)

A custom implementation of a multi-layer perceptron with three hidden layers.

#### Architecture:
- **Input Layer**: 4 features + bias
- **Hidden Layer 1**: 128 neurons + bias
- **Hidden Layer 2**: 64 neurons + bias  
- **Hidden Layer 3**: 32 neurons + bias
- **Output Layer**: 1 neuron (regression output)

#### Activation Function:
- **Leaky ReLU**: f(x) = max(0.01x, x) for hidden layers
- **Linear**: Identity function for output layer

#### Training Process:
1. Forward propagation through all layers
2. Calculate error using mean squared error
3. Backpropagate gradients through the network
4. Update weights using gradient descent

## Evaluation Metrics

### Primary Metrics:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MRE**: Mean Relative Error (percentage)

### Secondary Metrics:
- **R²**: Coefficient of Determination
- **TIC**: Theil's Inequality Coefficient
- **MAPE**: Mean Absolute Percentage Error

## Model Comparison

The models are compared based on:
1. **Accuracy**: RMSE and MAE on test data
2. **Generalization**: Performance on unseen data
3. **Interpretability**: ANFIS provides fuzzy rules
4. **Training Time**: Computational efficiency
5. **Robustness**: Performance across different data splits

## Experimental Setup

### Training Configuration:
- **Learning Rate**: 0.005 (BPNN)
- **Epochs**: 5000 (BPNN)
- **Clusters**: 3 (ANFIS)
- **Fuzziness Parameter**: 2.0 (ANFIS)

### Validation Strategy:
- **Train-Test Split**: 80-20
- **Cross-validation**: Not implemented (future work)
- **Random State**: 42 for reproducibility

## Results Summary

### BPNN Performance:
- Test RMSE: 3.512 minutes
- Test MAE: 2.553 minutes
- Test MRE: 37.81%
- Unseen Data RMSE: 2.173 minutes
- Unseen Data MAE: 0.879 minutes
- Unseen Data MRE: 13.10%

### ANFIS Performance:
- Implementation in progress
- Currently using Linear Regression baseline

## Future Improvements

1. **Hyperparameter Tuning**: Grid search for optimal parameters
2. **Cross-validation**: K-fold validation for robust evaluation
3. **Feature Engineering**: Additional derived features
4. **Ensemble Methods**: Combining multiple models
5. **Deep Learning**: LSTM/GRU for temporal patterns
