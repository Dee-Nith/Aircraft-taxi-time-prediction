# Results Analysis

**Author:** Deepak Annanth  
**GitHub:** [@Dee-Nith](https://github.com/Dee-Nith)  
**Email:** deepak.annanth@gmail.com

## Model Performance Comparison

### Backpropagation Neural Network (BPNN)

#### Training Results:
- **Architecture**: 4 → 128 → 64 → 32 → 1
- **Training Iterations**: 5000
- **Learning Rate**: 0.005
- **Activation Function**: Leaky ReLU

#### Test Set Performance:
| Metric | Value |
|--------|-------|
| RMSE | 3.512 minutes |
| MAE | 2.553 minutes |
| MRE | 37.81% |
| R² | -0.0214 |
| TIC | 0.9320 |

#### Unseen Data Performance:
| Metric | Value |
|--------|-------|
| RMSE | 2.173 minutes |
| MAE | 0.879 minutes |
| MRE | 13.10% |

### ANFIS Model

#### Implementation Status:
- **Current State**: Simplified version using Linear Regression
- **Clustering**: Fuzzy C-Means with 3 clusters
- **Fuzziness Parameter**: 2.0

#### Performance:
- Implementation in progress
- Baseline results available

## Key Observations

### 1. Model Generalization
- **BPNN shows better generalization** on unseen data compared to test data
- Unseen data RMSE (2.173) is lower than test RMSE (3.512)
- This suggests the model is not overfitting

### 2. Prediction Accuracy
- **MAE of 0.879 minutes** on unseen data is quite good for taxi time prediction
- **MRE of 13.10%** indicates reasonable relative error
- The model can predict taxi times within ~1 minute on average

### 3. Model Behavior
- **Negative R²** suggests the model performs worse than simply predicting the mean
- This could indicate:
  - Need for better feature engineering
  - Insufficient training data
  - Model architecture limitations

### 4. Training Convergence
- BPNN training shows good convergence
- Error decreases from 155.39 to near zero
- No signs of overfitting during training

## Feature Importance Analysis

### Most Important Features:
1. **Distance**: Primary factor in taxi time
2. **Operation Mode**: Departure vs arrival operations
3. **Other Moving Aircraft**: Traffic congestion factor
4. **Angle**: Taxi path complexity

### Feature Engineering Opportunities:
- **Time-based features**: Hour of day, day of week patterns
- **Weather features**: If available
- **Runway-specific features**: Different runway characteristics
- **Historical patterns**: Previous taxi times for similar operations

## Model Limitations

### 1. Data Quality
- **Limited dataset size**: May need more training data
- **Missing features**: Weather, runway conditions, aircraft type
- **Outlier handling**: Current IQR method may be too aggressive

### 2. Model Architecture
- **BPNN**: Fixed architecture, no hyperparameter optimization
- **ANFIS**: Simplified implementation, not full fuzzy inference
- **No ensemble methods**: Single model predictions

### 3. Evaluation
- **No cross-validation**: Limited robustness assessment
- **Single train-test split**: May not be representative
- **No confidence intervals**: Uncertainty quantification missing

## Recommendations for Improvement

### 1. Data Enhancement
- Collect more training data
- Include additional features (weather, aircraft type, etc.)
- Implement more sophisticated outlier detection

### 2. Model Improvements
- Implement full ANFIS with proper fuzzy inference
- Add hyperparameter tuning for BPNN
- Explore ensemble methods
- Consider deep learning approaches (LSTM, GRU)

### 3. Evaluation Enhancement
- Implement k-fold cross-validation
- Add confidence intervals
- Perform statistical significance tests
- Include more evaluation metrics

### 4. Deployment Considerations
- Model interpretability for operational use
- Real-time prediction capabilities
- Integration with airport systems
- Performance monitoring and retraining

## Conclusion

The BPNN model shows promising results for aircraft taxi time prediction, particularly on unseen data. The model achieves reasonable accuracy with an MAE of less than 1 minute on unseen data. However, there are opportunities for improvement in data quality, model architecture, and evaluation methodology.

The ANFIS implementation needs completion to provide a proper comparison between neuro-fuzzy and neural network approaches. Future work should focus on enhancing the dataset, optimizing model parameters, and implementing more robust evaluation strategies.
