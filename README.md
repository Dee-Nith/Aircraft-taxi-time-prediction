# Aircraft Taxi Time Prediction using Machine Learning

## Project Overview

This project implements and compares different machine learning approaches for predicting aircraft taxi times at airports. The goal is to develop accurate models that can help optimize airport operations and reduce delays by predicting how long aircraft will take to taxi from gate to runway or vice versa.

## Problem Statement

Aircraft taxi time prediction is a critical component of airport ground operations management. Accurate predictions can help:
- Optimize runway scheduling
- Reduce ground delays
- Improve fuel efficiency
- Enhance passenger experience
- Better resource allocation

## Dataset

The dataset contains aircraft taxi operations data with the following features:

### Input Features
- **Temporal**: Gate hour, runway hour, day of week
- **Spatial**: Distance, angle, shortest path, gate distance
- **Operational**: Operation mode, other moving aircraft count, queue information

### Target Variable
- **taxi_time**: Time taken for aircraft to taxi (in minutes)

## Machine Learning Approaches

### 1. ANFIS (Adaptive Neuro-Fuzzy Inference System)
- **Location**: `src/anfis/`
- **Description**: Hybrid neuro-fuzzy system combining fuzzy logic with neural networks
- **Implementation**: Fuzzy C-Means clustering + Linear Regression
- **Files**: 
  - `anfis_model.py` - Main ANFIS implementation
  - `anfis_final.ipynb` - Jupyter notebook with complete analysis

### 2. Neural Networks (Backpropagation)
- **Location**: `src/neural_networks/`
- **Description**: Custom 3-layer backpropagation neural network
- **Architecture**: Input → Hidden1(128) → Hidden2(64) → Hidden3(32) → Output
- **Activation**: Leaky ReLU
- **Files**:
  - `bpnn_model.py` - Custom neural network implementation
  - `taxi_times_nn.ipynb` - Complete training and evaluation notebook

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── processed_dataset.csv
│   ├── features.csv
│   └── pca_transformed_taxi_data.csv
├── src/
│   ├── anfis/
│   │   ├── anfis_model.py
│   │   └── anfis_final.ipynb
│   ├── neural_networks/
│   │   ├── bpnn_model.py
│   │   └── taxi_times_nn.ipynb
│   └── utils/
│       ├── data_preprocessing.py
│       └── evaluation_metrics.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_comparison.ipynb
├── results/
│   ├── model_performance.md
│   └── visualizations/
│       ├── dataset_overview.png
│       ├── bpnn_results.png
│       ├── anfis_results.png
│       └── model_comparison.png
└── docs/
    ├── methodology.md
    └── results_analysis.md
```

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aircraft-taxi-time-prediction.git
   cd aircraft-taxi-time-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running ANFIS Model
```bash
cd src/anfis
python anfis_model.py
```

### Running Neural Network Model
```bash
cd src/neural_networks
python bpnn_model.py
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Results

### Model Performance Comparison

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|----|---------| 
| **ANFIS** | **4.0488** | **3.2071** | **0.3291** | ✅ **BEST** |
| BPNN | 4.6474 | 3.2960 | 0.1161 | ✅ Working |

### 🏆 **Winner: ANFIS Model**
- **Lower RMSE**: 4.05 vs 4.65 (13% improvement)
- **Lower MAE**: 3.21 vs 3.30 (3% improvement) 
- **Better R²**: 0.33 vs 0.12 (much better fit)

### Visualization Results

#### Dataset Overview
![Dataset Overview](results/visualizations/dataset_overview.png)
*Distribution of taxi times, feature correlations, and dataset characteristics*

#### BPNN Model Results
![BPNN Results](results/visualizations/bpnn_results.png)
*Backpropagation Neural Network: Actual vs Predicted scatter plot and time series comparison*

#### ANFIS Model Results
![ANFIS Results](results/visualizations/anfis_results.png)
*Adaptive Neuro-Fuzzy Inference System: Actual vs Predicted scatter plot and time series comparison*

#### Model Comparison
![Model Comparison](results/visualizations/model_comparison.png)
*Comprehensive comparison of both models including performance metrics, residuals, and error distributions*

## Key Features

- **Custom Neural Network**: Built from scratch with 3 hidden layers
- **Data Preprocessing**: MinMax normalization and outlier removal
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, MRE, R², TIC)
- **Visualization**: Actual vs Predicted plots and performance analysis
- **Modular Design**: Separate modules for different approaches

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-fuzzy
- Jupyter Notebook

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Deepak Annanth**
- GitHub: [@Dee-Nith](https://github.com/Dee-Nith)
- Email: deepak.annanth@gmail.com

## Acknowledgments

- Course instructor and teaching assistants
- Aviation industry datasets and research
- Open source machine learning libraries
