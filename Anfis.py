# Importing necessary libraries for the refined ANFIS implementation
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skfuzzy.cluster import cmeans
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'Desktop/Machine Learning/NN/processed_dataset.csv'
data = pd.read_csv(file_path)

# Extract features and target
input_columns = ['distance', 'angle', 'operation_mode', 'other_moving_ac']
output_column = 'taxi_time'

X = data[input_columns].values
y = data[output_column].values

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Apply Fuzzy C-Means Clustering
n_clusters = 3
cntr, u, _, _, _, _, _ = cmeans(X_train.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

# Train a Linear Regression model to simulate ANFIS
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Rescale predictions back to original scale
y_train_pred_rescaled = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred_rescaled = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

# Rescale ground truth
y_train_rescaled = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate performance
rmse_train = np.sqrt(mean_squared_error(y_train_rescaled, y_train_pred_rescaled))
mae_train = mean_absolute_error(y_train_rescaled, y_train_pred_rescaled)

rmse_test = np.sqrt(mean_squared_error(y_test_rescaled, y_test_pred_rescaled))
mae_test = mean_absolute_error(y_test_rescaled, y_test_pred_rescaled)

# Calculate MRE
mre_test = (mae_test / np.mean(y_test_rescaled)) * 100

# Performance results
performance = {
    "Train RMSE": rmse_train,
    "Train MAE": mae_train,
    "Test RMSE": rmse_test,
    "Test MAE": mae_test,
    "Test MRE (%)": mre_test,
}

# Plotting actual vs. predicted for train data
plt.figure(figsize=(12, 6))
plt.scatter(y_train_rescaled, y_train_pred_rescaled, alpha=0.6, label='Train Data')
plt.plot([min(y_train_rescaled), max(y_train_rescaled)], [min(y_train_rescaled), max(y_train_rescaled)],
         color='red', linestyle='--', label='Ideal Fit')
plt.title("Actual vs Predicted - Train Data")
plt.xlabel("Actual Taxi Time")
plt.ylabel("Predicted Taxi Time")
plt.legend()
plt.show()

# Plotting actual vs. predicted for test data
plt.figure(figsize=(12, 6))
plt.scatter(y_test_rescaled, y_test_pred_rescaled, alpha=0.6, label='Test Data')
plt.plot([min(y_test_rescaled), max(y_test_rescaled)], [min(y_test_rescaled), max(y_test_rescaled)],
         color='red', linestyle='--', label='Ideal Fit')
plt.title("Actual vs Predicted - Test Data")
plt.xlabel("Actual Taxi Time")
plt.ylabel("Predicted Taxi Time")
plt.legend()
plt.show()

# Display performance
performance
