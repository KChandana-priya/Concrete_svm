# Concrete Price Prediction using Support Vector Machine (SVM)

## Introduction
This project focuses on predicting the price of concrete using machine learning techniques, specifically the Support Vector Machine (SVM) model. The objective is to develop a regression model that accurately estimates concrete prices based on various features such as material composition, location, and market conditions.

## Dataset
The dataset used in this project consists of historical concrete price data, including factors influencing the cost. The dataset includes the following features:
- **Cement Content (kg/m³)**: The amount of cement used in the mix.
- **Water Content (kg/m³)**: The water-to-cement ratio.
- **Aggregate Content (kg/m³)**: Coarse and fine aggregates.
- **Admixtures (kg/m³)**: Additional chemical additives.
- **Curing Age (days)**: The number of days the concrete has been cured.
- **Market Demand Index**: A numerical indicator of demand.
- **Location Factors**: Region-based pricing adjustments.
- **Concrete Strength (MPa)**: Measured compressive strength.
- **Target Variable - Price per Cubic Meter ($/m³)**: The cost of concrete.

## Data Preprocessing
To ensure data quality, preprocessing steps include:
1. **Handling Missing Values**: Imputation techniques for incomplete records.
2. **Feature Scaling**: Standardization using MinMaxScaler or StandardScaler.
3. **Encoding Categorical Variables**: One-hot encoding for non-numeric features.
4. **Data Splitting**: Training (80%) and testing (20%) split.
5. **Outlier Detection**: Removing anomalies using IQR or Z-score methods.

## Model Selection & Training
### Support Vector Regression (SVR)
Support Vector Regression (SVR) with different kernel functions is implemented:
- **Linear Kernel**: Basic linear relationship.
- **Polynomial Kernel**: Captures nonlinear relationships.
- **RBF (Radial Basis Function) Kernel**: Handles complex patterns.

### Hyperparameter Tuning
Grid Search Cross-Validation (GridSearchCV) is used to optimize:
- **C (Regularization Parameter)**
- **Epsilon (ε-insensitive loss function margin)**
- **Gamma (for RBF and polynomial kernels)**

### Training Execution
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Creating pipeline with standardization and SVR model
model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
model.fit(X_train, y_train)
```

## Evaluation & Results
The trained model is evaluated using regression metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²) Score**
- **Prediction vs Actual Visualization (Scatter Plot)**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Model predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Scatter plot of actual vs predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Price ($/m³)")
plt.ylabel("Predicted Price ($/m³)")
plt.title("Actual vs Predicted Concrete Prices")
plt.show()
```

## Requirements
To run the project, install the dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Execution
To train and test the model, run:
```bash
python concrete_price_prediction.py
```
