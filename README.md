# Demand Forecasting with Machine Learning

This project focuses on demand forecasting using machine learning models. The dataset is sourced from GitHub and various models are trained and evaluated to predict future demand.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Preparation](#data-preparation)
  - [Feature Engineering](#feature-engineering)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Forecasting](#forecasting)
  - [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to improve the performance of demand forecasting using various machine learning techniques. The dataset includes historical sales data, and the objective is to predict future sales based on this data.

## Dataset

The dataset used in this project is sourced from [this GitHub repository](https://github.com/Shahroz-Harral/demand-forecasting/blob/main/Dataset/train_0irEZ2H.csv).

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas scikit-learn xgboost matplotlib
```
## Usage

Clone this repository and run the provided Python script:

```bash
python demand_forecasting.py
```
## Methodology
### Data Preparation
1. Load Dataset:
```bash
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/Shahroz-Harral/demand-forecasting/main/Dataset/train_0irEZ2H.csv'
data = pd.read_csv(url)
```
2. Handle Missing Values:
```bash
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/Shahroz-Harral/demand-forecasting/main/Dataset/train_0irEZ2H.csv'
data = pd.read_csv(url)
```
3. Convert week Column to Datetime:
```bash
data['week'] = pd.to_datetime(data['week'], format='%y/%m/%d')
```

4. Extract Relevant Date Features:
```bash
data['year'] = data['week'].dt.year
data['month'] = data['week'].dt.month
data['day'] = data['week'].dt.day
data['week_number'] = data['week'].dt.isocalendar().week
```
5. Define Features and Target:
```bash
features = ['year', 'month', 'day', 'week_number', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku']
target = 'units_sold'
```
6. Split Dataset:
```bash
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
```
### Feature Engineering

1. Creating Interaction Terms:
```bash
X_train['price_diff'] = X_train['total_price'] - X_train['base_price']
X_test['price_diff'] = X_test['total_price'] - X_test['base_price']
```
2. Additional Feature Transformations:
```bash
X_train['total_base_ratio'] = X_train['total_price'] / X_train['base_price']
X_test['total_base_ratio'] = X_test['total_price'] / X_test['base_price']
```
### Model Training and Evaluation

1. Initialize Models:
```bash
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}
```

2. Train and Evaluate Models:
```bash
model_performance = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_performance[model_name] = mean_squared_error(y_test, y_pred)

for model_name, mse in model_performance.items():
    print(f"{model_name}: MSE = {mse:.4f}")
```
### Hyperparameter Tuning

1. Grid Search for Random Forest:
```bash
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=2,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
```
2. Evaluate the Best Model:
```bash
y_pred_best = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, y_pred_best)
print(f"Best Model MSE: {best_mse:.4f}")
```
### Forecasting

1. Create Future Data:
```bash
future_weeks = pd.date_range(start=data['week'].max(), periods=30, freq='W')
future_data = pd.DataFrame({
    'week': future_weeks,
    'year': future_weeks.year,
    'month': future_weeks.month,
    'day': future_weeks.day,
    'week_number': future_weeks.isocalendar().week,
    'store_id': [8091]*30,
    'sku_id': [216418]*30,
    'total_price': [100]*30,
    'base_price': [90]*30,
    'is_featured_sku': [0]*30,
    'is_display_sku': [0]*30
})

future_data['price_diff'] = future_data['total_price'] - future_data['base_price']
future_data['total_base_ratio'] = future_data['total_price'] / future_data['base_price']
future_data = future_data[features + ['price_diff', 'total_base_ratio']]

future_predictions = best_model.predict(future_data)
forecast = pd.DataFrame({'week': future_weeks, 'predicted_units_sold': future_predictions})
print(forecast)
```
### Visualization
1. Combine and Visualize Data:
```bash
import matplotlib.pyplot as plt

# Ensure future_data includes all necessary columns
future_data['units_sold'] = future_predictions
future_data['type'] = 'Predicted'
future_data['week'] = future_weeks

# Combine actual and predicted data for visualization
combined_data = pd.concat([X_test_with_week[['week', 'units_sold', 'type']], future_data[['week', 'units_sold', 'type']]])
combined_data = combined_data.sort_values(by='week')

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot actual data
actual_data = combined_data[combined_data['type'] == 'Actual']
axes[0].plot(actual_data['week'], actual_data['units_sold'], marker='o', linestyle='-', label='Actual')
axes[0].set_title('Actual Units Sold')
axes[0].set_ylabel('Units Sold')
axes[0].legend()
axes[0].grid(True)

# Plot predicted data
predicted_data = combined_data[combined_data['type'] == 'Predicted']
axes[1].plot(predicted_data['week'], predicted_data['units_sold'], marker='x', linestyle='--', label='Predicted')
axes[1].set_title('Predicted Units Sold')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Units Sold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Results
The models are trained and evaluated, and the best model is selected based on Mean Squared Error (MSE). The future demand is forecasted, and the results are visualized using subplots for clarity.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


