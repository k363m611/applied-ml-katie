# ==============================================================
# Project: California Housing Price Prediction
# Author: Katie McGaughey
# Date: 3-15-2025
# Objective: Predict the median house price in California using available housing features.
# ==============================================================

# ==========================
# 📌 SECTION 1: IMPORTS
# ==========================

# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For static visualizations
import seaborn as sns  # For statistical data visualization

# Import the California housing dataset from sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation

# ==========================
# 📌 SECTION 2: LOAD & EXPLORE DATA
# ==========================

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)
data_frame = data.frame  # Convert to pandas DataFrame

# Display first 10 rows
print("🔹 First 10 rows of the dataset:")
print(data_frame.head(10))

# Check for missing values and data types
print("\n🔹 Dataset Info:")
print(data_frame.info())

print("\n🔹 Summary Statistics:")
print(data_frame.describe())

print("\n🔹 Missing Values Per Column:")
print(data_frame.isnull().sum())

# ==========================
# 📌 SECTION 3: VISUALIZE DATA
# ==========================

# Histograms of numerical features
data_frame.hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.show()

# Generate boxenplots for each column
for column in data_frame.columns:
    plt.figure(figsize=(6, 4))
    sns.boxenplot(data=data_frame[column])
    plt.title(f'Boxenplot for {column}')
    plt.show()

# Generate all scatter plots (⚠️ This takes time)
sns.pairplot(data_frame)
plt.show()

# ==========================
# 📌 SECTION 4: FEATURE SELECTION
# ==========================

# Select input features and target variable
features = ['MedInc', 'AveRooms']  # Predictor variables
target = 'MedHouseVal'  # Target variable

# Create feature matrix (X) and target vector (y)
df_X = data_frame[features]
df_y = data_frame[target]

print("\n🔹 Selected Features (X):")
print(df_X.head())

print("\n🔹 Target Variable (y):")
print(df_y.head())

# ==========================
# 📌 SECTION 5: TRAIN A LINEAR REGRESSION MODEL
# ==========================

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

print("\n🔹 Training Set (X_train):")
print(X_train.head())

print("\n🔹 Testing Set (X_test):")
print(X_test.head())

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# ==========================
# 📌 SECTION 6: MODEL EVALUATION
# ==========================

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)  # R² Score
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root Mean Squared Error

# Print evaluation results
print("\n🔹 Model Performance Metrics:")
print(f'R² Score: {r2:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
