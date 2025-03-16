{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "project-title",
   "metadata": {},
   "source": [
    "# 📌 California Housing Price Prediction\n",
    "**Author:** Katie McGaughey  \n",
    "**Date:** 3-15-2025  \n",
    "**Objective:** Predict the median house price in California using available housing features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "imports",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================\n",
    "# 📌 SECTION 1: IMPORT LIBRARIES\n",
    "# ==========================\n",
    "\n",
    "import pandas as pd  # Data handling\n",
    "import numpy as np  # Numeric computing\n",
    "import matplotlib.pyplot as plt  # Visualization\n",
    "import seaborn as sns  # Statistical visualization\n",
    "\n",
    "from sklearn.datasets import fetch_california_housing  # Load dataset\n",
    "from sklearn.model_selection import train_test_split  # Data splitting\n",
    "from sklearn.linear_model import LinearRegression  # ML model\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "load-data",
   "metadata": {},
   "source": [
    "## 📌 SECTION 2: LOAD & EXPLORE DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "load-dataset",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load California housing dataset\n",
    "data = fetch_california_housing(as_frame=True)\n",
    "data_frame = data.frame  # Convert to Pandas DataFrame\n",
    "\n",
    "# Display first 10 rows\n",
    "data_frame.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "data-summary",
   "metadata": {},
   "source": [
    "## 📌 SECTION 3: DATA SUMMARY & MISSING VALUES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "check-data",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values and data types\n",
    "print(data_frame.info())\n",
    "print(data_frame.describe())\n",
    "print(data_frame.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "visualizations",
   "metadata": {},
   "source": [
    "## 📌 SECTION 4: VISUALIZE DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "visualize-data",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate histograms for numerical columns\n",
    "data_frame.hist(bins=30, figsize=(12, 8))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "feature-selection",
   "metadata": {},
   "source": [
    "## 📌 SECTION 5: FEATURE SELECTION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "select-features",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['MedInc', 'AveRooms']\n",
    "target = 'MedHouseVal'\n",
    "\n",
    "df_X = data_frame[features]\n",
    "df_y = data_frame[target]\n",
    "\n",
    "df_X.head(), df_y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "train-model",
   "metadata": {},
   "source": [
    "## 📌 SECTION 6: TRAIN A LINEAR REGRESSION MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "train-test-split",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split dataset into training and testing sets (80% train, 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Create and train the Linear Regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "make-predictions",
   "metadata": {},
   "source": [
    "## 📌 SECTION 7: MAKE PREDICTIONS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "predict",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "y_pred[:5]"  # Show first 5 predictions"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "model-evaluation",
   "metadata": {},
   "source": [
    "## 📌 SECTION 8: MODEL EVALUATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "evaluate-model",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate model\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "rmse = mean_squared_error(y_test, y_pred, squared=False)\n",
    "\n",
    "print(f'R²: {r2:.2f}')\n",
    "print(f'MAE: {mae:.2f}')\n",
    "print(f'RMSE: {rmse:.2f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
