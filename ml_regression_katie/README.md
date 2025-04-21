# ml_regression_katie

# Medical Cost Regression - Predicting Insurance Charges

Link to Notebook: https://github.com/k363m611/applied-ml-katie/blob/main/ml_regression_katie/regression_katie.ipynb

Link to Peer Review: https://github.com/egbogbo11/ml_regression_gbogbo/blob/main/peer_review.md 

## Project Setup

### Make repository in Github and clone to local workspace
```bash
git clone https://github.com/egbogbo11/ml_regression_gbogbo
```

### Create .gitignore file
Ensure the following entries are added to your .gitignore file to exclude unnecessary files from being committed:

```bash
# Python virtual environment folder
.venv/

# Visual Studio Code settings and workspace folder - Ignore entire .vscode folder except settings.json
.vscode/*
!.vscode/settings.json

# Compiled Python files
__pycache__/

# macOS system files
.DS_Store

# Jupyter Notebook checkpoint files
.ipynb_checkpoints
```

### Create and activate virtual environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.\.venv\Scripts\activate
```
### Create requirement.txt and download imports
Add the following libraries to your requirements.txt file:

```bash
pip
setuptools
wheel
jupyterlab
numpy
pandas
pyarrow
matplotlib
seaborn
scklit-learn
```

Install into your active project virtual environment with this command:

```bash
py -m pip install --upgrade pip setuptools wheel
py -m pip install -r requirements.txt
```
### Stage and Push Files to GitHub

Use the following Git commands to stage and commit changes:

```bash
git add .
git commit -m "commit: message"
git push origin main
```
## Project Outline
### Section 1. Load and Inspect the Data
- 1.1 Load the dataset.
- 1.2 Check for missing values and display summary statistics

### Section 2. Data Exploration and Preparation
- 2.1 Explore data patterns and distributions
- 2.2 Handle missing values and encode features

### Section 3. Feature Selection and Justification
- 3.1 Choose input features for predicting the target.
- 3.2 Define X (features) and y (target).

### Section 4. Train a Model - Linear Regression
- 4.1 Split the data
- 4.2 Train Model using Scikit-Learn model.fit
- 4.3 Evaluate Model Performance

### Section 5. Improve the Model or Try Alternatives (Implement Pipelines)
- 5.1 Implement Pipeline 1: Imputer → StandardScaler → Linear Regression
- 5.2 Implement Pipeline 2: Imputer → Polynomial Features (degree=3) → StandardScaler → Linear Regression.
- 5.3 Compare Performance

### Section 6. Final Thoughts and Insights
- 6.1 Summarize findings
- 6.2 Discuss challenges faced
- 6.3 If you have more time, what would you try next?
