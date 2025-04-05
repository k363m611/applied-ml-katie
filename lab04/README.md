# Lab 4 Project (Titanic)

# Objective
Predict a continuous numeric target like fare using the Titanic dataset by building a regression model using key features such as family size, sex, and age.

# What features were used?
'Sex' 

# Models Created
Linear, Ridge, ElasticNet, and Polynomial Regression was used in the project.

# Outline
### Section 1. Load and Inspect the Data
- 1.1 Load the dataset.

### Section 2. Data Exploration and Preparation
- Impute missing values for age using median
- Drop rows with missing fare (or impute if preferred)
- Create numeric variables (e.g., family_size from sibsp + parch + 1)
- Optional - convert categorical features (e.g. sex, embarked) if you think they might help your prediction model. (We do not know relationships until we evaluate things.)

### Section 3. Feature Selection and Justification
- Choose three input features for predicting the target. Justify your selection with reasoning.
- Define X (features) and y (target).
- 
### Section 4. Train a Classification Model
- 4.1 Split the data
- 4.2 Train and Evaluate Linear Regression Models (all 4 cases)
- 4.3 Report Performance

### Section 5. Compare Alternative Models
- 5.1 Ridge Regression (L2 penalty)
- 5.2 Elastic Net (L1 + L2 combined)
- 5.3 Polynomial Regression
- 5.4 Visualize Polynomial Cubic Fit (for 1 input feature)
- 5.5 Compare All Models
- 5.6 Visualize Higher Order Polynomial (for 1 same input feature)
