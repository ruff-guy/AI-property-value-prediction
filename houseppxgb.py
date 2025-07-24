# Step 1: Import the necessary libraries
# ----------------------------------------
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # We'll keep this to compare against
import xgboost as xgb # Import the XGBoost library
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 2: Load and Prepare the Dataset (Same as before)
# -----------------------------------------------------
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- For Comparison: Quickly retrain our old Linear Regression model ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_r2_score = lr_model.score(X_test, y_test)


# --- THIS IS THE UPGRADE ---
# Step 3: Create and Train the XGBoost Model
# ------------------------------------------
# We instantiate the XGBRegressor model.
# n_estimators: The number of "trees" (models) to build in sequence.
# learning_rate: How much each new tree corrects the mistakes of the previous one.
# random_state: Ensures we get the same result every time we run it.
print("--- Training the XGBoost Model ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

# We fit the model to our training data, just like before.
xgb_model.fit(X_train, y_train)
print("XGBoost model training complete!")
print("\n" + "="*50 + "\n")


# Step 4: Evaluate the New Model's Performance
# --------------------------------------------
# Let's see how much better our R-squared score is now.
xgb_r2_score = xgb_model.score(X_test, y_test)
xgb_mse = mean_squared_error(y_test, xgb_model.predict(X_test))

print("--- XGBoost Model Evaluation ---")
print(f"Mean Squared Error: {xgb_mse:.4f}")
print(f"R-squared Score: {xgb_r2_score:.4f}")
print("\n" + "="*50 + "\n")


# Step 5: The Final Showdown!
# ---------------------------
print("--- MODEL PERFORMANCE COMPARISON ---")
print(f"Linear Regression R-squared: {lr_r2_score:.4f}  (Explains ~{lr_r2_score:.0%} of price variation)")
print(f"XGBoost R-squared:           {xgb_r2_score:.4f}  (Explains ~{xgb_r2_score:.0%} of price variation)")
print("\n" + "="*50 + "\n")

print("The XGBoost model provides a significant improvement, capturing much more of the complexity in the housing market!")

