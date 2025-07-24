# Step 1: Import the necessary libraries
# ----------------------------------------
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 2: Load the Dataset
# ------------------------------------
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# --- THIS IS THE KEY CHANGE ---
# Step 3: Prepare the Data for Training (Using ALL features)
# ----------------------------------------------------------
# Instead of picking a few columns, we'll use all of them.
# 'X' will be our DataFrame minus the 'PRICE' column.
# 'y' remains the 'PRICE' column.

X = df.drop('PRICE', axis=1) # Use all columns except the price as features
y = df['PRICE']

# We still split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Using all 8 features for training ---")
print(X.columns.tolist())
print("\n" + "="*40 + "\n")


# Step 4: Create and Train the AI Model
# -------------------------------------
# The process is exactly the same, but now the model has more information to learn from.
model = LinearRegression()
model.fit(X_train, y_train)
print("--- Model training complete! ---")
print("\n" + "="*40 + "\n")


# Step 5: See What the Model Learned
# ----------------------------------
# Let's look at the new coefficients the model learned for ALL the features.
print("--- Model's Learned Coefficients ---")
learned_coefficients = pd.DataFrame(zip(X.columns, model.coef_), columns=['Feature', 'Coefficient'])
print(learned_coefficients)
print("\n" + "="*40 + "\n")


# Step 6: Evaluate the Model's Performance
# ----------------------------------------
# Let's see if our R-squared score has improved!
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2_score = model.score(X_test, y_test)

print("--- New Model Evaluation ---")
print(f"Mean Squared Error on Test Data: {mse:.4f}")
print(f"R-squared Score on Test Data: {r2_score:.4f}")
print("\n" + "="*40 + "\n")

print("--- Comparison ---")
print("Previous R-squared (3 features): 0.4842")
print(f"New R-squared (8 features):    {r2_score:.4f}")
print("\nBy giving the model more information (especially location), we've significantly improved its ability to explain house prices!")

