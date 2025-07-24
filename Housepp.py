# Step 1: Import the necessary libraries
# ----------------------------------------
# We need 'pandas' for handling data, 'train_test_split' to prepare our data for training,
# 'LinearRegression' for our AI model, and 'mean_squared_error' to evaluate it.
# We also import 'fetch_california_housing' to get our dataset.

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 2: Load and Explore the Dataset
# ------------------------------------
# Scikit-learn comes with several built-in datasets, which are perfect for learning.
# We'll load the California Housing dataset.

housing = fetch_california_housing()

# The data is stored in a special object. We can convert it to a pandas DataFrame,
# which is like a spreadsheet, to make it easier to view.
# 'housing.data' contains the features (like income, house age).
# 'housing.feature_names' has the column names for the features.
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# 'housing.target' contains the price of the house (our target for prediction).
# We'll add this as a new column in our DataFrame called 'PRICE'.
df['PRICE'] = housing.target

# Let's print the first 5 rows to see what our data looks like.
print("--- First 5 Rows of the Dataset ---")
print(df.head())
print("\n" + "="*40 + "\n")

# Step 3: Prepare the Data for Training
# -------------------------------------
# We need to separate our data into features (X) and the target (y).
# X contains all the columns that we'll use to make a prediction.
# y is the column that we WANT to predict ('PRICE').

# For this simple model, let's use only three features similar to our web app:
# MedInc: Median income in the block (a good proxy for a desirable area)
# AveRooms: Average number of rooms
# AveBedrms: Average number of bedrooms
features = ['MedInc', 'AveRooms', 'AveBedrms']
X = df[features]
y = df['PRICE']

# Now, we split our data into two sets: a training set and a testing set.
# The model will LEARN from the training set.
# We will then TEST its performance on the testing set, which it has never seen.
# 'test_size=0.2' means we'll use 20% of the data for testing.
# 'random_state=42' ensures we get the same split every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"--- Data Split ---")
print(f"Training set size: {len(X_train)} houses")
print(f"Testing set size: {len(X_test)} houses")
print("\n" + "="*40 + "\n")


# Step 4: Create and Train the AI Model
# -------------------------------------
# This is where the magic happens!
# We create an instance of the LinearRegression model.
model = LinearRegression()

# We then 'fit' the model to our training data.
# The model will analyze the relationship between the features (X_train) and the prices (y_train)
# to learn the best "coefficients" for its prediction formula.
print("--- Training the Model ---")
model.fit(X_train, y_train)
print("Model training complete!")
print("\n" + "="*40 + "\n")


# Step 5: See What the Model Learned
# ----------------------------------
# Our model has now learned a formula similar to our web app:
# price = (coeff_1 * feature_1) + (coeff_2 * feature_2) + ... + intercept
# Let's see the values it learned from the data.

print("--- Model's Learned Formula ---")
print(f"Intercept (Base Price): {model.intercept_}")
# We can zip the feature names with their learned coefficients.
learned_coefficients = pd.DataFrame(zip(features, model.coef_), columns=['Feature', 'Coefficient'])
print(learned_coefficients)
print("\nInterpretation: For every 1 unit increase in 'MedInc', the model predicts a price increase of ~$0.44 (in 100,000s), holding other features constant.")
print("\n" + "="*40 + "\n")


# Step 6: Evaluate the Model's Performance
# ----------------------------------------
# Now we use the test set (X_test) that the model has never seen to make predictions.
predictions = model.predict(X_test)

# We can then compare these predictions to the actual prices (y_test)
# to see how well our model did. We'll use Mean Squared Error.
# A lower MSE is better.
mse = mean_squared_error(y_test, predictions)
print("--- Model Evaluation ---")
print(f"Mean Squared Error on Test Data: {mse:.4f}")
# The R-squared score tells us what percentage of the price variation our model can explain. Higher is better.
r2_score = model.score(X_test, y_test)
print(f"R-squared Score on Test Data: {r2_score:.4f}")
print("\n" + "="*40 + "\n")


# Step 7: Make a Prediction on New Data
# -------------------------------------
# Let's pretend we have a new house and want to predict its price.
# We need to provide the data in the same format the model was trained on.
# Let's say: Median Income = 3.5, Avg Rooms = 6, Avg Bedrooms = 1.2
new_house_data = np.array([[3.5, 6, 1.2]]) 

# Use our trained model to predict the price
predicted_price = model.predict(new_house_data)

print("--- Predicting Price for a New House ---")
print(f"Data for new house: Median Income=${new_house_data[0][0]*10000}, Avg Rooms={new_house_data[0][1]}, Avg Bedrooms={new_house_data[0][2]}")
# The housing data target is in units of $100,000
print(f"Predicted Price: ${predicted_price[0] * 100000:,.2f}")

