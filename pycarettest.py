# Import libraries
from pycaret.regression import *
import pandas as pd

# Load the data
data = pd.read_csv("data/test.csv")

# Check the first few rows of the data
print(data.head())

# Initialize PyCaret for regression
exp1 = setup(data=data, target='tip', session_id=123, 
             categorical_features=['sex', 'smoker', 'day', 'time'],
             numeric_features=['total_bill', 'size'], 
             verbose=False)

# Compare and select the best model
best_model = compare_models()

# Train the best model
trained_model = finalize_model(best_model)

# Plot feature importance
plot_model(trained_model, plot='feature')

# Create new data for prediction
new_data = pd.DataFrame({
    'total_bill': [20.5],
    'sex': ['Male'],
    'smoker': ['No'],
    'day': ['Sun'],
    'time': ['Dinner'],
    'size': [2]
})

# Predict the tip for the new data
predictions = predict_model(trained_model, new_data)
print(predictions)
