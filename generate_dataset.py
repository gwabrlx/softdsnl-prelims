import pandas as pd
import numpy as np

# Random seed for reproducibility
np.random.seed(42)

# Number of rows
num_rows = 50

# Generate synthetic data
age = np.random.randint(18, 70, num_rows)  # Random ages between 18 and 70
height = np.random.randint(150, 190, num_rows)  # Random heights between 150 and 190 cm
weight = np.random.randint(40, 100, num_rows)  # Random weights between 40 and 100 kg

# Create target variable based on conditions
target = []
for i in range(num_rows):
    if age[i] < 30 and height[i] < 160 and weight[i] < 50:
        target.append("underweight")
    elif 30 <= age[i] <= 50 and 160 <= height[i] <= 175 and 50 <= weight[i] <= 75:
        target.append("normal")
    else:
        target.append("overweight")

# Create the DataFrame
data = pd.DataFrame({
    'feature1': age,
    'feature2': height,
    'feature3': weight,
    'target': target
})

# Save the DataFrame to a CSV file
data.to_csv('dataset.csv', index=False)

# Display the first few rows of the dataset
print(data.head())
