#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:50:25 2023

@author: akhilkhajuria
"""


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset from the Excel file
file_path = '/Users/akhilkhajuria/Downloads/merged_dfx_updated.xlsx'
df = pd.read_excel(file_path)

# Select the columns for modeling
columns_to_use = [
    'Order Qty Rate.9',
    'Inv Qty End of Period (Jul 30 - Aug 5)',
    'Traffic Jul 30 - Aug 5',
    'Conversion Rate Jul 30 - Aug 5',
    'Quantity Aug 19 - Aug 26',
    'Traffic Aug 19 - Aug 26',
    'Conversion Rate Aug 19 - Aug 26',
    'Quantity Sep 2 - Sep 9',
    'Traffic Sep 2 - Sep 9',
    'Conversion Rate Sep 2 - Sep 9',
    'Quantity Sep 9 - Sep 16',
    'Traffic Sep 9 - Sep 16',
    'Conversion Rate Sep 9 - Sep 16',
    'Quantity Sep 16 - Sep 23',
    'Traffic Sep 16 - Sep 23',
    'Conversion Rate Sep 16 - Sep 23',
    'Quantity Sep 23 - Oct 2',
	'Traffic Sep 23 - Oct 2',	
    'Conversion Rate Sep 23 - Oct 2',
    'Quantity Oct 2 - Oct 9',
	'Traffic Oct 2 - Oct 9',	
    'Conversion Rate Oct 2 - Oct 9',
    'Quantity Oct 9- Oct 16',
	'Traffic Oct 9 - Oct 16',
	'Conversion Rate Oct 9 - Oct 16',
    'Quantity Oct 16 - Oct 23',
	'Traffic Oct 16 - Oct 23',
	'Conversion Rate Oct 16 - Oct 23',
    
    
]

# Create a new DataFrame with only the selected columns
data = df[columns_to_use]

# Split the data into features and target
X = data.drop('Order Qty Rate.9', axis=1)
y = data['Order Qty Rate.9']

# Create and fit a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X, y)

# Make predictions for 'Order Qty Rate.9'
forecasted_data = rf.predict(X)

# Calculate evaluation metrics
mae = mean_absolute_error(y, forecasted_data)
mse = mean_squared_error(y, forecasted_data)
r2 = r2_score(y, forecasted_data)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Add the forecasted data as a new column in the DataFrame
df['Order Qty Rate.9 Forecast'] = forecasted_data

# Create a line chart to compare actual and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Order Qty Rate.9'], label='Actual Data', marker='o')
plt.plot(df.index, df['Order Qty Rate.9 Forecast'], label='Forecasted Data', marker='x')
plt.xlabel('Time Period')
plt.ylabel('Order Qty Rate.9')
plt.title('Actual vs. Forecasted Order Qty Rate.9')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the chart
plt.show()

# Save the DataFrame with actual and forecasted data to an Excel file
output_file_path = '/Users/akhilkhajuria/Desktop/Actual Sales vs Forecasted Sales Week 11.xlsx'  # Change this to the desired file path
output_df = df[['Order Qty Rate.9', 'Order Qty Rate.9 Forecast']]
output_df.to_excel(output_file_path, index=True)

# Show the chart
plt.show()

