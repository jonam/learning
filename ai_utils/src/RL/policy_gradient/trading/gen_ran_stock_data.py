import numpy as np
import json

# Generate sample historical data
np.random.seed(0)  # For reproducibility
prices_1 = np.random.normal(100, 10, 1000)
prices_2 = np.random.normal(200, 10, 1000)
historical_data = np.column_stack((prices_1, prices_2))

# Convert to list of lists for JSON serialization
historical_data_list = historical_data.tolist()

# Write to JSON file
with open('data/historical_data.json', 'w') as f:
    json.dump(historical_data_list, f)

# Provide the file path for download
file_path = 'data/historical_data.json'
file_path
