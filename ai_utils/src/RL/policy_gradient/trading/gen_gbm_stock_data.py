import numpy as np
import json

def generate_gbm_prices(S0, mu, sigma, T, steps):
    """
    Generate stock prices using Geometric Brownian Motion.

    Parameters:
    - S0: Initial stock price
    - mu: Expected return (drift coefficient)
    - sigma: Volatility (standard deviation of returns)
    - T: Time period in years
    - steps: Number of steps in the simulation

    Returns:
    - A NumPy array of simulated stock prices.
    """
    dt = T/steps
    t = np.linspace(0, T, steps)
    W = np.random.standard_normal(size = steps)
    W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) # geometric brownian motion
    return S

# Set parameters for the GBM model
S0 = 100  # Initial price
mu = 0.1  # Expected annual return
sigma = 0.2  # Annual volatility
T = 1/252/390 * 100000  # Adjust T to reflect 100,000 minutes in fraction of a trading years
steps = 100000  # Number of steps

# Generate two sets of stock prices
np.random.seed(0)  # For reproducibility
prices_1 = generate_gbm_prices(S0, mu, sigma, T, steps)
prices_2 = generate_gbm_prices(S0*2, mu, sigma, T, steps)  # Different initial price

historical_data = np.column_stack((prices_1, prices_2))

# Convert to list of lists for JSON serialization
historical_data_list = historical_data.tolist()

# Write to JSON file
with open('data/historical_data_gbm.json', 'w') as f:
    json.dump(historical_data_list, f)

# Provide the file path for download
file_path = 'data/historical_data_gbm.json'
print(file_path)
