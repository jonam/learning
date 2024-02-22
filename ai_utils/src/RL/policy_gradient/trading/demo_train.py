import gym
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gym
import numpy as np
from gym import spaces

from policy_network import PolicyNetwork

class PortfolioEnv(gym.Env):
    """A simple trading environment for portfolio management."""
    metadata = {'render.modes': ['human']}

    def __init__(self, historical_data, initial_investment=10000):
        super(PortfolioEnv, self).__init__()
        
        # Historical financial data, e.g., daily prices of assets
        self.data = historical_data
        
        # Initial investment amount
        self.initial_investment = initial_investment
        
        # Define action and observation space
        # Actions of the format: [proportion of asset 1, ..., proportion of asset N]
        # Assuming a simplified scenario with 2 assets for demonstration
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Observation space: historical prices and financial indicators
        # This is a simplified representation
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.data[0]),), dtype=np.float32)
        
        self.current_step = 0

    def step(self, action):
        # Implement the logic to move the environment to the next time step based on the action
        # Calculate reward, next_state, and done
        next_state = self.data[self.current_step]
        reward = self.calculate_reward(action)
        done = self.current_step == len(self.data) - 1
        self.current_step += 1
        return next_state, reward, done, {}

    def reset(self):
        # Reset the environment state to an initial state
        self.current_step = 0
        return self.data[self.current_step]

    def render(self, mode='human', close=False):
        # Render the environment to the screen (optional for this example)
        pass

    def calculate_reward(self, action):
        # Calculate the reward based on the action taken
        # This is a placeholder for the actual logic
        return np.random.rand()  # Placeholder for actual reward calculation

def train(env, policy, optimizer, episodes=1000):
    total_rewards = []  # To store total rewards per episode for averaging
    total_steps = []  # To store the number of steps per episode

    for episode in range(episodes):
        #state = env.reset()
        state = env.reset()
        log_probs = []
        rewards = []
        steps = 0  # Reset steps counter for the new episode
        done = False

        while not done:
            action, log_prob = policy.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            log_probs.append(log_prob)
            rewards.append(reward)
            steps += 1  # Increment step count

            if done:
                break

        total_rewards.append(sum(rewards))  # Total reward for this episode
        total_steps.append(steps)  # Total steps for this episode

        # Calculate discounted rewards
        discounts = np.array([0.99**i for i in range(len(rewards))])
        rewards = np.array(rewards)
        discounted_rewards = rewards * discounts
        discounted_rewards = discounted_rewards[::-1].cumsum()[::-1]

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
   
        # Example loss calculation that ensures the loss tensor is part of the computation graph
        #loss = 0  # Initialize a scalar for accumulating loss; actual tensor creation will be in the loop
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        for reward, log_prob in zip(rewards, log_probs):
            # Assuming log_prob is a tensor with requires_grad=True
            # And reward is converted to a tensor in a manner that keeps requires_grad=True for operations involving it
            reward_tensor = torch.tensor(reward, dtype=torch.float32)  # Convert to tensor if necessary
            #log_prob = torch.tensor(log_prob, dtype=torch.float32)  # Convert to tensor if necessary
            loss = loss - log_prob * reward  # Accumulate loss directly
        
        # Now, loss.backward() should work because loss is derived from model outputs and rewards

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f'Episode {episode}, Loss: {loss.item()}, Average Reward: {np.mean(total_rewards[-50:])}, Steps: {steps}')

    return total_rewards, total_steps

torch.set_num_threads(8)

# Initialize environment, policy, and optimizer as before

# Sample historical data for two assets over 100,000 events, one a minute
# Path to your downloaded JSON file
#file_path = 'data/historical_data.json'
file_path = 'data/historical_data_gbm.json'

# Read the JSON file into a Python list
with open(file_path, 'r') as f:
    historical_data_list = json.load(f)

# Convert the list to a NumPy array
historical_data = np.array(historical_data_list)

# Sample historical data for two assets over 5 days
# Format: [[price of asset 1 on day 1, price of asset 2 on day 1], ...]
"""
historical_data = np.array([
    [100, 200],  # Prices on day 1
    [105, 195],  # Prices on day 2
    [110, 190],  # Prices on day 3
    [115, 185],  # Prices on day 4
    [120, 180]   # Prices on day 5
])
"""

env = PortfolioEnv(historical_data)

# env = gym.make('CartPole-v1')
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Train the policy and capture performance metrics
total_rewards, total_steps = train(env, policy, optimizer)

torch.save(policy.state_dict(), 'policy_model.pth')
