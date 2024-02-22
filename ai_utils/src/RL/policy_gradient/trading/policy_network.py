import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),  # CartPole observation space is 4
            nn.ReLU(),
            nn.Linear(128, 2)  # CartPole action space is 2
        )

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)
        #return x

    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        with torch.no_grad():  # Ensure model is in inference mode
            raw_scores = self.forward(state)
        # Normalize raw scores to get allocation percentages
        positive_scores = torch.relu(raw_scores)  # Ensure non-negative
        allocation_percentages = positive_scores / positive_scores.sum(dim=1, keepdim=True)  # Normalize
        return allocation_percentages.numpy().flatten()  # Return as a flat numpy array


"""
    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
"""
