import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),  # CartPole observation space is 4
            nn.ReLU(),
            nn.Linear(128, 2)  # CartPole action space is 2
        )
    
    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def train(env, policy, optimizer, episodes=1000):
    total_rewards = []  # To store total rewards per episode for averaging
    total_steps = []  # To store the number of steps per episode

    for episode in range(episodes):
        #state = env.reset()
        state,_ = env.reset()
        log_probs = []
        rewards = []
        steps = 0  # Reset steps counter for the new episode
        done = False

        while not done:
            action, log_prob = policy.act(state)
            next_state, reward, done, _, _ = env.step(action)
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

        # Calculate loss
        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)
        loss = torch.cat(loss).sum()

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f'Episode {episode}, Loss: {loss.item()}, Average Reward: {np.mean(total_rewards[-50:])}, Steps: {steps}')

    return total_rewards, total_steps

# Initialize environment, policy, and optimizer as before
env = gym.make('CartPole-v1')
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Train the policy and capture performance metrics
total_rewards, total_steps = train(env, policy, optimizer)

