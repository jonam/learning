import gym
env = gym.make('CartPole-v1', render_mode='human')
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # replace this with your action selection logic
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation = env.reset()  # Reset the environment to start a new episode
