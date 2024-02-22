# Setup

## Create virtual environment

python3 -m venv gym-env
source gym-env/bin/activate

## Install

pip install -r requirements.txt

## Deactivate virtual environmment

```
deactivate
```

# Diagnostics

## Warning About render Method Without Specifying Render Mode

This warning is informing you that you've called the render method without specifying a render_mode. The render_mode parameter determines how the environment should be visualized. Since Gym version 0.21.0, specifying the render_mode during environment creation is recommended for clarity and to avoid deprecation warnings.

To address this warning, you should specify the render_mode when you create your Gym environment. If you're working on a headless server (like when using Google Colab) or if you simply want to process the images yourself, you might use render_mode='rgb_array'. For local development where you want to see a window with the environment, you might use render_mode='human'.

Example:

```
env = gym.make('CartPole-v1', render_mode='human')
```

## Warning About Calling step() After Environment is Terminated

This warning indicates that you're attempting to take an action in an environment after it has already signaled that the episode is terminated (i.e., terminated=True). This can happen if you continue to call env.step(action) after the environment has indicated the episode should end, typically because the task was completed or a failure condition was met.

To fix this, make sure your loop or logic that calls env.step(action) checks if the episode has terminated and calls env.reset() to start a new episode. Here is an example of handling this properly:

```
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # replace this with your action selection logic
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation = env.reset()  # Reset the environment to start a new episode
```

In this updated loop, we check for both terminated and truncated signals. The truncated signal indicates the episode was cut off (e.g., due to a time limit), a distinction introduced in newer versions of Gym for clearer episode termination semantics.
