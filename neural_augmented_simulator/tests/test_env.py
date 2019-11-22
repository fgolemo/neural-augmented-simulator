import gym
import numpy as np
import neural_augmented_simulator
env = gym.make('Nas-Pusher-3dof-Backlash01-v1')
obs = env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
    env.render()