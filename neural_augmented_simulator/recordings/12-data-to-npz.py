import numpy as np
from neural_augmented_simulator.arguments import get_args
import os

# Import files
args = get_args()
actions = np.load('/home/sharath/Desktop/reacher-recordings/02-reacher-actions.npy')
real_trajectory = np.load('/home/sharath/Desktop/reacher-recordings/02-reacher-real-trajectories.npy')
sim_trajectory = np.load('/home/sharath/Desktop/reacher-recordings/02-reacher-sim-trajectories.npy')

np.savez('/home/sharath/Desktop/reacher-recordings/02-reacher-recordings.npz', actions=actions, sim_trajectories=sim_trajectory,
         real_trajectories=real_trajectory)