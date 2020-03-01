import numpy as np
import matplotlib.pyplot as plt
from gym_ergojr.sim.single_robot import SingleRobot
from neural_augmented_simulator.arguments import get_args
import os

robot = SingleRobot(debug=False) # 6DOF Reacher Robot
robot.reset()
robot.step()
args = get_args()
rest_interval = 10 * 100
# Load the action file
file_name = "/home/sharath/Desktop/reacher-recordings/10-reacher-actions.npy"
actions = np.load(file_name)
sim_trajectories = np.zeros((actions.shape[0], 12))
for epi in range(actions.shape[0]):

    if epi % rest_interval == 0:  # Take rest after every 10 * 100 steps
        print('Taking Rest at {}'.format(epi))

    '''Perform action and record the observation and the tip position'''
    robot.act2(actions[epi, :])
    robot.step()
    obs = robot.observe()

    sim_trajectories[epi, :] = obs
np.save('/home/sharath/Desktop/reacher-recordings/10-reacher-sim-trajectories.npy', sim_trajectories)

