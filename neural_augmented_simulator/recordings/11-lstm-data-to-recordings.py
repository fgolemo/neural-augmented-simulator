import numpy as np
import matplotlib.pyplot as plt
from gym_ergojr.sim.single_robot import SingleRobot
from neural_augmented_simulator.arguments import get_args
import os

robot = SingleRobot(debug=False) # 6DOF Reacher Robot
args = get_args()