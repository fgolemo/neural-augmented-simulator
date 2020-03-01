import os
import numpy as np
from neural_augmented_simulator.arguments import get_args


args = get_args()

# Load the data.
file_name = os.getcwd() + '/data/reacher-recordings/{}-reacher-recordings.npz'.format(args.freq)
recordings = np.load(file_name)

# Print the actions, trajectories etc.
actions = recordings["actions"]
sim_trajectories = recordings["sim_trajectories"]
real_trajectories = recordings["real_trajectories"]

print(' Actions are \n {}'.format(actions[:100]))
print('Sim trajectories are \n {}'.format(np.around(sim_trajectories[:100], 3)))
print('real trajectories are \n {}'.format(np.around(real_trajectories[:100], 3)))

# Print Shapes
print('Actions shape is {}'.format(actions.shape))
print('Sim trajectories shape is {}'.format(sim_trajectories.shape))
print('Real trajectories shape is {}'.format(real_trajectories.shape))