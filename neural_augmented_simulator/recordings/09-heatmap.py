import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from neural_augmented_simulator.arguments import get_args


args = get_args()
file_path = os.getcwd() + '/data/robot_recordings/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

pos_action_noise = np.load(file_path + 'goals_and_positions.npz')
actions = np.load(file_path + 'actions_trajectories.npz')
print(actions["actions"][40:100])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
goals = pos_action_noise['goals']
position = pos_action_noise['positions']
print(position.shape)

ax1.hist2d(position[:, 0], position[:, 1], bins=100)
file_path = os.getcwd() + '/data/robot_recordings/{}/freq{}/motor-babbling/'.format(args.env_name, args.freq)
ax1.set_title('Goal Babbling')
end_pos = np.load(file_path + 'end_positions.npy'.format(args.freq))
position = end_pos
print(position.shape)
ax2.hist2d(position[:, 0], position[:, 1], bins=100)
ax2.set_title('Motor Babbling')
fig.suptitle("2D Histogram of end effector positions | Freq - {}".format(args.freq))
plt.savefig('freq-{}.png'.format(args.freq), figsize=(20, 10))
ax3.hist2d(goals[:, 0], goals[:, 1])
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
plt.show()
