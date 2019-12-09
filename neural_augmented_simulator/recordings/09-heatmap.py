import pickle
import os

import gym
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from neural_augmented_simulator.arguments import get_args

env = gym.make('Nas-Pusher-3dof-Vanilla-v1')
args = get_args()
file_path = os.getcwd() + '/data/robot_recordings/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)
print(file_path)
# pos_action_noise = np.load(file_path + 'goals_and_positions.npz')
actions = np.load(file_path + 'actions_trajectories.npz')
fig, (ax1, ax2) = plt.subplots(1, 2)
print(actions['actions'])
env.reset()
pos = []
for action in actions['actions'][:1000000]:
    new_obs, _, _, _ = env.step(action)
    pos.append(new_obs[6:8])
position = np.asarray(pos)
print(position.shape)
ax1.hist2d(position[:, 0], position[:, 1], bins=100)
file_path = os.getcwd() + '/data/robot_recordings/{}/freq{}/motor-babbling/'.format(args.env_name, args.freq)
print(file_path)
ax1.set_title('Goal Babbling')
end_pos = np.load(file_path + 'end_positions.npy'.format(args.freq))
position_mb = end_pos
print(position_mb.shape)
ax2.hist2d(position_mb[:1000000, 0], position_mb[:1000000, 1], bins=100)
ax2.set_title('Motor Babbling')
fig.suptitle("2D Histogram of end effector positions | Freq - {}".format(args.freq))
plt.savefig('freq-{}.png'.format(args.freq), figsize=(20, 10))
# ax3.set_xlim(-1, 1)
# ax3.set_ylim(-1, 1)
# ax1.set_xlim(-1, 1)
# ax1.set_ylim(-1, 1)
# ax2.set_xlim(-1, 1)
# ax2.set_ylim(-1, 1)
plt.show()
