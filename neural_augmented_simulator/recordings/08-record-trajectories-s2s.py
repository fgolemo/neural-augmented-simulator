import numpy as np
import os
import gym
import neural_augmented_simulator
from neural_augmented_simulator.arguments import get_args

args = get_args()
env = gym.make('Nas-Pusher-3dof-Vanilla-v1')
obs = env.reset()

file_path = os.getcwd() + '/data/robot_recordings/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

if not os.path.isdir(file_path):
    os.makedirs(file_path)

np.random.seed(seed=args.seed)
total_steps = args.total_steps
rest_interval = args.rest_interval
freq = args.freq
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, obs.shape[0])) # Dont hard code it.
actions = np.zeros((total_steps, env.action_space.shape[0]))  # Don't hard code it.
env.reset()
end_pos = []

for epi in range(total_steps):

    if epi % rest_interval == 0:
        print('Taking Rest at {}'.format(epi))
        env.reset()
    if epi % steps_until_resample == 0:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
    actions[epi, :] = action
    obs, _, _, _ = env.step(actions[epi, :])
    end_pos.append(obs[6:8])
    sim_trajectories[epi, :] = obs

np.save(file_path + 'end_positions.npy', end_pos)
np.savez(file_path + '/action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

