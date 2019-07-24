import math

import h5py
from fuel.datasets.hdf5 import H5PYDataset
import gym
import gym_reacher2
import numpy as np
from tqdm import tqdm
# from hyperdash import Experiment

env_sim = gym.make('Reacher2-v0')  # sim
env_real = gym.make('Reacher2-v0')  # real

env_sim.env._init(  # sim
    # colored=True
)
env_sim.reset()

env_real.env._init(
    xml="reacher-backlash.xml",
    # colored=False
)
env_real.reset()
observation_dim = int(env_sim.observation_space.shape[0])
action_dim = int(env_sim.action_space.shape[0])
rng = np.random.RandomState(seed=22)
max_steps = 2000
episode_length = 50  # how many steps max in each rollout?
split = 0.90
action_steps = 1

# Creating the h5 dataset
name = '/Tmp/alitaiga/mujoco_reacher_test.h5'
assert 0 < split <= 1
size_train = math.floor(max_steps * split)
size_val = math.ceil(max_steps * (1 - split))
f = h5py.File(name, mode='w')
observations = f.create_dataset('obs', (size_train+size_val, episode_length, observation_dim), dtype='float32')
actions = f.create_dataset('actions', (size_train+size_val, episode_length, action_dim), dtype='float32')
s_transition_obs = f.create_dataset('s_transition_obs', (size_train+size_val, episode_length, observation_dim), dtype='float32')
r_transition_obs = f.create_dataset('r_transition_obs', (size_train+size_val, episode_length, observation_dim), dtype='float32')
reward_sim = f.create_dataset('reward_sim', (size_train+size_val,episode_length), dtype='float32')
reward_real = f.create_dataset('reward_real', (size_train+size_val,episode_length), dtype='float32')

split_dict = {
    'train': {
        'obs': (0, size_train),
        'actions': (0, size_train),
        's_transition_obs': (0, size_train),
        'r_transition_obs': (0, size_train),
        'reward_sim': (0, size_train),
        'reward_real': (0, size_train)
    },
    'valid': {
        'obs': (size_train, size_train+size_val),
        'actions': (size_train, size_train+size_val),
        's_transition_obs': (size_train, size_train+size_val),
        'r_transition_obs': (size_train, size_train+size_val),
        'reward_sim': (size_train, size_train+size_val),
        'reward_real': (size_train, size_train+size_val),
    }
}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

def match_env(ev1, ev2):
    # set env1 (simulator) to that of env_real (real robot)
    ev1.unwrapped.set_state(
        ev2.unwrapped.model.data.qpos.ravel(),
        ev2.unwrapped.model.data.qvel.ravel()
    )

i = 0
# exp = Experiment("dataset pusher")

for i in tqdm(range(max_steps)):
    # exp.metric("episode", i)
    obs = env_sim.reset()
    obs2 = env_real.reset()
    match_env(env_sim, env_real)

    for j in range(episode_length):
        # env.render()
        # env_real.render()

        # if j % action_steps == 0:
        action = env_sim.action_space.sample()
        new_obs, reward, done, info = env_sim.step(action.copy())
        new_obs2, reward2, done2, info2 = env_real.step(action.copy())

        observations[i, j, :] = obs2.astype('float32')
        actions[i, j, :] = action.astype('float32')
        s_transition_obs[i, j, :] = new_obs.astype('float32')
        r_transition_obs[i, j, :] = new_obs2.astype('float32')
        reward_sim[i] = reward.astype('float32')
        reward_real[i] = reward2.astype('float32')

        # we have to set the state to be the old state in the next timestep.
        # Otherwise the old state is constant
        obs2 = new_obs2

        match_env(env_sim, env_real)
        if done2:
            break

    if i % 100 == 0:
        print ("{} done".format(i))
        f.flush()

f.flush()
f.close()
print('Created h5 dataset with {} elements'.format(max_steps))
