import time

import gym
import numpy as np
import neural_augmented_simulator

def match_env(ev1, ev2):
    # set env1 (simulator) to that of env2 (real robot)
    print(ev2.unwrapped.sim.data.qpos.ravel())
    ev1.unwrapped.set_state(
        ev2.unwrapped.sim.data.qpos.ravel()[:7],
        ev2.unwrapped.sim.data.qvel.ravel()[:7]
    )


env_sim = gym.make('Nas-Pusher-3dof-Vanilla-v1')
obs_sim = env_sim.reset()
env_real = gym.make('Nas-Pusher-3dof-Backlash01-v2')
obs_real = env_real.reset()

match_env(env_sim, env_real)
action = np.ones((env_sim.action_space.shape[0]))
sim_ob = []
for _ in range(10000):
    action = env_sim.action_space.sample()
    obs_sim, rew, done, _ = env_sim.step(action)
    sim_ob.append(list(obs_sim[6:8]))

    # obs_real, _, _, _ = env_real.step(action)
    # env_real.render()
    # print(f'Simulator Observation is : \n {np.around(obs_sim, 3)}')
    # print(f'Real Observation is : \n {np.around(obs_real, 3)}')
    if done:
        env_sim.reset()
        env_real.reset()
    # env_sim.render()
