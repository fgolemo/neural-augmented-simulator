import gym
import numpy as np
import neural_augmented_simulator

def match_env(ev1, ev2):
    # set env1 (simulator) to that of env2 (real robot)
    ev1.unwrapped.set_state(
        ev2.unwrapped.sim.model.data.qpos.ravel(),
        ev2.sim.model.data.qvel.ravel()
    )


env_sim = gym.make('Nas-Pusher-3dof-Vanilla-v1')
obs_sim = env_sim.reset()

env_real = gym.make('Nas-Pusher-3dof-Backlash01-v1')
obs_real = env_real.reset()
# match_env(env_sim, env_real)
action = np.ones((env_sim.action_space.shape[0]))
for _ in range(2):
    obs_sim, rew, done, _ = env_sim.step(action)
    print('==================')
    obs_real, _, _, _ = env_real.step(action)
    print(f'Simulator Observation is : \n {np.around(obs_sim, 3)}')
    print(f'Real Observation is : \n {np.around(obs_real, 3)}')
    if done:
        env_sim.reset()
        env_real.reset()
    # env_sim.render()