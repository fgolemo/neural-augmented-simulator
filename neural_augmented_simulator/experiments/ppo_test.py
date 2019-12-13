import os
import gym
import numpy as np

from neural_augmented_simulator.common.agent.ppo_agent import PPO
from neural_augmented_simulator.common.agent.actor_critic import Memory
from PIL import Image
from neural_augmented_simulator.arguments import get_args
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_args()
os.environ['approach'] = args.approach
os.environ['variant'] = args.variant
os.environ['task'] = args.task
############## Hyperparameters ##############
env_name = args.env_name
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

n_episodes = 25  # num of episodes to run
max_timesteps = 100  # max timesteps in one episode
render = True  # render the environment
save_gif = False  # png images are saved in gif folder

# filename and directory to load model from

action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0003  # parameters for Adam optimizer
betas = (0.9, 0.999)
#############################################

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std,
          lr, betas, gamma, K_epochs, eps_clip)
seed = [1, 2, 3]
freq = [1, 2, 10]
approach = ["motor-babbling", "goal-babbling"]
avg_rew = []
for f in freq:
    for app in approach:
        app_rew = []
        os.environ['approach'] = str(app)
        os.environ['variant'] = str(f)
        for s in seed:
            env = gym.make(env_name)
            ep_reward = 0
            env.seed(100 + s)
            filename = '/ppo_{}_{}_{}_{}.pth'.format(args.env_name,
                                                     f,
                                                     app,
                                                     s)

            directory = os.getcwd() + \
                        '/trained_models/ppo/{}/Variant-{}'.format(app, f)
            print(filename)
            torch.manual_seed(100 + s)
            np.random.seed(100 + s)
            ppo.policy_old.load_state_dict(torch.load(
                directory + filename, map_location=torch.device('cpu')))
            for ep in range(1, n_episodes + 1):
                state = env.reset()
                for t in range(max_timesteps):
                    action = ppo.select_action(state, memory)
                    state, reward, done, _ = env.step(action)
                    ep_reward += reward
                    if render:
                        env.render()
                    if save_gif:
                        img = env.render(mode='rgb_array')
                        img = Image.fromarray(img)
                        img.save('./gif/{}.jpg'.format(t))
                    if done:
                        break
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # if done:
            print('Episode: {}\tReward: {}'.format(ep, ep_reward / n_episodes))
            app_rew.append(ep_reward / n_episodes)
            env.close()
        avg_rew.append(app_rew)

print(avg_rew)
np.save(os.getcwd() + '/reacher_final_sim_results.npy', avg_rew)
labels = ["mb-1", "gb-1", "mb-2", "gb-2", "mb-10", "gb-10"]
_, ax1 = plt.subplots()
ax1.set_title('Evaluation of PPO in simulator (augmented)')
plt.xticks(range(1, len(labels) + 1), labels)
ax1.boxplot(avg_rew)
plt.show()
