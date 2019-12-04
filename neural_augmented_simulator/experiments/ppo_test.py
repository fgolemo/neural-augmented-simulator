import os
import gym
from neural_augmented_simulator.common.agent.ppo_agent import PPO
from neural_augmented_simulator.common.agent.actor_critic import Memory
from PIL import Image
from neural_augmented_simulator.arguments import get_args
import torch

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

n_episodes = 10  # num of episodes to run
max_timesteps = 1500  # max timesteps in one episode
render = True  # render the environment
save_gif = False  # png images are saved in gif folder

# filename and directory to load model from
filename = '/ppo_{}_{}_{}_{}.pth'.format(args.env_name,
                                         args.variant,
                                         args.approach,
                                         args.seed)

directory = os.getcwd() + \
    '/trained_models/ppo/{}/Variant-{}'.format(args.approach, args.variant)
print(filename)
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
ppo.policy_old.load_state_dict(torch.load(
    directory + filename, map_location=torch.device('cpu')))

for ep in range(1, n_episodes + 1):
    ep_reward = 0
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

    print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    ep_reward = 0
env.close()
