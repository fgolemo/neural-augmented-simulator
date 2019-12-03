import os
import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm
import torch
from neural_augmented_simulator.common.envs.pusher_envs import PusherVanillaEnv
from neural_augmented_simulator.common.nas.models.networks import LstmNetRealv1
GOAL_REACHED_DISTANCE = 0.01
RESTART_EVERY_N_EPISODES = 1000


class PusherVanillaAugmentedEnv(PusherVanillaEnv):

    def __init__(self, headless=False, is_cuda=False):
        super(PusherVanillaAugmentedEnv, self).__init__(headless=headless)

        self.hidden_layers = 128
        self.lstm_layers = 3
        self.is_cuda = is_cuda
        self.model = LstmNetRealv1(
            n_input_state_sim=12,
            n_input_state_real=12,
            n_input_actions=3,
            nodes=self.hidden_layers,
            layers=self.lstm_layers)
        self.modified_obs = torch.zeros(1, 12).float()
        self.modified_actions = torch.zeros(1, 3).float()
        self.model_path = os.path.abspath('model-exp1-h128-l3-v{}-{}-{}e5.pth'.format(os.environ['variant'], os.environ['approach'],
                                                                       os.environ['noise_type']))
        print('------------------------------------------------------------')
        print('Model Path is : {}'.format(self.model_path))
        print('------------------------------------------------------------')

        if self.is_cuda:
            self.cuda_convert()
        self.load_model()
        # observation = 3 joints + 3 velocities + 2 puck position + 2 coordinates for target

    def cuda_convert(self):
        self.model = self.model.cuda()
        self.modified_obs = self.modified_obs.cuda()
        self.modified_actions = self.modified_actions.cuda()

    def load_model(self):
        return self.model.load_state_dict(torch.load(self.model_path)) \
            if self.is_cuda else self.model.load_state_dict(torch.load(self.model_path,  map_location='cpu'))

    def augment(self, last_obs, action, new_obs):
        last_obs = self.obs2lstm(last_obs)
        new_obs = self.obs2lstm(new_obs)
        self.modified_actions = action
        action = self.modified_actions.clone()

        input_tensor = torch.cat((last_obs, action, new_obs), 1).unsqueeze(0)
        with torch.no_grad():
            diff = self.model.forward(input_tensor)

        return diff.squeeze(0)

    def convert_to_tensor(self, numpy_array):
        return torch.FloatTensor(np.expand_dims(numpy_array, 0)).cuda() \
            if self.is_cuda else torch.FloatTensor(np.expand_dims(numpy_array, 0))

    def obs2lstm(self, obs):
        self.modified_obs[:, :8] = obs[:, :8]
        return self.modified_obs.clone()

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        obs = super()._get_obs()
        new_obs, _, _, _ = super().step(action)

        obs = self.convert_to_tensor(obs)
        # print('observation is {}: '.format(obs))
        action = self.convert_to_tensor(action)
        new_obs = self.convert_to_tensor(new_obs)
        obs_diff = self.augment(obs, action, new_obs)
        corrected_obs = new_obs[:, :6] + obs_diff[:, :6]
        new_obs[:, :6] = corrected_obs
        corrected_obs = corrected_obs.cpu().numpy().squeeze(0)
        new_obs = new_obs.cpu().numpy()
        self._set_state_(corrected_obs[:6])

        reward, done, dist = super()._getReward()

        return new_obs, reward, done, {"distance": dist}

    def reset(self, forced=False):
        self.model.zero_hidden()  # !important
        self.model.hidden = (self.model.hidden[0].detach(),
                             self.model.hidden[1].detach())
        return super().reset()


if __name__ == '__main__':
    import gym
    import gym_ergojr
    import time

    env = gym.make("ErgoPusherAugmented-Graphical-v1")
    MODE = "manual"
    r = range(100)

    # env = gym.make("ErgoPusher-Headless-v1")
    # MODE = "timings"
    # r = tqdm(range(10000))

    env.reset()

    timings = []
    ep_count = 0

    start = time.time()

    for _ in r:
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)

            if MODE == "manual":
                # print("act {}, obs {}, rew {}, done {}".format(
                #     action, obs, rew, done))

                time.sleep(0.01)

            if MODE == "timings":
                ep_count += 1
                if ep_count >= 10000:
                    diff = time.time() - start
                    tqdm.write("avg. fps: {}".format(
                        np.around(10000 / diff, 3)))
                    np.savez("timings.npz", time=np.around(10000 / diff, 3))
                    ep_count = 0
                    start = time.time()

            if done:
                env.reset()
                break
