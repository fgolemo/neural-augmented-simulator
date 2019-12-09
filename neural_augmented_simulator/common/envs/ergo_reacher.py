import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from gym_ergojr.sim.objects import Ball
from gym_ergojr.envs.ergo_reacher_env import ErgoReacherEnv
import neural_augmented_simulator

GOAL_REACHED_DISTANCE = -0.016  # distance between robot tip and goal under which the task is considered solved
RADIUS = 0.2022
DIA = 2 * RADIUS
RESET_EVERY = 5  # for the gripper




class ErgoReacherNewEnv(ErgoReacherEnv):

    def __init__(self,
                 headless=False,
                 simple=False,
                 backlash=False,
                 max_force=1,
                 max_vel=18,
                 goal_halfsphere=False,
                 multi_goal=False,
                 goals=3,
                 terminates=True,
                 gripper=False,
                 is_cuda=False):

        super(ErgoReacherNewEnv, self).__init__(
            headless=headless,
            simple=simple,
            backlash=backlash,
            max_force=max_force,
            max_vel=max_vel,
            goal_halfsphere=goal_halfsphere,
            multi_goal=multi_goal,
            goals=goals,
            terminates=terminates,
            gripper=gripper)
        self.goal = None

    def reset(self, forced=False):
        return super().reset()

    def render(self, mode='human', close=False):
        super().render()

    def get_tip(self):
        return self.robot.get_tip()

    def get_goal_pos(self):
        return self.goal


if __name__ == '__main__':
    import gym
    import gym_ergojr
    import time
    import neural_augmented_simulator

    # MODE = "manual"
    env = gym.make("Nas-ErgoReacher-Graphical-MultiGoal-Halfdisk-Long-v2")

    MODE = "manual"
    # env = gym.make("ErgoReacher-Graphical-Simple-Halfdisk-v1")
    # env = gym.make("ErgoReacher-Graphical-Gripper-MobileGoal-v1")

    env.reset()

    timings = []
    ep_count = 0

    start = time.time()

    if MODE == "manual":
        r = range(100)
    else:
        r = tqdm(range(10000))

    for _ in r:
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)
            # obs, rew, done, misc = env.step([17/90,-29/90,-33/90,-61/90])

            if MODE == "manual":
                # print("act {}, obs {}, rew {}, done {}".format(
                #     action, obs, rew, done))
                print(env.unwrapped.get_goal_pos())
                time.sleep(0.01)

            if MODE == "timings":
                ep_count += 1
                if ep_count >= 10000:
                    diff = time.time() - start
                    print("avg. fps: {}".format(np.around(10000 / diff, 3)))
                    np.savez("timings.npz", time=np.around(10000 / diff, 3))
                    ep_count = 0
                    start = time.time()

            if done:
                env.reset()
                # env.unwrapped.goal = np.array([0., 0.01266761, 0.21479595])
                # env.unwrapped.dist.goal = np.array([0., 0.01266761, 0.21479595])
                # env.unwrapped.ball.changePos(env.unwrapped.goal, 4)
                break
