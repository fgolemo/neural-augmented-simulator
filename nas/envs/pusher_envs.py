import gym
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
import numpy as np
from tkinter import *

npa = np.array
BACKLASHES = ["01"]

class PusherVanillaEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, backlash = None):

        utils.EzPickle.__init__(self)
        # if hasattr(self, "_kwargs") and 'colored' in self._kwargs and self._kwargs["colored"]:
        #     model_path = '3link_gripper_push_2d-colored.xml'
        # else:
        if backlash is None:
            model_path = '3link_gripper_push_2d.xml'
        else:
            assert backlash in BACKLASHES
            model_path = f'3link_gripper_push_2d_backlash-colored-new-b{backlash}.xml'
        full_model_path = os.path.join(
            os.path.dirname(__file__), "assets", model_path)
        mujoco_env.MujocoEnv.__init__(self, full_model_path, 5)

    def step(self, action):

        # vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        a = 3 * np.clip(
            npa(action), -1, 1
        )  # important - in the original env the range of actions is doubled

        # reward_near = - np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(action).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        # reward = reward_dist + 0.1 * reward_ctrl
        reward = reward_dist

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        img = None
        # img = self.render('rgb_array')
        # img = scipy.misc.imresize(img, (128, 128, 3))

        return ob, reward, done, dict(img=img)

    def viewer_setup(self):
        coords = [.7, -.5, 0]
        for i in range(3):
            self.viewer.cam.lookat[i] = coords[i]
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 2

    def reset_model(self):

        qpos = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        while True:
            self.cylinder_pos = [
                np.random.uniform(low=-1.0, high=-0.4),
                np.random.uniform(low=0.3, high=1.2)
            ]
            goal = [
                np.random.uniform(low=-1.2, high=-0.8),
                np.random.uniform(low=0.8, high=1.2)
            ]
            if np.linalg.norm(npa(self.cylinder_pos) - npa(goal)) > 0.45:
                break

        qpos[-4:-2] = npa(self.cylinder_pos)
        qpos[-2:] = npa(goal)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:-4],
            self.sim.data.qvel.flat[:-4],
            self.get_body_com("distal_4")[:2],
            self.get_body_com("object")[:2],
            self.get_body_com("goal")[:2],
        ])


class PusherCtrlGui:

    def __init__(self, env):
        self.env = env
        self.root = Tk()
        self.sliders = []

        b = Button(self.root, text="STEP", command=self.button_step)
        b.pack()

        b = Button(self.root, text="RESET", command=self.button_reset)
        b.pack()

        for i in range(3):
            slider = Scale(
                self.root,
                # variable=0.1,
                from_=-1,
                to=1,
                resolution=.1,
                orient="horizontal")
            self.sliders.append(slider)
            slider.pack()

        self.root.bind("<space>", self.key_step)

        self.root.mainloop()

    def key_step(self, event):
        self.button_step()

    def button_step(self):
        action = [s.get() for s in self.sliders]
        print("UI:", action)

        obs, rew, done, misc = self.env.step(action)
        print(np.around(obs, 2), rew)
        self.env.render()

    def button_reset(self):
        [s.set(0.0) for s in self.sliders]
        self.env.reset()
        self.env.render()


if __name__ == '__main__':
    import gym
    import nas

    # env = gym.make("Nas-Pusher-3dof-Vanilla-v1")
    env = gym.make("Nas-Pusher-3dof-Backlash01-v1")
    env.reset()
    # env.render()

    app = PusherCtrlGui(env)

    #
    # for i in range(100):
    #     env.render()
    #     action = env.action_space.sample()
    #     print(action)
    #     obs, reward, done, misc = env.step(action)
    #     print(obs, reward, done, misc)
    #     # time.sleep(1)
