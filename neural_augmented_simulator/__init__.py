from gym.envs.registration import register

register(
    id='Nas-Pusher-3dof-Vanilla-v1',
    entry_point='neural_augmented_simulator.common.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={})

register(
    id='Nas-Pusher-3dof-Backlash01-v1',
    entry_point='neural_augmented_simulator.common.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "backlash": "01"
    })

register(
    id='Nas-Pusher-3dof-Backlash01-v2',
    entry_point='neural_augmented_simulator.common.envs:PusherVanillaEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={
        "backlash": "01",
        "backlash_version": 2
    })
register(
    id='Nas-Pusher-3dof-Augmented-Vanilla-v1',
    entry_point='neural_augmented_simulator.common.envs:PusherVanillaAugmentedEnv',
    max_episode_steps=100,
    reward_threshold=0,
    kwargs={})

for headlessness in ["Graphical", "Headless"]:
    headlessness_switch = True if headlessness == "Headless" else False

    for terminates, name in [(True, ""), (False, "-Long")]:
        register(
            id='Nas-ErgoReacher-{}-MultiGoal-Halfdisk{}-v2'.format(headlessness, name),
            entry_point='neural_augmented_simulator.common.envs.ergo_reacher:ErgoReacherNewEnv',
            max_episode_steps=100,
            reward_threshold=0,
            kwargs={
                'headless': headlessness_switch,
                'simple': True,
                'goal_halfsphere': True,
                'multi_goal': True,
                'goals': 3,
                'terminates': terminates
            })

        register(
            id='Nas-ErgoReacherAugmented-{}-MultiGoal-Halfdisk{}-v2'.format(headlessness, name),
            entry_point='neural_augmented_simulator.common.envs.ergo_reacher_augmented:ErgoReacherAugmentedEnv',
            max_episode_steps=100,
            reward_threshold=0,
            kwargs={
                'headless': headlessness_switch,
                'simple': True,
                'goal_halfsphere': True,
                'multi_goal': True,
                'goals': 3,
                'terminates': terminates
            })
